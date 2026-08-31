#!/usr/bin/env python3
"""overlap_scan.py —— 图文叠印检测。

扫描 PDF 每页：图片矩形 与 文本 span 矩形求交，交叠面积超过阈值即报告。
浮动图的双语题注紧贴图片下缘属正常（间距 >0 不相交）；真正叠印时
span 会落进图片矩形内部。另检测 表格线(横线) 与文本 span 的穿越
（线 y 在 span y0+2..y1-2 内且 x 区间重叠 >30%，典型为三线表规则线
横穿题注/正文文字）。

用法： python3 overlap_scan.py 文件.pdf [--min-area 4]
退出码：0 = 无叠印，1 = 检出叠印。
"""
import sys

import fitz


def scan(path, min_area=4.0):
    doc = fitz.open(path)
    hits = 0
    for pno in range(len(doc)):
        page = doc[pno]
        spans = []
        for b in page.get_text('dict')['blocks']:
            for line in b.get('lines', ()):
                for sp in line.get('spans', ()):
                    txt = sp['text'].strip()
                    if txt:
                        spans.append((fitz.Rect(sp['bbox']), txt))
        # 1) 图片 × 文本
        img_rects = []
        for info in page.get_image_info():
            img_rects.append(fitz.Rect(info['bbox']))
        for ir in img_rects:
            for sr, txt in spans:
                inter = ir & sr
                if not inter.is_empty and inter.get_area() >= min_area \
                        and sr.y0 >= ir.y0 - 1 and sr.y1 <= ir.y1 + 1:
                    hits += 1
                    print(f'第{pno + 1}页 图文叠印: 图{tuple(round(v, 1) for v in ir)} '
                          f'压住文本 "{txt[:40]}" @y={round(sr.y0, 1)}')
        # 1b) 图片 × 图片（两个浮动块叠置，LibreOffice 不执行 tblOverlap 互斥）
        for i in range(len(img_rects)):
            for j in range(i + 1, len(img_rects)):
                inter = img_rects[i] & img_rects[j]
                if not inter.is_empty and inter.get_area() >= 100:
                    hits += 1
                    print(f'第{pno + 1}页 图图叠印: '
                          f'{tuple(round(v, 1) for v in img_rects[i])} × '
                          f'{tuple(round(v, 1) for v in img_rects[j])} '
                          f'交叠 {round(inter.get_area(), 0)} pt²')
        # 2) 横线 × 文本（线穿字）
        for d in page.get_drawings():
            r = d['rect']
            if r.height > 1.5:      # 只看水平细线
                continue
            for sr, txt in spans:
                if sr.y0 + 2.5 < r.y0 < sr.y1 - 2.5:
                    ox = min(r.x1, sr.x1) - max(r.x0, sr.x0)
                    if ox > 0.3 * (sr.x1 - sr.x0) and ox > 6:
                        hits += 1
                        print(f'第{pno + 1}页 线穿文字: 线y={round(r.y0, 1)} '
                              f'x=[{round(r.x0, 1)},{round(r.x1, 1)}] 穿过 '
                              f'"{txt[:40]}" @[{round(sr.y0, 1)},{round(sr.y1, 1)}]')
    print(f'—— {path}: 共 {hits} 处疑似叠印')
    return 1 if hits else 0


if __name__ == '__main__':
    pdf = sys.argv[1]
    ma = 4.0
    if '--min-area' in sys.argv:
        ma = float(sys.argv[sys.argv.index('--min-area') + 1])
    sys.exit(scan(pdf, ma))
