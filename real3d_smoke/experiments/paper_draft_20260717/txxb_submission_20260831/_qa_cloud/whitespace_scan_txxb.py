#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""whitespace_scan_txxb.py —— 图学学报版式逐页空白带检测（云端 QA 副本）

自 oep_submission_20260827/_qa/whitespace_scan.py 移植，页面几何改为
图学学报投稿模板 sectPr 实测值：A4，上 1588 / 下 963 / 左右 1134 twip，
双栏栏宽 4600 twip、栏间距 438 twip。

用法： python3 whitespace_scan_txxb.py 渲染稿.pdf [阈值cm，默认3.0]
"""
import sys

import fitz
import numpy as np

MARGIN_T_IN = 1588 / 1440.0
MARGIN_B_IN = 963 / 1440.0
MARGIN_LR_IN = 1134 / 1440.0
COL_W_IN = 4600 / 1440.0
COL_GAP_IN = 438 / 1440.0


def _max_band(blank):
    best_len = cur = 0
    best_end = -1
    for j, v in enumerate(blank):
        cur = cur + 1 if v else 0
        if cur > best_len:
            best_len, best_end = cur, j
    return best_len, best_end


def scan(pdf_path, thresh_cm=3.0, dpi=100):
    doc = fitz.open(pdf_path)
    n = len(doc)
    flagged = []
    print(f'{pdf_path}: {n} 页, 空白带阈值 {thresh_cm} cm'
          f'（整幅/左栏/右栏三路，TXXB 几何）')
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width)
        l = int(MARGIN_LR_IN * dpi)
        t = int(MARGIN_T_IN * dpi)
        b = int(MARGIN_B_IN * dpi)
        area = arr[t:pix.height - b, l:pix.width - l]
        colw = int(COL_W_IN * dpi)
        gap = int(COL_GAP_IN * dpi)
        bands = {
            '整幅': (area > 250).all(axis=1),
            '左栏': (area[:, :colw] > 250).all(axis=1),
            '右栏': (area[:, colw + gap:] > 250).all(axis=1),
        }
        page_flag = []
        msgs = []
        for name, blank in bands.items():
            blen, bend = _max_band(blank)
            cm = blen / dpi * 2.54
            tail = ''
            if i == n - 1 and bend >= len(blank) - 2:
                tail = '(末页结束不计)'
            msgs.append(f'{name} {cm:5.2f} cm{tail}')
            if cm > thresh_cm and not tail:
                page_flag.append((name, round(cm, 2)))
        print(f'  第{i + 1:2d}页  ' + '  '.join(msgs))
        if page_flag:
            flagged.append((i + 1, page_flag))
    print('—— 超阈值页：', flagged if flagged else '无')
    return flagged


if __name__ == '__main__':
    path = sys.argv[1]
    th = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    sys.exit(1 if scan(path, th) else 0)
