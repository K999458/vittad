#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文字问题扫描：英文大小写 / 中英文标点混用 / 多余空格 / 公式表示。

用法： python3 text_issue_scan.py ../GCAS_光学精密工程_排版初稿_v12_20260830.docx
"""
import re
import sys
import zipfile

from lxml import etree

DOCX = sys.argv[1]
NS = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
}


def q(tag):
    p, t = tag.split(':')
    return '{%s}%s' % (NS[p], t)


z = zipfile.ZipFile(DOCX)
doc = etree.fromstring(z.read('word/document.xml'))
body = doc.find(q('w:body'))


def para_text(p):
    out = []
    for r in p.iter(q('w:r')):
        for t in r.iter():
            if t.tag == q('w:t'):
                out.append(t.text or '')
            elif t.tag == q('w:tab'):
                out.append('\t')
    return ''.join(out)


# 收集全部段落文本（含表格内），并记录所在位置
items = []          # (where, text)
for i, p in enumerate(body.findall(q('w:p'))):
    items.append((f'p[{i}]', para_text(p)))
for ti, tb in enumerate(body.findall(q('w:tbl'))):
    for p in tb.iter(q('w:p')):
        items.append((f'tbl[{ti}]', para_text(p)))

CJK = r'\u4e00-\u9fff\u3400-\u4dbf'
issues = {}


def add(cat, where, text, mark):
    issues.setdefault(cat, []).append((where, mark, text))


for where, txt in items:
    if not txt.strip():
        continue
    # 1. 多空格（两个及以上连续半角空格；全角空格\u3000 用于"摘 要"排版除外）
    for m in re.finditer(r'[^\S\u3000]{2,}', txt):
        add('A1 连续多空格', where, txt, repr(m.group(0)))
    # 2. 中文字符之间夹半角空格（排除中英文之间的合法空格）
    for m in re.finditer(f'[{CJK}] +[{CJK}]', txt):
        add('A2 中文间空格', where, txt, m.group(0))
    # 3. 全角标点前后空格
    for m in re.finditer(r' [，。；：、）】》"]|[，。；：、（【《"] ', txt):
        add('A3 全角标点旁空格', where, txt, repr(m.group(0)))
    # 4. 中文语境中的半角逗号/分号/冒号/句号（数字小数点、URL、英文串除外）
    for m in re.finditer(f'[{CJK}][,;:.](?![0-9])', txt):
        add('B1 中文后接半角标点', where, txt, m.group(0))
    for m in re.finditer(f'[,;:][{CJK}]', txt):
        add('B2 半角标点后接中文', where, txt, m.group(0))
    # 5. 英文语境中的全角标点（英文字母/数字紧邻全角逗号句号等）
    for m in re.finditer(r'[A-Za-z][，。；：、]', txt):
        add('B3 英文后接全角标点', where, txt, m.group(0))
    # 6. 半角括号内含中文 / 全角括号内全英文
    for m in re.finditer(f'\\([^()]*[{CJK}][^()]*\\)', txt):
        add('B4 半角括号含中文', where, txt, m.group(0)[:40])
    for m in re.finditer(r'（[A-Za-z0-9 ,.\-+%/=×±·；;:]*）', txt):
        add('B5 全角括号纯西文', where, txt, m.group(0)[:40])
    # 7. 空格紧邻半角标点（英文规范："x ," / "x ."）
    for m in re.finditer(r'\s[,.;:](?=\s|$)', txt):
        add('A4 半角标点前空格', where, txt, repr(m.group(0)))
    # 8. 英文句子内可疑大小写：句号+空格后接小写字母（英文摘要等）
    for m in re.finditer(r'[a-z][.!?] +[a-z]', txt):
        add('C1 句后小写', where, txt, m.group(0))

print(f'>>> 共扫描片段 {len(items)}')
for cat in sorted(issues):
    lst = issues[cat]
    print(f'\n==== {cat}（{len(lst)} 处）====')
    for where, mark, text in lst[:40]:
        t = text if len(text) <= 90 else text[:60] + ' … ' + text[-25:]
        print(f'  {where} 【{mark}】 {t!r}')

# 术语大小写一致性统计
print('\n==== C2 术语大小写统计 ====')
alltext = '\n'.join(t for _, t in items)
for pat in [r'\bkNN\b|\bKNN\b|\bknn\b|\bk-NN\b|\bk_nn\b',
            r'\bsoftmax\b|\bSoftmax\b|\bSoftMax\b',
            r'\bfitness\b|\bFitness\b',
            r'\btop-k\b|\bTop-k\b|\bTop-K\b|\btop-K\b',
            r'\bsink\b|\bSink\b',
            r'AUROC|AUC\b|auroc',
            r'\bAP\b|\bap\b',
            r'Fig\.\s?\d|Fig\s\d|图\s?\d',
            r'Tab\.\s?\d|Table\s\d|表\s?\d']:
    from collections import Counter
    c = Counter(re.findall(pat, alltext))
    if len(c) > 1:
        print(f'  {pat} -> {dict(c)}')

# 公式段检查
print('\n==== D 公式段 ====')
for i, p in enumerate(body.findall(q('w:p'))):
    txt = para_text(p)
    if re.search(r'\t\(\d+\)\s*$', txt):
        om = len(p.findall('.//' + q('m:oMath')))
        print(f'  p[{i}] oMath={om} | {txt!r}')
