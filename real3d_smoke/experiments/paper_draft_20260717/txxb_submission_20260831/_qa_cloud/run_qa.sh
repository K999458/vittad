#!/bin/bash
# run_qa.sh —— 图学学报排版稿一键 QA（云端副本）
# 用法： bash run_qa.sh ../GCAS_图学学报_排版初稿_xxx.docx
# 步骤： node1 soffice 渲染 PDF → TXXB 几何空白带扫描 → 逐段文字完整性校验
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
DOCX="$(readlink -f "$1")"
TAG="${2:?用法: run_qa.sh 稿.docx ascii标签(渲染目录名)}"
# node1 的 ssh 非交互 shell + soffice 对含中文的 cwd 会报
# "source file could not be loaded"，渲染目录与文件名必须纯 ASCII
WORK="$HERE/render_${TAG}"
mkdir -p "$WORK"
cp -f "$DOCX" "$WORK/${TAG}.docx"
echo "== 渲染（node1 soffice，独立 profile 防并发锁冲突）=="
ssh -o BatchMode=yes node1 "cd '$WORK' && soffice -env:UserInstallation=file:///tmp/lo_profile_cloud_qa --headless --convert-to pdf '${TAG}.docx' --outdir . >/dev/null 2>&1; ls -la *.pdf"
PDF="$WORK/${TAG}.pdf"
[ -f "$PDF" ] || { echo "渲染失败：无 PDF"; exit 2; }
echo "== 空白带扫描（阈值 3.0 cm）=="
python3 "$HERE/whitespace_scan_txxb.py" "$PDF" 3.0
WS=$?
echo "== 文字完整性校验 =="
python3 "$HERE/../../jig_submission_20260831/_qa/text_integrity_check.py" "$DOCX" "$PDF"
TI=$?
echo "== 汇总 == 空白扫描 exit=$WS（0=无超标），完整性 exit=$TI（0=零丢字）"
exit $(( WS || TI ))
