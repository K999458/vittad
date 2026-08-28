"""统一中文字体设置。import 本模块即可让 matplotlib 正常显示中文。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

PREFER = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Micro Hei",
    "Noto Serif CJK SC", "Noto Serif CJK JP", "Droid Sans Fallback",
]

for p in ("/store/zkyang/.fonts", "/usr/share/fonts"):
    try:
        for f in fm.findSystemFonts(p):
            try:
                fm.fontManager.addfont(f)
            except Exception:
                pass
    except Exception:
        pass

have = {f.name for f in fm.fontManager.ttflist}
CHOSEN = next((n for n in PREFER if n in have), None)
if CHOSEN:
    plt.rcParams["font.sans-serif"] = [CHOSEN] + PREFER + ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
