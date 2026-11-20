"""Side-by-side example tiles, DA vs source-only collagen prediction.
  python src/analyses/v10_source_only_vs_DA/make_tile_figure.py

Example 1 (by rule): interior ci 0 tile with tissue >= 99% whose per-tile collagen
  ratio of both models is closest to the cohort medians (DA 19.6%, source-only 54.1%).
Example 2 (tubular region): of the 12 interior (tissue >= 95%) ci 0 tiles closest to
  the medians, the one chosen by eye for clear tubular cross-sections and no glomerulus.
Writes results/v10_source_only_vs_DA/figure/<tag>_{he,da_overlay,source_only_overlay,da_M,source_only_M}.png
"""
import numpy as np, tifffile, zarr
from PIL import Image

EXAMPLES = [("ex1", "133H-0", 17000, 28000), ("ex2_tubular", "110H-0", 67000, 58000)]
T = 1000
OUT = "results/v10_source_only_vs_DA/figure"


def make(tag, slide, x, y):
    x3, y3 = x // 3000 * 3000, y // 3000 * 3000
    svs = f"F:/Davy_Jones_Locker/Farzad_Fibrosis/aim3/images/{slide}/"

    def region(name):
        z = zarr.open(tifffile.imread(svs + name, aszarr=True, level=0), mode="r")
        return np.asarray(z[y:y + T, x:x + T])

    m_da = region("M.svs")[..., 0]
    b_svs = region("B.svs")
    name = f"{slide}__x{x3}_y{y3}"
    sl = np.s_[y - y3:y - y3 + T, x - x3:x - x3 + T]
    b = np.array(Image.open(f"E:/f.afsari/4-DUET/Aim3/Tiles_3000/B/{name}.png").convert("RGB"))[sl]
    m_src = np.array(Image.open(f"E:/f.afsari/4-DUET/Aim3/Pred_CGPL_source_only/Testing_Output/ds1/{name}.jpg")
                     .convert("L"))[sl]
    print(f"{tag} {slide} x{x} y{y}: B tile vs B.svs mean abs diff "
          f"{np.abs(b.astype(int) - b_svs.astype(int)).mean():.2f}; pixels M>127: "
          f"DA {100 * (m_da > 127).mean():.1f}%  source-only {100 * (m_src > 127).mean():.1f}%")

    def overlay(m):
        o = b.astype(float).copy()
        k = m > 127
        o[k] = 0.4 * o[k] + 0.6 * np.array([0, 190, 170])
        return Image.fromarray(o.astype(np.uint8))

    Image.fromarray(b).save(f"{OUT}/{tag}_he.png")
    overlay(m_da).save(f"{OUT}/{tag}_da_overlay.png")
    overlay(m_src).save(f"{OUT}/{tag}_source_only_overlay.png")
    Image.fromarray(m_da).save(f"{OUT}/{tag}_da_M.png")
    Image.fromarray(m_src).save(f"{OUT}/{tag}_source_only_M.png")


for ex in EXAMPLES:
    make(*ex)
