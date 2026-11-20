"""
Clinical validation of domain adaptation: does the DA collagen model (M) agree
with the pathologist better than the source-only model?

  python src/analyses/v10_source_only_vs_DA/source_only_vs_da.py            # extract + stats
  python src/analyses/v10_source_only_vs_DA/source_only_vs_da.py --stats    # stats only (cached tiles)

Inputs
  DA          data/T0_T1_v5_M_values/tiles/<slide>_Tile_Collagen_Features.csv
              (1000 px tiles, extractor v5, from M.svs on HiPerGator)
  source-only E:/.../Pred_CGPL_source_only/Testing_Output/ds1/<slide>__x<X>_y<Y>.jpg
              (3000 px tiles, sigmoid x 255), H&E input tiles in Tiles_3000/B
  pathology   data/analysis_ready_clinical.csv: ci (Banff), if_pct (IF%)

Same tiles for both models: each 3000 px source-only tile is split into its
nine 1000 px sub-tiles, which are exactly the DA grid; only DA tiles whose
3000 px parent was predicted are used, for BOTH models. Only slides with every
tile predicted are analysed.

Source-only features follow the v5 extractor rules (FeatureExtractor_wsi_v5):
  tissue   H&E saturation > slide Otsu threshold, at 1/4 resolution, closing
           disk(1), objects < 2000 px dropped, holes < 25 um^2 filled
           (lumen_mode "exclude"); union the collagen mask
  collagen M > 127 within 20 um of tissue
  m_density      sum M / tissue area over all tiles
  m_mean_iqr     IQR across tiles (Tissue Fraction >= 0.5) of per-tile mean M
  collagen_ratio collagen area / tissue area, all tiles
Two differences from DA, both reported: the slide Otsu threshold is taken on
the slide's 3000 px tiles (DA: 2048 px thumbnail), and glomeruli are not
masked out (DA uses CE = 0; CE is not available locally). The DA features are
recomputed from the DA tile CSVs on the same tile subset.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_closing
from scipy.stats import spearmanr
from skimage.filters import threshold_otsu
from skimage.morphology import disk, remove_small_holes, remove_small_objects

E = Path("E:/f.afsari/4-DUET/Aim3")
PRED = E / "Pred_CGPL_source_only/Testing_Output/ds1"
B_DIR = E / "Tiles_3000/B"
TILES = E / "Tiles_3000/tiles.csv"
DA_DIR = Path("data/T0_T1_v5_M_values/tiles")
CLIN = Path("data/analysis_ready_clinical.csv")
OUT = Path("results/v10_source_only_vs_DA")
CACHE = OUT / "source_only_tiles"
T, DS, MPP = 1000, 4, 0.5
MIN_OBJ, SPECKLE_UM2, NEAR_UM = 2000, 25, 20
FEATURES = [("m_mean_iqr", "M variability (tile to tile)"),
            ("m_density", "M density (mean M over tissue)"),
            ("collagen_ratio", "collagen ratio, M > 0.5 mask")]
N_BOOT = 5000


def slide_threshold(files):
    """Otsu on H&E saturation over all of the slide's tiles at 1/16 resolution."""
    sat = []
    for f in files:
        b = np.array(Image.open(B_DIR / f).convert("RGB"))
        small = cv2.resize(b, (b.shape[1] // 16, b.shape[0] // 16), interpolation=cv2.INTER_AREA)
        sat.append(cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[..., 1].ravel())
    return float(threshold_otsu(np.concatenate(sat)))


def stained_mask(b, thr):
    h, w = b.shape[:2]
    small = cv2.resize(b, (max(1, w // DS), max(1, h // DS)), interpolation=cv2.INTER_AREA)
    m = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[..., 1] > thr
    m = binary_closing(m, disk(1))
    m = remove_small_objects(m, min_size=max(1, MIN_OBJ // (DS * DS)))
    m = remove_small_holes(m, area_threshold=max(1, int(SPECKLE_UM2 / (MPP * DS) ** 2)))
    return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def near(tissue):
    r = max(1, int(round(NEAR_UM / MPP)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(tissue.astype(np.uint8), k).astype(bool)


def process_slide(args):
    slide, tiles = args
    out = CACHE / f"{slide}.csv"
    if out.exists():
        return slide, "cached"
    thr = slide_threshold(tiles["file"])
    rows = []
    for _, t in tiles.iterrows():
        b = np.array(Image.open(B_DIR / t["file"]).convert("RGB"))
        m = np.array(Image.open(PRED / t["file"].replace(".png", ".jpg")).convert("L"))
        for i in range(3):
            for j in range(3):
                sl = np.s_[j * T:(j + 1) * T, i * T:(i + 1) * T]
                bt, mt = b[sl], m[sl]
                if bt.shape[:2] != (T, T):
                    continue
                stained = stained_mask(bt, thr)
                coll = (mt > 127) & near(stained)
                tissue = stained | coll
                p = mt.astype(np.float64) / 255.0
                rows.append({"Tile X": int(t["x"]) + i * T, "Tile Y": int(t["y"]) + j * T,
                             "Tissue Area": int(tissue.sum()), "Collagen Area": int(coll.sum()),
                             "M Sum": float(p[tissue].sum()),
                             "M Mean": float(p[tissue].mean()) if tissue.any() else np.nan,
                             "sat_threshold": thr})
    pd.DataFrame(rows).to_csv(out, index=False)
    return slide, f"{len(rows)} sub-tiles"


def slide_features(d):
    """d: per-tile rows with Tissue Area, Collagen Area, M Sum, M Mean, Tissue Fraction."""
    d = d[d["Tissue Area"] > 0]
    interior = d[d["Tissue Fraction"] >= 0.5]
    dist = d if len(interior) < 5 else interior
    v = dist["M Mean"].dropna()
    return {"m_density": d["M Sum"].sum() / d["Tissue Area"].sum(),
            "m_mean_iqr": v.quantile(0.75) - v.quantile(0.25),
            "collagen_ratio": d["Collagen Area"].sum() / d["Tissue Area"].sum(),
            "n_tiles": len(d)}


def boot_diff(x_da, x_src, y, rng):
    """Paired bootstrap over patients of rho(DA) - rho(source-only)."""
    n, diffs = len(y), []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if np.unique(y[i]).size < 2:
            continue
        diffs.append(spearmanr(x_da[i], y[i])[0] - spearmanr(x_src[i], y[i])[0])
    diffs = np.array(diffs)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return lo, hi, min(p, 1.0)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    tiles = pd.read_csv(TILES)
    done = {p.stem for p in PRED.glob("*.jpg")}
    tiles["done"] = tiles["file"].str.replace(".png", "", regex=False).isin(done)
    cov = tiles.groupby("slide")["done"].agg(["sum", "size"])
    complete = sorted(cov.index[cov["sum"] == cov["size"]])
    print(f"[INFO] {len(complete)} of {len(cov)} slides fully predicted "
          f"({tiles['done'].sum()} of {len(tiles)} tiles)")

    if "--stats" not in sys.argv:
        jobs = [(s, tiles[tiles["slide"] == s]) for s in complete]
        with ProcessPoolExecutor(max_workers=6) as ex:
            for slide, msg in ex.map(process_slide, jobs):
                print(f"[INFO] {slide}: {msg}", flush=True)

    rows, mask_check = [], []
    for s in complete:
        if not (CACHE / f"{s}.csv").exists():
            continue
        da = pd.read_csv(DA_DIR / f"{s}_Tile_Collagen_Features.csv")
        src = pd.read_csv(CACHE / f"{s}.csv")
        key = ["Tile X", "Tile Y"]
        m = da.merge(src, on=key, suffixes=("_da", "_src"))
        lost = len(da) - len(m)
        d_da = m[[c + "_da" for c in ["Tissue Area", "Collagen Area", "M Sum", "M Mean"]]].copy()
        d_da.columns = ["Tissue Area", "Collagen Area", "M Sum", "M Mean"]
        d_src = m[[c + "_src" for c in ["Tissue Area", "Collagen Area", "M Sum", "M Mean"]]].copy()
        d_src.columns = ["Tissue Area", "Collagen Area", "M Sum", "M Mean"]
        d_da["Tissue Fraction"] = d_src["Tissue Fraction"] = m["Tissue Fraction"]  # DA's: same tiles
        f_da, f_src = slide_features(d_da), slide_features(d_src)
        # DA density as defined in v5 (glomeruli excluded), for reference
        f_da["m_density"] = m["M Sum_da"].sum() / m["M Valid Area"].sum()
        rows.append({"slide_name": s, **{f"{k}_da": v for k, v in f_da.items()},
                     **{f"{k}_src": v for k, v in f_src.items()}, "da_tiles_not_predicted": lost})
        mask_check.append({"slide_name": s,
                           "tissue_area_ratio_src_over_da": m["Tissue Area_src"].sum() / m["Tissue Area_da"].sum(),
                           "tile_tissue_rho": spearmanr(m["Tissue Area_src"], m["Tissue Area_da"])[0],
                           "tile_M_mean_rho": spearmanr(m["M Mean_src"], m["M Mean_da"], nan_policy="omit")[0]})
    feat = pd.DataFrame(rows)
    mc = pd.DataFrame(mask_check)
    clin = pd.read_csv(CLIN).rename(columns={"Slide Name": "slide_name"})[["slide_name", "ci", "if_pct"]]
    clin["slide_name"] = clin["slide_name"].replace({"99H2-0": "99H-0"})  # tiles/predictions use the old name
    df = feat.merge(clin, on="slide_name", how="left")
    df.to_csv(OUT / "slide_features_da_vs_source_only.csv", index=False)
    mc.to_csv(OUT / "mask_check.csv", index=False)

    rng = np.random.default_rng(0)
    res = []
    for ref, ref_label in [("ci", "Banff ci"), ("if_pct", "IF%")]:
        for f, label in FEATURES:
            d = df.dropna(subset=[ref, f + "_da", f + "_src"])
            y, a, b = d[ref].to_numpy(), d[f + "_da"].to_numpy(), d[f + "_src"].to_numpy()
            r_da, p_da = spearmanr(a, y)
            r_src, p_src = spearmanr(b, y)
            lo, hi, p_diff = boot_diff(a, b, y, rng)
            res.append({"reference": ref_label, "feature": label, "n": len(d),
                        "rho_DA": r_da, "p_DA": p_da, "rho_source_only": r_src, "p_source_only": p_src,
                        "rho_diff": r_da - r_src, "diff_ci_lo": lo, "diff_ci_hi": hi, "p_diff_boot": p_diff,
                        "rho_DA_vs_source_feature": spearmanr(a, b)[0]})
    res = pd.DataFrame(res)
    res.to_csv(OUT / "rho_da_vs_source_only.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n== mask check (source-only tissue mask vs DA, same tiles)")
    print(mc.describe().loc[["mean", "50%", "min", "max"]].round(3).to_string())
    print("\n== slide features, median (DA vs source-only)")
    print(df[[c for c in df.columns if c.endswith(("_da", "_src"))]].median().round(4).to_string())
    print("\n== Spearman with pathology")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
