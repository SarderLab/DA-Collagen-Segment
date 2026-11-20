"""Collagen morphometry per biopsy, on the same 1000 px (500 um) tiles and collagen mask as extraction v5.
  python src/analyses/v11_morphometry/extract_morphometry.py [--only 133H-0] [--jobs 8]

Collagen mask per tile = M > 127 and CE > 0 (v5 rule, without the 20 um near-tissue step: per-tile area within ~1%
of v5's Collagen Area). Regions < 20 um^2 dropped. Tissue area per tile is taken from the v5 tile CSV.
Interior tiles only (v5 Tissue Fraction >= 0.5) with >= 1% collagen.

Per tile:
  region_density       collagen regions per mm^2 tissue
  region_area_mean     mean region area (um^2); region_area_p90 = 90th percentile
  eccentricity         area-weighted mean region eccentricity (0 round, 1 line)
  solidity             area-weighted mean region solidity (area / convex hull area)
  major_axis_p50       median region major axis length (um)
  perimeter_area       total perimeter / total collagen area (1/um): fine, ragged collagen is high
  fibre_length_density skeleton length per tissue area (mm / mm^2)
  branch_density       skeleton branch points per mm^2 tissue
  orientation_coh      structure-tensor coherence of M inside the mask (0 random, 1 aligned)
  lacunarity_16um / _64um  gappiness of the mask at 16 and 64 um boxes
  nnd_mean             mean nearest-neighbour distance between region centroids (um)
Per slide: collagen-area-weighted mean over tiles (_mean) and tile-to-tile IQR (_iqr).
Writes results/v11_morphometry/tiles/<slide>.csv and results/v11_morphometry/morphometry_features.csv
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tifffile
import zarr
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure, morphology
from skimage.feature import structure_tensor

IMG = "F:/Davy_Jones_Locker/Farzad_Fibrosis/aim3/images"
TILES = "data/T0_T1_v5_M_values/tiles"
OUT = "results/v11_morphometry"
T, MPP = 1000, 0.5
MIN_PX = int(20 / MPP ** 2)  # 20 um^2
FEATS = ["region_density", "region_area_mean", "region_area_p90", "eccentricity", "solidity", "major_axis_p50",
         "perimeter_area", "fibre_length_density", "branch_density", "orientation_coh", "lacunarity_16um",
         "lacunarity_64um", "nnd_mean"]


def lacunarity(mask, box):
    h, w = (mask.shape[0] // box) * box, (mask.shape[1] // box) * box
    b = mask[:h, :w].reshape(h // box, box, w // box, box).sum(axis=(1, 3)).astype(float)
    return float(b.var() / b.mean() ** 2 + 1) if b.mean() > 0 else np.nan


def tile_features(m, ce, tissue_px):
    mask = morphology.remove_small_objects((m > 127) & (ce > 0), MIN_PX)
    area = mask.sum()
    tissue_mm2 = tissue_px * MPP ** 2 / 1e6
    f = {"collagen_px": int(area)}
    if area == 0:
        return f
    lab = measure.label(mask, connectivity=2)
    props = measure.regionprops(lab)
    a = np.array([p.area for p in props], float)
    w = a / a.sum()
    f["region_density"] = len(props) / tissue_mm2
    f["region_area_mean"] = a.mean() * MPP ** 2
    f["region_area_p90"] = np.percentile(a, 90) * MPP ** 2
    f["eccentricity"] = float(np.sum(w * [p.eccentricity for p in props]))
    f["solidity"] = float(np.sum(w * [p.solidity for p in props]))
    f["major_axis_p50"] = float(np.median([p.major_axis_length for p in props])) * MPP
    f["perimeter_area"] = sum(p.perimeter for p in props) / a.sum() / MPP
    sk = morphology.skeletonize(mask)
    f["fibre_length_density"] = sk.sum() * MPP / 1000 / tissue_mm2
    nb = ndimage.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant")
    f["branch_density"] = int(((nb >= 4) & sk).sum()) / tissue_mm2
    a_rr, a_rc, a_cc = structure_tensor(m.astype(float) / 255, sigma=2, order="rc")
    coh = np.sqrt((a_rr - a_cc) ** 2 + 4 * a_rc ** 2) / (a_rr + a_cc + 1e-9)
    f["orientation_coh"] = float(coh[mask].mean())
    f["lacunarity_16um"] = lacunarity(mask, int(16 / MPP))
    f["lacunarity_64um"] = lacunarity(mask, int(64 / MPP))
    if len(props) >= 2:
        c = np.array([p.centroid for p in props])
        d, _ = cKDTree(c).query(c, k=2)
        f["nnd_mean"] = float(d[:, 1].mean()) * MPP
    return f


def iqr(x):
    x = x.dropna()
    return float(x.quantile(0.75) - x.quantile(0.25)) if len(x) >= 3 else np.nan


def slide(name):
    out = f"{OUT}/tiles/{name}.csv"
    if os.path.exists(out):
        return pd.read_csv(out)
    t = pd.read_csv(f"{TILES}/{name}_Tile_Collagen_Features.csv")
    t = t[(t["Tissue Fraction"] >= 0.5) & (t["Collagen Area Ratio"] >= 0.01)]
    z = {k: zarr.open(tifffile.TiffFile(f"{IMG}/{name}/{k}.svs").series[0].aszarr(level=0), mode="r")
         for k in ["M", "CE"]}
    rows = []
    for _, r in t.iterrows():
        x, y = int(r["Tile X"]), int(r["Tile Y"])
        m, ce = (np.asarray(z[k][y:y + T, x:x + T]) for k in ["M", "CE"])
        m, ce = (v[..., 0] if v.ndim == 3 else v for v in (m, ce))
        f = tile_features(m, ce, r["Tissue Area"])
        f.update({"slide": name, "x": x, "y": y, "tissue_px": r["Tissue Area"], "v5_collagen_px": r["Collagen Area"]})
        rows.append(f)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(name, len(df), "tiles", flush=True)
    return df


def aggregate(df):
    w = df["collagen_px"]
    row = {"slide": df["slide"].iloc[0], "n_tiles": len(df)}
    for f in FEATS:
        ok = df[f].notna() & (w > 0)
        row[f"{f}_mean"] = float(np.average(df.loc[ok, f], weights=w[ok])) if ok.any() else np.nan
        row[f"{f}_iqr"] = iqr(df[f])
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(f"{OUT}/tiles", exist_ok=True)
    clin = pd.read_csv("data/analysis_ready_clinical.csv")
    names = [a.only] if a.only else sorted(s for s in clin["Slide Name"].dropna()
                                           if os.path.exists(f"{TILES}/{s}_Tile_Collagen_Features.csv"))
    print(len(names), "slides", flush=True)
    with Pool(a.jobs) as p:
        dfs = p.map(slide, names)
    res = pd.DataFrame([aggregate(d) for d in dfs if len(d)])
    if not a.only:
        res.to_csv(f"{OUT}/morphometry_features.csv", index=False)
    print(res.describe().T.round(3).to_string())
