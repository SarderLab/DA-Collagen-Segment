"""One row per v5 slide-level imaging feature (59): column, group, what it is measured on, summary, meaning.
Definitions follow E:/f.afsari/Codes/DA-Collagen-Segment/Fibrosis/FeatureExtractor_wsi_v5.py.
  python src/analyses/make_v5_feature_list.py   ->  results/v5_feature_list.csv
"""
import pandas as pd

META = ["slide_name", "timepoint", "interior_fallback", "mean_tile_tissue_fraction", "n_tiles_collagen_positive",
        "n_tiles_interior", "n_tiles_total", "m_masked_out_share", "ce_outside_share"]
INTERIOR = "tiles with >= 50% tissue"
TEX = {"contrast": "texture: contrast (how different neighbouring pixels are)",
       "homogeneity": "texture: homogeneity (how similar neighbouring pixels are)",
       "energy": "texture: energy (how regular / repetitive the pattern is)",
       "correlation": "texture: correlation (how predictable a pixel is from its neighbour)"}
GROUPS = {"collagen": "Collagen amount and shape", "m": "Collagen amount and shape", "f": "DUET fluorescence",
          "fw": "DUET fluorescence", "ce": "Collagen x fluorescence (CE)", "b": "H&E"}

SPECIAL = {
    "collagen_ratio": ("collagen mask (M > 0.5)", "area-weighted over all tiles",
                       "collagen area / tissue area for the whole biopsy"),
    "collagen_median": ("collagen mask (M > 0.5)", f"median over {INTERIOR}", "typical per-tile collagen fraction"),
    "collagen_iqr": ("collagen mask (M > 0.5)", f"IQR over {INTERIOR}",
                     "tile-to-tile spread of collagen fraction (patchiness)"),
    "collagen_thickness_mean": ("collagen mask skeleton", f"mean over {INTERIOR}",
                                "collagen strand thickness (2 x distance to edge along the skeleton; relative)"),
    "collagen_thickness_median": ("collagen mask skeleton", f"median over {INTERIOR}", "typical strand thickness"),
    "collagen_thickness_iqr": ("collagen mask skeleton", f"IQR over {INTERIOR}",
                               "tile-to-tile spread of strand thickness"),
    "m_density": ("model output M (0-1) over tissue, glomeruli excluded", "sum M / tissue area, all tiles",
                  "soft collagen fraction: every pixel counts by the model's collagen probability"),
    "m_mean_mean": ("model output M over tissue", f"mean over {INTERIOR}", "average per-tile soft collagen fraction"),
    "m_mean_iqr": ("model output M over tissue", f"IQR over {INTERIOR}",
                   "M variability: tile-to-tile spread of soft collagen fraction"),
    "m_collagen_mean_mean": ("model output M inside the collagen mask", f"mean over {INTERIOR}",
                             "model confidence inside collagen"),
    "m_collagen_mean_iqr": ("model output M inside the collagen mask", f"IQR over {INTERIOR}",
                            "tile-to-tile spread of model confidence inside collagen"),
    "ce_density": ("CE (0-1) over tissue", "sum CE / tissue area, all tiles",
                   "collagen amount weighted by green brightness, no threshold"),
    "ce_density_median": ("CE over tissue", f"median over {INTERIOR}", "typical per-tile CE density"),
    "ce_density_iqr": ("CE over tissue", f"IQR over {INTERIOR}", "tile-to-tile spread of CE density"),
    "ce_collagen_density": ("CE inside the collagen mask", "sum / tissue area, all tiles",
                            "collagen density (CE): CE-weighted collagen amount, haze outside collagen excluded"),
}


def describe(col):
    if col in SPECIAL:
        return SPECIAL[col]
    stem, stat = col.rsplit("_", 1)
    summary = f"{'mean' if stat == 'mean' else 'IQR'} over {INTERIOR}"
    spread = "" if stat == "mean" else " (tile-to-tile spread)"
    prefix, feat = stem.split("_", 1)
    on = {"b": "H&E eosin channel inside the collagen mask", "f": "DUET green inside the collagen mask",
          "fw": "DUET green over tissue, weighted by CE"}.get(prefix)
    if prefix == "ce":
        on = "CE inside the collagen mask" if feat.startswith("collagen_") else "CE over tissue"
        feat = feat.replace("collagen_", "")
    what = {"mean_intensity": "brightness", "std_intensity": "pixel-to-pixel brightness spread",
            "mean": "brightness", "std": "pixel-to-pixel spread", "p90": "90th-percentile value (bright end)"}
    meaning = TEX.get(feat) or what[feat]
    return on, summary, meaning + spread


cols = [c for c in pd.read_csv("data/T0_T1_v5_M_values/collagen_scores.csv").columns if c not in META]
rows = []
for c in cols:
    on, summary, meaning = describe(c)
    rows.append({"feature": c, "group": GROUPS[c.split("_")[0]], "measured_on": on,
                 "per_biopsy_summary": summary, "meaning": meaning})
out = pd.DataFrame(rows)
order = ["Collagen amount and shape", "DUET fluorescence", "Collagen x fluorescence (CE)", "H&E"]
out = out.sort_values("group", key=lambda s: s.map(order.index), kind="stable")
out.to_csv("results/v5_feature_list.csv", index=False)
print(out.groupby("group", sort=False).size().to_string(), "\ntotal", len(out))
