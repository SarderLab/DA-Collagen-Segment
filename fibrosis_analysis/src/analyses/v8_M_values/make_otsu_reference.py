"""
Previous (CE > slide Otsu mask, extraction v4) slide features, renamed *_otsu,
for comparison next to the v5 (M mask + M values) features in the mixed model.
  python results/v8_M_values/make_otsu_reference.py
Writes data/T0_T1_v5_M_values/previous_otsu_features.csv (config: extra_features)
"""
import pandas as pd

KEEP = ["collagen_ratio", "collagen_iqr", "ce_density", "ce_density_iqr", "ce_collagen_density",
        "ce_collagen_mean_iqr", "f_mean_intensity_mean", "f_mean_intensity_iqr"]
old = pd.read_csv("data/collagen_scores.csv")
out = old[["slide_name"] + KEEP].rename(columns={c: f"{c}_otsu" for c in KEEP})
out.to_csv("data/T0_T1_v5_M_values/previous_otsu_features.csv", index=False)
print(out.shape)
