"""
Each feature on its own (no age, sex or other covariates) in Dr. Zee's mixed
model, with prediction error.
  python results/v9_unadjusted_mse/unadjusted_mse.py
Model 0: egfr ~ time + time2 + (1 + time | id)
Model 1: egfr ~ (time + time2) * feature + (1 + time | id)
  LR p        Model 1 vs Model 0, 3 df, all patients with the feature
  MSE         leave-one-patient-out: fit without the patient, predict all of
              that patient's eGFR values from the fixed effects only (time and
              the feature; a new patient's own random effects are unknown),
              squared error averaged over all held-out eGFR values
  % change    MSE(Model 1) vs MSE(Model 0) on the same patients
  Wilcoxon p  per-patient mean squared error, Model 1 vs Model 0 (paired)
A linear rescaling of the feature leaves predictions unchanged (it has its own
intercept and time terms), so z-scoring once on all patients leaks nothing.
Second run (python ... adjusted): the same with age and sex in both models
(Model 0 = time * (age, sex)), to compare with the deck's adjusted p values.
Writes unadjusted_mse.csv / adjusted_mse.csv
"""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, "src")
import run_association_analysis_v2 as v2
import run_association_analysis_v3 as v3

warnings.filterwarnings("ignore")
ADJ = "adjusted" in sys.argv[1:]
MMASK = "mmask" in sys.argv[1:]          # green features inside the M > 0.5 mask instead of collagen_0/1
COV = ["recipient_age_tx", "recipient_sex_male"] if ADJ else []
OUT = "results/v9_unadjusted_mse"
S = "./data/T0T1 Clinical Data Cleaned 7-27-2026.csv"
FEATURES = [
    ("f_mean_intensity_iqr_otsu", "green variability"),
    ("f_mean_intensity_mean_otsu", "green brightness"),
    ("m_density", "M density"),
    ("m_mean_iqr", "M variability"),
    ("collagen_ratio", "collagen ratio, M mask"),
    ("collagen_ratio_otsu", "collagen_0/1 ratio (before)"),
    ("ce_density_otsu", "collagen density (CE)"),
    ("ce_collagen_mean_iqr_otsu", "CE variability"),
    ("ci", "Banff ci"),
    ("if_pct", "IF%"),
    ("recipient_age_tx", "recipient age"),
    ("recipient_sex_male", "recipient sex"),
]

df = v2.load_inputs("data/analysis_ready_clinical.csv", "data/T0_T1_v5_M_values/collagen_scores.csv",
                    {"99H-0": "99H2-0"})
ex = pd.read_csv("data/T0_T1_v5_M_values/previous_otsu_features.csv")
ex["slide_name"] = ex["slide_name"].replace({"99H-0": "99H2-0"})
df = df.merge(ex, on="slide_name")
_, extra = v3.read_series(S)
long = v3.build_long(df.merge(extra, on="study_id", how="left"), None, 1.0, S)


if ADJ:
    FEATURES = [f for f in FEATURES if f[0] not in COV]
if MMASK:
    FEATURES = [("f_mean_intensity_iqr", "green variability (M mask)"),
                ("f_mean_intensity_mean", "green brightness (M mask)")]


def lopo(d, terms):
    """Per-patient mean squared error of leave-one-patient-out predictions."""
    err = {}
    for pid in d["id"].unique():
        train, test = d[d["id"] != pid], d[d["id"] == pid]
        res = v3.fit_lmm(train, COV + terms, [], "1 + time")[0]
        err[pid] = ((test["egfr"] - res.predict(test)) ** 2).to_numpy()
    return err


rows = []
for feat, label in FEATURES:
    d = v3.prepare_terms(long.dropna(subset=[feat] + COV), [feat], COV)
    r0 = v3.fit_lmm(d, COV, [], "1 + time")[0]
    r1 = v3.fit_lmm(d, COV + [feat], [], "1 + time")[0]
    p = v3.lr_test(r0, r1)[2]
    e0, e1 = lopo(d, []), lopo(d, [feat])
    mse0 = np.concatenate(list(e0.values())).mean()
    mse1 = np.concatenate(list(e1.values())).mean()
    per0 = np.array([e0[k].mean() for k in e0])
    per1 = np.array([e1[k].mean() for k in e0])
    fe = r1.fe_params
    rows.append({"feature": label, "column": feat, "n_patients": d["id"].nunique(), "n_egfr": len(d),
                 "lr_p": p, "post_knot_slope_change_per_SD": fe[f"time:{feat}"] + fe[f"time2:{feat}"],
                 "mse_time_only": mse0, "mse_with_feature": mse1,
                 "mse_change_pct": 100 * (mse1 - mse0) / mse0,
                 "patients_improved": int((per1 < per0).sum()),
                 "wilcoxon_p": wilcoxon(per1, per0).pvalue})
    print(f"{label:30s} n={rows[-1]['n_patients']:2d}  p={p:.3f}  MSE {mse0:6.1f} -> {mse1:6.1f} "
          f"({rows[-1]['mse_change_pct']:+.1f}%)", flush=True)
out = pd.DataFrame(rows)
out.to_csv(f"{OUT}/{'adjusted' if ADJ else 'unadjusted'}_mse{'_mmask' if MMASK else ''}.csv", index=False)
print(out.round(3).to_string(index=False))
