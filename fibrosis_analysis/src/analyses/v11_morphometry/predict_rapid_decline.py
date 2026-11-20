"""Do the handcrafted features (v5 + morphometry) predict rapid eGFR decline? Leave-one-patient-out.
  python src/analyses/v11_morphometry/predict_rapid_decline.py [--perms 100]

Outcome: rapid_decline (Theil-Sen eGFR slope < -5 mL/min/yr; defined only with a slope) and the slope itself.
Every model includes recipient age and sex. Feature sets: none, morphometry, v5 imaging, v5 + morphometry,
Banff ci, IF%. Models:
  logistic  L2, C picked by inner 5-fold CV (0.01, 0.1, 1), median imputation, standardised
  xgboost   200 trees, depth 2, lr 0.05, subsample 0.8, colsample 0.5 (fixed, not tuned)
  ridge / xgboost regressor for the slope (same settings)
AUC is computed on the pooled leave-one-out predictions. Chance range: the same pipeline with the outcome
shuffled (--perms times), 5th-95th percentile.
Also: Spearman of each morphometry feature with Banff ci and IF% (construct validity).
Writes results/v11_morphometry/{loocv_auc.csv, loocv_slope.csv, morphometry_vs_pathology.csv}
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")
R = "results/v11_morphometry"
COV = ["recipient_age_tx", "male"]

clin = pd.read_csv("data/analysis_ready_clinical.csv")
clin["male"] = (clin["recipient_sex"] == "Male").astype(float)
clin.loc[clin["egfr_slope_theilsen"].isna(), "rapid_decline"] = np.nan
v5 = pd.read_csv("data/T0_T1_v5_M_values/collagen_scores.csv")
meta = ["slide_name", "timepoint", "interior_fallback", "mean_tile_tissue_fraction", "n_tiles_collagen_positive",
        "n_tiles_interior", "n_tiles_total", "m_masked_out_share", "ce_outside_share"]
V5 = [c for c in v5.columns if c not in meta]
mo = pd.read_csv(f"{R}/morphometry_features.csv")
MORPH = [c for c in mo.columns if c not in ("slide", "n_tiles")]
df = (clin.merge(v5.rename(columns={"slide_name": "Slide Name"}), on="Slide Name")
      .merge(mo.rename(columns={"slide": "Slide Name"}), on="Slide Name"))
SETS = {"age, sex only": [], "+ morphometry": MORPH, "+ v5 imaging": V5, "+ v5 imaging + morphometry": V5 + MORPH,
        "+ Banff ci": ["ci"], "+ IF%": ["if_pct"]}


def clf(kind):
    if kind == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegressionCV(Cs=[0.01, 0.1, 1], cv=5, scoring="roc_auc", max_iter=5000))
    return XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                         n_jobs=1, verbosity=0)


def reg(kind):
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 9)))
    return XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                        n_jobs=1, verbosity=0)


def loo(make, X, y):
    p = np.empty(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        m = make().fit(X[tr], y[tr])
        p[i] = m.predict_proba(X[i:i + 1])[0, 1] if hasattr(m, "predict_proba") else m.predict(X[i:i + 1])[0]
    return p


def auc_perm(kind, X, y, seed):
    yp = np.random.default_rng(seed).permutation(y)
    return roc_auc_score(yp, loo(lambda: clf(kind), X, yp))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=100)
    a = ap.parse_args()

    rows, srows = [], []
    for name, feats in SETS.items():
        need = COV + [f for f in feats if f in ("ci", "if_pct")]
        d = df.dropna(subset=["rapid_decline"] + need)
        X, y = d[COV + feats].to_numpy(float), d["rapid_decline"].to_numpy(int)
        for kind in ["logistic", "xgboost"]:
            auc = roc_auc_score(y, loo(lambda: clf(kind), X, y))
            null = Parallel(n_jobs=16)(delayed(auc_perm)(kind, X, y, s) for s in range(a.perms))
            rows.append({"features": name, "model": kind, "n": len(y), "events": int(y.sum()),
                         "n_features": X.shape[1], "auc": round(auc, 3),
                         "chance_5": round(np.percentile(null, 5), 2), "chance_95": round(np.percentile(null, 95), 2),
                         "p_perm": round((1 + np.sum(np.array(null) >= auc)) / (1 + len(null)), 3)})
            print(rows[-1], flush=True)
        ys = d["egfr_slope_theilsen"].to_numpy(float)
        for kind in ["ridge", "xgboost"]:
            p = loo(lambda: reg(kind), X, ys)
            srows.append({"features": name, "model": kind, "n": len(ys),
                          "loo_r2": round(1 - np.sum((ys - p) ** 2) / np.sum((ys - ys.mean()) ** 2), 3),
                          "spearman_pred_vs_obs": round(spearmanr(p, ys)[0], 2)})
            print(srows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{R}/loocv_auc.csv", index=False)
    pd.DataFrame(srows).to_csv(f"{R}/loocv_slope.csv", index=False)

    vrows = []
    for f in MORPH:
        r = {"feature": f}
        for ref in ["ci", "if_pct"]:
            d = df.dropna(subset=[f, ref])
            rho, p = spearmanr(d[f], d[ref])
            r.update({f"rho_{ref}": round(rho, 2), f"p_{ref}": p, f"n_{ref}": len(d)})
        vrows.append(r)
    v = pd.DataFrame(vrows)
    for ref in ["ci", "if_pct"]:
        v[f"q_{ref}"] = multipletests(v[f"p_{ref}"], method="fdr_bh")[1].round(3)
        v[f"p_{ref}"] = v[f"p_{ref}"].round(4)
    v.sort_values("p_ci").to_csv(f"{R}/morphometry_vs_pathology.csv", index=False)
    print(v.sort_values("p_ci").to_string(index=False))
