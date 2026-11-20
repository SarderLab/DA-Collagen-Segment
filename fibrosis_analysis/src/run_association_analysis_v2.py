"""
DUET fibrosis study - Arms A1/A2/A2b: association and head-to-head analysis.
Runs once the DUET-derived collagen scores are available.

Usage (config-driven):
  python run_association_analysis_v2.py --config ./config/association_analysis.json
  python run_association_analysis_v2.py --config ./config/association_analysis.json --classifier logistic

Config JSON keys:
  clinical    (required) path to analysis_ready_clinical.csv
  collagen    (required) path to collagen_scores.csv with columns:
                  slide_name (or study_id)  join key
                  timepoint                           optional
                  collagen_ratio                      slide-level DUET collagen area ratio (0-1)
                  collagen_median                     optional: median patch ratio
                  collagen_iqr                        optional: patch-ratio IQR (heterogeneity)
  exposures   optional list, default ["collagen_ratio", "collagen_median", "collagen_iqr"]
              — used for the original targeted A1 table (kept for backward compatibility).
  blocks      optional dict overriding DEFAULT_BLOCKS for A2b
  slide_aliases  optional {"collagen-file slide name": "clinical Slide Name"}
              for slides named differently in the two files, e.g.
              {"99H-0": "99H2-0"}
  scan_features  optional list — features for the generalized A1b single-feature scan
              (default: auto-detected, every numeric column that isn't a covariate,
              outcome, id, or IF%/TA% — so collagen_*, b_*, f_*, and any custom feature
              we add later, e.g. thickness, are picked up automatically without editing
              this script or DEFAULT_BLOCKS).
  sensitivity_covariates  optional list, default ["recipient_age_tx", "recipient_sex_male"]
              — A1 is re-run with this reduced set (donor_age is missing for ~40%)
  classifier  optional, "logistic" (default) or "xgboost" — backend used for every
              LOOCV AUC / DeLong comparison (A1's LOOCV-AUC row, all of A2, all of A2b,
              and A1b's LOOCV-AUC rows).
              The linear/logistic regression rows (betas, CIs, p-values) always use
              statsmodels regardless of this setting, since those need interpretable
              coefficients, not just predictive probabilities.
  xgb_params  optional dict of XGBClassifier keyword overrides (n_estimators, max_depth,
              learning_rate, etc.), only used when classifier == "xgboost"
  out_dir     optional output folder (default "./results"); created if missing

CAVEAT on classifier="xgboost": gradient-boosted trees have far more capacity than a
regularized logistic regression and are more prone to overfitting at small n / low
events-per-variable (see epv column in the A2b output). Defaults below are deliberately
conservative (shallow trees, few rounds, heavy shrinkage). Treat xgboost results here as
exploratory alongside, not a replacement for, the logistic results.

CAVEAT on A1b (the feature scan): it tests every feature individually, which is a much
larger search than A1's original 3 collagen exposures. The BH-FDR correction accounts for
the number of tests, but not for the small n / low events-per-variable in this cohort —
treat A1b as a ranked list of candidates for a validation cohort, not a confirmed finding
on its own, especially for any feature whose single-exposure model has EPV well under 10
(EPV isn't computed in A1b since it's a single-feature model with fixed covariates, but the
same small-n instability that affects A2b's LOOCV AUCs applies here too).

Outputs (written to out_dir, tagged with the classifier so
different backends run on the same input don't overwrite each other):
  <collagen>_<classifier>_a1_association.csv        effect estimates, CIs, p-values, BH-FDR
                                                     (targeted: exposures list only)
  <collagen>_<classifier>_a1b_feature_scan.csv      same per-feature treatment, but looped
                                                     over every candidate feature column
  <collagen>_<classifier>_a1_sensitivity_covariates.csv  A1 with the reduced covariate set
  <collagen>_<classifier>_a2_head_to_head.csv       DUET vs if_pct vs combined, DeLong tests
  <collagen>_<classifier>_a2_nested_vs_pathology.csv  LR test of DUET over IF% / Banff ci,
                                                     R^2 of each model (plan A2)
  <collagen>_<classifier>_a2b_combination_table.csv all C/F/B block combinations vs IF%
"""

import argparse
import json
import os
import re
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

COVARIATES = ["recipient_age_tx", "recipient_sex_male", "donor_age", "cold_ischemia_min"]
PRIMARY_OUTCOMES = ["egfr_slope_theilsen", "rapid_decline"]

# Minimum count of each class required before an AUC is considered estimable.
MIN_CLASS_COUNT = 3

# Conservative defaults for XGBClassifier given small-n clinical cohorts:
# shallow trees, few rounds, strong shrinkage and subsampling to limit variance.
DEFAULT_XGB_PARAMS = dict(
    n_estimators=50,
    max_depth=2,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="logloss",
    random_state=0,
)

# Pre-specified feature blocks for the A2b combination table.
# C = collagen geometry (CE-derived); F = DUET fluorescence (CE-masked);
# B = H&E brightfield (CE-masked). The _iqr columns are the spatial
# heterogeneity features motivated by the A1 finding.
DEFAULT_BLOCKS = {
    "C": ["collagen_ratio", "collagen_iqr"],
    "F": ["f_mean_intensity_mean", "f_mean_intensity_iqr",
          "f_contrast_mean", "f_homogeneity_mean"],
    "B": ["b_mean_intensity_mean", "b_mean_intensity_iqr",
          "b_contrast_mean", "b_homogeneity_mean"],
}


def load_inputs(clinical_path, collagen_path, slide_aliases=None):
    clin = pd.read_csv(clinical_path)
    col = pd.read_csv(collagen_path)
    # slide_aliases: {"name in collagen file": "name in clinical file"} for
    # slides whose folder name differs from the clinical 'Slide Name'.
    if slide_aliases and "slide_name" in col.columns:
        col["slide_name"] = col["slide_name"].replace(slide_aliases)
        print(f"[INFO] slide name aliases applied: {slide_aliases}")
    # Imaging features come from the imaging file only. A clinical column with
    # the same name (e.g. a pasted-in collagen_ratio) is dropped: it would
    # otherwise be suffixed away by the merge, and a copy pasted by row order
    # rather than by slide name puts values on the wrong patients.
    clash = [c for c in clin.columns if c in col.columns and c not in ("study_id", "slide_name")]
    if clash:
        print(f"[WARNING] clinical file also has imaging column(s) {clash}; ignoring the clinical "
              f"copy and using {collagen_path}.")
        clin = clin.drop(columns=clash)
    clin = clin.loc[:, ~clin.columns.str.startswith("Unnamed")]
    # join key: study_id when present, otherwise slide_name -> clinical 'Slide Name'
    join_key = "study_id" if "study_id" in col.columns else "slide_name"
    if join_key == "slide_name":
        clin = clin.rename(columns={"Slide Name": "slide_name"})
        unmatched = set(col["slide_name"]) - set(clin["slide_name"])
        if unmatched:
            print(f"[WARNING] {len(unmatched)} slide(s) not in clinical table: {sorted(unmatched)[:5]}...")

    # optional timepoint handling: pivot long -> wide, default base columns to T0,
    # and add delta_* = T1 - T0 for every value column present.
    if "timepoint" in col.columns:
        value_cols = [c for c in col.columns if c not in (join_key, "timepoint")]
        wide = col.pivot_table(index=join_key, columns="timepoint",
                                values=value_cols, aggfunc="first")
        wide.columns = [f"{a}_t{b}" for a, b in wide.columns]
        col = wide.reset_index()
        for m in value_cols:
            if f"{m}_tT0" in col.columns:
                col[m] = col[f"{m}_tT0"]
            if f"{m}_tT1" in col.columns and f"{m}_tT0" in col.columns:
                col[f"delta_{m}"] = col[f"{m}_tT1"] - col[f"{m}_tT0"]

    df = clin.merge(col, on=join_key, how="inner")
    df["recipient_sex_male"] = (df["recipient_sex"].astype(str).str.upper().str[0] == "M").astype(int)

    # pathology bands like '<5' -> midpoint 2.5; '0'/'5'/... -> numeric
    def band_to_num(v):
        s = str(v).strip()
        if s.startswith("<"):
            return float(s[1:]) / 2
        try:
            return float(s)
        except ValueError:
            return np.nan

    for band_col in ["if_pct", "ta_pct"]:
        if band_col in df.columns:
            df[band_col] = df[band_col].map(band_to_num)

    # rapid_decline is only defined when an eGFR slope exists. The clinical
    # table codes "Insufficient data" patients as 0, which would silently
    # count them as non-decliners.
    if "egfr_slope_theilsen" in df.columns and "rapid_decline" in df.columns:
        undefined = df["egfr_slope_theilsen"].isna() & df["rapid_decline"].notna()
        if undefined.any():
            print(f"[INFO] {int(undefined.sum())} patient(s) without an eGFR slope had "
                  f"rapid_decline coded; set to missing.")
            df.loc[undefined, "rapid_decline"] = np.nan

    for a, b in [("if_pct", "ta_pct"), ("ci", "ct")]:
        if a in df.columns and b in df.columns and df[a].fillna(-1).equals(df[b].fillna(-1)):
            print(f"[WARNING] {a} and {b} are identical in every row — likely a data-prep "
                  f"copy error; {b} carries no separate information.")

    print(f"Merged: {len(df)} samples with both clinical data and collagen scores")
    return df


def _failed_fit(n, reason):
    print(f"[WARNING] model fit failed ({reason}); row set to NaN")
    return {"n": n, "beta": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan,
            "beta_per_sd": np.nan, "ci_lo_per_sd": np.nan, "ci_hi_per_sd": np.nan,
            "model_r2": np.nan}


def _effect_row(fit, d, exposure, model_r2):
    """beta/CI per unit of the exposure, plus per SD of the exposure in the
    fitted sample. Per-unit betas for a 0-1 ratio describe a 0 -> 100%
    collagen change and are hard to read; per-SD values are comparable across
    features. For logistic models these are log-odds (exp() for an OR)."""
    ci = fit.conf_int().loc[exposure]
    sd = float(d[exposure].std())
    return {"n": len(d), "beta": fit.params[exposure], "ci_lo": ci[0], "ci_hi": ci[1],
            "p": fit.pvalues[exposure],
            "beta_per_sd": fit.params[exposure] * sd, "ci_lo_per_sd": ci[0] * sd,
            "ci_hi_per_sd": ci[1] * sd, "model_r2": model_r2}


def fit_linear(df, exposure, outcome, covariates):
    d = df[[exposure, outcome] + covariates].dropna()
    X = sm.add_constant(d[[exposure] + covariates])
    try:
        fit = sm.OLS(d[outcome], X).fit()
    except (np.linalg.LinAlgError, ValueError) as e:
        return _failed_fit(len(d), f"{exposure} -> {outcome}: {e}")
    return _effect_row(fit, d, exposure, model_r2=fit.rsquared)


def fit_logistic(df, exposure, outcome, covariates):
    # Small n + few events -> (quasi-)separation is possible; report NaN
    # rather than aborting the whole run.
    d = df[[exposure, outcome] + covariates].dropna()
    X = sm.add_constant(d[[exposure] + covariates])
    try:
        fit = sm.Logit(d[outcome], X).fit(disp=0)
    except (np.linalg.LinAlgError, ValueError) as e:
        return _failed_fit(len(d), f"{exposure} -> {outcome}: {e}")
    row = _effect_row(fit, d, exposure, model_r2=np.nan)
    row["n_events"] = int(d[outcome].sum())
    return row


# Columns that are never candidate imaging/collagen *features* for the A1b
# scan below — covariates, outcomes, ids, and the pathologist reference
# (if_pct/ta_pct are tested separately as the A2/A2b reference, not folded
# into the generic feature scan).
NON_FEATURE_COLUMNS = set(COVARIATES) | set(PRIMARY_OUTCOMES) | {
    "study_id", "slide_name", "timepoint", "if_pct", "ta_pct",
    "n_tiles_total", "n_tiles_collagen_positive", "recipient_sex",
}


# Imaging feature families produced by FeatureExtractor_wsi*.py. Restricting
# discovery to these prefixes keeps clinical columns (egfr_6m/1y/3y/5y,
# traj_*, Banff ci/ct/ti) out of the scan — several of those are derived from
# the outcome itself and would be pure leakage.
FEATURE_PREFIXES = ("collagen_", "b_", "f_", "delta_")
TIMEPOINT_SUFFIX = re.compile(r"_t[^_]+$")


def discover_feature_columns(df, exclude=None):
    """Auto-detect candidate imaging feature columns: numeric columns with an
    imaging prefix (collagen_*, b_*, f_*, delta_*), so a newly added feature
    (e.g. collagen_thickness_*) is picked up without editing this script.
    Per-timepoint copies (<m>_tT0, <m>_tT1) are skipped when the base column
    <m> exists, so the T0 value isn't tested twice.
    """
    exclude = set(exclude or []) | NON_FEATURE_COLUMNS
    feats = []
    for c in df.columns:
        if c in exclude or not c.startswith(FEATURE_PREFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if TIMEPOINT_SUFFIX.search(c) and TIMEPOINT_SUFFIX.sub("", c) in df.columns:
            continue
        feats.append(c)
    return feats


def a1_feature_scan(df, covariates, features=None, classifier="logistic", xgb_params=None):
    """
    A1b: the same per-feature treatment A1 gives its 3 hand-picked collagen
    exposures (adjusted linear regression on eGFR slope, adjusted logistic
    regression + single-feature LOOCV AUC on rapid_decline) — but looped
    over EVERY candidate feature column, not just collagen_ratio/median/iqr.

    This closes a real gap in the original design: A1 only ever scanned
    collagen's 3 summary stats one at a time, while every B/F feature was
    only ever tested bundled 4-at-a-time inside A2b. A feature with a
    collagen_iqr-sized individual effect could be invisible in that bundle
    (diluted by 3 correlated co-features, competing for the same
    regularization budget). This function gives every feature the same
    fair, isolated shot A1 gave collagen_iqr, then applies ONE BH-FDR
    correction across the whole scan so results are directly comparable
    regardless of which block they came from.
    """
    features = features or discover_feature_columns(df)
    rows = []
    for feat in features:
        if feat not in df.columns:
            continue
        rows.append({"feature": feat, "outcome": "egfr_slope_theilsen", "model": "linear+adj",
                     **fit_linear(df, feat, "egfr_slope_theilsen", covariates)})
        rows.append({"feature": feat, "outcome": "rapid_decline", "model": "logistic+adj",
                     **fit_logistic(df, feat, "rapid_decline", covariates)})
        auc, n = loocv_auc(df, feat, "rapid_decline", covariates,
                            classifier=classifier, xgb_params=xgb_params)
        rows.append({"feature": feat, "outcome": "rapid_decline", "model": f"LOOCV AUC ({classifier})",
                     "n": n, "beta": auc, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan,
                     "model_r2": np.nan})
    scan = pd.DataFrame(rows)
    mask = scan["model"].str.contains("linear|logistic") & scan["p"].notna()
    if mask.any():
        scan.loc[mask, "p_fdr"] = stats.false_discovery_control(scan.loc[mask, "p"].values)
    return scan.sort_values(["outcome", "model", "p"], na_position="last").reset_index(drop=True)


def make_classifier(classifier, c_reg=1.0, scale_pos_weight=1.0, xgb_params=None):
    """Build the LOOCV base estimator. Both branches expose predict_proba,
    so callers don't need to know which backend is in use."""
    if classifier == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError(
                "classifier='xgboost' requested but xgboost is not installed. "
                "Install it with: pip install xgboost"
            )
        params = dict(DEFAULT_XGB_PARAMS)
        params["scale_pos_weight"] = scale_pos_weight
        if xgb_params:
            params.update(xgb_params)
        return XGBClassifier(**params)
    elif classifier == "logistic":
        return LogisticRegression(max_iter=1000, class_weight="balanced", C=c_reg)
    else:
        raise ValueError(f"Unknown classifier '{classifier}'; use 'logistic' or 'xgboost'.")


def loocv_probs(df, exposures, outcome, covariates, standardize=True, c_reg=1.0,
                 classifier="logistic", xgb_params=None):
    """Out-of-fold predicted probabilities via LOOCV (small-n honest estimate).

    standardize=True (default) z-scores features using training-fold statistics
    only (no leakage). This matters a great deal for classifier="logistic":
    without it, exposures/features on a small raw scale (e.g. a 0-1 collagen
    ratio) get outcompeted by covariates on a larger scale (e.g. donor_age,
    cold_ischemia_min) under L2 regularization, since the penalty acts on raw
    coefficient magnitude — the exposure can end up contributing ~nothing to
    the fitted model regardless of its true association with the outcome.
    Harmless for classifier="xgboost", which is invariant to monotonic
    per-feature rescaling; only pass standardize=False if you have a specific
    reason to compare on raw covariate scales.

    classifier: "logistic" (default, L2-regularized, class_weight="balanced")
    or "xgboost" (gradient-boosted trees; requires xgboost, see caveat in the
    module docstring re: overfitting risk at small n). Class imbalance is
    handled per training fold either via class_weight (logistic) or
    scale_pos_weight = n_neg/n_pos in that fold (xgboost).
    """
    d = df[exposures + [outcome] + covariates].dropna()
    X = d[exposures + covariates].values
    y = d[outcome].values
    prob = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        y_tr = y[tr]
        n_pos_tr = y_tr.sum()
        spw = (len(y_tr) - n_pos_tr) / n_pos_tr if n_pos_tr > 0 else 1.0
        clf = make_classifier(classifier, c_reg=c_reg, scale_pos_weight=spw,
                               xgb_params=xgb_params)
        if standardize:
            mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
            sd[sd == 0] = 1.0
            clf.fit((X[tr] - mu) / sd, y_tr)
            prob[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
        else:
            clf.fit(X[tr], y_tr)
            prob[te] = clf.predict_proba(X[te])[:, 1]
    return y, prob, len(y)


def auc_estimable(y):
    """True if both classes have enough members for a stable AUC estimate."""
    n_pos = int(np.sum(y))
    n_neg = len(y) - n_pos
    return n_pos >= MIN_CLASS_COUNT and n_neg >= MIN_CLASS_COUNT


def events_per_variable(y, n_predictors):
    """Events-per-variable: minority-class count / number of predictors.

    Standard rule-of-thumb diagnostic for logistic model stability (commonly
    cited threshold: EPV >= 10). n_predictors should include covariates.
    """
    if n_predictors <= 0:
        return np.nan
    n_pos = int(np.sum(y))
    n_neg = len(y) - n_pos
    return min(n_pos, n_neg) / n_predictors


def loocv_auc(df, exposure, outcome, covariates, classifier="logistic", xgb_params=None):
    """Out-of-sample AUC via LOOCV (small-n honest estimate)."""
    y, prob, n = loocv_probs(df, [exposure], outcome, covariates,
                              classifier=classifier, xgb_params=xgb_params)
    if not auc_estimable(y):
        return np.nan, n
    return roc_auc_score(y, prob), n


def delong_auc_variance(y, prob_a, prob_b):
    """Paired DeLong test for two AUCs on the same samples (Sun & Xu 2014 vectorized form).

    AUC and its variance are both derived from the same tie-corrected placement
    values (v01/v10), so the returned AUCs are consistent with the variance
    estimate even when predicted probabilities contain ties.
    """
    n1, n0 = y.sum(), (1 - y).sum()

    def v_components(prob):
        v01 = (prob[y == 1][:, None] > prob[y == 0][None, :]).mean(axis=1) + \
              0.5 * (prob[y == 1][:, None] == prob[y == 0][None, :]).mean(axis=1)
        v10 = (prob[y == 1][:, None] > prob[y == 0][None, :]).mean(axis=0) + \
              0.5 * (prob[y == 1][:, None] == prob[y == 0][None, :]).mean(axis=0)
        return v01, v10

    v01a, v10a = v_components(prob_a)
    v01b, v10b = v_components(prob_b)
    # AUC = mean placement value of the positives (tie-corrected); equivalent
    # to roc_auc_score, and consistent with the variance terms below.
    auc_a, auc_b = v01a.mean(), v01b.mean()

    s01 = np.cov(np.vstack([v01a, v01b]))
    s10 = np.cov(np.vstack([v10a, v10b]))
    var = (s01[0, 0] / n1 + s10[0, 0] / n0, s01[1, 1] / n1 + s10[1, 1] / n0)
    cov = s01[0, 1] / n1 + s10[0, 1] / n0
    se = np.sqrt(var[0] + var[1] - 2 * cov)
    if se == 0 or np.isnan(se):
        return auc_a, auc_b, np.nan
    z = (auc_a - auc_b) / se
    return auc_a, auc_b, 2 * norm.sf(abs(z))


def a2b_combination_table(df, covariates, blocks=None, classifier="logistic", xgb_params=None):
    """
    A2b: all non-empty combinations of the C/F/B imaging blocks vs pathologist
    IF% for predicting rapid decline. Each row = one feature block combination
    (+ covariates). LOOCV AUC (in-fold standardization, regularized), paired
    DeLong test vs IF% on the shared complete-case subset, BH-FDR across the
    combination tests. Exploratory: EPV-limited at this n (see epv column;
    doubly true for classifier="xgboost", which has more capacity to overfit
    than logistic regression at the same EPV).
    """
    blocks = blocks or DEFAULT_BLOCKS
    available = {k: [c for c in v if c in df.columns] for k, v in blocks.items()}
    missing = [k for k, v in blocks.items() if not available.get(k)]
    if missing:
        print(f"[WARNING] Missing feature columns for block(s) {missing} — "
              f"re-run FeatureExtractor_wsi.py (new schema) to add them. "
              f"Running available blocks only.")
    usable = {k: v for k, v in available.items() if v}
    if not usable:
        print("[ERROR] No imaging feature blocks available; A2b skipped.")
        return None
    if "if_pct" not in df.columns:
        print("[ERROR] if_pct not in clinical table; A2b needs the pathology reference.")
        return None

    combo_names = []
    for r in range(1, len(usable) + 1):
        combo_names.extend("".join(c) for c in combinations(sorted(usable), r))

    # shared complete-case subset for paired DeLong tests
    all_feats = [f for v in usable.values() for f in v]
    d = df.dropna(subset=["if_pct", "rapid_decline"] + covariates + all_feats)
    y_ref, p_if, n_shared = loocv_probs(d, ["if_pct"], "rapid_decline", covariates,
                                         classifier=classifier, xgb_params=xgb_params)
    ref_auc = roc_auc_score(y_ref, p_if) if auc_estimable(y_ref) else np.nan
    ref_n_predictors = 1 + len(covariates)

    rows = [{"model": "IF% (reference)", "blocks": "IF", "features": ["if_pct"], "n": n_shared,
             "n_shared": n_shared, "loocv_auc": ref_auc,
             "n_predictors": ref_n_predictors,
             "epv": events_per_variable(y_ref, ref_n_predictors),
             "p_delong_vs_if": np.nan, "p_fdr_vs_if": np.nan}]

    for combo in combo_names:
        feats = [f for k in combo for f in usable[k]]
        n_predictors = len(feats) + len(covariates)

        # row AUC on the block's own complete cases
        y_k, p_k, n_k = loocv_probs(df, feats, "rapid_decline", covariates,
                                     standardize=True, c_reg=0.1,
                                     classifier=classifier, xgb_params=xgb_params)
        row_auc = roc_auc_score(y_k, p_k) if auc_estimable(y_k) else np.nan

        # paired DeLong vs IF% on the shared subset (only if the shared
        # labels support an estimable AUC)
        p_delong = np.nan
        if auc_estimable(y_ref):
            _, p_k_shared, _ = loocv_probs(d, feats, "rapid_decline", covariates,
                                            standardize=True, c_reg=0.1,
                                            classifier=classifier, xgb_params=xgb_params)
            _, _, p_delong = delong_auc_variance(y_ref, p_k_shared, p_if)

        rows.append({"model": combo, "blocks": combo, "features": feats,
                     "loocv_auc": row_auc, "n": n_k,
                     "n_shared": n_shared,
                     "n_predictors": n_predictors,
                     "epv": events_per_variable(y_k, n_predictors),
                     "p_delong_vs_if": p_delong,
                     "p_fdr_vs_if": np.nan})

    a2b = pd.DataFrame(rows)
    mask = (a2b["model"] != "IF% (reference)") & a2b["p_delong_vs_if"].notna()
    if mask.any():
        a2b.loc[mask, "p_fdr_vs_if"] = stats.false_discovery_control(
            a2b.loc[mask, "p_delong_vs_if"].values)
    return a2b


def loocv_auc_vs_covariates(df, exposure, outcome, covariates, classifier="logistic",
                            xgb_params=None):
    """LOOCV AUC of exposure+covariates AND of covariates alone on the SAME
    rows, with a paired DeLong test. The exposure's own contribution is the
    difference; an exposure+covariates AUC on its own mixes the two."""
    d = df.dropna(subset=[exposure, outcome] + covariates)
    y, p_exp, n = loocv_probs(d, [exposure], outcome, covariates,
                              classifier=classifier, xgb_params=xgb_params)
    out = {"n": n, "auc": np.nan, "auc_covariates_only": np.nan, "p_delong_vs_covariates": np.nan}
    if not auc_estimable(y):
        return out
    out["auc"] = roc_auc_score(y, p_exp)
    if covariates:
        _, p_cov, _ = loocv_probs(d, [], outcome, covariates,
                                  classifier=classifier, xgb_params=xgb_params)
        out["auc_covariates_only"] = roc_auc_score(y, p_cov)
        out["p_delong_vs_covariates"] = delong_auc_variance(y, p_exp, p_cov)[2]
    return out


def run_a1(df, exposures, covariates, classifier="logistic", xgb_params=None):
    """A1 table: adjusted linear (eGFR slope) and logistic (rapid decline)
    regressions per exposure, plus LOOCV AUC vs the covariates-only model.
    BH-FDR across the regression p-values (the pre-specified primary tests)."""
    rows = []
    for exp in exposures:
        if exp not in df.columns:
            continue
        rows.append({"exposure": exp, "outcome": "egfr_slope_theilsen", "model": "linear+adj",
                     **fit_linear(df, exp, "egfr_slope_theilsen", covariates)})
        rows.append({"exposure": exp, "outcome": "rapid_decline", "model": "logistic+adj",
                     **fit_logistic(df, exp, "rapid_decline", covariates)})
        r = loocv_auc_vs_covariates(df, exp, "rapid_decline", covariates,
                                    classifier=classifier, xgb_params=xgb_params)
        rows.append({"exposure": exp, "outcome": "rapid_decline", "model": f"LOOCV AUC ({classifier})",
                     "n": r["n"], "beta": r["auc"], "auc_covariates_only": r["auc_covariates_only"],
                     "p_delong_vs_covariates": r["p_delong_vs_covariates"]})
    a1 = pd.DataFrame(rows)
    mask = a1["model"].str.contains("linear|logistic") & a1["p"].notna()
    if mask.any():
        a1.loc[mask, "p_fdr"] = stats.false_discovery_control(a1.loc[mask, "p"].values)
    a1.insert(0, "covariates", "+".join(covariates) if covariates else "none")
    return a1


def a2_nested_tests(df, reference, exposure, covariates):
    """Incremental value of the DUET exposure over a pathology reference
    (analysis plan A2): on identical complete cases, fit covariates+reference
    vs covariates+reference+exposure and report the likelihood-ratio test,
    for eGFR slope (OLS, with R^2 of each model) and rapid decline (logistic).
    Also covariates+exposure alone, so the two single-marker models can be
    compared on the same patients."""
    rows = []
    for outcome, family in [("egfr_slope_theilsen", "linear"), ("rapid_decline", "logistic")]:
        d = df.dropna(subset=[reference, exposure, outcome] + covariates)

        def fit(cols):
            X = sm.add_constant(d[cols + covariates])
            return sm.OLS(d[outcome], X).fit() if family == "linear" else \
                sm.Logit(d[outcome], X).fit(disp=0)

        def r2(f):
            return f.rsquared if family == "linear" else f.prsquared

        try:
            f_ref, f_exp, f_both = fit([reference]), fit([exposure]), fit([reference, exposure])
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"[WARNING] A2 nested {outcome} ({reference}) failed: {e}")
            continue
        lr = 2 * (f_both.llf - f_ref.llf)
        rows.append({
            "reference": reference, "exposure": exposure, "outcome": outcome, "n": len(d),
            "n_events": int(d[outcome].sum()) if family == "logistic" else np.nan,
            "r2_type": "R2" if family == "linear" else "McFadden pseudo-R2",
            "r2_reference": r2(f_ref), "r2_exposure": r2(f_exp), "r2_both": r2(f_both),
            "p_reference_in_both": f_both.pvalues[reference],
            "p_exposure_in_both": f_both.pvalues[exposure],
            "lr_stat_exposure_over_reference": lr,
            "p_lr_exposure_over_reference": stats.chi2.sf(lr, df=1),
        })
    return rows


def main(args):
    df = load_inputs(args.clinical, args.collagen, getattr(args, "slide_aliases", None))
    cov = [c for c in COVARIATES if c in df.columns and df[c].notna().sum() > 30]
    exposures = getattr(args, "exposures", None) or ["collagen_ratio", "collagen_median", "collagen_iqr"]
    classifier = getattr(args, "classifier", "logistic")
    xgb_params = getattr(args, "xgb_params", None)
    if classifier == "xgboost" and not HAS_XGBOOST:
        raise ImportError(
            "config specifies classifier='xgboost' but xgboost is not installed. "
            "Install it with: pip install xgboost"
        )
    print(f"[INFO] LOOCV classifier backend: {classifier}")
    # Tag output filenames with the classifier so runs with different
    # backends don't overwrite each other's results for the same --collagen input.
    out_dir = getattr(args, "out_dir", None) or "./results"
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.collagen))[0]
    def out_path(tag):
        return os.path.join(out_dir, f"{stem}_{classifier}_{tag}.csv")

    # ---- A1: association ----
    a1 = run_a1(df, exposures, cov, classifier=classifier, xgb_params=xgb_params)
    a1.to_csv(out_path("a1_association"), index=False)
    print("\n=== A1 association ===")
    print(a1.drop(columns="covariates").to_string(index=False))

    # ---- A1 sensitivity: reduced covariate set ----
    # donor_age is missing for ~40% of patients, so the full set drops the
    # analysable n from ~66 to ~42. Re-run A1 with a reduced set so a finding
    # that exists only in the donor-age-complete subset is visible.
    sens_cov = [c for c in (getattr(args, "sensitivity_covariates", None)
                            or ["recipient_age_tx", "recipient_sex_male"]) if c in df.columns]
    if sens_cov != cov:
        a1s = run_a1(df, exposures, sens_cov, classifier=classifier, xgb_params=xgb_params)
        a1s.to_csv(out_path("a1_sensitivity_covariates"), index=False)
        print(f"\n=== A1 sensitivity (covariates: {', '.join(sens_cov)}) ===")
        print(a1s.drop(columns="covariates").to_string(index=False))

    # ---- A1b: generalized single-feature scan (every candidate feature) ----
    scan_features = getattr(args, "scan_features", None)
    a1b = a1_feature_scan(df, cov, features=scan_features, classifier=classifier, xgb_params=xgb_params)
    a1b.to_csv(out_path("a1b_feature_scan"), index=False)
    print(f"\n=== A1b feature scan ({len(a1b['feature'].unique())} features tested) ===")
    top = a1b[a1b["model"].str.contains("linear|logistic")].sort_values("p_fdr").head(10)
    print(top[["feature", "outcome", "model", "n", "beta", "p", "p_fdr"]].to_string(index=False))

    # ---- A2: head-to-head vs pathologist if_pct ----
    d = df.dropna(subset=["if_pct", "collagen_ratio", "rapid_decline"] + cov)
    print(f"\n=== A2 head-to-head (n={len(d)} complete cases) ===")
    res = []
    # paired out-of-fold probabilities on identical LOOCV splits
    y_d, p_duet, n_d = loocv_probs(d, ["collagen_ratio"], "rapid_decline", cov,
                                    classifier=classifier, xgb_params=xgb_params)
    y_p, p_if, _ = loocv_probs(d, ["if_pct"], "rapid_decline", cov,
                                classifier=classifier, xgb_params=xgb_params)
    y_c, p_comb, _ = loocv_probs(d, ["collagen_ratio", "if_pct"], "rapid_decline", cov,
                                  classifier=classifier, xgb_params=xgb_params)

    estimable = auc_estimable(y_d)
    res.append({"model": "DUET collagen",
                "loocv_auc": roc_auc_score(y_d, p_duet) if estimable else np.nan,
                "n": n_d, "p_delong_vs_duet": np.nan})
    res.append({"model": "Pathologist IF%",
                "loocv_auc": roc_auc_score(y_p, p_if) if estimable else np.nan,
                "n": n_d, "p_delong_vs_duet": np.nan})
    res.append({"model": "DUET + IF% combined",
                "loocv_auc": roc_auc_score(y_c, p_comb) if estimable else np.nan,
                "n": n_d, "p_delong_vs_duet": np.nan})
    # paired DeLong test: DUET vs pathologist IF% (primary A2 comparison)
    if estimable:
        auc_duet, auc_if, p_delong = delong_auc_variance(y_d, p_duet, p_if)
    else:
        p_delong = np.nan
    # NOTE: this row is the only one where p_delong_vs_duet is populated; the
    # column is reused above as a placeholder for the other rows.
    res.append({"model": "DeLong: DUET vs IF%", "loocv_auc": np.nan, "n": n_d,
                "p_delong_vs_duet": p_delong})
    if estimable:
        _, _, p_comb_vs_if = delong_auc_variance(y_d, p_comb, p_if)
        res.append({"model": "DeLong: DUET + IF% vs IF%", "loocv_auc": np.nan, "n": n_d,
                    "p_delong_vs_duet": p_comb_vs_if})
    a2 = pd.DataFrame(res)
    a2.to_csv(out_path("a2_head_to_head"), index=False)
    print(a2.to_string(index=False))

    # ---- A2 nested: incremental value of DUET over the pathology read ----
    # (analysis plan: likelihood-ratio test of DUET-over-IF%, and R^2 for slope)
    # Banff ci is included as a second reference: it is recorded for more
    # patients than IF% (ta_pct/ct duplicate if_pct/ci in this dataset).
    nested = []
    for ref in [r for r in ["if_pct", "ci"] if r in df.columns]:
        nested.extend(a2_nested_tests(df, ref, "collagen_ratio", cov))
    if nested:
        a2n = pd.DataFrame(nested)
        a2n.to_csv(out_path("a2_nested_vs_pathology"), index=False)
        print("\n=== A2 nested: DUET collagen_ratio over pathology reference ===")
        print(a2n.round(4).to_string(index=False))

    # ---- A2b: C/F/B combination table vs pathologist if_pct ----
    print("\n=== A2b combination table (exploratory) ===")
    blocks = getattr(args, "blocks", None)
    a2b = a2b_combination_table(df, cov, blocks=blocks, classifier=classifier, xgb_params=xgb_params)
    if a2b is not None:
        a2b.to_csv(out_path("a2b_combination_table"), index=False)
        print(a2b[["model", "loocv_auc", "n", "n_shared", "n_predictors", "epv",
                    "p_delong_vs_if", "p_fdr_vs_if"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DUET fibrosis association analysis (A1/A2/A2b)")
    ap.add_argument('--config', type=str, default="./config/association_analysis.json",
                     help='Path to config JSON')
    ap.add_argument('--classifier', type=str, choices=["logistic", "xgboost"], default=None,
                     help='Override the classifier set in the config')
    cli_args, _ = ap.parse_known_args()
    with open(cli_args.config) as f:
        data = json.load(f)
    if cli_args.classifier:
        data["classifier"] = cli_args.classifier
    run_args = argparse.Namespace(**data)
    main(run_args)