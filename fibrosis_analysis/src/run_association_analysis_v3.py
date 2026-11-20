"""
DUET fibrosis study - v3: longitudinal linear mixed model for eGFR (Dr. Zee's
proposal), added on top of the v2 cross-sectional arms (A1/A1b/A2/A2b).

Usage:
  python run_association_analysis_v3.py --config ./config/association_analysis.json
  python run_association_analysis_v3.py --skip-v2          # mixed model only

Model (lme4 notation):
  base:      egfr ~ (time + time2) * (slope covariates) + baseline covariates
                    + (random effects | id)
  + one:     base + (time + time2) * collagen_k           -> LR test, 3 df
  joint:     base + (time + time2) * (all collagen_k)     -> LR test, 3k df
  time2 = max(time - knot, 0): linear spline, so `time` is the slope before
  the knot and `time2` the CHANGE in slope after it. The collagen_k main effect
  is its association with eGFR level at time 0; collagen_k:time with the slope
  before the knot; collagen_k:time2 with the change in slope after the knot.
  All models are fitted by maximum likelihood (REML = FALSE) so nested models
  can be compared by likelihood-ratio test.

Config keys (in addition to v2's; all optional):
  egfr_series           path to the source clinical file with the full wide
                        eGFR series: every column named egfr_<n>m or egfr_<n>y
                        (e.g. egfr_3m, egfr_6m, egfr_1y ... egfr_8y) becomes one
                        visit at n/12 or n years. Joined on study_id. Its
                        graft_status column, if present, is used in the
                        missing-data table. Recommended:
                        "./data/T0T1 Clinical Data Cleaned 7-27-2026.csv"
  egfr_long             alternatively, a long-format file with columns
                        [study_id or slide_name, time_years, egfr]
                        If neither is given, the long format is built from the
                        4 eGFR columns in the clinical file (0.5/1/3/5 y).
  lmm_knot_years        spline knot, default 1.0
  lmm_slope_covariates  clinical factors allowed to affect the slope (they get
                        time and time2 interactions); default
                        ["recipient_age_tx", "recipient_sex_male"]
  lmm_baseline_covariates  clinical factors that affect eGFR level only;
                        default []
  lmm_sensitivity_covariates  {"slope": [...], "baseline": [...]}; a second,
                        fuller adjustment run. Default adds donor_age (slope)
                        and cold_ischemia_min (baseline); n drops to ~40.
  lmm_predictors        default ["collagen_ratio", "collagen_iqr",
                        "collagen_thickness_mean", "f_mean_intensity_mean",
                        "b_mean_intensity_mean"]
  lmm_joint_predictors  default = lmm_predictors
  lmm_added_predictor   predictor tested on top of each pathology read;
                        default = the first entry of lmm_predictors
  lmm_feature_groups    {"group name": [features], ...}, e.g.
                        {"collagen": ["ce_density"],
                         "fluorescence": ["f_mean_intensity_iqr"]}.
                        Runs the nested group experiment (see below) for the
                        primary and both sensitivity analyses. Omit to skip.
  extra_features        list of CSVs with a slide_name column whose other
                        columns are merged into the collagen features (e.g.
                        data/T0_T1/fibrotic_area_features.csv from
                        compute_fibrotic_area.py)
  lmm_references        pathology reads run through the same procedure, and
                        used as the base for "collagen over pathology" tests;
                        default ["ci", "if_pct"]
  lmm_random_effects    "auto" (default): the richest of "1 + time + time2",
                        "1 + time", "1" that converges without a singular
                        fit on the base model, then held fixed for every model
                        in that analysis (LR tests need identical random
                        effects). Or give one of the three strings explicitly.
  lmm_exclude_imputed   also run a sensitivity analysis without the patients
                        whose 6m/1y eGFR was imputed; default true

Predictors are z-scored across patients (effects are per SD); continuous
covariates are mean-centred and kept in their units.

Outputs (out_dir, prefix lmm_):
  lmm_egfr_long.csv          the modelling data, one row per patient-visit,
                             with the transformed predictors (use it to
                             reproduce the models in lme4)
  lmm_missingness.csv        step 1: patients with vs without eGFR after the knot
  lmm_base_<tag>.csv         fixed effects of the base model
  lmm_single_<tag>.csv       one collagen predictor at a time: LR test, BH-FDR,
                             and its level / slope / slope-change coefficients
  lmm_joint_<tag>.csv        all predictors together
  lmm_over_pathology_<tag>.csv  lmm_added_predictor added on top of each pathology read
  lmm_groups_<tag>.csv       nested group experiment: likelihood-ratio tests
  lmm_groups_coefs_<tag>.csv its level / slope / slope-change coefficients
  lmm_check.R                lme4 script for the primary single-predictor models

Nested group experiment (lmm_feature_groups). On ONE patient set (complete for
every feature in every group) it fits the base model (covariates only), one
model per group, and one model with all groups, then tests:
  <group> alone                 base + group     vs base
  all groups together           base + all       vs base
  <group> on top of the others  base + all       vs base + all-but-group
With two groups (collagen, fluorescence) these are exactly: collagen alone,
fluorescence alone, collagen + fluorescence, fluorescence on top of collagen,
collagen on top of fluorescence. Each feature adds 3 terms (level, slope
before the knot, change after the knot), so df = 3 x number of features.
  <tag> is "primary", "sensitivity_covariates" or "sensitivity_no_imputed".

CAVEAT: the mixed model assumes eGFR is missing at random given the model
terms. If short follow-up reflects graft failure or death rather than recent
transplantation, results are biased; see lmm_missingness.csv.
"""

import argparse
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import run_association_analysis_v2 as v2

WIDE_EGFR = {"egfr_6m": 0.5, "egfr_1y": 1.0, "egfr_3y": 3.0, "egfr_5y": 5.0}
DEFAULT_PREDICTORS = ["collagen_ratio", "collagen_iqr", "collagen_thickness_mean",
                      "f_mean_intensity_mean", "b_mean_intensity_mean"]
RE_LADDER = ["1 + time + time2", "1 + time", "1"]
# Singular-fit rule, matching lme4's isSingular(): a diagonal element of the
# Cholesky factor of cov_re / residual variance near 0 means one random effect
# is not supported by the data. statsmodels' optimiser tends to stop just
# inside the boundary where lme4 lands on it, so the tolerance is 0.05 (that
# random effect's SD, conditional on the others, is < 5% of the residual SD)
# rather than lme4's 1e-4; with this rule both pick the same structure.
SINGULAR_TOL = 0.05


# =====================================================================
#                          DATA PREPARATION
# =====================================================================
def id_column(df):
    return "study_id" if "study_id" in df.columns else "slide_name"


SERIES_COL = re.compile(r"egfr_(\d+)([my])")


def read_series(path):
    """Wide source file -> (long eGFR [study_id, time, egfr], extra patient columns)."""
    src = pd.read_csv(path)
    cols = [c for c in src.columns if SERIES_COL.fullmatch(c)]
    times = {c: int(SERIES_COL.fullmatch(c).group(1)) / (12 if c.endswith("m") else 1) for c in cols}
    long = src.melt(id_vars=["study_id"], value_vars=cols, var_name="visit", value_name="egfr")
    long["time"] = long["visit"].map(times)
    extra = [c for c in ["graft_status"] if c in src.columns]
    print(f"[INFO] full eGFR series from {path}: visits {sorted(set(times.values()))} y")
    return long[["study_id", "time", "egfr"]], src[["study_id"] + extra]


def build_long(df, egfr_long_path=None, knot=1.0, egfr_series_path=None):
    """One row per patient-visit. Baseline columns are repeated per row."""
    idc = id_column(df)
    if egfr_series_path:
        series, _ = read_series(egfr_series_path)
        drop = [c for c in df.columns if SERIES_COL.fullmatch(c)]
        long = series.merge(df.drop(columns=drop), on="study_id", how="inner")
    elif egfr_long_path:
        long = pd.read_csv(egfr_long_path)
        key = idc if idc in long.columns else "slide_name"
        long = long[[key, "time_years", "egfr"]].rename(columns={"time_years": "time"})
        long = long.merge(df, on=key, how="inner")
        print(f"[INFO] long-format eGFR from {egfr_long_path}")
    else:
        cols = [c for c in WIDE_EGFR if c in df.columns]
        long = df.melt(id_vars=[c for c in df.columns if c not in cols], value_vars=cols,
                       var_name="visit", value_name="egfr")
        long["time"] = long["visit"].map(WIDE_EGFR)
        print(f"[INFO] long-format eGFR built from {cols} (times {[WIDE_EGFR[c] for c in cols]} y)")
        if "egfr_n_points" in df.columns and df["egfr_n_points"].max() > len(cols):
            print(f"[WARNING] egfr_n_points goes up to {int(df['egfr_n_points'].max())}, but only "
                  f"{len(cols)} time points are in the clinical file. Export the full series "
                  f"and pass it as egfr_long to use every measurement.")
    long = long.dropna(subset=["egfr", "time"]).copy()
    long["id"] = long[idc].astype(str)
    long["time2"] = np.maximum(long["time"] - knot, 0.0)
    return long.sort_values(["id", "time"]).reset_index(drop=True)


def prepare_terms(long, predictors, covariates):
    """z-score predictors and mean-centre continuous covariates, both on the
    patient level (one value per patient), so every patient counts once."""
    long = long.copy()
    per_patient = long.drop_duplicates("id").set_index("id")
    for c in predictors:
        m, s = per_patient[c].mean(), per_patient[c].std()
        long[c] = (long[c] - m) / s if s > 0 else np.nan
    for c in covariates:
        if per_patient[c].nunique() > 2:
            long[c] = long[c] - per_patient[c].mean()
    return long


# =====================================================================
#                              FITTING
# =====================================================================
def _formula(slope_terms, baseline_terms):
    rhs = "(time + time2)"
    if slope_terms:
        rhs += " * (" + " + ".join(slope_terms) + ")"
    if baseline_terms:
        rhs += " + " + " + ".join(baseline_terms)
    return "egfr ~ " + rhs


def fit_lmm(data, slope_terms, baseline_terms, re):
    """ML fit. Returns (result, ok, note). ok is False when the optimiser did
    not converge or the random-effects covariance is on the boundary."""
    re_formula = None if re == "1" else "~" + re.replace("1 + ", "")
    model = smf.mixedlm(_formula(slope_terms, baseline_terms), data, groups=data["id"],
                        re_formula=re_formula)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = model.fit(reml=False, method=["lbfgs", "powell", "cg"])
        except (np.linalg.LinAlgError, ValueError) as e:
            return None, False, f"fit failed: {e}"
    notes = sorted({str(w.message).split("\n")[0] for w in caught
                    if issubclass(w.category, (ConvergenceWarning, RuntimeWarning, UserWarning))})
    rel = np.atleast_2d(res.cov_re.values) / res.scale
    try:
        singular = np.diag(np.linalg.cholesky(rel)).min() < SINGULAR_TOL
    except np.linalg.LinAlgError:
        singular = True
    ok = bool(res.converged) and not singular
    note = "; ".join(([] if res.converged else ["not converged"]) +
                     (["singular random-effects covariance"] if singular else []) + notes[:2])
    return res, ok, note


def choose_random_effects(data, slope_terms, baseline_terms, requested="auto"):
    ladder = RE_LADDER if requested == "auto" else [requested]
    for re in ladder:
        res, ok, note = fit_lmm(data, slope_terms, baseline_terms, re)
        print(f"   random effects ({re} | id): {'OK' if ok else 'rejected - ' + note}")
        if ok or requested != "auto":
            return re, res, ok, note
    return ladder[-1], res, ok, note


def lr_test(res_small, res_big):
    stat = 2 * (res_big.llf - res_small.llf)
    df = len(res_big.fe_params) - len(res_small.fe_params)
    return max(stat, 0.0), df, stats.chi2.sf(max(stat, 0.0), df)


def term_rows(res, var):
    """Level / pre-knot slope / post-knot slope-change coefficients of `var`."""
    out = {}
    for label, name in [("level", var), ("slope_pre_knot", f"time:{var}"),
                        ("slope_change_post_knot", f"time2:{var}")]:
        if name in res.fe_params.index:
            out[f"{label}_est"] = res.fe_params[name]
            out[f"{label}_se"] = res.bse_fe[name]
            out[f"{label}_p"] = res.pvalues[name]
    return out


# =====================================================================
#                      ANALYSIS (one covariate set)
# =====================================================================
def run_lmm_analysis(long_all, predictors, joint_predictors, references, slope_cov,
                     base_cov, re_requested, tag, out_dir, added_predictor=None):
    print(f"\n=== Mixed model [{tag}]: slope covariates {slope_cov}, "
          f"baseline covariates {base_cov} ===")
    covs = slope_cov + base_cov
    data = long_all.dropna(subset=covs + predictors).copy()
    data = prepare_terms(data, predictors, covs)
    n_pat, n_obs = data["id"].nunique(), len(data)
    print(f"   {n_pat} patients, {n_obs} eGFR measurements "
          f"({n_obs / max(n_pat, 1):.1f} per patient)")

    re, base, ok, note = choose_random_effects(data, slope_cov, base_cov, re_requested)
    if base is None:
        print("[ERROR] base model could not be fitted; skipping this analysis.")
        return
    pd.DataFrame({"estimate": base.fe_params, "se": base.bse_fe,
                  "p": base.pvalues[base.fe_params.index]}).assign(
        random_effects=re, n_patients=n_pat, n_obs=n_obs, fit_ok=ok, fit_note=note
    ).to_csv(os.path.join(out_dir, f"lmm_base_{tag}.csv"))
    print(f"   base model: slope before knot {base.fe_params['time']:+.2f} "
          f"(p={base.pvalues['time']:.3f}), change after knot {base.fe_params['time2']:+.2f} "
          f"(p={base.pvalues['time2']:.3f}) mL/min/1.73m2 per year at mean covariates")

    # ---- step 4: one collagen predictor at a time ----
    rows = []
    for pred in predictors:
        res, ok_k, note_k = fit_lmm(data, slope_cov + [pred], base_cov, re)
        row = {"predictor": pred, "n_patients": n_pat, "n_obs": n_obs, "random_effects": re,
               "fit_ok": ok_k, "fit_note": note_k}
        if res is not None:
            stat, df, p = lr_test(base, res)
            row.update({"lr_stat": stat, "lr_df": df, "lr_p": p, **term_rows(res, pred)})
        rows.append(row)
    single = pd.DataFrame(rows)
    if single["lr_p"].notna().any():
        m = single["lr_p"].notna()
        single.loc[m, "lr_p_fdr"] = stats.false_discovery_control(single.loc[m, "lr_p"])
    single.to_csv(os.path.join(out_dir, f"lmm_single_{tag}.csv"), index=False)
    show = [c for c in ["predictor", "lr_stat", "lr_df", "lr_p", "lr_p_fdr", "level_est", "level_p",
                        "slope_pre_knot_est", "slope_pre_knot_p", "slope_change_post_knot_est",
                        "slope_change_post_knot_p", "fit_ok"] if c in single.columns]
    print("\n   One predictor at a time (per SD; LR test vs base model):")
    print(single[show].round(4).to_string(index=False))

    # ---- step 5: joint model ----
    joint = [p for p in joint_predictors if p in data.columns]
    if len(joint) > 1:
        res, ok_j, note_j = fit_lmm(data, slope_cov + joint, base_cov, re)
        if res is not None:
            stat, df, p = lr_test(base, res)
            jrows = [{"predictor": pr, **term_rows(res, pr)} for pr in joint]
            pd.DataFrame(jrows).assign(joint_lr_stat=stat, joint_lr_df=df, joint_lr_p=p,
                                       fit_ok=ok_j, fit_note=note_j, n_patients=n_pat).to_csv(
                os.path.join(out_dir, f"lmm_joint_{tag}.csv"), index=False)
            print(f"\n   Joint model ({len(joint)} predictors, {df} extra parameters for "
                  f"{n_pat} patients): LR p = {p:.4f}, fit {'OK' if ok_j else 'problem: ' + note_j}")

    # ---- pathology reads, and the added predictor on top of them ----
    added = added_predictor or predictors[0]
    prow = []
    for ref in [r for r in references if r in long_all.columns]:
        d = long_all.dropna(subset=covs + predictors + [ref]).copy()
        d = prepare_terms(d, predictors + [ref], covs)
        b0, ok0, _ = fit_lmm(d, slope_cov, base_cov, re)
        b_ref, ok1, _ = fit_lmm(d, slope_cov + [ref], base_cov, re)
        b_both, ok2, _ = fit_lmm(d, slope_cov + [ref, added], base_cov, re)
        if None in (b0, b_ref, b_both):
            continue
        s_ref, df_ref, p_ref = lr_test(b0, b_ref)
        s_add, df_add, p_add = lr_test(b_ref, b_both)
        prow.append({"reference": ref, "added_predictor": added,
                     "n_patients": d["id"].nunique(), "n_obs": len(d),
                     "lr_p_reference_over_base": p_ref,
                     "lr_p_added_over_reference": p_add,
                     **{f"ref_{k}": v for k, v in term_rows(b_ref, ref).items()},
                     "fit_ok": ok0 and ok1 and ok2})
    if prow:
        pr = pd.DataFrame(prow)
        pr.to_csv(os.path.join(out_dir, f"lmm_over_pathology_{tag}.csv"), index=False)
        print(f"\n   Pathology read (per SD) and {added} added on top of it:")
        print(pr[["reference", "n_patients", "lr_p_reference_over_base",
                  "lr_p_added_over_reference", "ref_slope_pre_knot_est",
                  "ref_slope_pre_knot_p", "fit_ok"]].round(4).to_string(index=False))
    return data


# =====================================================================
#                 NESTED FEATURE-GROUP EXPERIMENT
# =====================================================================
def run_group_experiment(long_all, groups, slope_cov, base_cov, re_requested, tag, out_dir):
    feats = [f for g in groups.values() for f in g]
    missing = [f for f in feats if f not in long_all.columns]
    if missing:
        print(f"[WARNING] group experiment [{tag}] skipped: features not in the data: {missing}")
        return None
    data = long_all.dropna(subset=slope_cov + base_cov + feats).copy()
    data = prepare_terms(data, feats, slope_cov + base_cov)
    n_pat = data["id"].nunique()
    print(f"\n=== Group experiment [{tag}]: {groups} | {n_pat} patients, {len(data)} eGFR values ===")
    re, base, ok0, note0 = choose_random_effects(data, slope_cov, base_cov, re_requested)
    names = list(groups)
    fits = {(): (base, ok0)}

    def fit(combo):
        if combo not in fits:
            terms = [f for g in combo for f in groups[g]]
            res, ok, _ = fit_lmm(data, slope_cov + terms, base_cov, re)
            fits[combo] = (res, ok)
        return fits[combo]

    full = tuple(names)
    tests = [(f"{g} alone", (), (g,)) for g in names]
    if len(names) > 1:
        tests.append(("all groups together" if len(names) > 2 else " + ".join(names), (), full))
        tests += [(f"{g} on top of {' + '.join(h for h in names if h != g)}",
                   tuple(h for h in names if h != g), full) for g in names]
    rows = []
    for label, small, big in tests:
        (r0, ok_a), (r1, ok_b) = fit(small), fit(big)
        stat, df_, p = lr_test(r0, r1)
        rows.append({"test": label, "model_1": "covariates" + "".join(f" + {g}" for g in small),
                     "model_2": "covariates" + "".join(f" + {g}" for g in big),
                     "lr_stat": stat, "df": df_, "p": p, "n_patients": n_pat, "n_obs": len(data),
                     "random_effects": re, "fits_ok": ok_a and ok_b})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, f"lmm_groups_{tag}.csv"), index=False)
    coefs = [{"model": " + ".join(c) or "covariates", "feature": f, **term_rows(fits[c][0], f)}
             for c in fits if c for g in c for f in groups[g]]
    pd.DataFrame(coefs).to_csv(os.path.join(out_dir, f"lmm_groups_coefs_{tag}.csv"), index=False)
    print(out[["test", "df", "lr_stat", "p", "fits_ok"]].round(4).to_string(index=False))
    return out


# =====================================================================
#                      STEP 1: MISSING eGFR
# =====================================================================
def missingness_table(df, long, knot, out_dir):
    """Compare patients with and without any eGFR after the knot. Differences
    in 1-year eGFR or pathology suggest follow-up is NOT missing at random
    (e.g. graft failure); no differences are consistent with, but do not
    prove, missing at random. Transplant dates and graft-loss/death records
    are needed to settle it."""
    has_late = long[long["time"] > knot].groupby("id").size()
    idc = id_column(df)
    d = df.copy()
    d["id"] = d[idc].astype(str)
    d["has_egfr_after_knot"] = d["id"].isin(has_late.index).astype(int)
    if "graft_status" in d.columns:
        print("   graft status by group:")
        print(pd.crosstab(d["graft_status"].fillna("not recorded"),
                          d["has_egfr_after_knot"].map({1: "eGFR after knot", 0: "none after knot"}))
              .to_string())
    rows = []
    for var in ["egfr_6m", "egfr_1y", "recipient_age_tx", "donor_age", "cold_ischemia_min",
                "collagen_ratio", "ci", "if_pct", "egfr_slope_theilsen"]:
        if var not in d.columns:
            continue
        a = d.loc[d.has_egfr_after_knot == 1, var].dropna()
        b = d.loc[d.has_egfr_after_knot == 0, var].dropna()
        p = stats.mannwhitneyu(a, b).pvalue if len(a) > 2 and len(b) > 2 else np.nan
        rows.append({"variable": var, "n_with_later_egfr": len(a), "median_with": a.median(),
                     "n_without_later_egfr": len(b), "median_without": b.median(), "p_mannwhitney": p})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out_dir, "lmm_missingness.csv"), index=False)
    print(f"\n=== Step 1: patients with vs without eGFR after {knot:g} y ===")
    print(f"   {int(d.has_egfr_after_knot.sum())} with, {int((1 - d.has_egfr_after_knot).sum())} without")
    print(t.round(3).to_string(index=False))


# =====================================================================
#                       lme4 CROSS-CHECK SCRIPT
# =====================================================================
def write_r_check(out_dir, slope_cov, base_cov, predictors, re):
    rhs_cov = " + ".join(slope_cov)
    base_extra = (" + " + " + ".join(base_cov)) if base_cov else ""
    rscript = f'''# lme4 reproduction of the primary single-predictor models
# (run from the project root:  Rscript {out_dir}/lmm_check.R)
library(lme4)
dat <- read.csv("{out_dir}/lmm_egfr_long.csv")
dat <- dat[complete.cases(dat[, c({", ".join(repr(c) for c in slope_cov + base_cov + predictors)})]), ]
ctrl <- lmerControl(optimizer = "bobyqa", check.conv.singular = "warning")
base <- lmer(egfr ~ (time + time2) * ({rhs_cov}){base_extra} + ({re} | id),
             data = dat, REML = FALSE, control = ctrl)
print(summary(base)$coefficients)
for (p in c({", ".join(repr(c) for c in predictors)})) {{
  f <- as.formula(paste0("egfr ~ (time + time2) * ({rhs_cov} + ", p, "){base_extra} + ({re} | id)"))
  m <- lmer(f, data = dat, REML = FALSE, control = ctrl)
  a <- anova(base, m)
  cat(sprintf("%-26s LR = %6.3f  df = %d  p = %.4f\\n", p, a$Chisq[2], a$Df[2], a$`Pr(>Chisq)`[2]))
}}
'''
    with open(os.path.join(out_dir, "lmm_check.R"), "w") as f:
        f.write(rscript.replace("'", '"'))


# =====================================================================
#                                MAIN
# =====================================================================
def main(args):
    out_dir = getattr(args, "out_dir", None) or "./results"
    os.makedirs(out_dir, exist_ok=True)
    knot = float(getattr(args, "lmm_knot_years", 1.0))
    predictors = getattr(args, "lmm_predictors", None) or DEFAULT_PREDICTORS
    joint = getattr(args, "lmm_joint_predictors", None) or predictors
    refs = getattr(args, "lmm_references", None) or ["ci", "if_pct"]
    slope_cov = getattr(args, "lmm_slope_covariates", None) or ["recipient_age_tx", "recipient_sex_male"]
    base_cov = getattr(args, "lmm_baseline_covariates", None) or []
    sens = getattr(args, "lmm_sensitivity_covariates", None) or {
        "slope": ["recipient_age_tx", "recipient_sex_male", "donor_age"],
        "baseline": ["cold_ischemia_min"]}
    re_req = getattr(args, "lmm_random_effects", "auto")

    aliases = getattr(args, "slide_aliases", None)
    df = v2.load_inputs(args.clinical, args.collagen, aliases)
    for path in getattr(args, "extra_features", None) or []:
        extra_feats = pd.read_csv(path)
        if aliases:
            extra_feats["slide_name"] = extra_feats["slide_name"].replace(aliases)
        df = df.merge(extra_feats, on="slide_name", how="left")
        print(f"[INFO] merged {[c for c in extra_feats.columns if c != 'slide_name']} from {path}")
    missing = [p for p in predictors if p not in df.columns]
    if missing:
        print(f"[WARNING] predictors not found and skipped: {missing}")
    predictors = [p for p in predictors if p in df.columns]
    added = getattr(args, "lmm_added_predictor", None)
    series_path = getattr(args, "egfr_series", None)
    if series_path:
        _, extra = read_series(series_path)
        df = df.merge(extra, on="study_id", how="left")
    long = build_long(df, getattr(args, "egfr_long", None), knot, series_path)
    print(f"[INFO] {long['id'].nunique()} patients, {len(long)} eGFR measurements; "
          f"per patient: {long.groupby('id').size().value_counts().sort_index().to_dict()}")
    missingness_table(df, long, knot, out_dir)

    data = run_lmm_analysis(long, predictors, joint, refs, slope_cov, base_cov, re_req,
                            "primary", out_dir, added)
    if data is not None:
        data.to_csv(os.path.join(out_dir, "lmm_egfr_long.csv"), index=False)
        base_re = pd.read_csv(os.path.join(out_dir, "lmm_base_primary.csv"))["random_effects"].iloc[0]
        write_r_check(out_dir, slope_cov, base_cov, predictors, base_re)

    run_lmm_analysis(long, predictors, joint, refs, sens["slope"], sens.get("baseline", []),
                     re_req, "sensitivity_covariates", out_dir, added)

    groups = getattr(args, "lmm_feature_groups", None)
    if groups:
        no_imp = long[long.get("egfr_6m_1y_imputed", pd.Series("NO", index=long.index))
                      .astype(str).str.upper() != "YES"]
        for tag, data_, sc, bc in [("primary", long, slope_cov, base_cov),
                                   ("sensitivity_covariates", long, sens["slope"], sens.get("baseline", [])),
                                   ("sensitivity_no_imputed", no_imp, slope_cov, base_cov)]:
            run_group_experiment(data_, groups, sc, bc, re_req, tag, out_dir)

    if getattr(args, "lmm_exclude_imputed", True) and "egfr_6m_1y_imputed" in long.columns:
        keep = long[long["egfr_6m_1y_imputed"].astype(str).str.upper() != "YES"]
        print(f"\n[INFO] sensitivity without imputed 6m/1y values: "
              f"{long['id'].nunique() - keep['id'].nunique()} patients removed")
        run_lmm_analysis(keep, predictors, joint, refs, slope_cov, base_cov, re_req,
                         "sensitivity_no_imputed", out_dir, added)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DUET fibrosis analysis v3 (v2 arms + eGFR mixed model)")
    ap.add_argument("--config", type=str, default="./config/association_analysis.json")
    ap.add_argument("--classifier", type=str, choices=["logistic", "xgboost"], default=None)
    ap.add_argument("--skip-v2", action="store_true", help="run only the mixed-model arm")
    cli, _ = ap.parse_known_args()
    with open(cli.config) as f:
        cfg = json.load(f)
    if cli.classifier:
        cfg["classifier"] = cli.classifier
    run_args = argparse.Namespace(**cfg)
    if not cli.skip_v2:
        v2.main(run_args)
    main(run_args)
