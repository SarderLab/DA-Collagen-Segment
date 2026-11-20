# DUET Fibrosis → Transplant Outcome Study: Design and Analysis Plan

## 1. Research question and hypothesis

**Question.** Does collagen quantification derived from DUET dual-mode imaging (paired H&E brightfield + collagen-highlighting fluorescence capture of the same slide) predict post-transplant graft function decline, and does it outperform pathologist-read interstitial fibrosis (IF) and tubular atrophy (TA) percentages?

**Primary hypothesis.** Slide-level DUET collagen area ratio at the index biopsy (T0) is associated with eGFR decline and improves prediction of KDIGO rapid progression over pathologist-read IF%.

**Secondary hypothesis.** The collagen signal is recoverable from H&E alone (virtual collagen staining), and T0→T1 collagen change tracks functional decline.

## 2. Cohort and data (verified against the clinical workbook)

- 69 unique patients (unique `study_id`, unique slide names of the form `1H-0`); 56 with both T0 and T1 images, 7 without, 6 unrecorded.
- **Outcomes (usable):**
  - eGFR slope, Theil-Sen robust (n=67) — primary continuous outcome
  - KDIGO rapid decline, slope < −5 (14 Yes / 53 No, n=67) — co-primary binary
  - 3-class trajectory: Improving 24 / Stable 29 / Declining 14 — secondary
  - eGFR at 3m–8y for secondary cross-sectional analyses
- **Outcomes (excluded as too sparse):** graft status (1 failure), year-1 rejection (2 non-null), BK nephropathy (2 non-null), KDPI (14 non-null).
- **Pathology comparators:** IF% and TA% (n=48, recorded in bands: 0, <5, 5, 10, 15, 30; bands converted to numeric with `<x` → x/2). Banff i-IFTA/t-IFTA available in only ~30.
- **Covariates (defensible adjustment set):** recipient age (69/69), recipient sex (69/69), donor age (43/69), cold ischemia time (58/69). Donor DM (15/69) and recipient DM/HTN (33/69) are too sparse; models report complete-case n (≈42–48 with full adjustment).
- **QC performed:** trajectory class and rapid-decline flags are internally consistent with the Theil-Sen slopes (0 mismatches). 33/69 patients have ≤1 year of eGFR follow-up (2-point slopes); 8/69 slopes use imputed 6m/1y values (flagged in `egfr_6m_1y_imputed`).

## 3. Study arms

### A1 — Association (core)
Slide-level DUET collagen metrics from T0 slides: **collagen area ratio** (primary), median patch ratio, and patch-ratio IQR (heterogeneity). Linear regression for eGFR slope; logistic regression for rapid decline; ordinal model for trajectory class; each adjusted for the covariate set. BH-FDR across the pre-specified primary tests.

### A2 — Head-to-head vs pathology read
Same models substituting pathologist IF%/TA% (n=48). Comparisons: LOOCV AUC for rapid decline (DeLong paired test), R² difference for slope (bootstrap), likelihood-ratio test of DUET-over-IF% incremental value, and discordance review of cases where DUET and the pathologist disagree.

### A3 — H&E-only distillation (virtual collagen staining)
- **Stage 1 (patch level):** frozen foundation-model embeddings of H&E patches (Prov-GigaPath primary — pretrained on 1.3B tiles from 171k slides spanning 31 tissue types; DINOv2 fallback) → ridge/elastic-net regression predicting the DUET collagen ratio for that patch. Feature selection inside CV folds. Patch-level acceptance: held-out ICC > 0.6.
- **Stage 2 (slide level):** aggregate patch predictions (mean + quartiles) → slide-level virtual fibrosis score → prognostic value tested as in A1/A2.
- **Stain-variation ablation:** Macenko/Reinhard normalization vs raw H&E, to quantify how much the frozen FM backbone already absorbs staining differences.
- End-to-end fine-tuning of transformer encoders and attention-MIL are deferred: at n≈50 slides with 14 events, they carry high overfitting risk. Revisit if the cohort grows or an external cohort is added.

### A4 — Secondary: progression phenotype
T0→T1 Δcollagen ratio as a phenotype; T0 collagen predicting T1 IF%/trajectory; Δcollagen vs eGFR slope.

## 4. Statistical and validation rules

- No external cohort → **discovery-only** framing; nested validation: LOOCV at slide level for A3, repeated k-fold for A1/A2.
- All feature selection, band conversion, and aggregation learned inside CV folds.
- Class imbalance (14/53): stratified folds, class weights, report AUC + sensitivity at fixed specificity.
- Maximum 2–4 adjusted covariates given complete-case n≈42–48; EPV reported.
- Patch sampling: uniform grid, 20× (10× sensitivity), 512×512, ≥50% tissue; per-slide patch counts recorded.

## 5. Execution split and current status

**Completed in this session:**
- Clinical QC (Section 2 findings)
- `analysis_ready_clinical.csv` — analysis-ready dataset (69 × 30)
- `table1_baseline_by_rapid_decline.csv` — Table 1
- `run_association_analysis.py` — A1/A2 pipeline, smoke-tested end-to-end (handles timepoint pivots, IF% band conversion, covariate missingness, LOOCV AUC, DeLong test)
- `collagen_scores_TEMPLATE.csv` — expected input format

**Needed from your side (GPU/WSI infra):**
1. DUET segmentation inference on all T0 (and T1) slides → per-patch collagen ratios
2. Patch extraction (512×512, 20×, ≥50% tissue) + frozen FM embeddings → per-slide parquet of patch embeddings
3. Deliver both as: `collagen_scores.csv` (study_id, timepoint, collagen_ratio, collagen_median, collagen_iqr) + embedding files; then `run_association_analysis.py` produces A1/A2 results immediately.

**Estimated compute:** the statistical layer runs in minutes on the sandbox (8 CPU). Patch extraction + FM inference is ~2–4 h/slide CPU-only — run on your GPU infra or in chunked background batches.

## 6. Limitations to state in the paper

Single center, discovery-only (no external validation), heterogeneous eGFR follow-up (33/69 ≤ 1 year), 2-point slopes for short follow-ups, 8/69 slopes with imputed 6m/1y values, zero-inflated banded IF% comparator, missing DSA/immunosuppression/timing covariates, and 14 events limiting the binary outcome model.
