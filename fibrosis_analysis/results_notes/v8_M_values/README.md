# Extraction v5: collagen prediction map M vs eGFR

Run: `python run_association_analysis_v3.py --config ./config/association_analysis.json`

- **Collagen file:** `data/T0_T1_v5_M_values/collagen_scores.csv`.
- **Mask:** M > 0.5 (and CE > 0, which excludes the glomeruli).
- **Previous features:** the CE > slide-Otsu versions from extraction v4 (`data/collagen_scores.csv`) are merged in as `*_otsu` by `make_otsu_reference.py`.
- **Columns that changed in v5:** every column defined "inside collagen" (green, H&E, CE, ratio, thickness), because the mask changed. Green variability under the M mask vs the Otsu mask: ρ = 0.98.

## M is a valid fibrosis measure (Spearman with the pathologist's scores)

| feature | ci (n = 63) | IF% (n = 48) |
|---|---|---|
| `m_mean_iqr` (tile-to-tile IQR of M) | **0.48** | 0.36 |
| `m_density` (mean M over tissue) | 0.37 | 0.26 |
| collagen ratio, M mask | 0.35 | 0.27 |
| collagen ratio, Otsu mask | 0.34 | 0.22 |
| CE density (Otsu run) | 0.19 | 0.21 |
| green variability | 0.01 | 0.17 |

## eGFR mixed model (age, sex; LR test, 3 df)

| predictor | all (69) | no imputed (61) | + donor age, CIT (43) |
|---|---|---|---|
| `m_density` | 0.75 | 0.80 | 0.72 |
| `m_mean_iqr` | 0.43 | 0.71 | 0.58 |
| `m_collagen_mean_mean` | 0.68 | 0.78 | 0.28 |
| collagen ratio, M mask | 0.67 | 0.81 | 0.68 |
| collagen IQR, M mask | 0.43 | 0.85 | 0.71 |
| collagen ratio, Otsu (previous) | 0.71 | 0.79 | 0.43 |
| CE density, Otsu (previous) | 0.49 | 0.33 | 0.39 |
| **green variability, M mask** | **0.028** | **0.015** | 0.12 |
| **green variability, Otsu (previous)** | **0.026** | **0.014** | 0.071 |

- **FDR:** across the 9 predictors, the q-value is 0.125 for both green variability versions.
- **Pathology reference:** ci on its own is p = 0.13, and M density adds nothing on top of it (p = 0.85).

## Group tests (as on deck slide 6): M collagen = `m_density` + `m_mean_iqr`

| test | all | no imputed | + donor age, CIT |
|---|---|---|---|
| M collagen alone | 0.45 | 0.80 | 0.84 |
| green variability alone | **0.028** | **0.015** | 0.12 |
| M collagen on top of green | 0.41 | 0.86 | 0.85 |
| green on top of M collagen | **0.024** | **0.019** | 0.13 |

## Reading

- **What M measures:** the same fibrosis the pathologist scores. It matches ci better than the Otsu or CE measures did.
- **What predicts decline:** fibrosis at biopsy, whether scored by the pathologist (ci) or measured by M, does not predict later eGFR decline in these 69 patients.
- **What green adds:** green variability predicts decline, is unrelated to ci and to M, and still predicts on top of M.
- **Not yet strong:** the green result does not survive the FDR correction across the 9 predictors (q = 0.125), and with the fuller covariates (n = 43) it is p = 0.07–0.12.
