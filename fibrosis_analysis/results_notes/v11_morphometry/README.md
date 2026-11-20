# v11: collagen morphometry, and prediction of rapid eGFR decline (2026-11-10)

Run: `python src/analyses/v11_morphometry/extract_morphometry.py --jobs 10`, then `python src/analyses/v11_morphometry/predict_rapid_decline.py --perms 100`.

- **Features:** 13 morphometry measures per 500 µm tile, each kept as a collagen-weighted mean and a tile-to-tile IQR, so 26 per biopsy.
  - The measures are region density, area (mean and 90th percentile), eccentricity, solidity, major axis, perimeter/area, fibre length density, branch density, orientation coherence, lacunarity (16 and 64 µm) and nearest-neighbour distance.
  - They use the v5 collagen mask (M > 127 and CE > 0); the total collagen area is 1.009× v5's. 27,055 interior tiles from 68 slides.
- **Models:** leave-one-patient-out. Every model includes age and sex. The chance range is the 5th–95th percentile with the outcome shuffled 100 times.

## Rapid decline (66 patients, 13 events), LOOCV AUC

| features | logistic | xgboost | chance range |
|---|---|---|---|
| age, sex only | 0.08 | 0.42 | 0.00–0.64 |
| + morphometry (26) | 0.49 | 0.42 | 0.15–0.68 |
| + v5 imaging (59) | 0.35 | 0.25 | 0.24–0.63 |
| + v5 + morphometry (85) | 0.57 | 0.22 | 0.22–0.65 |
| + Banff ci (n 60) | 0.22 | 0.32 | 0.00–0.64 |
| + IF% (n 45) | 0.30 | 0.73 (p 0.01) | 0.01–0.66 |

eGFR slope: the leave-one-out R² is below 0 for every set, so no model beats predicting the average.

## Morphometry vs pathology (Spearman, n = 62 ci, 47 IF%)

- 13 of 26 features track Banff ci with FDR q < 0.05.
- **Largest:**
  - mean region area: ρ 0.45 with ci, 0.38 with IF%
  - 90th-percentile region area: ρ 0.43
  - eccentricity: ρ −0.43
- **Reading:** in more fibrotic biopsies, collagen forms larger and less elongated (more confluent) patches.
