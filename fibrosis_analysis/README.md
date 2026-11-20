# Fibrosis analysis (DUET Aim3, T0 kidney transplant biopsies)

Current version of the handcrafted collagen/fluorescence features and their analysis against pathology and eGFR.
Run scripts from this folder; they expect `data/` and write to `results/` (neither is in git: patient data).

- `src/FeatureExtractor_wsi_v5.py` (config `config/feat_extract_wsi_v5.json`): whole-slide feature extraction from
  B (H&E), F (DUET fluorescence), CE and the collagen prediction map M; 1000 px (500 um) tiles, collagen mask
  M > 0.5 and CE > 0, per-slide mean and tile-to-tile IQR. `src/analyses/make_v5_feature_list.py` writes the list
  of the 59 features (`results_notes/v5_feature_list.csv`).
- `src/run_association_analysis_v3.py` (uses `run_association_analysis_v2.py`, config
  `config/association_analysis.json`): eGFR linear mixed model (Dr. Jarcy Zee's specification),
  `egfr ~ (time + time2) * feature + (1 + time | id)`, knot at 1 year, likelihood-ratio tests.
- `src/analyses/`:
  - `v8_M_values`: features of the model output M; lme4 check in R.
  - `v9_unadjusted_mse`: leave-one-patient-out prediction error of the mixed model.
  - `v10_source_only_vs_DA`: domain-adapted vs source-only collagen model, agreement with Banff ci and IF%.
  - `v11_morphometry`: collagen morphometry and leave-one-out models of rapid eGFR decline.
- `results_notes/`: the written summary of v8, v10 and v11.
- `study_design_and_analysis_plan.md`: the study plan.

Athena upload of the collagen masks: `../claude_utils/collagen_json_athena/`.
