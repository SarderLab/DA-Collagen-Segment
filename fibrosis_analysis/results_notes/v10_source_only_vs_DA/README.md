# v10: source-only vs DA collagen model, agreement with pathology (PRELIMINARY)

Run: `python src/analyses/v10_source_only_vs_DA/source_only_vs_da.py` (re-run when more slides are predicted; finished slides are cached in `source_only_tiles/`).

- **Slides:** 26 of 69 T0 slides were fully predicted by the source-only model by 2026-11-08 (the run stopped at 17H-0, 10 of 44 tiles); ci for 24, IF% for 15.
- **Same tiles for both models:** each 3000 px source-only tile is split into its nine 1000 px DA tiles. All DA tiles of these slides were covered.
- **Tissue mask:** v5 rules re-implemented on the H&E tiles. Total area is 0.98× DA's and per-tile ρ = 0.98. Glomeruli are not masked out for source-only.

## Spearman ρ with pathology (same 24 / 15 patients)

| feature | ci: DA | ci: source-only | IF%: DA | IF%: source-only |
|---|---|---|---|---|
| M variability | 0.41 (p 0.047) | 0.43 (p 0.037) | 0.45 | 0.22 |
| M density | 0.32 | 0.37 | 0.15 | 0.39 |
| collagen ratio, M > 0.5 | 0.34 | 0.30 | 0.30 | 0.32 |

Bootstrap 95% CIs for ρ(DA) − ρ(source-only) all include 0.

## Amount of collagen called

| | DA | source-only |
|---|---|---|
| median collagen ratio | 20% | 54% |
| median M density | 0.25 | 0.54 |

The pathologists scored 21 of these 24 biopsies ci 0 and IF% ≤ 5%.

## Reading

- **Ranking:** no DA advantage on these 26 slides. This group has little range (3 of 24 are ci ≥ 1), so the test has almost no power.
- **Amount:** source-only labels about half of the tissue as collagen in kidneys scored ci 0. DA labels about a fifth. Collagen ratio includes normal collagen (interstitium, vessels, basement membranes), so neither number is directly the pathologist's IF%.
