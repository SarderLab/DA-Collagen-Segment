# Collagen masks to Athena annotations

Uploads the DUET Aim3 H&E slides to Athena (Digital Slide Archive) with two kinds of mask overlays:
the collagen segmentation (green) and the cortex/medulla prediction (blue/yellow).

| File | What it does |
|---|---|
| `duet_collagen_mask_json.py` | Collagen mask of each slide (M > 127 and CE > 0, the v5 feature-extraction mask) traced at 2 µm/px into one DSA annotation JSON, `<slide>_B_collagen_mask.json`, level-0 coordinates |
| `duet_athena_upload.py` | Uploads `<slide>/B.svs` as item `<slide>_B.svs` and posts annotation JSONs onto it; reruns skip what is already there |
| `duet_athena_upload.sh` | SLURM job: slides + cortex/medulla masks (`*_mask_prediction.json`) |
| `duet_athena_upload_collagen.sh` | SLURM job: slides + collagen masks (`*_collagen_mask.json`) |

## Run

1. Your own Athena API key, once (Athena > your name > My account > API keys, scope read + write data):
   `echo 'export ATHENA_API_KEY=<key>' > ~/.athena_key && chmod 600 ~/.athena_key`. Never commit the key.
2. Collagen JSONs: `python duet_collagen_mask_json.py --images <aim3/images> --out <dir> --jobs 4`
   (reads M.svs and CE.svs from each slide folder).
3. From this folder: `sbatch duet_athena_upload_collagen.sh` (or `... 133H-0` to test one slide), and
   `sbatch duet_athena_upload.sh` for the cortex/medulla masks.

Paths, the collection ID (`6b005d3fec7333818fc7b969`) and `--qos=pinaki.sarder` are set in the `.sh` files.
The cortex/medulla masks come from `duet_cortex_medulla.sh` in the cortex-medullary-identification repo
(TRIDENT + UNI-v1 + Jamie Fermin's KPMP SVC), which is not copied here because it runs inside that repo.
