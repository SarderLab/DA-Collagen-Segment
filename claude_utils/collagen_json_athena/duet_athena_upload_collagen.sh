#!/usr/bin/env bash
#SBATCH --qos=pinaki.sarder
#SBATCH --job-name=duet_athena_collagen
#SBATCH --partition=hpg-turin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=24:00:00
#SBATCH --output=/blue/pinaki.sarder/f.afsari/4-DUET/Aim3/cortex_medulla/logs/%x_%j.out

# Upload the collagen mask JSONs (duet_collagen_mask_json.py, green layer) onto the DUET slides to Athena collection 6b005d3fec7333818fc7b969.
# One-time setup (your own key, Athena > your name > My account > API keys, scope: read + write data):
#   echo 'export ATHENA_API_KEY=<your key>' > ~/.athena_key && chmod 600 ~/.athena_key
# Submit from this repository directory:
#   sbatch duet_athena_upload_collagen.sh              # all slides
#   sbatch duet_athena_upload_collagen.sh 133H-0       # just one, to test
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

[[ -f ~/.athena_key ]] || { echo "missing ~/.athena_key (see header)" >&2; exit 2; }
source ~/.athena_key

LIB=/blue/pinaki.sarder/f.afsari/4-DUET/Aim3/cortex_medulla/pylibs
module load python
if ! PYTHONPATH=$LIB python -c "import girder_client" 2>/dev/null; then
  python -m pip install --quiet --target "$LIB" girder-client
fi
export PYTHONPATH=$LIB

ONLY=()
[[ $# -gt 0 ]] && ONLY=(--only "$@")
python duet_athena_upload.py \
  --collection 6b005d3fec7333818fc7b969 \
  --images /orange/pinaki.sarder/Davy_Jones_Locker/Farzad_Fibrosis/aim3/images \
  --masks /blue/pinaki.sarder/f.afsari/4-DUET/Aim3/collagen_mask_json --mask-suffix _collagen_mask.json \
  "${ONLY[@]}"
