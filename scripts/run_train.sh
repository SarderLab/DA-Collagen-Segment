#!/bin/sh
#SBATCH --qos=pinaki.sarder
#SBATCH --job-name=cluster
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20gb
#SBATCH --time=60:00:00
#SBATCH --output=logs/cluster.out

date; pwd; hostname
#module load singularity
module load conda
ml
start_time=$(date +%s)
#nvidia-smi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhODE0Mzc0MC03MmQ5LTQ0NjYtOGU2OC1jZDQ1NGZhODUxNDAifQ==",
export TORCH_HOME=/blue/pinaki.sarder/f.afsari/.torch_cache
conda activate collagen_segment


python3 main.py Config/cluster_CGPL_Targets.json
end_time=$(date +%s)
time_diff=$((end_time - start_time)); hours=$((time_diff / 3600)); minutes=$(((time_diff % 3600) / 60)); seconds=$((time_diff % 60))
echo "Running completed in: ${hours}:${minutes}:${seconds} Hours:Minutes:Seconds"
