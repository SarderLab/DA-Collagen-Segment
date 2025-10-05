#!/bin/sh
#SBATCH --qos=pinaki.sarder
#SBATCH --job-name=CG_PA_G_RGB_D6
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=25gb
#SBATCH --time=60:00:00
#SBATCH --output=logs/data_v6/train_CGPL_UCD_PATH_Green_RGB.out

date; pwd; hostname
#module load singularity
module load conda
ml
start_time=$(date +%s)
conda activate duet9-gpu
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZWQ3NDYzZS1jYzc4LTQ5ZDgtOWNhNS0yMDE2YTI1ODk2NmUifQ=="


python3 main.py Config/train_CGPL_UCD_PATH_Green_RGB.json

end_time=$(date +%s)
time_diff=$((end_time - start_time)); hours=$((time_diff / 3600)); minutes=$(((time_diff % 3600) / 60)); seconds=$((time_diff % 60))
echo "Running completed in: ${hours}:${minutes}:${seconds} Hours:Minutes:Seconds"
