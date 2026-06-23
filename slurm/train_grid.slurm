#!/bin/bash
#SBATCH --job-name=track_train
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# =====================================================
# User configuration
# =====================================================

TIFF=src/trackrecon/data/Airyscan_data/YOUR_FILE.ome.tiff
CONFIG=config.yaml

# =====================================================
# Environment
# =====================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlbio

REPO_DIR=$SLURM_SUBMIT_DIR

cd $REPO_DIR

echo "====================================="
echo "Working directory:"
pwd
echo "====================================="

which python
which uv

PARAMS=(
"1e-4 100"
"1e-4 200"
"1e-4 400"
"1e-5 100"
"1e-5 200"
"1e-5 400"
"1e-6 100"
"1e-6 200"
"1e-6 400"
)

read LR PATCH <<< "${PARAMS[$SLURM_ARRAY_TASK_ID]}"

echo "====================================="
echo "LR               = ${LR}"
echo "PATCHES/EPOCH    = ${PATCH}"
echo "====================================="

uv run python ${REPO_DIR}/run_experiment_main_pipeline_script.py \
    --tiff ${TIFF} \
    --infile_yaml ${CONFIG} \
    --roi \
    --run_train \
    --lr ${LR} \
    --patches_per_epoch ${PATCH} \
    --preprocess_dir preprocessing \
    --outdir_local experiments \
    --save_plots
