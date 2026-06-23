#!/bin/bash
#SBATCH --job-name=track_eval
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH --array=0-80
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

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

MODELS=(
"lr_1e-4_patch_100"
"lr_1e-4_patch_200"
"lr_1e-4_patch_400"
"lr_1e-5_patch_100"
"lr_1e-5_patch_200"
"lr_1e-5_patch_400"
"lr_1e-6_patch_100"
"lr_1e-6_patch_200"
"lr_1e-6_patch_400"
)

THRESHOLDS=(0.7 0.8 0.9)
MINSIZES=(50 100 250)

MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 9))
SCAN_INDEX=$((SLURM_ARRAY_TASK_ID % 9))

THR_INDEX=$((SCAN_INDEX / 3))
MIN_INDEX=$((SCAN_INDEX % 3))

MODEL=${MODELS[$MODEL_INDEX]}
THRESHOLD=${THRESHOLDS[$THR_INDEX]}
MIN_SIZE=${MINSIZES[$MIN_INDEX]}

MODEL_PATH=experiments/${MODEL}/models/model.pth

echo "====================================="
echo "MODEL      = ${MODEL}"
echo "MODEL_PATH = ${MODEL_PATH}"
echo "THRESHOLD  = ${THRESHOLD}"
echo "MIN_SIZE   = ${MIN_SIZE}"
echo "====================================="

uv run python ${REPO_DIR}/run_experiment_main_pipeline_script.py \
    --tiff ${TIFF} \
    --infile_yaml ${CONFIG} \
    --roi \
    --run_infer \
    --pred_all \
    --pred_septs \
    --preprocess_dir preprocessing \
    --model_path ${MODEL_PATH} \
    --threshold ${THRESHOLD} \
    --min_size ${MIN_SIZE} \
    --outdir_local experiments \
    --save_plots \
    --save_result
