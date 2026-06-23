#!/bin/bash
#SBATCH --job-name=track_preprocess
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# =====================================================
# User configuration
# =====================================================

TIFF=/src/trackrecon/data/Airyscan_data/YOUR_FILE.ome.tiff
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

echo "Files:"
ls
echo "====================================="

which python
which uv

mkdir -p logs

# =====================================================
# Preprocessing
# =====================================================

uv run python ${REPO_DIR}/run_experiment_main_pipeline_script.py \
    --tiff ${TIFF} \
    --infile_yaml ${CONFIG} \
    --roi \
    --run_detect \
    --run_filter \
    --run_mask \
    --save_plots \
    --outdir_local preprocessing
