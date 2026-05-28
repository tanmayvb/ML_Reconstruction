#!/bin/bash
#SBATCH --job-name=spine_ml
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=02:00:00
#SBATCH --mem=8G

source ~/.bashrc
conda activate mlbio

python run_experiment.py $1
