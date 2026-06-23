#!/bin/bash

PRE=$(sbatch slurm/preprocess.slurm | awk '{print $4}')

TRAIN=$(sbatch \
    --dependency=afterok:$PRE \
    slurm/train_grid.slurm | awk '{print $4}')

sbatch \
    --dependency=afterok:$TRAIN \
    slurm/eval_grid.slurm
