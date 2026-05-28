#!/bin/bash

for p in 60 70 80
do
for sigma in 4 5 6
do
for th in 0.4 0.5 0.6
do

cat > tmp_config.yaml <<EOF
filter:
  percentile: $p
  min_distance: 3

mask:
  sigma_z: 2
  sigma_xy: $sigma

training:
  lr: 3e-5
  sampling_ratio: 0.7
  epochs: 30

inference:
  threshold: $th
  min_size: 20

data:
  image_path: data/processed/vol1.tiff
  points_path: data/annotations/points.csv
EOF

python run_experiment.py tmp_config.yaml

done
done
done
