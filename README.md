# ML\_Reconstruction

## Machine Learning-Based 3D U-Net Detection of Brain Spines, Axons, and Dendrites

A complete deep learning pipeline for 3D fluorescence microscopy 
* Reconstruction, 
* Segmentation, 
* Biological structure analysis 
* Using a 3D U-Net framework.

The pipeline supports:

* automatic preprocessing,
* ROI extraction,
* Gaussian mask generation,
* 3D segmentation,
* spine detection,
* axon vs dendrite classification,
* visualization,
* quantitative evaluation.

---

# Features

## Preprocessing

* CZI/TIFF microscopy volume loading
* Background suppression
* Noise reduction
* Automatic ROI extraction
* Percentile-based normalization

## Point Processing

* Automatic point generation
* Point filtering
* Spatial QC analysis
* Gaussian supervision mask generation

## Deep Learning

* 3D U-Net segmentation
* Dice + BCE loss
* Automatic LR finder
* Cosine/step LR scheduler
* Patch-based volumetric training

## Inference

* Full-volume prediction
* Connected component analysis
* Threshold-based segmentation
* Axon vs dendrite classification

## Visualization

* Raw microscopy visualization
* Prediction overlays
* Gaussian mask overlays
* Max intensity projections
* Spatial distribution plots
* Precision/recall curves
* F1-score optimization plots

## Evaluation

* Precision
* Recall
* F1-score
* Connected component statistics
* Ground-truth comparison

---

# Pipeline Overview

```text
Raw Microscopy Volume
        ↓
Background Suppression
        ↓
Noise Reduction
        ↓
Automatic ROI Selection
        ↓
Normalization
        ↓
Automatic Point Generation
        ↓
Point Filtering
        ↓
Gaussian Mask Generation
        ↓
3D U-Net Training
        ↓
Inference / Prediction
        ↓
Thresholding + Connected Components
        ↓
Axon/Dendrite Classification
        ↓
Evaluation + Visualization
```

---

# Repository Structure

```text
ML_Reconstruction/
  └──src
       └──trackrecon
            ├── data/
            ├── models/
            ├── patching/
            ├── inference/
            ├── scripts/
            ├── training/
  ├── run_experiment_main_pipeline_script.py, pyproject.toml ...etc ...
  └── README.md
  └── slurm
```
---

# Installation

## Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/ML_Reconstruction.git

cd ML_Reconstruction
```

---

# Python Environment

Recommended Python version:

```text
Python >= 3.10
```

---

# Environment Setup

## Install Miniconda

Download and install Miniconda from:

https://docs.conda.io/en/latest/miniconda.html

Verify installation:

```bash
conda --version
```

---

## Create Conda Environment

Create a dedicated environment for the project:

```bash
conda create -n mlbio python=3.11
```

Activate the environment:

```bash
conda activate mlbio
```

Verify Python version:

```bash
python --version
```

Expected output:

```text
Python 3.11.x
```

---

## Install Project Dependencies

Using pip:

```bash
pip install numpy
pip install scipy
pip install matplotlib
pip install pandas
pip install tifffile
pip install scikit-image
pip install torch
pip install torchvision
pip install tqdm
pip install opencv-python
pip install czifile
```

Alternatively:

```bash
pip install -r requirements.txt
```

---

## Optional: Install UV

For faster package management and execution:

```bash
pip install uv
```

Run the pipeline using:

```bash
uv run python run_experiment_main_pipeline_script.py
```
---

# Optional

For faster dependency management:

```bash
pip install uv
```

Run with:

```bash
uv run python run_experiment_main_pipeline_script.py
```

---

# Input Data

Supported microscopy formats:

* `.czi`
* `.tiff`
* `.ome.tiff`

The pipeline internally converts data into normalized 3D volumes.

---

# Example Run

## Full pipeline for local Testing/Run

### Recommended sanity test before launching GPU jobs:

**Step1: Preprocessing**

```bash
uv run python run_experiment_main_pipeline_script.py \
  --tiff input_volume.ome.tiff \
  --infile_yaml config.yaml \
  --raw_stat \ 
  --roi \
  --run_detect \ 
  --run_filter \
  --run_mask \
  --save_plots \
  --outdir_local preprocessing
```
## Output:

preprocessing/<br>
├── csv/<br>
├── masks/<br>
└── plots/<br>

**Step2: Single Training**

```bash
uv run python run_experiment_main_pipeline_script.py \ 
  --tiff INPUT.ome.tiff \ 
  --infile_yaml config.yaml \ 
  --roi \ 
  --run_train \ 
  --lr 1e-4 \ 
  --patches_per_epoch 100 \ 
  --preprocess_dir preprocessing \
  --outdir_local experiments \ 
  --save_plots
```
## Output structure:

experiments/<br>
└── `lr_1e-4_patch_100/`<br>
    ├── models/<br>
    └── plots/<br>

**Step 3: Inference and Evaluation : Single Evaluation**

```bash
uv run python run_experiment_main_pipeline_script.py \ 
  --tiff INPUT.ome.tiff \ 
  --infile_yaml config.yaml \ 
  --roi \ 
  --run_infer \ 
  --pred_all \ 
  --pred_septs \ 
  --preprocess_dir preprocessing \ 
  --model_path experiments/lr_1e-4_patch_100/models/model.pth \ 
  --threshold 0.8 \ 
  --min_size 50 \ 
  --outdir_local experiments \ 
  --save_result \ 
  --save_plots
```
## Output structure:

experiments/
└── `lr_1e-4_patch_100/`<br>
    └── `eval_thr_0.8_min_50/`<br>
        ├── `prediction_volume.ome.tiff`<br>
        ├── results.csv<br>
        └── plots/<br>
## Verify:

**model.pth
prediction_volume.ome.tiff
results.csv**

are generated successfully.

---

## Run all step in once (need to be update better to use the abobe local run steps):

```bash
uv run python run_experiment_main_pipeline_script.py \
    --tiff input_volume.ome.tiff \
    --infile_yaml config.yaml
    --roi **If needed**
    --run_all
```
---

# Example Pipeline Stages

## Statistical information from RAW data

```bash
--raw_stat
```

## ROI extraction

```bash
--roi
```

## Mask generation

```bash
--run_mask
```

## Training

```bash
--train
```

## Inference

```bash
--predict
```

## Evaluation

```bash
--evaluate
```

---

# Training Configuration

Example:

```yaml
training:

    epochs: 100

    batch_size: 1

    patches_per_epoch: 200

    lr: 0.0001

    auto_lr: false

    scheduler:

        type: cosine
```

---

# Current Model

## Architecture

* 3D U-Net

## Loss Function

* Dice + BCEWithLogits loss

## Training Strategy

* Patch-based volumetric training
* Foreground-aware patch sampling
* Gaussian supervision masks

---

# Current Results

Example inference results:

| Metric    | Value  |
| --------- | ------ |
| Precision | ~0.89  |
| Recall    | ~0.016 |
| F1-score  | ~0.033 |

The model currently demonstrates:

* strong localization precision,
* robust detection quality,
* successful axon/dendrite separation.

Future work focuses on improving recall and segmentation completeness.

---

# Visualization Outputs

The pipeline automatically generates:

* raw image plots,
* Gaussian mask overlays,
* prediction maps,
* threshold overlays,
* max projections,
* spine detection overlays,
* axon/dendrite visualizations,
* learning rate curves,
* precision/recall curves,
* F1 optimization plots.

---

# Axon vs Dendrite Classification

The pipeline includes preliminary morphology-based classification:

* elongated structures → axons
* clustered/localized structures → dendrites/spines

This functionality is modular and can be extended for:

* multi-class segmentation,
* morphology-aware deep learning,
* graph-based biological analysis.

---

# Hyperparameter Optimization and GPU Execution
## Overview

The reconstruction pipeline supports:

* Preprocessing (Detection → Filtering → Mask Generation)
* Model Training (3D U-Net)
* Inference and Post-processing
* Hyperparameter Optimization
* Large-scale GPU Grid Search using SLURM

The recommended workflow is:

Preprocessing
      ↓
9 Training Jobs
      ↓
81 Evaluation Jobs
      ↓
Summary Ranking

## Step 1: Preprocessing (Instruction same as in local Testing/Run section)

**Run preprocessing only once:

This stage:

* Detects candidate points
* Filters noisy detections
* Generates Gaussian masks
* Generates ROI volume (optional)
* Save preprocessing plots(optional)


## Step 2: Training (Instruction same as in local Testing/Run section)

Training uses preprocessing outputs generated in Step 1.

## Step 3: Inference and Evaluation (Instruction same as in local Testing/Run section)

Inference uses a trained model and evaluates post-processing parameters.

---

# Hyperparameter Scan

## Training Grid

Learning rates: [1e-4, 1e-5, 1e-6]

Patches per epoch: [100, 200, 400]

Total training jobs: 3 × 3 = 9

## Evaluation Grid

Threshold: [0.7, 0.8, 0.9]

Minimum connected-component size: [50, 100, 250]

Total evaluations per model: 3 × 3 = 9

Total evaluations: 9 models × 9 evaluations = 81

## Final Directory Structure

preprocessing/<br>
├── csv/<br>
├── masks/<br>
├── plots/<br>

experiments/<br>
├── `lr_1e-4_patch_100/`<br>
├── `lr_1e-4_patch_200/`<br>
├── `lr_1e-4_patch_400/`<br>
├── `lr_1e-5_patch_100/`<br>
├── `lr_1e-5_patch_200/`<br>
├── `lr_1e-5_patch_400/`<br>
├── `lr_1e-6_patch_100/`<br>
├── `lr_1e-6_patch_200/`<br>
└── `lr_1e-6_patch_400/`<br>

and plot sub directory

---
# GPU Execution with SLURM

## The pipeline supports large-scale parameter scans using SLURM arrays.

**Workflow:**

Preprocessing<br>
      ↓<br>
9 Training Jobs<br>
      ↓<br>
81 Evaluation Jobs<br>
      ↓<br>
summary.csv<br>

## Recommended resources:

### Training:

* GPU: 1
* CPU: 8
* Memory: 64 GB
* Time: 48 hours

### Evaluation:

* GPU: 1
* CPU: 4
* Memory: 32 GB
* Time: 12 hours
* Result Aggregation

## Submission SLURM files:

Files are here
 * `ML_Reconstruction`/<br>
      └── slurm/<br>
            ├── preprocess.slurm<br>
            ├── `train_grid.slurm`<br>
            └── `eval_grid.slurm`<br>

From the repository root:

```bash
cd ~/MLtorch/ML_Reconstruction
mkdir -p logs
```

```bash
PRE=$(sbatch slurm/preprocess.slurm | awk '{print $4}')

TRAIN=$(sbatch \
    --dependency=afterok:$PRE \
    slurm/train_grid.slurm | awk '{print $4}')

sbatch \
    --dependency=afterok:$TRAIN \
    slurm/eval_grid.slurm
```

**OR** 

```bash
sh submit_slurm.sh
```
This will automatically run:

* 1 preprocessing job
* 9 training jobs
* 81 evaluation jobs
* Total = 91 jobs, in the correct order.

---

## After all evaluations complete:

```bash
uv run python scripts/build_summary.py
```

Produces: summary.csv

containing:

* lr
* `patches_per_epoch`
* threshold
* `min_size`
* precision
* recall
* f1

sorted by descending F1 score.

The top-ranked configuration can then be selected as the final optimized model.

---

# Future Development

Planned improvements:

* multi-class segmentation
* automatic hyperparameter optimization
* advanced denoising
* adaptive thresholding
* transformer-based segmentation
* support for additional microscopy modalities
* large-scale inference benchmarking
* Separate the plot codes/scripts from main run

---


# License

MIT License

