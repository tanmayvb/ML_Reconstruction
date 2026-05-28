# ML_Reconstruction

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
Axon / Dendrite Classification
        ↓
Evaluation + Visualization
```

---

# Repository Structure

```text
ML_Reconstruction/

├── configs/
├── data/
├── models/
├── preprocessing/
├── patching/
├── inference/
├── scripts/
├── plots/
├── training/
├── Documentation/
├── run_experiment_main_pipeline_script.py
└── README.md
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

## Full pipeline

```bash
uv run python run_experiment_main_pipeline_script.py \
    --tiff input_volume.ome.tiff \
    --run_all
```

---

# Example Pipeline Stages

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

# Future Development

Planned improvements:

* multi-class segmentation
* automatic hyperparameter optimization
* advanced denoising
* adaptive thresholding
* transformer-based segmentation
* support for additional microscopy modalities
* large-scale inference benchmarking

---


# License

MIT License

