import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import pandas as pd

from trackrecon.scripts.plot_utils import (
    setup_plot_style,
    create_figure,
    apply_legend_style,
    save_plot
)

# =========================================================
# GLOBAL SETTINGS
# =========================================================

RAW_CMAP = "magma"

PRED_CMAP = "inferno"

MASK_CMAP = "jet"

GT_CMAP = "Greens"

FIGSIZE_WIDE = (20, 6)

FIGSIZE_GRID = (20, 12)

FIGSIZE_SMALL = (8, 6)


# =========================================================
# Helper
# =========================================================

def get_contrast_limits(img):

    p1 = np.percentile(img, 1)

    p99 = np.percentile(img, 99)

    return p1, p99


# =========================================================
# Raw image
# =========================================================

def show_slice(volume, save_path, z=None):

    setup_plot_style()

    if z is None:
        z = volume.shape[0] // 2

    img = volume[z]

    p1, p99 = get_contrast_limits(img)

    fig, ax = create_figure(
        title=f"Raw Slice Z={z}",
        xlabel="X pixels",
        ylabel="Y pixels",
        figsize=FIGSIZE_WIDE
    )

    im = ax.imshow(
        img,
        cmap=RAW_CMAP,
        vmin=p1,
        vmax=p99,
        aspect="auto",
        interpolation="nearest"
    )
    apply_legend_style(ax)
    plt.colorbar(
        im,
        ax=ax,
        label="Intensity"
    )

    save_plot(
        save_path,
        "raw_slice"
    )


# =========================================================
# Histogram
# =========================================================

def show_histogram(volume, save_path):

    setup_plot_style()

    plt.figure(figsize=FIGSIZE_SMALL)

    plt.hist(
        volume.ravel(),
        bins=200
    )

    plt.yscale("log")

    plt.xlabel("Intensity")

    plt.ylabel("Count")

    plt.title("Intensity Histogram")

    plt.grid(True)

    plt.tight_layout()

    save_plot(
        save_path,
        "debug_histogram"
    )


# =========================================================
# Max Projection
# =========================================================

def show_max_projection_raw(volume, save_path):

    setup_plot_style()

    mip = volume.max(axis=0)

    p1, p99 = get_contrast_limits(mip)

    plt.figure(figsize=FIGSIZE_WIDE)

    plt.imshow(
        mip,
        cmap=RAW_CMAP,
        vmin=p1,
        vmax=p99,
        aspect="auto",
        interpolation="nearest"
    )

    plt.xlabel("X pixels")

    plt.ylabel("Y pixels")

    plt.title("Max Projection")

    plt.grid(False)

    plt.tight_layout()

    save_plot(
        save_path,
        "debug_max_projection"
    )


# =========================================================
# Contrast Visualization
# =========================================================

def show_contrast(volume, save_path, z=None):

    setup_plot_style()

    if z is None:
        z = volume.shape[0] // 2

    img = volume[z]

    p1, p99 = get_contrast_limits(img)

    plt.figure(figsize=FIGSIZE_WIDE)

    plt.imshow(
        img,
        cmap=RAW_CMAP,
        vmin=p1,
        vmax=p99,
        aspect="auto",
        interpolation="nearest"
    )

    plt.xlabel("X pixels")

    plt.ylabel("Y pixels")

    plt.title(f"Contrast Stretch Z={z}")

    plt.tight_layout()

    save_plot(
        save_path,
        "debug_contrast"
    )


# =========================================================
# Auto vs Filtered Points
# =========================================================

def plot_points_comparison(
        volume,
        auto_points,
        filtered_points,
        save_path,
        z=None
    ):

    setup_plot_style()

    if z is None:
        z = volume.shape[0] // 2

    img = volume[z]

    p1, p99 = get_contrast_limits(img)

    plt.figure(figsize=FIGSIZE_WIDE)

    plt.imshow(
        img,
        cmap=RAW_CMAP,
        vmin=p1,
        vmax=p99,
        aspect="auto",
        interpolation="nearest"
    )

    first_auto = True

    for p in auto_points:

        zz, yy, xx = p

        if int(zz) == z:

            plt.scatter(
                xx,
                yy,
                c="yellow",
                s=10,
                label="Auto" if first_auto else None
            )

            first_auto = False

    first_filtered = True

    for p in filtered_points:

        zz, yy, xx = p

        if int(zz) == z:

            plt.scatter(
                xx,
                yy,
                c="cyan",
                s=20,
                label="Filtered" if first_filtered else None
            )

            first_filtered = False

    plt.xlabel("X pixels")

    plt.ylabel("Y pixels")

    plt.title(f"Auto vs Filtered Points (Z={z})")

    #plt.legend()

    save_plot(
        save_path,
        f"auto_filter_points_overlay_z{z}"
    )


# =========================================================
# Filtered Points
# =========================================================

def plot_filtered_points(
        volume,
        filtered_points,
        save_path,
        z=None
    ):

    setup_plot_style()

    if z is None:
        z = volume.shape[0] // 2

    img = volume[z]

    p1, p99 = get_contrast_limits(img)

    plt.figure(figsize=FIGSIZE_WIDE)

    plt.imshow(
        img,
        cmap=RAW_CMAP,
        vmin=p1,
        vmax=p99,
        aspect="auto",
        interpolation="nearest"
    )

    for p in filtered_points:

        zz, yy, xx = p

        if int(zz) == z:

            plt.scatter(
                xx,
                yy,
                c="cyan",
                s=15
            )

    plt.xlabel("X pixels")

    plt.ylabel("Y pixels")

    plt.title(f"Filtered Points (Z={z})")

    save_plot(
        save_path,
        f"filtered_points_overlay_z{z}"
    )

#=========================================================
# Filter Retention Curve
# Purpose:
# How many points survive filtering
# vs percentile threshold.
#=========================================================

def plot_filter_retention(
        percentiles,
        counts,
        save_path
    ):

    setup_plot_style()

    fig, ax = create_figure(
        title="Filtering Retention Curve",
        xlabel="Percentile Threshold",
        ylabel="Remaining Points",
        figsize=(10,6)
    )

    ax.plot(
        percentiles,
        counts,
        "-o",
        linewidth=3,
        markersize=8,
        color="cyan",
        label="Retained Points"
    )

    ax.set_xlabel(
        "Percentile Threshold",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Remaining Points",
        fontweight="bold"
    )

    ax.set_title(
        "Filtering Retention Curve",
        fontweight="bold"
    )

    ax.grid(
        True,
        alpha=0.3,
        linestyle="--"
    )

    #ax.legend()

    plt.tight_layout()

    save_plot(
        save_path,
        "filter_retention_curve"
    )

# =========================================================
# Mask Overlay
# =========================================================

def plot_mask_overlay(
        volume,
        mask,
        save_path,
        z=None
    ):

    setup_plot_style()

    if z is None:
        z = volume.shape[0] // 2

    img = volume[z]

    p1, p99 = get_contrast_limits(img)

    plt.figure(figsize=FIGSIZE_WIDE)

    plt.imshow(
        img,
        cmap=RAW_CMAP,
        vmin=p1,
        vmax=p99,
        aspect="auto",
        interpolation="nearest"
    )

    plt.imshow(
        mask[z],
        cmap=MASK_CMAP,
        alpha=0.4,
        aspect="auto",
        interpolation="nearest"
    )

    plt.xlabel("X pixels")

    plt.ylabel("Y pixels")

    plt.title(f"Gaussian Mask Overlay (Z={z})")

    save_plot(
        save_path,
        f"gaussian_mask_overlay_z{z}"
    )

#=========================================================
# Filtered points + mask overlay
# Did every filtered point generate proper Gaussian
#=========================================================

def plot_flt_points_and_mask(
        volume,
        filtered_points,
        mask,
        save_path,
        z=None
    ):

    setup_plot_style()

    if z is None:
        z = volume.shape[0] // 2

    img = volume[z]

    p1 = np.percentile(img, 1)

    p99 = np.percentile(img, 99)

    fig, ax = create_figure(
        title=f"Filtered Points + Gaussian Mask Overlay (Z={z})",
        xlabel="X pixels",
        ylabel="Y pixels",
        figsize=(20,8)
    )

    # ---------------------------------------------------
    # Raw microscopy image
    # ---------------------------------------------------
    ax.imshow(
        img,
        cmap="magma",
        vmin=p1,
        vmax=p99,
        aspect="auto",
        interpolation="nearest"
    )

    # ---------------------------------------------------
    # Gaussian mask overlay
    # ---------------------------------------------------
    ax.imshow(
        mask[z],
        cmap="inferno",
        alpha=0.45,
        aspect="auto",
        interpolation="nearest"
    )

    # ---------------------------------------------------
    # Filtered points
    # ---------------------------------------------------
    first_point = True

    for p in filtered_points:

        zz, yy, xx = p

        if int(zz) == z:

            ax.scatter(
                xx,
                yy,
                c="cyan",
                s=18,
                edgecolors="white",
                linewidths=0.5,
                label=(
                    "Filtered Points"
                    if first_point else None
                )
            )

            first_point = False

    ax.set_xlabel(
        "X pixels",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Y pixels",
        fontweight="bold"
    )

    ax.set_title(
        f"Filtered Points + Gaussian Mask Overlay (Z={z})",
        fontweight="bold"
    )

    #ax.legend()

    plt.tight_layout()

    save_plot(
        save_path,
        f"filter_points_mask_overlay_z{z}"
    )

#=========================================================
# Max projection versions
# Single Z slices can be misleading.
#=========================================================

def show_max_projection_raw_and_mask(
        volume,
        mask,
        save_path
    ):

    setup_plot_style()

    # ---------------------------------------------------
    # Max projections
    # ---------------------------------------------------
    img_proj = volume.max(axis=0)

    mask_proj = mask.max(axis=0)

    # ---------------------------------------------------
    # Contrast normalization
    # ---------------------------------------------------
    p1 = np.percentile(img_proj, 1)

    p99 = np.percentile(img_proj, 99)

    # ---------------------------------------------------
    # Figure
    # ---------------------------------------------------
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(20,8)
    )

    # ===================================================
    # Raw projection
    # ===================================================
    im0 = ax[0].imshow(
        img_proj,
        cmap="magma",
        vmin=p1,
        vmax=p99,
        aspect="auto",
        interpolation="nearest"
    )

    ax[0].set_title(
        "Raw Max Projection",
        fontweight="bold"
    )

    ax[0].set_xlabel(
        "X pixels",
        fontweight="bold"
    )

    ax[0].set_ylabel(
        "Y pixels",
        fontweight="bold"
    )

    cbar0 = plt.colorbar(
        im0,
        ax=ax[0]
    )

    cbar0.set_label(
        "Intensity",
        fontweight="bold"
    )

    # ===================================================
    # Overlay
    # ===================================================
    im1 = ax[1].imshow(
        img_proj,
        cmap="magma",
        vmin=p1,
        vmax=p99,
        aspect="auto",
        interpolation="nearest"
    )

    ax[1].imshow(
        mask_proj,
        cmap="inferno",
        alpha=0.45,
        aspect="auto",
        interpolation="nearest"
    )

    ax[1].set_title(
        "Gaussian Mask Overlay (Max Projection)",
        fontweight="bold"
    )

    ax[1].set_xlabel(
        "X pixels",
        fontweight="bold"
    )

    ax[1].set_ylabel(
        "Y pixels",
        fontweight="bold"
    )

    cbar1 = plt.colorbar(
        im1,
        ax=ax[1]
    )

    cbar1.set_label(
        "Intensity",
        fontweight="bold"
    )

    plt.tight_layout()

    save_plot(
        save_path,
        "mask_overlay_max_projection"
    )

#=========================================================
# Spatial Distribution
# Very useful QC plot
#=========================================================

def spatial_distribution(
        filtered_points,
        save_path
    ):

    setup_plot_style()

    # ---------------------------------------------------
    # Extract coordinates
    # ---------------------------------------------------
    z, y, x = filtered_points.T

    # ---------------------------------------------------
    # Figure
    # ---------------------------------------------------
    fig, ax = create_figure(
        title="Spatial Distribution of Filtered Points",
        xlabel="X pixels",
        ylabel="Y pixels",
        figsize=(10,10)
    )

    # ---------------------------------------------------
    # Scatter plot
    # ---------------------------------------------------
    scatter = ax.scatter(
        x,
        y,
        c=z,
        cmap="viridis",
        s=4,
        alpha=0.7,
        edgecolors="none"
    )

    # ---------------------------------------------------
    # Labels
    # ---------------------------------------------------
    ax.set_xlabel(
        "X pixels",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Y pixels",
        fontweight="bold"
    )

    ax.set_title(
        "Spatial Distribution of Filtered Points",
        fontweight="bold"
    )

    # ---------------------------------------------------
    # Grid
    # ---------------------------------------------------
    ax.grid(
        True,
        alpha=0.2,
        linestyle="--"
    )

    # ---------------------------------------------------
    # Colorbar for Z slices
    # ---------------------------------------------------
    cbar = plt.colorbar(
        scatter,
        ax=ax
    )

    cbar.set_label(
        "Z slice",
        fontweight="bold"
    )

    plt.tight_layout()

    save_plot(
        save_path,
        "spatial_distribution_points"
    )


#=========================================================
# show_slice_overlay
# Validates:
# filtered points -> Gaussian mask generation
# Critical QC for training quality
#=========================================================

def show_slice_overlay(
        volume,
        mask,
        save_path,
        z=None
    ):

    setup_plot_style()

    # ---------------------------------------------------
    # Select slice
    # ---------------------------------------------------
    Z = volume.shape[0]

    if z is None:
        z = Z // 2

    img = volume[z]

    msk = mask[z]

    # ---------------------------------------------------
    # Contrast normalization
    # ---------------------------------------------------
    p1 = np.percentile(img, 1)

    p99 = np.percentile(img, 99)

    # ---------------------------------------------------
    # Figure
    # ---------------------------------------------------
    fig, ax = plt.subplots(
        1,
        3,
        figsize=(22,8)
    )

    # ===================================================
    # Raw Image
    # ===================================================
    im0 = ax[0].imshow(
        img,
        cmap="magma",
        vmin=p1,
        vmax=p99,
        aspect="auto",
        interpolation="nearest"
    )

    ax[0].set_title(
        f"Raw Microscopy Image (Z={z})",
        fontweight="bold"
    )

    ax[0].set_xlabel(
        "X pixels",
        fontweight="bold"
    )

    ax[0].set_ylabel(
        "Y pixels",
        fontweight="bold"
    )

    cbar0 = plt.colorbar(
        im0,
        ax=ax[0]
    )

    cbar0.set_label(
        "Intensity",
        fontweight="bold"
    )

    # ===================================================
    # Gaussian Mask
    # ===================================================
    im1 = ax[1].imshow(
        msk,
        cmap="inferno",
        aspect="auto",
        interpolation="nearest"
    )

    ax[1].set_title(
        "Gaussian Mask",
        fontweight="bold"
    )

    ax[1].set_xlabel(
        "X pixels",
        fontweight="bold"
    )

    ax[1].set_ylabel(
        "Y pixels",
        fontweight="bold"
    )

    cbar1 = plt.colorbar(
        im1,
        ax=ax[1]
    )

    cbar1.set_label(
        "Mask Intensity",
        fontweight="bold"
    )

    # ===================================================
    # Overlay
    # ===================================================
    im2 = ax[2].imshow(
        img,
        cmap="magma",
        vmin=p1,
        vmax=p99,
        aspect="auto",
        interpolation="nearest"
    )

    ax[2].imshow(
        msk,
        cmap="inferno",
        alpha=0.45,
        aspect="auto",
        interpolation="nearest"
    )

    ax[2].set_title(
        "Overlay: Raw Image + Gaussian Mask",
        fontweight="bold"
    )

    ax[2].set_xlabel(
        "X pixels",
        fontweight="bold"
    )

    ax[2].set_ylabel(
        "Y pixels",
        fontweight="bold"
    )

    cbar2 = plt.colorbar(
        im2,
        ax=ax[2]
    )

    cbar2.set_label(
        "Raw Intensity",
        fontweight="bold"
    )

    # ---------------------------------------------------
    # Layout
    # ---------------------------------------------------
    plt.tight_layout()

    save_plot(
        save_path,
        "raw_gaussian_mask_spatial_image_with_masks"
    )

#=========================================================
# Score Distribution Comparison
# Did filtering remove weak/noisy points
#=========================================================

def score_distribution_compare(
        scores,
        filtered_scores,
        save_path
    ):

    setup_plot_style()

    # ---------------------------------------------------
    # Figure
    # ---------------------------------------------------
    fig, ax = create_figure(
        title="Score Distribution Before vs After Filtering",
        xlabel="Detection Score",
        ylabel="Count",
        figsize=(12,7)
    )

    # ---------------------------------------------------
    # Histogram: Before filtering
    # ---------------------------------------------------
    ax.hist(
        scores,
        bins=100,
        alpha=0.5,
        color="orange",
        label="Before Filtering"
    )

    # ---------------------------------------------------
    # Histogram: After filtering
    # ---------------------------------------------------
    ax.hist(
        filtered_scores,
        bins=100,
        alpha=0.7,
        color="cyan",
        label="After Filtering"
    )

    # ---------------------------------------------------
    # Labels
    # ---------------------------------------------------
    ax.set_xlabel(
        "Detection Score",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Count",
        fontweight="bold"
    )

    ax.set_title(
        "Score Distribution Before vs After Filtering",
        fontweight="bold"
    )

    # ---------------------------------------------------
    # Grid
    # ---------------------------------------------------
    ax.grid(
        True,
        alpha=0.3,
        linestyle="--"
    )

    # ---------------------------------------------------
    # Legend
    # ---------------------------------------------------
    #ax.legend()

    plt.tight_layout()

    save_plot(
        save_path,
        "score_distribution_before_after_filtering"
    )

#=========================================================
# Z-distribution of detections
# Shows whether detections are biased across Z
#=========================================================

def plot_z_distribution(
        filtered_points,
        save_path
    ):

    setup_plot_style()

    # ---------------------------------------------------
    # Extract Z coordinates
    # ---------------------------------------------------
    z = filtered_points[:, 0]

    # ---------------------------------------------------
    # Figure
    # ---------------------------------------------------
    fig, ax = create_figure(
        title="Z Distribution of Filtered Points",
        xlabel="Z Slice",
        ylabel="Detection Count",
        figsize=(12,7)
    )

    # ---------------------------------------------------
    # Histogram
    # ---------------------------------------------------
    ax.hist(
        z,
        bins=30,
        color="cyan",
        edgecolor="white",
        alpha=0.85
    )

    # ---------------------------------------------------
    # Labels
    # ---------------------------------------------------
    ax.set_xlabel(
        "Z Slice",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Detection Count",
        fontweight="bold"
    )

    ax.set_title(
        "Z Distribution of Filtered Points",
        fontweight="bold"
    )

    # ---------------------------------------------------
    # Grid
    # ---------------------------------------------------
    ax.grid(
        True,
        alpha=0.3,
        linestyle="--"
    )

    plt.tight_layout()

    save_plot(
        save_path,
        "z_distribution"
    )

#=========================================================
# Training Plots
#=========================================================

# -------------------------------------------------
# LR Finder
# -------------------------------------------------
def lr_smooth(
        lrs,
        losses,
        save_path
    ):

    setup_plot_style()

    # ---------------------------------------------------
    # Smooth loss
    # ---------------------------------------------------
    smooth = pd.Series(losses).rolling(
        5,
        min_periods=1
    ).mean()

    # ---------------------------------------------------
    # Figure
    # ---------------------------------------------------
    fig, ax = create_figure(
        title="Learning Rate Finder",
        xlabel="Learning Rate",
        ylabel="Loss",
        figsize=(12,7)
    )

    # ---------------------------------------------------
    # Plot
    # ---------------------------------------------------
    ax.plot(
        lrs,
        smooth,
        linewidth=3,
        color="cyan"
    )

    ax.set_xscale("log")

    # ---------------------------------------------------
    # Labels
    # ---------------------------------------------------
    ax.set_xlabel(
        "Learning Rate",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Loss",
        fontweight="bold"
    )

    ax.set_title(
        "Learning Rate Finder",
        fontweight="bold"
    )

    # ---------------------------------------------------
    # Grid
    # ---------------------------------------------------
    ax.grid(
        True,
        alpha=0.3,
        linestyle="--"
    )

    plt.tight_layout()

    save_plot(
        save_path,
        "lr_finder"
    )


# -------------------------------------------------
# Loss Curve
# -------------------------------------------------
def loss_curve(
        loss_history,
        save_path
    ):

    setup_plot_style()

    # ---------------------------------------------------
    # Smooth curve
    # ---------------------------------------------------
    smooth = pd.Series(loss_history).rolling(
        5,
        min_periods=1
    ).mean()

    # ---------------------------------------------------
    # Figure
    # ---------------------------------------------------
    fig, ax = create_figure(
        title="Training Loss Curve",
        xlabel="Epoch",
        ylabel="Loss",
        figsize=(12,7)
    )

    # ---------------------------------------------------
    # Raw loss
    # ---------------------------------------------------
    ax.plot(
        loss_history,
        alpha=0.35,
        linewidth=2,
        color="orange",
        label="Raw Loss"
    )

    # ---------------------------------------------------
    # Smooth loss
    # ---------------------------------------------------
    ax.plot(
        smooth,
        linewidth=3,
        color="cyan",
        label="Smoothed Loss"
    )

    # ---------------------------------------------------
    # Labels
    # ---------------------------------------------------
    ax.set_xlabel(
        "Epoch",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Loss",
        fontweight="bold"
    )

    ax.set_title(
        "Training Loss Curve",
        fontweight="bold"
    )

    # ---------------------------------------------------
    # Grid
    # ---------------------------------------------------
    ax.grid(
        True,
        alpha=0.3,
        linestyle="--"
    )

    # ---------------------------------------------------
    # Legend
    # ---------------------------------------------------
    #ax.legend()

    plt.tight_layout()

    save_plot(
        save_path,
        "epoch_loss_curve"
    )


# -------------------------------------------------
# Learning Rate Schedule
# -------------------------------------------------
def lr_schedule(
        lr_history,
        save_path
    ):

    setup_plot_style()

    # ---------------------------------------------------
    # Figure
    # ---------------------------------------------------
    fig, ax = create_figure(
        title="Learning Rate Schedule",
        xlabel="Epoch",
        ylabel="Learning Rate",
        figsize=(12,7)
    )

    # ---------------------------------------------------
    # Plot
    # ---------------------------------------------------
    ax.plot(
        lr_history,
        linewidth=3,
        color="lime"
    )

    ax.set_yscale("log")

    # ---------------------------------------------------
    # Labels
    # ---------------------------------------------------
    ax.set_xlabel(
        "Epoch",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Learning Rate",
        fontweight="bold"
    )

    ax.set_title(
        "Learning Rate Schedule",
        fontweight="bold"
    )

    # ---------------------------------------------------
    # Grid
    # ---------------------------------------------------
    ax.grid(
        True,
        alpha=0.3,
        linestyle="--"
    )

    plt.tight_layout()

    save_plot(
        save_path,
        "epoch_lr_schedule"
    )

# -------------------------------------------------
# Precision Recall
# -------------------------------------------------
def precision_recall_curve(thresholds,
                           precisions,
                           recalls,
                           save_path):

    plt.figure(figsize=(6,5))

    plt.plot(thresholds,
             precisions,
             label="Precision")

    plt.plot(thresholds,
             recalls,
             label="Recall")

    plt.xlabel("Threshold")
    plt.ylabel("Score")

    plt.title("Precision / Recall")

    #plt.legend()

    plt.grid(True)

    save_plot(save_path, f"threshold_vs_score_precision")

# -------------------------------------------------
# F1 curve
# -------------------------------------------------
def f1_curve(thresholds,
             f1s,
             save_path):

    best_idx = np.argmax(f1s)

    plt.figure(figsize=(6,5))

    plt.plot(thresholds,
             f1s)
    """
    plt.scatter(
        thresholds[best_idx],
        f1s[best_idx],
        s=50
    )
    """
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")

    plt.title("F1 vs Threshold")

    plt.grid(True)

    save_plot(save_path, f"f1_vs_threshold")

# -------------------------------------------------
# Threshold Overlay
# Visualizes binary thresholding results
# -------------------------------------------------
def plot_threshold_overlay(
        volume,
        prediction,
        threshold,
        save_path,
        z=None
    ):

    setup_plot_style()

    # ---------------------------------------------------
    # Select slice
    # ---------------------------------------------------
    if z is None:
        z = volume.shape[0] // 2

    img = volume[z]

    pred = prediction[z]

    binary = pred > threshold

    # ---------------------------------------------------
    # Contrast normalization
    # ---------------------------------------------------
    p1 = np.percentile(img, 1)

    p99 = np.percentile(img, 99)

    # ---------------------------------------------------
    # Figure
    # ---------------------------------------------------
    fig, ax = plt.subplots(
        1,
        3,
        figsize=(22,8)
    )

    # ===================================================
    # Raw Image
    # ===================================================
    im0 = ax[0].imshow(
        img,
        cmap="magma",
        vmin=p1,
        vmax=p99,
        aspect="auto",
        interpolation="nearest"
    )

    ax[0].set_title(
        f"Raw Microscopy Image (Z={z})",
        fontweight="bold"
    )

    ax[0].set_xlabel(
        "X pixels",
        fontweight="bold"
    )

    ax[0].set_ylabel(
        "Y pixels",
        fontweight="bold"
    )

    cbar0 = plt.colorbar(
        im0,
        ax=ax[0]
    )

    cbar0.set_label(
        "Intensity",
        fontweight="bold"
    )

    # ===================================================
    # Binary Threshold Map
    # ===================================================
    im1 = ax[1].imshow(
        binary,
        cmap="Reds",
        aspect="auto",
        interpolation="nearest"
    )

    ax[1].set_title(
        f"Binary Threshold Map (Threshold={threshold})",
        fontweight="bold"
    )

    ax[1].set_xlabel(
        "X pixels",
        fontweight="bold"
    )

    ax[1].set_ylabel(
        "Y pixels",
        fontweight="bold"
    )

    cbar1 = plt.colorbar(
        im1,
        ax=ax[1]
    )

    cbar1.set_label(
        "Binary Detection",
        fontweight="bold"
    )

    # ===================================================
    # Overlay
    # ===================================================
    im2 = ax[2].imshow(
        img,
        cmap="magma",
        vmin=p1,
        vmax=p99,
        aspect="auto",
        interpolation="nearest"
    )

    ax[2].imshow(
        binary,
        cmap="Reds",
        alpha=0.45,
        aspect="auto",
        interpolation="nearest"
    )

    ax[2].set_title(
        "Overlay: Raw Image + Threshold Detection",
        fontweight="bold"
    )

    ax[2].set_xlabel(
        "X pixels",
        fontweight="bold"
    )

    ax[2].set_ylabel(
        "Y pixels",
        fontweight="bold"
    )

    cbar2 = plt.colorbar(
        im2,
        ax=ax[2]
    )

    cbar2.set_label(
        "Raw Intensity",
        fontweight="bold"
    )

    # ---------------------------------------------------
    # Layout
    # ---------------------------------------------------
    plt.tight_layout()

    save_plot(
        save_path,
        "raw_threshold_overlay"
    )

# -------------------------------------------------
# Spine Detections
# Visualize detected spine centroids
# -------------------------------------------------
def plot_spines(
        volume,
        spines,
        save_path,
        z=None
    ):

    setup_plot_style()

    # ---------------------------------------------------
    # Select slice
    # ---------------------------------------------------
    if z is None:
        z = volume.shape[0] // 2

    img = volume[z]

    # ---------------------------------------------------
    # Contrast normalization
    # ---------------------------------------------------
    p1 = np.percentile(img, 1)

    p99 = np.percentile(img, 99)

    # ---------------------------------------------------
    # Figure
    # ---------------------------------------------------
    fig, ax = create_figure(
        title=f"Detected Spines (Z={z})",
        xlabel="X pixels",
        ylabel="Y pixels",
        figsize=(20,8)
    )

    # ===================================================
    # Raw microscopy image
    # ===================================================
    im = ax.imshow(
        img,
        cmap="magma",
        vmin=p1,
        vmax=p99,
        aspect="auto",
        interpolation="nearest"
    )

    cbar = plt.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(
        "Intensity",
        fontweight="bold"
    )

    # ===================================================
    # Spine detections
    # ===================================================
    first_spine = True

    for s in spines:

        zz, yy, xx = s["centroid"]

        if abs(zz - z) <= 1:

            ax.scatter(
                xx,
                yy,
                c="red",
                s=30,
                edgecolors="white",
                linewidths=0.6,
                marker="o",
                label="Detected Spine" if first_spine else None
            )

            first_spine = False

    # ---------------------------------------------------
    # Labels
    # ---------------------------------------------------
    ax.set_xlabel(
        "X pixels",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Y pixels",
        fontweight="bold"
    )

    ax.set_title(
        f"Detected Spines (Z={z})",
        fontweight="bold"
    )

    # ---------------------------------------------------
    # Legend
    # ---------------------------------------------------
    #ax.legend()

    # ---------------------------------------------------
    # Layout
    # ---------------------------------------------------
    plt.tight_layout()

    save_plot(
        save_path,
        f"detected_spines_z{z}"
    )

#=========================================================
# Prediction Histogram
# VERY important for threshold selection
#=========================================================

def plot_prediction_histogram(
        prediction,
        save_path
    ):

    setup_plot_style()

    # ---------------------------------------------------
    # Flatten prediction volume
    # ---------------------------------------------------
    pred = prediction.flatten()

    # ---------------------------------------------------
    # Figure
    # ---------------------------------------------------
    fig, ax = create_figure(
        title="Prediction Value Distribution",
        xlabel="Prediction Value",
        ylabel="Voxel Count",
        figsize=(12,7)
    )

    # ---------------------------------------------------
    # Histogram
    # ---------------------------------------------------
    ax.hist(
        pred,
        bins=100,
        color="orange",
        edgecolor="white",
        alpha=0.85
    )

    # ---------------------------------------------------
    # Mean prediction line
    # ---------------------------------------------------
    mean_pred = np.mean(pred)

    ax.axvline(
        mean_pred,
        color="cyan",
        linestyle="--",
        linewidth=3,
        label=f"Mean = {mean_pred:.4f}"
    )

    # ---------------------------------------------------
    # Labels
    # ---------------------------------------------------
    ax.set_xlabel(
        "Prediction Value",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Voxel Count",
        fontweight="bold"
    )

    ax.set_title(
        "Prediction Value Distribution",
        fontweight="bold"
    )

    # ---------------------------------------------------
    # Grid
    # ---------------------------------------------------
    ax.grid(
        True,
        alpha=0.3,
        linestyle="--"
    )

    # ---------------------------------------------------
    # Legend
    # ---------------------------------------------------
    #ax.legend()

    plt.tight_layout()

    save_plot(
        save_path,
        "prediction_distribution"
    )

#=========================================================
# Connected Component Size Distribution
# Distribution of detected spine volumes
#=========================================================

def plot_component_size_distribution(
        spines,
        save_path
    ):

    setup_plot_style()

    # ---------------------------------------------------
    # Extract component sizes
    # ---------------------------------------------------
    sizes = [s["volume"] for s in spines]

    # ---------------------------------------------------
    # Figure
    # ---------------------------------------------------
    fig, ax = create_figure(
        title="Detected Spine Size Distribution",
        xlabel="Connected Component Size (voxels)",
        ylabel="Count",
        figsize=(12,7)
    )

    # ---------------------------------------------------
    # Histogram
    # ---------------------------------------------------
    ax.hist(
        sizes,
        bins=30,
        color="cyan",
        edgecolor="white",
        alpha=0.85
    )

    # ---------------------------------------------------
    # Mean size line
    # ---------------------------------------------------
    mean_size = np.mean(sizes)

    ax.axvline(
        mean_size,
        color="red",
        linestyle="--",
        linewidth=3,
        label=f"Mean Size = {mean_size:.1f}"
    )

    # ---------------------------------------------------
    # Median size line
    # ---------------------------------------------------
    median_size = np.median(sizes)

    ax.axvline(
        median_size,
        color="yellow",
        linestyle=":",
        linewidth=3,
        label=f"Median Size = {median_size:.1f}"
    )

    # ---------------------------------------------------
    # Labels
    # ---------------------------------------------------
    ax.set_xlabel(
        "Connected Component Size (voxels)",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Count",
        fontweight="bold"
    )

    ax.set_title(
        "Detected Spine Size Distribution",
        fontweight="bold"
    )

    # ---------------------------------------------------
    # Grid
    # ---------------------------------------------------
    ax.grid(
        True,
        alpha=0.3,
        linestyle="--"
    )

    # ---------------------------------------------------
    # Legend
    # ---------------------------------------------------
    #ax.legend()

    plt.tight_layout()

    save_plot(
        save_path,
        "detected_spine_size_distribution"
    )

#=========================================================
# Learning Rate vs F1
# Useful for:
# - Multiple experiments
# - Hyperparameter scans
# - Grid search optimization
#=========================================================

def learning_rate_vs_f1(
        results_csv,
        save_path
    ):

    setup_plot_style()

    # ---------------------------------------------------
    # Load results
    # ---------------------------------------------------
    df = pd.read_csv(results_csv)

    # ---------------------------------------------------
    # Figure
    # ---------------------------------------------------
    fig, ax = create_figure(
        title="Learning Rate vs F1 Score",
        xlabel="Learning Rate",
        ylabel="F1 Score",
        figsize=(12,7)
    )

    # ---------------------------------------------------
    # Scatter plot
    # ---------------------------------------------------
    scatter = ax.scatter(
        df["lr"],
        df["f1"],
        c=df["f1"],
        cmap="viridis",
        s=120,
        edgecolors="white",
        linewidths=1.2,
        alpha=0.9
    )

    # ---------------------------------------------------
    # Highlight best point
    # ---------------------------------------------------
    best_idx = df["f1"].idxmax()

    best_lr = df.loc[best_idx, "lr"]

    best_f1 = df.loc[best_idx, "f1"]

    ax.scatter(
        best_lr,
        best_f1,
        c="red",
        s=220,
        marker="*",
        edgecolors="white",
        linewidths=1.5,
        label=f"Best F1 = {best_f1:.4f}"
    )

    # ---------------------------------------------------
    # Log scale
    # ---------------------------------------------------
    ax.set_xscale("log")

    # ---------------------------------------------------
    # Labels
    # ---------------------------------------------------
    ax.set_xlabel(
        "Learning Rate",
        fontweight="bold"
    )

    ax.set_ylabel(
        "F1 Score",
        fontweight="bold"
    )

    ax.set_title(
        "Learning Rate vs F1 Score",
        fontweight="bold"
    )

    # ---------------------------------------------------
    # Grid
    # ---------------------------------------------------
    ax.grid(
        True,
        alpha=0.3,
        linestyle="--"
    )

    # ---------------------------------------------------
    # Colorbar
    # ---------------------------------------------------
    cbar = plt.colorbar(
        scatter,
        ax=ax
    )

    cbar.set_label(
        "F1 Score",
        fontweight="bold"
    )

    # ---------------------------------------------------
    # Legend
    # ---------------------------------------------------
    #ax.legend()

    plt.tight_layout()

    save_plot(
        save_path,
        "learning_rate_vs_F1"
    )

#=========================================================
# You only need it if you have GT points
#It compares: predicted detections vs
#             ground-truth detections
#on the same image.
#=========================================================
def compare_points(volume,
                   pred_points,
                   gt_points,
                   save_path,
                   z=None):

    if z is None:
        z = volume.shape[0] // 2

    plt.figure(figsize=(20,14))

    plt.imshow(
        volume[z], 
        cmap="gray",
        aspect="auto",
        interpolation="nearest"
    )

    pred = [p for p in pred_points if int(p[0]) == z]
    gt = [p for p in gt_points if int(p[0]) == z]

    if len(pred):
        pred = np.array(pred)

        plt.scatter(pred[:,2],
                    pred[:,1],
                    c="red",
                    s=10,
                    label="Prediction")

    if len(gt):
        gt = np.array(gt)

        plt.scatter(gt[:,2],
                    gt[:,1],
                    c="lime",
                    s=10,
                    label="Ground Truth")

    #plt.legend()

    plt.title(f"Prediction vs GT (Z={z})")

    save_plot(save_path, f"prediction_vs_GT_{z}")

#=========================================================
#This plot directly shows: 
#ground truth mask vs predicted mask
#over the raw image. where the model succeeds
#=========================================================

def show_gt_vs_pred(volume,
                    pred,
                    gt,
                    save_path,
                    z=None):

    if z is None:
        z = volume.shape[0] // 2

    fig, ax = plt.subplots(1, 4, figsize=(16,5))

    # -----------------------------
    # Raw
    # -----------------------------
    ax[0].imshow(
        volume[z], 
        cmap="gray",
        aspect="auto",
        interpolation="nearest"
    )
    ax[0].set_title("Raw")
    ax[0].axis("off")

    # -----------------------------
    # GT
    # -----------------------------
    ax[1].imshow(
        gt[z], 
        cmap="Greens",
        aspect="auto",
        interpolation="nearest"
    )
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")

    # -----------------------------
    # Prediction
    # -----------------------------
    ax[2].imshow(
        pred[z], 
        cmap="Reds",
        aspect="auto",
        interpolation="nearest"
    )
    ax[2].set_title("Prediction")
    ax[2].axis("off")

    # -----------------------------
    # Overlay
    # -----------------------------
    ax[3].imshow(
        volume[z], 
        cmap="gray",
        aspect="auto",
        interpolation="nearest"
    )

    ax[3].imshow(gt[z],
                 cmap="Greens",
                 alpha=0.3,
                 aspect="auto",
                 interpolation="nearest"
                )

    ax[3].imshow(pred[z],
                 cmap="Reds",
                 alpha=0.3,
                 aspect="auto",
                 interpolation="nearest"
                )

    ax[3].set_title("GT vs Prediction")
    ax[3].axis("off")

    save_plot(save_path, f"gt_vs_prediction")

def show_with_points(volume,
                     points,
                     save_path,
                     z=None):

    if z is None:
        z = volume.shape[0] // 2

    plt.figure(figsize=(20,14))

    plt.imshow(
        volume[z], 
        cmap="gray",
        aspect="auto",
        interpolation="nearest"
    )

    pts = [p for p in points if int(p[0]) == z]

    if len(pts) > 0:

        pts = np.array(pts)

        plt.scatter(
            pts[:,2],
            pts[:,1],
            s=10,
            c="red"
        )

    plt.title(f"Points Overlay (Z={z})")

    plt.axis("off")

    save_plot(save_path, f"points_overlay_{z}")




# =========================================================
# Prediction Visualization
# =========================================================

def show_prediction(
        volume,
        prediction,
        save_path,
        z=None
    ):

    setup_plot_style()

    if z is None:
        z = volume.shape[0] // 2

    img = volume[z]

    p1, p99 = get_contrast_limits(img)

    fig, ax = plt.subplots(
        1,
        3,
        figsize=FIGSIZE_WIDE
    )

    # --------------------------------
    # Raw
    # --------------------------------
    ax[0].imshow(
        img,
        cmap=RAW_CMAP,
        vmin=p1,
        vmax=p99,
        aspect="auto",
        interpolation="nearest"
    )

    ax[0].set_title("Raw Image")

    ax[0].set_xlabel("X")

    ax[0].set_ylabel("Y")

    # --------------------------------
    # Prediction
    # --------------------------------
    ax[1].imshow(
        prediction[z],
        cmap=PRED_CMAP,
        aspect="auto",
        interpolation="nearest"
    )

    ax[1].set_title("Prediction Map")

    ax[1].set_xlabel("X")

    ax[1].set_ylabel("Y")

    # --------------------------------
    # Overlay
    # --------------------------------
    ax[2].imshow(
        img,
        cmap=RAW_CMAP,
        vmin=p1,
        vmax=p99,
        aspect="auto",
        interpolation="nearest"
    )

    ax[2].imshow(
        prediction[z],
        cmap=PRED_CMAP,
        alpha=0.4,
        aspect="auto",
        interpolation="nearest"
    )

    ax[2].set_title("Prediction Overlay")

    ax[2].set_xlabel("X")

    ax[2].set_ylabel("Y")

    plt.tight_layout()

    save_plot(
        save_path,
        "prediction_volume_overlay"
    )


# =========================================================
# Axon / Dendrite Visualization
# =========================================================

def visualize_prediction_results(
        volume,
        prediction,
        structures,
        save_path,
        z=None
    ):

    setup_plot_style()

    if z is None:
        z = volume.shape[0] // 2

    img = volume[z]

    p1, p99 = get_contrast_limits(img)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=FIGSIZE_GRID
    )

    # =================================
    # RAW IMAGE
    # =================================
    axes[0,0].imshow(
        img,
        cmap=RAW_CMAP,
        vmin=p1,
        vmax=p99,
        aspect="auto",
        interpolation="nearest"
    )

    axes[0,0].set_title("Raw Image")

    # =================================
    # Prediction
    # =================================
    axes[0,1].imshow(
        prediction[z],
        cmap=PRED_CMAP,
        aspect="auto",
        interpolation="nearest"
    )

    axes[0,1].set_title("Prediction Map")

    # =================================
    # Axons
    # =================================
    axes[1,0].imshow(
        img,
        cmap=RAW_CMAP,
        vmin=p1,
        vmax=p99,
        aspect="auto",
        interpolation="nearest"
    )

    first_axon = True

    for s in structures:

        if s["type"] != "axon":
            continue

        zz, yy, xx = s["centroid"]

        if abs(zz - z) > 2:
            continue

        axes[1,0].scatter(
            xx,
            yy,
            c="red",
            s=40,
            marker="o",
            label="Axon" if first_axon else None
        )

        first_axon = False

    #axes[1,0].legend()

    axes[1,0].set_title("Axon Overlay")

    # =================================
    # Dendrites
    # =================================
    axes[1,1].imshow(
        img,
        cmap=RAW_CMAP,
        vmin=p1,
        vmax=p99,
        aspect="auto",
        interpolation="nearest"
    )

    first_dendrite = True

    for s in structures:

        if s["type"] != "dendrite":
            continue

        zz, yy, xx = s["centroid"]

        if abs(zz - z) > 2:
            continue

        axes[1,1].scatter(
            xx,
            yy,
            c="cyan",
            s=25,
            marker="o",
            label="Dendrite" if first_dendrite else None
        )

        first_dendrite = False

    #axes[1,1].legend()

    axes[1,1].set_title("Dendrite Overlay")

    plt.tight_layout()

    save_plot(
        save_path,
        "axon_dendrite_visualization"
    )

    plt.close()


"""
# This one is the cross check of the above function
def debug_dataset(dataset, save_path):

    x, y = dataset[0]

    img = x.squeeze().numpy()
    mask = y.squeeze().numpy()

    z = img.shape[0] // 2

    plt.figure(figsize=(12,4))

    # -----------------------------
    # Raw image
    # -----------------------------
    plt.subplot(1,3,1)

    plt.imshow(img[z], cmap="gray")

    plt.title("Image")

    plt.axis("off")

    # -----------------------------
    # Mask
    # -----------------------------
    plt.subplot(1,3,2)

    plt.imshow(mask[z], cmap="hot")

    plt.title("Mask")

    plt.axis("off")

    # -----------------------------
    # Overlay
    # -----------------------------
    plt.subplot(1,3,3)

    plt.imshow(img[z], cmap="gray")

    plt.imshow(
        mask[z],
        cmap="jet",
        alpha=0.4
    )

    plt.title("Overlay")

    plt.axis("off")

    plt.tight_layout()

    save_plot(
        save_path,
        "raw_image_mask_overlay"
    )
"""
