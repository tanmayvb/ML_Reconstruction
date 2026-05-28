import matplotlib.pyplot as plt
import numpy as np


def show_slice(volume, prediction, z=None):
    if z is None:
        z = volume.shape[0] // 2

    fig, ax = plt.subplots(1, 3, figsize=(12,4))

    ax[0].imshow(volume[z], cmap="gray")
    ax[0].set_title("Raw")

    ax[1].imshow(prediction[z], cmap="hot")
    ax[1].set_title("Prediction")

    ax[2].imshow(volume[z], cmap="gray")
    ax[2].imshow(prediction[z] > 0.5, alpha=0.4, cmap="Reds")
    ax[2].set_title("Overlay")

    plt.savefig("Raw_predection_thresholdzerop6_minsize20_Overlay_plot.png")
    plt.show()


def plot_spines(volume, spines, z):
    plt.imshow(volume[z], cmap="gray")

    for s in spines:
        zz, yy, xx = s["centroid"]
        if int(zz) == z:
            plt.scatter(xx, yy, c="red", s=10)

    plt.title(f"Detected spines (z={z})")
    plt.savefig("Detected_spines_thresholdzerop6_minsize20.png")
    plt.show()
