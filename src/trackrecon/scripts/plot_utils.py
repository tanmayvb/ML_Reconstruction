from pathlib import Path
import matplotlib.pyplot as plt

"""

| Colormap    | Style                  | Good For                  |
| ----------- | ---------------------- | ------------------------- |
| `"gray"`    | white on black         | classic raw microscopy    |
| `"magma"`   | black→orange→yellow    | BEAUTIFUL + high contrast |
| `"inferno"` | black→red→yellow       | excellent fluorescence    |
| `"viridis"` | dark blue→green→yellow | quantitative maps         |
| `"cividis"` | colorblind-friendly    | scientific publication    |
| `"hot"`     | black→red→yellow→white | prediction heatmaps       |
| `"bone"`    | soft grayscale         | elegant publication look  |

raw image      → magma
prediction map → inferno
overlays       → magma + colored markers
"""

def setup_plot_style():

    plt.rcParams.update({

        # ---------------------------------
        # Figure
        # ---------------------------------
        "figure.figsize": (18, 6),
        "figure.dpi": 300,

        # ---------------------------------
        # Fonts
        # ---------------------------------
        "axes.titlesize": 18,
        "axes.titleweight": "bold",

        "axes.labelsize": 16,
        "axes.labelweight": "bold",

        "xtick.labelsize": 13,
        "ytick.labelsize": 13,

        "legend.fontsize": 12,

        # ---------------------------------
        # Background
        # ---------------------------------
        "axes.facecolor": "black",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",

        # ---------------------------------
        # Image
        # ---------------------------------
        "image.cmap": "magma",

        # ---------------------------------
        # Save
        # ---------------------------------
        "savefig.bbox": "tight",

    })

def create_figure(
        title=None,
        xlabel="X",
        ylabel="Y",
        figsize=(18,6)
    ):

    fig, ax = plt.subplots(
        figsize=figsize
    )

    ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    return fig, ax

def apply_legend_style(ax):

    legend = ax.legend(
        facecolor="black",
        edgecolor="white",
        framealpha=0.8
    )

    if legend is not None:

        for text in legend.get_texts():
            text.set_color("white")

# -------------------------------------
# SAVE FIGURE
# -------------------------------------
def save_plot(save_dir,
              name):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()

    plt.savefig(
        save_dir / f"{name}.png",
    )

    plt.savefig(
        save_dir / f"{name}.pdf",
    )

    plt.close()

    print(f"\n[INFO] Saved plot: {save_dir}: \n{name}.pdf \n{name}.png\n")
