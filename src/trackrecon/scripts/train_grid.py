# scripts/train_grid.py

import subprocess
from itertools import product
from pathlib import Path

LRS = [
    3e-5,
    1e-4,
    3e-4
]

PATCHES = [
    100,
    200,
    400
]

TIFF = "/path/to/ESI008_20260412_slice02_cell01-Airyscan-Processing-08-Stitching-09.ome.tiff"
CONFIG = "config.yaml"

for lr, patches in product(LRS, PATCHES):
    lr_str = f"{lr:.0e}".replace("e-0", "e-")
    run_name = f"ESI008_lr_{lr_str}_patch_{patches}"
    run_dir = Path(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "python",
        "run_experiment_main_pipeline_script.py",
        "--tiff",
        TIFF,
        "--infile_yaml",
        CONFIG,
        "--run_detect",
        "--run_filter",
        "--run_mask",
        "--run_train",
        "--lr",
        str(lr),
        "--patches_per_epoch",
        str(patches),
        "--outdir_local",
        str(run_dir)
    ]

    print(f"\nRunning stage 1 training: {run_name}")
    subprocess.run(cmd, check=True)
