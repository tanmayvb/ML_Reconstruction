import subprocess
from itertools import product
from pathlib import Path

THRESHOLDS = [
    0.7,
    0.8,
    0.9
]

MIN_SIZES = [
    20,
    50,
    100
]

MODEL_RUNS = [
    "ESI008_lr_3e-05_patch_100",
    "ESI008_lr_3e-05_patch_200",
    "ESI008_lr_3e-05_patch_400",
    "ESI008_lr_1e-04_patch_100",
    "ESI008_lr_1e-04_patch_200",
    "ESI008_lr_1e-04_patch_400",
    "ESI008_lr_3e-04_patch_100",
    "ESI008_lr_3e-04_patch_200",
    "ESI008_lr_3e-04_patch_400"
]

TIFF = "/path/to/ESI008_20260412_slice02_cell01-Airyscan-Processing-08-Stitching-09.ome.tiff"
CONFIG = "config.yaml"

for model_run in MODEL_RUNS:
    model_path = Path(model_run) / "models" / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")

    for threshold, min_size in product(THRESHOLDS, MIN_SIZES):
        eval_name = f"eval_thr_{threshold}_min_{min_size}"
        eval_dir = Path(model_run) / eval_name
        eval_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "uv",
            "run",
            "python",
            "run_experiment_main_pipeline_script.py",
            "--tiff",
            TIFF,
            "--infile_yaml",
            CONFIG,
            "--run_infer",
            "--pred_all",
            "--pred_septs",
            "--threshold",
            str(threshold),
            "--min_size",
            str(min_size),
            "--model_path",
            str(model_path),
            "--outdir_local",
            str(Path(model_run) / eval_name)
        ]

        print(f"\nRunning stage 2 evaluation: {model_run} / threshold={threshold} / min_size={min_size}")
        subprocess.run(cmd, check=True)
