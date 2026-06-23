# scripts/submit_9_training_jobs.py

import subprocess
from itertools import product

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

TIFF = (
"/Users/tanmay/LenevoINFN/Work/UTokyo/V1/MLtorch/"
"ML_Reconstruction/src/trackrecon/data/Airyscan_data/"
"ESI008_20260412_slice02_cell01-Airyscan-Processing-08-Stitching-09.ome.tiff"
)

for lr, patches in product(
        LRS,
        PATCHES
    ):

    outdir = (
        f"train_lr_{lr:.0e}"
        f"_patches_{patches}"
    )

    cmd = [

        "uv",
        "run",
        "python",
        "run_experiment_main_pipeline_script.py",

        "--tiff",
        TIFF,

        "--infile_yaml",
        "config.yaml",

        "--run_all",

        "--save_plots",

        "--save_result",

        "--pred_all",

        "--pred_septs",

        "--roi",

        "--lr",
        str(lr),

        "--patches_per_epoch",
        str(patches),

        "--outdir_local",
        outdir
    ]

    print(
        "\nSubmitting:",
        outdir
    )

    subprocess.run(
        cmd,
        check=True
    )
