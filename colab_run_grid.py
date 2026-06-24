# run_grid_colab.py

import itertools
import subprocess

lrs = [1e-4, 1e-5, 1e-6]

patches = [100, 200, 400]

thresholds = [0.7, 0.8, 0.9]

minsizes = [50, 100, 250]

TIFF = "INPUT.ome.tiff"

CFG = "config.yaml"

for lr, patch in itertools.product(
        lrs,
        patches
    ):

    print(
        f"\nTraining: "
        f"lr={lr} "
        f"patch={patch}"
    )

    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "run_experiment_main_pipeline_script.py",

            "--tiff", TIFF,
            "--infile_yaml", CFG,

            "--roi",

            "--run_train",

            "--lr", str(lr),

            "--patches_per_epoch",
            str(patch),

            "--preprocess_dir",
            "preprocessing",

            "--outdir_local",
            "experiments",

            "--save_plots"
        ],
        check=True
    )

    lr_str = f"{lr:.0e}".replace(
        "e-0",
        "e-"
    )

    model_path = (
        f"experiments/"
        f"lr_{lr_str}_patch_{patch}"
        f"/models/model.pth"
    )

    for thr, msize in itertools.product(
            thresholds,
            minsizes
        ):

        print(
            f"Eval "
            f"thr={thr} "
            f"min={msize}"
        )

        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "run_experiment_main_pipeline_script.py",

                "--tiff", TIFF,

                "--infile_yaml", CFG,

                "--roi",

                "--run_infer",

                "--pred_all",

                "--pred_septs",

                "--preprocess_dir",
                "preprocessing",

                "--model_path",
                model_path,

                "--threshold",
                str(thr),

                "--min_size",
                str(msize),

                "--outdir_local",
                "experiments",

                "--save_plots",

                "--save_result"
            ],
            check=True
        )
