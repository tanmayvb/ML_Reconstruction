import yaml
import argparse
import pandas as pd
import os
import time
import tifffile as tiff
import torch
from pathlib import Path

from trackrecon.scripts.output_build_dir import build_output_dir_name
from trackrecon.scripts.background_suppression import suppress_background
from trackrecon.scripts.noise_reduction import denoise_volume
from trackrecon.createmasks.load_data import load_volume
from trackrecon.createmasks.crop_volume import crop_volume
from trackrecon.scripts.plot_script import *
from trackrecon.createmasks.auto_detect_points import auto_points

from trackrecon.scripts.file_utils import (
    save_points_csv, 
    load_points_csv
)

from trackrecon.createmasks.filter_points_advance import filter_points
from trackrecon.createmasks.create_gaussian_mask import gaussian_mask
from trackrecon.training.train_model import train_model
from trackrecon.models.model_3dunet import UNet3D
from trackrecon.inference.predict_full_volume import predict_full_volume
from trackrecon.inference.postprocess import extract_spines
from trackrecon.inference.classification import extract_structures
from trackrecon.inference.metrics import match_spines


#================================================================
#Example Run Guidelines
#Step scheduler (default params): python train.py --scheduler step
#Step scheduler (custom params):  python train.py --scheduler step --step_size 5 --gamma 0.3

#================================================================

#=============================
# Arguments 
#=============================

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="Machine Learning, Neural Network ",
        description="3D UNet-based deep learning pipeline for automated dendrite, axon, and spine detection from microscopy volumes",
    )

    parser.add_argument(
        "--tiff",
        type=Path,
        help="Folder containing raw Thorlabs TIFF files",
    )

    parser.add_argument(
        "--infile_yaml",
        help="YAML file containing list inputs"
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default="Output",
        help="Directory to write all outputs Tiff and cvs ",
    )

    parser.add_argument(
        "--diff_outdirpath",
        type=str,
        default=None,
        help="Optional different base output directory"
    )

    parser.add_argument(
        "--base_path",
        type=str,
        default=None,
        help="Base path of dataset (auto-detected provide only for test run)"
    )

    parser.add_argument(
        "--outdir_local", 
        type=Path,
        default="dir_debug",
        help="Easy in the debuging step"
    )

    parser.add_argument(
        "--raw_stat",
        action="store_true"
    )

    parser.add_argument(
        "--run_detect", 
        action="store_true"
    )

    parser.add_argument(
        "--run_filter", 
        action="store_true"
    )

    parser.add_argument(
        "--run_mask", 
        action="store_true"
    )

    parser.add_argument(
        "--run_train", 
        action="store_true"
    )

    parser.add_argument(
        "--roi", 
        action="store_true"
    )

    parser.add_argument(
        "--crop_point", 
        action="store_true"
    )

    parser.add_argument(
        "--run_infer",
        action="store_true"
    )

    parser.add_argument(
        "--pred_all",
        action="store_true"
    )

    parser.add_argument(
        "--pred_septs",
        action="store_true"
    )

    parser.add_argument(
        "--input_modelpath",
        type=Path,
        help="Provide saved mask model path"
    )

    parser.add_argument(
        "--input_filterpoints",
        type=Path,
        help="Provide saved train model path"
    )

    parser.add_argument(
        "--save_result",
        action="store_true"
    )

    parser.add_argument(
        "--save_plots",
        action="store_true"
    )

    parser.add_argument(
        "--run_all", 
        action="store_true"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run full pipeline without writing any output files"
    )

    parser.add_argument("--verbose", action="store_true")

    return parser

#=============================================
# Loading Input Config file
#=============================================

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

#=============================================
# Checks to run the pipeline 
#=============================================

def validate_pipeline(image_path, cfg):
    print("\n========================")
    print("Validating pipeline...")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not cfg:
        raise ValueError("Please provide config file!!")

    if "filter" not in cfg:
        raise ValueError("Missing config: filter")

    if "mask" not in cfg:
        raise ValueError("Missing config: mask")

    if "training" not in cfg:
        raise ValueError("Missing config: training parameters")

    if "inference" not in cfg:
        raise ValueError("Missing config: inference parameters")

    print("✔ Config structure OK")


#=============================================
# Validation of the config file
#=============================================

def validate_config(cfg):
    required = {
        "filter": ["percentile", "min_distance"],
        "mask": ["sigma_z", "sigma_xy"],
        "training": ["epochs", "batch_size"],
        "inference": ["threshold", "min_size"]
    }

    for section, keys in required.items():
        if section not in cfg:
            raise ValueError(f"Missing section: {section}")

        for key in keys:
            if key not in cfg[section]:
                raise ValueError(f"Missing key: {section}.{key}")


def debug_volume_stats(volume):

    print(f"Shape      : {volume.shape}")
    print(f"Dtype      : {volume.dtype}")

    print(f"Min        : {volume.min():.5f}")
    print(f"Max        : {volume.max():.5f}")
    print(f"Mean       : {volume.mean():.5f}")
    print(f"Std        : {volume.std():.5f}")

    print(f"NaN values : {np.isnan(volume).sum()}")

    # ---------------------------------
    # Percentiles
    # ---------------------------------
    for p in [1, 5, 50, 95, 99]:

        print(
            f"P{p}:",
            np.percentile(volume, p)
        )

    # ---------------------------------
    # Sparsity
    # ---------------------------------
    print(
        "Fraction >0.05 :",
        (volume > 0.05).mean()
    )

    print(
        "Fraction >0.1 :",
        (volume > 0.1).mean()
    )

    if volume.mean() < 0.01:
        print("WARNING: extremely sparse volume")


def crop_volume(
        volume,
        z0, z1,
        y0, y1,
        x0, x1
    ):

    cropped = volume[
        z0:z1,
        y0:y1,
        x0:x1
    ]

    print(
        f"[DEBUG] Cropped shape: {cropped.shape}"
    )

    return cropped
  
def crop_point(
        points,
        z0, z1,
        y0, y1,
        x0, x1
    ):

    mask = (
        (points[:,0] >= z0) &
        (points[:,0] <  z1) &
        (points[:,1] >= y0) &
        (points[:,1] <  y1) &
        (points[:,2] >= x0) &
        (points[:,2] <  x1)
    )

    pts = points[mask].copy()

    pts[:,0] -= z0
    pts[:,1] -= y0
    pts[:,2] -= x0

    return pts


def main():

    # ====================================
    # Build parser and load input cfg file
    # ====================================

    parser = build_parser()
    args = parser.parse_args()
    
    config_file = args.infile_yaml

    if config_file is None:
        raise ValueError("Please provide --config-file")

    cfg = load_config(config_file)

    # ====================================
    # Build Output directory
    # ====================================

    if args.diff_outdirpath:
        change_output_dir_path = args.diff_outdirpath
    else:
        change_output_dir_path = None

    if args.base_path:
        dataset_name = args.base_path
    else:
        dataset_name = args.tiff.stem.replace(".ome", "")

    if args.outdir_local: 
        output_dir = Path(args.outdir_local)
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )
    else:
        output_dir = build_output_dir_name("airyscan", args.output_dir, f"{dataset_name}", change_output_dir_path)

    print(f"\n Output directory name : {output_dir} {args.outdir_local}")

    # -------------------------------------------------
    # Main directories
    # -------------------------------------------------
    plot_dir = output_dir / "plots"
    model_dir = output_dir / "models"
    mask_dir = output_dir / "masks"
    csv_dir = output_dir / "csv"

    # -------------------------------------------------
    # Plot subdirectories
    # -------------------------------------------------
    preprocess_plot_dir = plot_dir / "preprocessing"
    training_plot_dir = plot_dir / "training"
    inference_plot_dir = plot_dir / "inference"
    evaluation_plot_dir = plot_dir / "evaluation"

    # -------------------------------------------------
    # Create all sub/directories
    # -------------------------------------------------
    dirs = [
        plot_dir,
        model_dir,
        mask_dir,
        csv_dir,
        preprocess_plot_dir,
        training_plot_dir,
        inference_plot_dir,
        evaluation_plot_dir
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Run all main steps once
    # Need some switch on/off by hand like --roi
    # -------------------------------------------------
    if args.run_all:
        args.raw_stat = args.run_detect = args.run_filter = args.run_mask = args.run_train = args.run_infer = args.save_result = args.save_plots = True

    # ====================================
    # All args and cfg parameters
    # ====================================
    tiff_file = args.tiff
    volume = load_volume(tiff_file)

    base = tiff_file.stem.replace(".ome", "")
    raw_csv = csv_dir / f"{base}_points.csv"
    filt_csv = csv_dir / f"{base}_points_filtered.csv"

    validate_pipeline(tiff_file, cfg)
    validate_config(cfg)

    if args.roi:
        volume = crop_volume(
            volume,
            z0=cfg["crop"]["z0"],
            z1=cfg["crop"]["z1"],
            y0=cfg["crop"]["y0"],
            y1=cfg["crop"]["y1"],
            x0=cfg["crop"]["x0"],
            x1=cfg["crop"]["x1"]
        )

    # ---------------------------------
    # Normalize
    # ---------------------------------
    p1 = np.percentile(volume, 1)
    p99 = np.percentile(volume, 99)

    volume = np.clip(
                (volume - p1) / (p99 - p1 + 1e-8),
                0,
                1
             )

    # ---------------------------------
    # Save processed ROI image
    # ---------------------------------
    if args.roi:
        roi_tiff = mask_dir / f"{base}_roi.ome.tiff"

        tiff.imwrite(
            roi_tiff,
            volume.astype(np.float32)
        )
        print(f"[INFO] ROI image saved: {roi_tiff}")
        print(f"[INFO] ROI shape: {volume.shape}")

    if args.dry_run:
        print("\n=== DRY RUN ===")

        print(f"Input image: {tiff_file}")
        print(f"Output dir: {output_dir}")

        print("\nSteps to execute:")
        print("1. Detect points")
        print("2. Filter points")
        print("3. Create mask")
        print("4. Train model")
        print("5. Run inference")
        print("6. Postprocess")

    # ----------------------------------
    # RAW IMAGE
    # ----------------------------------
    if args.raw_stat:
        print("============== RAW IMAGE STATISTIC ========================= \n")

        debug_volume_stats(volume)

        if args.save_plots:
            show_slice(volume, preprocess_plot_dir)
            show_histogram(volume, preprocess_plot_dir)
            show_max_projection_raw(volume, preprocess_plot_dir)
            show_contrast(volume, preprocess_plot_dir)

    print("=========================================================\n")
    # ----------------------------------
    # AUTO POINT DETECTION
    # ----------------------------------
    if args.run_detect:
        print("============== Running AUTO POINT DITECTION=========================")

        points = auto_points(
            volume,
            threshold_abs=cfg["detect"]["threshold_abs"],
            min_distance=cfg["detect"]["min_distance"]
        )

        if args.crop_point:
            points = crop_point(
            points,
            z0=20,
            z1=60,
            y0=500,
            y1=1200,
            x0=1000,
            x1=2000
        )

        print(f"[INFO] Detected Points : {len(points)}")
        save_points_csv(points, raw_csv)

    # -------------------
    # FILTER POINTS
    # contrast = local_mean - background
    #score = contrast
    #score = how strong / distinct a candidate spine point is
    #Each detected point gets a quality value:
    #high score → strong spine candidate
    #low score → noise / background
    #patch_small → local signal
    #patch_large → background
    #Then: score = (signal − background)
    # -------------------

    # ----------------------------------
    # FILTERING
    # ----------------------------------

    if args.run_filter: 
        print("========================Running FILTER STEP============================\n")
        
        detect_points = load_points_csv(raw_csv)

        filtered_points, scores= filter_points(
            volume,
            detect_points,
            percentile = cfg["filter"]["percentile"],
            min_distance = cfg["filter"]["min_distance"],
            contrast_thresh = cfg["filter"]["contrast_thresh"]
        )

        print(f"[DEBUG] Original Points : {len(points)}")
        print(f"[DEBUG] Filtered Points : {len(filtered_points)}")

        save_points_csv(filtered_points, filt_csv, scores=scores)
        print(f"[INFO] Filter  Points save: {filt_csv}")

        # ----------------------------------
        # Make Plots
        # ----------------------------------
        if args.save_plots: 
            plot_points_comparison(
                volume, 
                detect_points,
                filtered_points,
                preprocess_plot_dir,
                z = None
            )

            plot_filtered_points(
                volume,
                filtered_points,
                preprocess_plot_dir,
                z = None
            )

            percentiles = [10,20,30,40,50,60,70,80,90]
            counts = []

            for p in percentiles:
                threshold = np.percentile(scores, p)
                remaining = np.sum(scores > threshold)
                counts.append(remaining)

            plot_filter_retention(
                percentiles=[10,20,30,40,50,60,70,80,90],
                counts=counts,
                save_path=preprocess_plot_dir
            )

    # ----------------------------------
    # CREATE MASK: GAUSSIAN
    # ----------------------------------
    if args.run_mask:
        print("====================MASKING============================== \n")

        filt_points = load_points_csv(filt_csv)

        mask = gaussian_mask(
            volume.shape,
            filt_points,
            sigma_z=cfg["mask"]["sigma_z"],
            sigma_xy=cfg["mask"]["sigma_xy"]
        )

        mask_file = mask_dir / f"{base}_mask.ome.tiff"
        tiff.imwrite(
            mask_file,
            mask.astype("float32")
        )
       
        print(f"[INFO] Mask file saved: {mask_file}")
        print(f"[INFO] Mask Shape: {mask.shape}")
        print(f"Min        : {mask.min()}")
        print(f"Max        : {mask.max()}")
        print(f"Mean       : {mask.mean()}")
        print(f"Std        : {mask.std()}")

        print("====================MASKING============================== \n")

        # ----------------------------------
        # FILTERING
        # ----------------------------------
        if args.save_plots: 
            """
            debug_dataset(
                [tiff_file, mask], 
                preprocess_plot_dir
            )
            """
            plot_mask_overlay(
                volume,
                mask,
                preprocess_plot_dir,
                z = None
            )
           
            plot_flt_points_and_mask(
                volume,
                filt_points,
                mask,
                preprocess_plot_dir,
                z = None
            )

            show_max_projection_raw_and_mask(
                volume,
                mask,
                preprocess_plot_dir
            )

            show_slice_overlay(
                volume,
                mask,
                preprocess_plot_dir,
                z=None
            )
          
            plot_z_distribution(
                filt_points,
                preprocess_plot_dir
            )
         
            spatial_distribution(
                filtered_points,
                preprocess_plot_dir
            )
   
    # -------------------
    # TRAIN MODEL
    # -------------------
    if args.run_train:
        print("=============== Training... ============================== \n")
        if args.roi:
            image_paths = [roi_tiff]     
        else:
            image_paths = [tiff_file] 
        
        mask_paths = [mask_file]

        if mask_file.exists(): 
            print(f"[DEBUG : ] Mask File found: {mask_file}")
        else:
            print(f"[DEBUG : ] Check File and OR name !!")

        if tiff_file.exists(): 
            print(f"[DEBUG : ] Tiff File found: {tiff_file}")
        else:
            print(f"[DEBUG : ] Check File !!")

        model_path, loss_history, lr_history = train_model(image_paths, mask_paths, model_dir, cfg) #args=args
        #model_path = train_model(volume, image_paths, mask_paths, output_dir)

        # ----------------------------------
        # Make Plots
        # ----------------------------------
        if args.save_plots:
            lr_smooth(
                lr_history, 
                loss_history, 
                training_plot_dir
            )         
         
            loss_curve(
                loss_history, 
                training_plot_dir
            ) 

            lr_schedule(
                lr_history, 
                training_plot_dir
            )

    # -------------------
    # INFERENCE
    # -------------------
    if args.run_infer:
        print("===============Running INFERENCE==============================\n")

        if args.input_modelpath:
            model_path = args.input_modelpath
        else:
            model_path = model_dir/"model.pth"
        
        print(f"Model Path: { model_path}")

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = UNet3D().to(device)

        model.load_state_dict(
            torch.load(
                model_path, 
                map_location=device,
                weights_only=True
            )
        )
        #model.eval()
        #print(f"Run: Perdiction")
        #print(next(model.parameters()).mean())
        start = time.time()

        prediction = predict_full_volume(
                         model, 
                         volume, 
                         device=device
                     )

        print("Inference time:", time.time() - start)
        print(f"Shape Volume: {volume.shape}")
        print(f"Shape      : {volume.shape}")
        print(f"Dtype      : {volume.dtype}")
        print(f"Min        : {volume.min():.5f}")
        print(f"Max        : {volume.max():.5f}")
        print(f"Mean       : {volume.mean():.5f}")
        print(f"Std        : {volume.std():.5f}")

        print(f"NaN values : {np.isnan(volume).sum()}")

        print("==================================\n")

        # -------------------
        # POSTPROCESS
        # -------------------
        print("===============Analysis, Detection==============================")

        threshold=cfg["inference"]["threshold"]
        min_size=cfg["inference"]["min_size"]

        print(f"Threshold: {threshold}")
        print(f"Min size: {min_size}")

        print(
            f"Prediction range: "
            f"{prediction.min():.4f} -> "
            f"{prediction.max():.4f}"
        )

        if args.pred_all:
            pred_spines, labeled = extract_spines(
                prediction,
                threshold,
                min_size
            )

        print(f"Detected spines: {len(pred_spines)}")

        if args.pred_septs:
            structures, labeled = extract_structures(
                prediction,
                threshold,
                min_size
            )

            axons = [
                s for s in structures
                if s["type"] == "axon"
            ]

            dendrites = [
                s for s in structures
                if s["type"] == "dendrite"
            ]

            print(f"Axons detected     : {len(axons)}")
            print(f"Dendrites detected : {len(dendrites)}")
 
        # -------------------
        # EVALUATION
        # -------------------
        print("\n===============Run EVALUATION==============================")

        if args.input_filterpoints:
            filtered_points = load_points_csv(
                args.input_filterpoints
            )
        else:
            filtered_points = load_points_csv(
            filt_csv
            )

        print(f"Spines\n")

        if args.pred_all:
            precision, recall = match_spines(pred_spines, filtered_points)

            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            print(
                f"Precision = {precision:.4f},"
                f"Recall = {recall:.4f},"
                f"F1:= {f1}"
            )

            print(f"\n========================================\n")

        print(f"Dendrites\n")

        if args.pred_septs:
            precision_dens, recall_dens = match_spines(
                dendrites,
                filtered_points
            )
            f1_dens = 2 * precision_dens * recall_dens / (precision_dens + recall_dens + 1e-8)
 
            print(
                f"Precision_dendrites = {precision_dens:.4f},"
                f"Recall_dendrites = {recall_dens:.4f},"
                f"F1_dendrites:= {f1_dens}"
            )

            precision_axons, recall_axons = match_spines(
                axons,
                filtered_points
            )

            f1_axons = 2 * precision_axons * recall_axons / (precision_axons + recall_axons + 1e-8)

            print(
                f"Precision_dendrites = {precision_dens:.4f},"
                f"Recall_dendrites = {recall_dens:.4f},"
                f"F1_dendrites:= {f1_dens}"
            )

            print(f"\n========================================\n")
            
            print(f"Axons\n")
            print(
                f"Precision_axons = {precision_axons:.4f},"
                f"Recall_axons = {recall_axons:.4f},"
                f"F1_axons:= {f1_axons}"
            )

        #---------------------------------------------------------
        # Make Plots
        #---------------------------------------------------------
        if args.save_plots:
            precision_recall_curve(
                threshold,
                precision,
                recall,
                evaluation_plot_dir
            )

            f1_curve(
                threshold,
                f1,
                evaluation_plot_dir
            ) 

            show_prediction(
                volume,
                prediction,
                evaluation_plot_dir,
                z=None
            )
           
            plot_threshold_overlay(
                volume,
                prediction,
                threshold,
                evaluation_plot_dir,
                z=None
           )     

            plot_spines(
                volume,
                pred_spines,
                evaluation_plot_dir,
                z=None
            )
            
            plot_prediction_histogram(
                prediction,
                evaluation_plot_dir
            )

            plot_component_size_distribution(
                pred_spines,
                evaluation_plot_dir
            )

            visualize_prediction_results(
                volume=volume,
                prediction=prediction,
                structures=structures,
                save_path=(
                    evaluation_plot_dir
                )
            ) 


    # -------------------
    # SAVE RESULT
    # -------------------
    if args.save_result:
        results_file = output_dir / "results.csv"

        result = {
            "dataset": base,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_spines": len(pred_spines),
            **cfg.get("filter", {}),
            **cfg.get("mask", {}),
            **cfg.get("training", {}),
            **cfg.get("inference", {})
        }

        df = pd.DataFrame([result])

        if not results_file.exists():
            df.to_csv(results_file, index=False)
        else:
            df.to_csv(results_file, mode="a", header=False, index=False)

        if args.save_plots:
            learning_rate_vs_f1(
            results_file,
            evaluation_plot_dir
          )

if __name__ == "__main__":
    import sys
    main()




