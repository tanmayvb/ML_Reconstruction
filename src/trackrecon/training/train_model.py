import os
import numpy as np
import torch
import random
import time
from torch.utils.data import DataLoader
import torch.nn as nn
#from torch.utils.data import Dataset

from trackrecon.patching.dataset_multivolume import MultiVolumeDataset
from trackrecon.models.model_3dunet import UNet3D
from trackrecon.models.loss_fun import get_loss
from trackrecon.scripts.lr_utils import run_lr_finder, pick_best_lr

#==================================================
#Correct flow:
#Forward pass : Compute loss : Backward pass
#Optimizer step (updates weights using LR)
#Scheduler step (updates LR for next epoch)
#smooth curve
#smooth = np.convolve(losses, np.ones(5)/5, mode='valid')

#lr = [1e-4, 5e-5, 3e-5, 1e-5] scan these values
#def train_model(volume, mask, cfg):
#==================================================


def train_model(image_paths, mask_paths, output_dir, cfg):

    # -----------------------------
    # DATASET + DATALOADER
    # -----------------------------

    #image_paths = sorted(glob.glob("data/processed/*.tiff"))
 
    assert len(image_paths) == len(mask_paths), "Mismatch in images and masks"

    print("==========================================================")
    print(f"[INFO] Found {len(image_paths)} volumes")

    dataset = MultiVolumeDataset(image_paths, mask_paths)

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"], 
        shuffle=True
    )

    # -----------------------------
    # DEVICE
    # -----------------------------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[DEBUG] Using device: {device}")

    # -----------------------------
    # MODEL
    # -----------------------------
    model = UNet3D().to(device)
    loss_fn = get_loss("dice_bce")

    epochs = cfg["training"]["epochs"]
    # -----------------------------
    # LR FINDER
    # -----------------------------
    if(cfg["training"]["auto_lr"]):    
        print("Running LR finder...")
        lrs, losses = run_lr_finder(model, loader, loss_fn, device)
        best_lr = pick_best_lr(lrs, losses)
        print("Best LR:", best_lr)

        # Reset Model very important
        model = UNet3D().to(device)

    else:
        #best_lr = args.lr
        best_lr = float(cfg["training"]["lr"])


# -----------------------------
# OPTIMIZER + SCHEDULER
# -----------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    scheduler_type = cfg["training"]["scheduler"]["type"]
    step_size = cfg["training"]["scheduler"]["step_size"]
    gamma = cfg["training"]["scheduler"]["gamma"]

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
            #T_max=cfg["training"]["epochs"]
        )
    
    # Step scheduler simpler one
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    else:
        raise ValueError("Unknown scheduler")

    # -----------------------------
    # TRAINING LOOP
    # -----------------------------
    model.train()

    loss_history = []
    lr_history = []

    for epoch in range(epochs):
        #for epoch in range(cfg["training"]["epochs"]):
        start = time.time()
        total_loss = 0

        criterion = nn.BCEWithLogitsLoss()

        for x, y in loader:

            batch_start = time.time()
            x = x.float().to(device)
            y = y.float().to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            if torch.isnan(loss):
                print("NaN detected. Stopping training.")
                return

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
      
            #print(f"[DEBUG] Batch took {time.time()-batch_start:.2f}s")
            total_loss += loss.item()
        #print(f"[DEBUG] Epoch took {time.time()-start:.2f}s")
          
        scheduler.step()
        #print(f"[DEBUG] Using Scheduler: {scheduler_type}")
        if (scheduler_type == "step"): 
            print(f"[INFO] step_size={step_size}, gamma={gamma}")
            
        avg_loss = total_loss / max(1, len(loader)) # Avoid NAN
        loss_history.append(avg_loss)

        lr = optimizer.param_groups[0]["lr"]
        lr_history.append(lr)

        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, LR={lr:.2e}, Len Loader={len(loader)}")

    # -----------------------------
    # SAVE MODEL
    # -----------------------------

    model_path = output_dir/"model.pth"
    torch.save(model.state_dict(), model_path)

    print(f"[INFO :] training complete")
    print(f"[INFO :] model save :  {model_path}")
    print("==========================================================")

    return model_path, loss_history, lr_history
