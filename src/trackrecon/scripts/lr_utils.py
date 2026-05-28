import torch
import numpy as np


def run_lr_finder(model, loader, loss_fn, device,
                  start_lr=1e-6, end_lr=1e-2, num_iter=100):

    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)

    lrs = []
    losses = []

    lr = start_lr
    mult = (end_lr / start_lr) ** (1 / num_iter)

    model.train()

    for i, (x, y) in enumerate(loader):
        if i >= num_iter:
            break

        x = x.float().to(device)
        y = y.float().to(device)

        for g in optimizer.param_groups:
            g["lr"] = lr

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lrs.append(lr)
        losses.append(loss.item())

        lr *= mult

    return lrs, losses


def pick_best_lr(lrs, losses):
    losses = np.array(losses)
    lrs = np.array(lrs)

    # pick LR at minimum loss
    idx = np.argmin(losses)

    return lrs[idx]
