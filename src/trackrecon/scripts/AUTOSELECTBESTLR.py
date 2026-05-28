def pick_best_lr(lrs, losses):
    # smooth losses
    smooth = np.convolve(losses, np.ones(5)/5, mode='valid')

    # find minimum loss point
    idx = np.argmin(smooth)

    return lrs[idx]
