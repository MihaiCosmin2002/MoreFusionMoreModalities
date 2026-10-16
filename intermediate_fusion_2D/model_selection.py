"""
Model-selection protocol from Raab et al. (Sci. Rep. 2023, "Methods > Implementation details"):

  * during training save the best-val-loss network AND every 3rd epoch
  * run a moving window of `window` (=50) epochs over the last `last_n` (=150) epochs,
    moving one epoch at a time, and compute the variance of the validation loss in each window
  * pick the window with the smallest variance
  * inside that window rank the epochs by validation loss (lowest first) and take the first one
    that was actually saved to disk; if none was saved, fall back to the next-smallest-variance window
"""
from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np


@dataclass
class Selection:
    epoch: int            # 0-based epoch index
    val_loss: float
    window_start: int     # 0-based, inclusive
    window_end: int       # 0-based, inclusive
    window_var: float
    window_rank: int      # 0 = smallest-variance window
    loss_rank: int        # 0 = lowest loss inside that window


def select_best_checkpoint(val_losses, saved_epochs, last_n=150, window=50):
    losses = np.asarray(val_losses, dtype=np.float64)
    saved = {int(e) for e in saved_epochs}
    n_ep = len(losses)
    if n_ep == 0 or not saved:
        return None

    base = max(0, n_ep - last_n)            # only the last `last_n` epochs are considered
    hist = losses[base:]
    w = min(window, len(hist))              # short runs (early stopping) -> window shrinks
    n_win = len(hist) - w + 1
    variances = np.array([hist[s:s + w].var() for s in range(n_win)])

    for w_rank, s in enumerate(np.argsort(variances, kind="stable")):
        epochs = np.arange(base + s, base + s + w)
        by_loss = epochs[np.argsort(losses[epochs], kind="stable")]
        for l_rank, e in enumerate(by_loss):
            if int(e) in saved:
                return Selection(epoch=int(e), val_loss=float(losses[e]),
                                 window_start=int(epochs[0]), window_end=int(epochs[-1]),
                                 window_var=float(variances[s]), window_rank=w_rank, loss_rank=l_rank)
    return None


class CheckpointKeeper:
    """Saves best-val-loss + every `save_every`-th epoch, and prunes checkpoints that can no
    longer fall into the final `last_n`-epoch range (keeps disk usage bounded)."""

    def __init__(self, folder, save_every=3, last_n=150):
        self.dir = Path(folder) / "epoch_ckpts"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_every, self.last_n = save_every, last_n
        self.saved = {}                      # epoch(0-based) -> path
        self.best_loss, self.best_epoch = float("inf"), -1

    @property
    def saved_epochs(self):
        return set(self.saved)

    def step(self, model, epoch, val_loss):
        import torch
        improved = val_loss < self.best_loss
        if improved:
            self.best_loss, self.best_epoch = val_loss, epoch
        if improved or (epoch + 1) % self.save_every == 0:
            path = self.dir / f"epoch_{epoch + 1:04d}.pth"
            torch.save(model.state_dict(), path)
            self.saved[epoch] = path
        for e in list(self.saved):
            if e < epoch - self.last_n + 1 and e != self.best_epoch:
                self.saved.pop(e).unlink(missing_ok=True)

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)