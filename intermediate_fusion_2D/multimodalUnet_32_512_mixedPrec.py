# Author: Mihai Cosmin (Modified for PyTorch)
# Institution: West University of Timisoara, Romania

import csv
import gc
import os
import argparse
import shutil
import numpy as np
import random as rn
from glob import glob
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

# Import your converted local modules
import slicer
import losses
import architecture
import data_generator
import folder_structure
import model_selection
from tqdm import tqdm, trange
import psutil
from utils import print_and_save, epoch_time, calculate_metrics


# ----------------------------------------------------------------------------
# ARGUMENTS
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mask", type=str, default="mask1", required=False,
                    help="Number of the Mask! Default is Mask 1.")
parser.add_argument("-ep", "--epochs", type=int, default=1000, required=False,
                    help="Number of the Epochs! Default is 1000.")
parser.add_argument("-f", "--filters", type=int, default=32, required=False,
                    help="Number of the filters in first downsampling stage! Default is 32!")
parser.add_argument("-exp", "--experiment", type=str, default="ISBI", required=False,
                    help="Name of the experiment. Default is ISBI!")
parser.add_argument("-bs_tr", "--batchsize_train", type=int, default=15, required=False,
                    help="Batch size for training. Default is 15!")
parser.add_argument("-bs_val", "--batchsize_val", type=int, default=50, required=False,
                    help="Batch size for validation. Default is 50!")
parser.add_argument("-n", "--num", type=int, default=0, required=False,
                    help="(loocv only) Declares, which combinations should be trained with this network! Default is 0")
parser.add_argument("-s", "--slice", type=bool, default=False, required=False,
                    help="Declares, if data should be sliced or not! Default is False")
parser.add_argument("--preview_every", type=int, default=5, required=False,
                    help="Save a preview image every N epochs. Default is 5.")

# --- fusion related ---
parser.add_argument("--fusion", choices=["early", "intermediate", "multibranch"], default="early",
                    help="early: stacked channels | intermediate: pretrained frozen encoders + 1x1 fusion | "
                         "multibranch: one encoder per modality + MMFF")
parser.add_argument("--modalities", nargs="+", default=["flair", "t1", "t2"],
                    help="Any of: flair t1 t2 pd (the order is the branch order)")
parser.add_argument("--p1_epochs", type=int, default=50,
                    help="Phase-1 encoder pretraining epochs (intermediate only)")
parser.add_argument("--no_pretrain", action="store_true",
                    help="Intermediate fusion WITHOUT encoder pretraining/freezing (ablation)")
parser.add_argument("--num_workers", type=int, default=12,
                    help="DataLoader workers per loader. Default is 12.")
parser.add_argument("--patience", type=int, default=200,
                    help="Early stopping: stop after N epochs without improvement. 0 disables it. Default is 200.")
parser.add_argument("--min_delta", type=float, default=1e-4,
                    help="Minimum F1 increase that counts as an improvement.")

# --- dataset split protocol (paper: Table A1 vs. Table A4) ---
parser.add_argument("--protocol", choices=["loocv", "timepoint"], default="loocv",
                    help="loocv: nested leave-one-subject-out (Table A1, 3 train / 1 val subjects) | "
                         "timepoint: Table A4 -> all subjects for training, one time point per subject for validation")
parser.add_argument("--tp_combo", choices=["first", "second", "third", "fourth", "all"], default="first",
                    help="(timepoint only) Table A4 combination: first=last time point, second=3, third=2, "
                         "fourth=1, all=run the four combinations one after another")

# --- best-model selection (paper: moving-window variance protocol) ---
parser.add_argument("--select", choices=["window", "f1"], default="window",
                    help="window: paper protocol (smallest val-loss-variance window, then lowest loss, "
                         "among saved checkpoints) | f1: best validation F1 (BestModel.pth)")
parser.add_argument("--save_every", type=int, default=3,
                    help="Save a checkpoint every N epochs (plus every new best val-loss). Paper: 3.")
parser.add_argument("--sel_last", type=int, default=150,
                    help="Only the last N epochs are considered by the selection. Paper: 150.")
parser.add_argument("--sel_window", type=int, default=50,
                    help="Moving-window size in epochs. Paper: 50.")
parser.add_argument("--keep_ckpts", action="store_true",
                    help="Keep all intermediate epoch checkpoints after selection (large!).")

args = parser.parse_args()
msk = args.mask
exp_name = args.experiment
num_of_combinations = args.num
to_slice = args.slice
PREVIEW_EVERY = args.preview_every

NUM_OF_EPOCHS = args.epochs
BS_tr = args.batchsize_train
BS_val = args.batchsize_val
FILTERS = args.filters
NUM_WORKERS = args.num_workers
PATIENCE = args.patience
MIN_DELTA = args.min_delta

PROTOCOL = args.protocol
SELECT = args.select
SAVE_EVERY = args.save_every
SEL_LAST = args.sel_last
SEL_WINDOW = args.sel_window

FUSION = args.fusion
# validate + normalise modality names early (raises a clear error on typos)
_mod_idx = data_generator.resolve_modalities(args.modalities)
MODALITIES = [data_generator.MODALITY_NAMES[i] for i in _mod_idx]
N_MOD = len(MODALITIES)
P1_EPOCHS = args.p1_epochs
PRETRAIN = (FUSION == "intermediate") and (not args.no_pretrain)
RUN_NAME = f"{exp_name}_{FUSION}{'' if (FUSION != 'intermediate' or PRETRAIN) else '-nopre'}_{'-'.join(MODALITIES)}"

# Reproducibility Seeds
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(37)
rn.seed(1254)
torch.manual_seed(90)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(90)

# Hardware Config for RTX 5070 Ti
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONSTANTS
SEED = 12345678
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
access_rights = 0o777

subj_identifier = ["training01", "training02",
                   "training03", "training04", "training05"]
timepoint_identifier = ["_01_", "_02_", "_03_", "_04_", "_05_"]
msk_identifier = ["mask1", "mask2"]

modalities = ["flair_pp", "mprage_pp", "t2_pp", "pd_pp"]

# Derive repository root relative to this script and point dataset folders there by default.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
base_data = os.path.join(REPO_ROOT, 'training')  # path where your base volumes are stored
dest_data = os.path.join(REPO_ROOT, 'training_sliced')  # path where your sliced data should be stored
dest_trained_nets = os.path.join(REPO_ROOT, 'trained_models')  # path where your trained networks should be stored

# --- COMBINATION LOGIC ---
one, two, three, four, five = subj_identifier
train_combinations = [
    [one, two, three, four, "_123_4"], [one, two, three, five, "_123_5"],
    [one, two, four, three, "_124_3"], [one, two, four, five, "_124_5"],
    [one, three, four, two, "_134_2"], [one, three, four, five, "_134_5"],
    [two, three, four, one, "_234_1"], [two, three, four, five, "_234_5"],
    [one, two, five, three, "_125_3"], [one, two, five, four, "_125_4"],
    [one, three, five, two, "_135_2"], [one, three, five, four, "_135_4"],
    [two, three, five, one, "_235_1"], [two, three, five, four, "_235_4"],
    [one, four, five, two, "_145_2"], [one, four, five, three, "_145_3"],
    [two, four, five, one, "_245_1"], [two, four, five, three, "_245_3"],
    [three, four, five, one, "_345_1"], [three, four, five, two, "_345_2"]
]

# NOTE: must be a LIST of combinations (the loop below iterates over it)
if num_of_combinations == 0: train_subs = [train_combinations[0]]
elif num_of_combinations == 1: train_subs = train_combinations[:5]
elif num_of_combinations == 2: train_subs = train_combinations[5:10]
elif num_of_combinations == 3: train_subs = train_combinations[10:15]
elif num_of_combinations == 4: train_subs = train_combinations[15:20]
else: train_subs = train_combinations


def build_folds():
    """Unified fold description for both protocols."""
    folds = []
    if PROTOCOL == "loocv":
        for a, b, c, v, tag in train_subs:
            folds.append(dict(tag=tag, train=[a, b, c], val=[v], val_tp=None))
    else:  # Table A4: every subject is in train AND val, split by acquisition time point
        names = (list(data_generator.TIMEPOINT_COMBINATIONS) if args.tp_combo == "all"
                 else [args.tp_combo])
        for name in names:
            folds.append(dict(tag=f"_A4_{name}", train=list(subj_identifier),
                              val=list(subj_identifier),
                              val_tp=data_generator.TIMEPOINT_COMBINATIONS[name]))
    return folds


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def compute_loss(target, logits):
    # Original Keras model trained with dice_loss on sigmoid outputs, in float32
    return losses.DiceBCELoss(target, logits.float())


def pick_preview_sample(dataset):
    """Return the index of the val slice with the largest lesion area."""
    best_i, best_area = 0, -1.0
    for i, p in enumerate(dataset.mask_paths):
        area = data_generator._load(p, dataset.img_size).sum()
        if area > best_area:
            best_i, best_area = i, area
    return best_i


@torch.no_grad()
def save_preview(model, sample, epoch, out_dir, device, thr=0.5):
    inputs, mask = sample                       # inputs: (n_mod, H, W) since stack=True
    model.eval()
    with autocast(device_type='cuda'):
        logits = model(inputs.unsqueeze(0).to(device))
    prob = torch.sigmoid(logits.float())[0, 0].cpu().numpy()

    gt = mask[0].numpy()
    pred = (prob > thr).astype(np.float32)
    img = inputs[0].numpy()                     # first selected modality
    name = MODALITIES[0].upper()
    dice = 2 * (gt * pred).sum() / (gt.sum() + pred.sum() + 1e-9)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(img, cmap="gray");                      ax[0].set_title(name)
    ax[1].imshow(gt, cmap="gray");                       ax[1].set_title("Ground truth")
    ax[2].imshow(prob, cmap="gray", vmin=0, vmax=1);     ax[2].set_title("Predicted probability")
    ax[3].imshow(img, cmap="gray")
    if gt.any():
        ax[3].contour(gt, levels=[0.5], colors="lime", linewidths=1)
    if pred.any():
        ax[3].contour(pred, levels=[0.5], colors="red", linewidths=1)
    ax[3].set_title(f"Overlay (green=GT, red=pred) Dice={dice:.3f}")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"Epoch {epoch} | {FUSION} | {'+'.join(MODALITIES)}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"epoch_{epoch:04d}.png"), dpi=80)
    plt.close(fig)


def train(model, loader, optimizer, device, scaler, max_steps, frozen=(), scheduler=None):
    model.train()
    for m in frozen:            # frozen encoders must not update BN running stats
        m.eval()
    running_loss, n = 0.0, 0

    bar = tqdm(loader, desc="Train", leave=False, total=max_steps)
    for batch_idx, (x, target) in enumerate(bar):
        if batch_idx >= max_steps:          # enforce steps_per_epoch
            break

        x = x.to(device, dtype=torch.float32, non_blocking=True)
        target = target.to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda'):
            output = model(x)
        loss = compute_loss(target, output.float())   # fp32, outside autocast

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
        if scheduler:
            scheduler.step()

        running_loss += loss.item()
        n += 1
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(n, 1)


def evaluate(model, loader, device, max_steps, apply_sigmoid=True):
    model.eval()
    epoch_loss, n = 0.0, 0
    metrics_sum = np.zeros(6)   # jac, f1, recall, precision, acc, f2

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(tqdm(loader, desc="Eval", leave=False, total=max_steps)):
            if batch_idx >= max_steps:
                break

            x = x.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.float32)

            with autocast(device_type='cuda'):
                output = model(x)
            output = output.float()
            epoch_loss += compute_loss(target, output).item()

            y_prob = torch.sigmoid(output) if apply_sigmoid else output
            y_hat = (y_prob > 0.5).float()

            batch_scores = [calculate_metrics(yt, yp) for yt, yp in zip(target, y_hat)]
            metrics_sum += np.mean(batch_scores, axis=0)
            n += 1

    return epoch_loss / max(n, 1), (metrics_sum / max(n, 1)).tolist()


def finalize_selection(model, keeper, history, checkp_folder, log_path, tag):
    """Pick the final model of this fold and copy it to SelectedModel.pth.
    Returns (selected_path, info_dict)."""
    selected_path = os.path.join(checkp_folder, "SelectedModel.pth")
    best_f1_path = os.path.join(checkp_folder, "BestModel.pth")
    info = dict(fold=tag, method=SELECT, epoch="", val_loss="", val_f1="",
                win_start="", win_end="", win_var="")

    sel = None
    if SELECT == "window":
        sel = model_selection.select_best_checkpoint(history['val_loss'], keeper.saved_epochs,
                                                     SEL_LAST, SEL_WINDOW)

    if sel is not None:
        shutil.copy2(keeper.saved[sel.epoch], selected_path)
        info.update(epoch=sel.epoch + 1, val_loss=sel.val_loss, val_f1=history['f1'][sel.epoch],
                    win_start=sel.window_start + 1, win_end=sel.window_end + 1, win_var=sel.window_var)
        print_and_save(log_path,
            f"[selection:window] epoch {sel.epoch + 1} | val loss {sel.val_loss:.4f} | "
            f"val F1 {history['f1'][sel.epoch]:.4f} | window epochs {sel.window_start + 1}-{sel.window_end + 1} "
            f"(var {sel.window_var:.3e}, window rank {sel.window_rank}, loss rank {sel.loss_rank})")
    else:
        if SELECT == "window":
            print_and_save(log_path, "[selection:window] no usable checkpoint -> falling back to best-F1 model")
        src = best_f1_path if os.path.exists(best_f1_path) else os.path.join(checkp_folder, "LastEpoch.pth")
        shutil.copy2(src, selected_path)
        best_ep = int(np.argmax(history['f1']))
        info.update(method="f1", epoch=best_ep + 1, val_loss=history['val_loss'][best_ep],
                    val_f1=history['f1'][best_ep])
        print_and_save(log_path, f"[selection:f1] epoch {best_ep + 1} | F1 {history['f1'][best_ep]:.4f}")
    return selected_path, info


# ----------------------------------------------------------------------------
# PHASE 1 (intermediate fusion only): per-modality encoder pretraining
# ----------------------------------------------------------------------------
def pretrain_encoders(fold_dir, dest_folders_tr, steps):
    """Train one single-modality UNet per modality; save its encoder + bottom block."""
    enc_dir = Path(fold_dir) / "encoders"
    enc_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for mod in MODALITIES:
        ckpt = enc_dir / f"enc_{mod}.pth"
        paths[mod] = ckpt
        if ckpt.exists():
            print(f"[{mod}] encoder checkpoint found, skipping.")
            continue

        print(f"\n--- Phase 1 | modality: {mod} ---")
        loader = data_generator.datagenerator(dest_folders_tr, IMG_SIZE, BS_tr, modalities=[mod],
                                              stack=True, augment=True, num_workers=NUM_WORKERS)
        net = architecture.UNet(FILTERS, n_modalities=1, fusion="early",
                                img_size=IMAGE_HEIGHT).to(device)
        opt = optim.Adam(net.parameters(), lr=1e-4, eps=1e-7)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5)
        sc = GradScaler("cuda")
        best = float("inf")

        for ep in range(P1_EPOCHS):
            loss = train(net, loader, opt, device, sc, steps)
            sched.step(loss)
            if loss < best:
                best = loss
                torch.save({"enc": net.encoders[0].state_dict(),
                            "btm": net.bottoms[0].state_dict()}, ckpt)
            if ep == 0 or (ep + 1) % 5 == 0:
                print(f"  [{mod}] epoch {ep + 1:02}/{P1_EPOCHS}  loss={loss:.4f}  best={best:.4f}")
        print(f"[{mod}] phase 1 done, best train loss {best:.4f}")

        del loader, net, opt, sched, sc      # persistent workers + VRAM
        gc.collect()
        torch.cuda.empty_cache()
    return paths


def load_pretrained_encoders(model, paths):
    for b, mod in enumerate(MODALITIES):
        ck = torch.load(paths[mod], map_location=device)
        model.encoders[b].load_state_dict(ck["enc"])
        model.bottoms[b].load_state_dict(ck["btm"])


def freeze_encoders(model):
    for p in list(model.encoders.parameters()) + list(model.bottoms.parameters()):
        p.requires_grad = False


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    folds = build_folds()
    print("Base data path is: ", base_data)
    print("Value for slice is: ", to_slice)
    print(f"Protocol: {PROTOCOL} | folds: {[f['tag'] for f in folds]}")
    print(f"Run: {RUN_NAME} | fusion={FUSION} | modalities={MODALITIES} | pretrain={PRETRAIN} | select={SELECT}")

    if to_slice:
        for subj in subj_identifier:
            dest_folders_valid = folder_structure.create_paths(base_data, str(msk), str(subj), access_rights, False)
            dest_folders_train = folder_structure.create_paths(base_data, str(msk),  str(subj), access_rights, True)

            for i in range(0, 5):
                # original dataset has fewer entries for some subjects
                if i == 4 and subj != "training03":
                    continue
                else:
                    flair_img, t1_img, t2_img, pd_img, msk1_img, msk2_img = \
                        slicer.find_and_load_data(modalities, msk_identifier, base_data,
                                                  str(subj), str(f"{subj}{timepoint_identifier[i]}"))
                    subj_acq = f"{subj}{timepoint_identifier[i]}"

                    if msk == msk_identifier[0]:
                        print("mask 1 was chosen!")
                        cnt_valid = slicer.slice_and_save_vol_img(IMG_SIZE, msk1_img, flair_img, t1_img, t2_img, pd_img,
                                                                  subj_acq, dest_folders_valid[0],
                                                                  dest_folders_valid[1], dest_folders_valid[2],
                                                                  dest_folders_valid[3], dest_folders_valid[4], False)

                        cnt_train = slicer.slice_and_save_vol_img(IMG_SIZE, msk1_img, flair_img, t1_img, t2_img, pd_img,
                                                                  subj_acq, dest_folders_train[0],
                                                                  dest_folders_train[1], dest_folders_train[2],
                                                                  dest_folders_train[3], dest_folders_train[4], True)
                    elif msk == msk_identifier[1]:
                        print("mask 2 was chosen!")
                        cnt_valid = slicer.slice_and_save_vol_img(IMG_SIZE, msk2_img, flair_img, t1_img, t2_img, pd_img,
                                                                  subj_acq, dest_folders_valid[0],
                                                                  dest_folders_valid[1], dest_folders_valid[2],
                                                                  dest_folders_valid[3], dest_folders_valid[4], False)

                        cnt_train = slicer.slice_and_save_vol_img(IMG_SIZE, msk2_img, flair_img, t1_img, t2_img, pd_img,
                                                                  subj_acq, dest_folders_train[0],
                                                                  dest_folders_train[1], dest_folders_train[2],
                                                                  dest_folders_train[3], dest_folders_train[4], True)
                    else:
                        print("ERROR! The rater you have chosen is not available in this training procedure!")
                        exit()

    run_dir = os.path.join(dest_trained_nets, RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)
    summary_csv = os.path.join(run_dir, "selection_summary.csv")
    fold_results = []

    for fold in folds:
        tag = fold["tag"]

        # --- 1. FOLDER MANAGEMENT ---
        dest_folders_tr, dest_folders_val, checkp_folder, trained_folder = folder_structure.create_folder_for_fold(
            dest_trained_nets, dest_data, RUN_NAME, tag, msk, access_rights)

        if PROTOCOL == "loocv":
            # Table A1: 3 subjects train (lesion slices + every 30th empty), 1 subject validation (all slices)
            for s in fold["train"]:
                folder_structure.copy_data_for_fold(
                    folder_structure.get_paths(base_data, msk, str(s), True), dest_folders_tr)
            folder_structure.copy_data_for_fold(
                folder_structure.get_paths(base_data, msk, str(fold["val"][0]), False), dest_folders_val)
            val_desc = f"val subject {fold['val'][0]}"
        else:
            # Table A4: all subjects; one acquisition time point per subject held out for validation
            train_src = {s: folder_structure.get_paths(base_data, msk, str(s), True) for s in fold["train"]}
            valid_src = {s: folder_structure.get_paths(base_data, msk, str(s), False) for s in fold["val"]}
            chosen_tp = data_generator.build_timepoint_fold(train_src, valid_src, dest_folders_tr,
                                                            dest_folders_val, fold["val_tp"])
            val_desc = f"val time points {chosen_tp}"

        # --- 2. STEP CALCULATION ---
        NUM_TR = len(glob(dest_folders_tr[0] + '/*.png'))
        EP_STEP_TR = NUM_TR // BS_tr

        NUM_VAL = len(glob(dest_folders_val[0] + '/*.png'))
        EP_STEP_VAL = max(NUM_VAL // BS_val, 1)

        print(f"\nSuccessfully created fold's dataset! ({val_desc})")
        print(f"Train slices: {NUM_TR} | Val slices: {NUM_VAL}")
        print(f"Epoch steps train is: {EP_STEP_TR}")
        print(f"Epoch steps val is: {EP_STEP_VAL}")

        log_path = Path(checkp_folder) / "train_log.txt"
        print_and_save(log_path, f"Fold {tag} | Protocol: {PROTOCOL} | {val_desc} | Fusion: {FUSION} | "
                                 f"Modalities: {MODALITIES} | Batch Size: {BS_tr} | LR: 0.0003 | Epochs: {NUM_OF_EPOCHS}")

        # --- 2b. PHASE 1: encoder pretraining (intermediate only) ---
        # Done BEFORE the main loaders exist, so phase-1 workers are gone first.
        enc_paths = None
        if PRETRAIN:
            print_and_save(log_path, "========== PHASE 1: Encoder Pretraining ==========")
            enc_paths = pretrain_encoders(checkp_folder, dest_folders_tr, EP_STEP_TR)
            print_and_save(log_path, "========== PHASE 2: Full Model Training ==========")

        # --- 3. DATA PIPELINE ---
        train_loader = data_generator.datagenerator(dest_folders_tr, IMG_SIZE, BS_tr, modalities=MODALITIES,
                                                    stack=True, augment=True, num_workers=NUM_WORKERS, fixed_shuffle_seed=SEED)
        # validation order is permuted ONCE (fixed seed): batches mix slices of all subjects/time points
        val_loader = data_generator.datagenerator(dest_folders_val, IMG_SIZE, BS_val, modalities=MODALITIES,
                                                  stack=True, augment=False, shuffle=False,
                                                  num_workers=NUM_WORKERS, fixed_shuffle_seed=SEED)

        # --- PREVIEW SETUP (one fixed validation slice per fold) ---
        preview_dir = os.path.join(checkp_folder, "previews")
        os.makedirs(preview_dir, exist_ok=True)
        preview_idx = pick_preview_sample(val_loader.dataset)
        preview_sample = val_loader.dataset[preview_idx]   # augment=False for val
        print(f"Preview slice index: {preview_idx}")

        # --- 4. MODEL & OPTIMIZER ---
        model = architecture.UNet(FILTERS, n_modalities=N_MOD, fusion=FUSION,
                                  img_size=IMAGE_HEIGHT).to(device)
        frozen = []
        if PRETRAIN:
            load_pretrained_encoders(model, enc_paths)
            freeze_encoders(model)
            frozen = [model.encoders, model.bottoms]

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print_and_save(log_path, f"Trainable params: {trainable:,} / {total:,}")

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=0.0003, eps=1e-7)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.9)
        scaler = GradScaler("cuda")   # fresh scaler per fold
        keeper = model_selection.CheckpointKeeper(checkp_folder, SAVE_EVERY, SEL_LAST)

        best_valid_f1 = 0.0
        patience_counter = 0
        best_epoch = 0
        history = {'loss': [], 'val_loss': [], 'f1': []}
        time_start = time.time()

        for epoch in trange(NUM_OF_EPOCHS, desc=f"Fold {tag}"):
            start_time = time.time()

            train_loss = train(model, train_loader, optimizer, device, scaler, EP_STEP_TR, frozen=frozen, scheduler=scheduler)
            valid_loss, valid_metrics = evaluate(model, val_loader, device, EP_STEP_VAL)

            history['loss'].append(train_loss)
            history['val_loss'].append(valid_loss)
            history['f1'].append(valid_metrics[1])

            print_and_save(log_path, f"Current LR: {optimizer.param_groups[0]['lr']:.6e}")

            # --- paper protocol: keep best-val-loss + every N-th epoch for the final selection ---
            keeper.step(model, epoch, valid_loss)

            # --- checkpoint on best validation F1 (Dice) + early stopping ---
            if valid_metrics[1] > best_valid_f1 + MIN_DELTA:
                print_and_save(log_path,
                    f"Valid F1 improved from {best_valid_f1:2.4f} to {valid_metrics[1]:2.4f}. Saving checkpoint.")
                best_valid_f1 = valid_metrics[1]
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(checkp_folder, "BestModel.pth"))
            else:
                patience_counter += 1

            # --- preview on fixed val slice ---
            if epoch % PREVIEW_EVERY == 0:
                save_preview(model, preview_sample, epoch, preview_dir, device)

            epoch_mins, epoch_secs = epoch_time(start_time, time.time())
            print_and_save(log_path,
                f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
                f"\tTrain Loss: {train_loss:.4f}\n"
                f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} "
                f"- Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f} "
                f"- Acc: {valid_metrics[4]:.4f} - F2: {valid_metrics[5]:.4f}\n")
            # --- early stopping ---
            if PATIENCE > 0 and patience_counter >= PATIENCE:
                print_and_save(log_path,
                    f"Early stopping at epoch {epoch+1}: no F1 improvement for {PATIENCE} epochs "
                    f"(best F1 {best_valid_f1:.4f} at epoch {best_epoch+1}).")
                break

        # --- 5. POST-TRAINING SAVING ---
        total_training_time = time.time() - time_start
        print_and_save(log_path, f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(total_training_time))}")

        for name, key in [("loss_history.txt", 'loss'), ("valid_loss_history.txt", 'val_loss'),
                          ("valid_f1_history.txt", 'f1')]:
            np.savetxt(os.path.join(checkp_folder, name), np.array(history[key]), delimiter=",")

        torch.save(model.state_dict(), os.path.join(checkp_folder, 'LastEpoch.pth'))

        # --- 6. FINAL MODEL SELECTION (paper protocol) ---
        selected_path, info = finalize_selection(model, keeper, history, checkp_folder, log_path, tag)
        if not args.keep_ckpts:
            keeper.cleanup()
        write_header = not os.path.exists(summary_csv)
        with open(summary_csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(info.keys()))
            if write_header:
                w.writeheader()
            w.writerow(info)
        fold_results.append(dict(tag=tag, f1=float(info["val_f1"]), ckpt=selected_path))

        print(f"RAM used: {psutil.virtual_memory().percent}%")
        print(f"VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB, "
              f"reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

        # --- CLEANUP between folds ---
        del model, optimizer, scheduler, scaler, frozen, keeper
        del train_loader, val_loader, preview_sample
        gc.collect()
        torch.cuda.empty_cache()

    # --- 7. BEST MODEL OVERALL (across folds, by validation F1/Dice of the selected checkpoint) ---
    if fold_results:
        best = max(fold_results, key=lambda r: r["f1"])
        overall_path = os.path.join(run_dir, "BestOverall.pth")
        shutil.copy2(best["ckpt"], overall_path)
        print("\n=== Selected model per fold ===")
        for r in fold_results:
            print(f"  {r['tag']:>14}  val F1 = {r['f1']:.4f}")
        print(f"Best overall: fold {best['tag']} (F1 {best['f1']:.4f}) -> {overall_path}")