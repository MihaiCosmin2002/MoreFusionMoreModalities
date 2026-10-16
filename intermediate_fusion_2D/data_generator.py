import os
import re
import shutil
from glob import glob

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from torch.utils.data import Dataset, DataLoader

# order of the `folders` list passed to datagenerator (mask is last)
MODALITY_NAMES = ("flair", "t1", "t2", "pd")

# ---------------------------------------------------------------------------
# Table A4 (Raab et al.): ALL subjects are used for training; one acquisition
# time point of every subject is reserved for validation.
#   first  -> last time point of every subject
#   second -> time point 3,  third -> time point 2,  fourth -> time point 1
# ---------------------------------------------------------------------------
TIMEPOINT_COMBINATIONS = {"first": "last", "second": 3, "third": 2, "fourth": 1}


def _load(path, size_hw):
    h, w = size_hw
    img = Image.open(path).convert("L")
    if img.size != (w, h):
        img = img.resize((w, h), Image.NEAREST)
    return np.asarray(img, dtype=np.float32)          # 0..255


def _center(matrix, h, w):
    ox, oy = h / 2 - 0.5, w / 2 - 0.5
    off = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]])
    rst = np.array([[1, 0, -ox], [0, 1, -oy], [0, 0, 1]])
    return off @ matrix @ rst


def _warp(x, m, order):
    return ndi.affine_transform(x, m[:2, :2], m[:2, 2],
                                order=order, mode="nearest", cval=0.0)


def resolve_modalities(modalities):
    """Accepts names ('flair', 'pd') and/or indices (0..3). Returns indices."""
    idx = []
    for m in modalities:
        if isinstance(m, str):
            key = m.lower()
            if key not in MODALITY_NAMES:
                raise ValueError(f"Unknown modality '{m}'. Use one of {MODALITY_NAMES}")
            idx.append(MODALITY_NAMES.index(key))
        else:
            if not 0 <= int(m) < len(MODALITY_NAMES):
                raise ValueError(f"Modality index {m} out of range")
            idx.append(int(m))
    if len(idx) == 0:
        raise ValueError("Select at least one modality")
    if len(set(idx)) != len(idx):
        raise ValueError("Duplicate modalities selected")
    return idx


# ---------------------------------------------------------------------------
# Time-point based split (Table A4)
# ---------------------------------------------------------------------------
def timepoint_prefix(subj, tp):
    """'training01', 3 -> 'training01_03_' (same naming as subj_acq in the slicer)."""
    return f"{subj}_{int(tp):02d}_"


def available_timepoints(folder, subj):
    pat = re.compile(rf"^{re.escape(subj)}_(\d+)_")
    tps = set()
    for f in glob(os.path.join(folder, "*.png")):
        m = pat.match(os.path.basename(f))
        if m:
            tps.add(int(m.group(1)))
    return sorted(tps)


def resolve_validation_timepoint(spec, available, subj=""):
    if not available:
        raise RuntimeError(f"No time points found for {subj}: sliced files are expected to be "
                           f"named '{subj}_<tp>_*.png'. Check the naming used by slicer.py.")
    tp = max(available) if spec == "last" else int(spec)
    if tp not in available:
        raise ValueError(f"{subj}: time point {tp} not available (have {available})")
    return tp


def _clear_pngs(folder):
    os.makedirs(folder, exist_ok=True)
    for f in glob(os.path.join(folder, "*.png")):
        os.remove(f)


def _copy_pngs(src_folder, dst_folder, keep):
    n = 0
    for f in sorted(glob(os.path.join(src_folder, "*.png"))):
        if keep(os.path.basename(f)):
            shutil.copy2(f, os.path.join(dst_folder, os.path.basename(f)))
            n += 1
    return n


def build_timepoint_fold(train_src, valid_src, dest_tr, dest_val, spec):
    """
    train_src / valid_src: {subject: [folder_flair, folder_t1, folder_t2, folder_pd, folder_mask]}
        train_src = sliced TRAIN folders (lesion slices + every 30th empty slice)
        valid_src = sliced VALID folders (every slice)
    dest_tr / dest_val: destination folder lists (same order as the source lists).
    spec: 'last' or an int (1, 2, 3 ...) -> time point reserved for validation in EVERY subject.

    Training  = all subjects, every time point except the reserved one.
    Validation = all subjects, only the reserved time point (every slice).
    Returns {subject: validation_time_point}.
    """
    for d in list(dest_tr) + list(dest_val):
        _clear_pngs(d)

    chosen, n_tr, n_val = {}, 0, 0
    for subj in train_src:
        tp = resolve_validation_timepoint(spec, available_timepoints(valid_src[subj][-1], subj), subj)
        chosen[subj] = tp
        prefix = timepoint_prefix(subj, tp)
        for s, d in zip(train_src[subj], dest_tr):
            n_tr += _copy_pngs(s, d, lambda name: not name.startswith(prefix))
        for s, d in zip(valid_src[subj], dest_val):
            n_val += _copy_pngs(s, d, lambda name: name.startswith(prefix))

    counts_tr = {len(glob(os.path.join(d, "*.png"))) for d in dest_tr}
    counts_val = {len(glob(os.path.join(d, "*.png"))) for d in dest_val}
    if len(counts_tr) != 1 or len(counts_val) != 1 or 0 in counts_tr or 0 in counts_val:
        raise RuntimeError(f"Inconsistent/empty fold: train counts {counts_tr}, val counts {counts_val}")
    return chosen


# ---------------------------------------------------------------------------
class MultiModalDataset(Dataset):
    """
    img_folders: list of folders, one per SELECTED modality (in the order you want them returned).
    stack=False -> returns ([mod_0, mod_1, ...], mask), each mod is (1, H, W)
    stack=True  -> returns (tensor of shape (n_mod, H, W), mask)   # early fusion
    fixed_shuffle_seed: if set, the sample order is permuted ONCE (deterministically). Used for the
        validation set (paper: validation set is shuffled so batch-wise metrics are stable despite the
        heavy foreground/background imbalance; consecutive slices of one subject would otherwise form
        very biased batches).
    """
    def __init__(self, img_folders, folder_mask, img_size, augment=False,
                 mask_order=1, brightness_on_mask=True, stack=False,
                 fixed_shuffle_seed=None):
        self.img_paths = [sorted(glob(os.path.join(f, "*.png"))) for f in img_folders]
        self.mask_paths = sorted(glob(os.path.join(folder_mask, "*.png")))
        n = len(self.mask_paths)
        assert n > 0, "no masks found"
        assert all(len(p) == n for p in self.img_paths), "folder sizes differ"

        if fixed_shuffle_seed is not None:
            perm = np.random.default_rng(fixed_shuffle_seed).permutation(n)
            self.img_paths = [[p[j] for j in perm] for p in self.img_paths]
            self.mask_paths = [self.mask_paths[j] for j in perm]

        self.img_size = tuple(img_size)
        self.augment = augment
        self.mask_order = mask_order
        self.brightness_on_mask = brightness_on_mask
        self.stack = stack

    def __len__(self):
        return len(self.mask_paths)

    def _augment(self, imgs, mask):
        h, w = self.img_size
        rng = np.random.default_rng()

        theta = np.deg2rad(rng.uniform(-360, 360))
        tx = rng.uniform(-0.3, 0.3) * h
        ty = rng.uniform(-0.3, 0.3) * w
        zx, zy = rng.uniform(0.5, 1.5, 2)
        flip_h = rng.random() < 0.5
        flip_v = rng.random() < 0.5
        brightness = rng.uniform(0.5, 1.5)

        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        shift = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        zoom = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])
        m = _center(rot @ shift @ zoom, h, w)

        def geo(x, order):
            x = _warp(x, m, order)
            if flip_h:
                x = x[:, ::-1]
            if flip_v:
                x = x[::-1, :]
            return np.ascontiguousarray(x)

        # same geometry + brightness for every modality -> they stay aligned
        imgs = [np.clip(geo(x, 1) * brightness, 0, 255) for x in imgs]
        mask = geo(mask, self.mask_order)
        if self.brightness_on_mask:
            mask = np.clip(mask * brightness, 0, 255)
        return imgs, mask

    def __getitem__(self, i):
        imgs = [_load(p[i], self.img_size) for p in self.img_paths]
        mask = _load(self.mask_paths[i], self.img_size)

        if self.augment:
            imgs, mask = self._augment(imgs, mask)

        imgs = [torch.from_numpy(np.ascontiguousarray(x / 255.0)).unsqueeze(0) for x in imgs]
        mask = torch.from_numpy(np.ascontiguousarray(mask / 255.0)).unsqueeze(0)

        if self.stack:
            return torch.cat(imgs, dim=0), mask       # (n_mod, H, W)
        return imgs, mask


def datagenerator(folders, img_size, batch_size, modalities=("flair", "t1", "t2"),
                  augment=False, shuffle=True, num_workers=12, mask_order=1,
                  brightness_on_mask=True, stack=False, fixed_shuffle_seed=None):
    # folders = [folder_flair, folder_t1, folder_t2, folder_pd, folder_mask]
    idx = resolve_modalities(modalities)
    img_folders = [folders[i] for i in idx]
    folder_mask = folders[-1]

    dataset = MultiModalDataset(img_folders, folder_mask, img_size=img_size,
                                augment=augment, mask_order=mask_order,
                                brightness_on_mask=brightness_on_mask, stack=stack,
                                fixed_shuffle_seed=fixed_shuffle_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=augment,
        persistent_workers=num_workers > 0,
    )