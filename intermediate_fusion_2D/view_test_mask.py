"""
Overlay predicted masks (red) on an MRI volume and save PNG montages.

One volume:
  python view_masks.py --image testdata/test01/preprocessed/test01_01_flair_pp.nii \
                       --mask  test_predictions/<model_tag>/test01_01_mask.nii \
                       --out   test01_01.png

Every volume of one model (uses test_manifest.json to find the images):
  python view_masks.py --manifest test_manifest.json --modality flair \
                       --masks_dir test_predictions/<model_tag> --out_dir previews/<model_tag>

--modality is only the background image to draw on (flair, t1, t2 or pd), it does not have to be
one the model was trained on. The mask has the same geometry as the image, so they line up.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

AXES = {"sagittal": 0, "coronal": 1, "axial": 2}


def pick_slices(mask, axis, n):
    """n slices spread over the extent of the lesions (or evenly over the brain if there are none)."""
    other = tuple(a for a in range(3) if a != axis)
    nz = np.flatnonzero(mask.sum(axis=other))
    if len(nz) == 0:
        size = mask.shape[axis]
        return np.linspace(size * 0.25, size * 0.75, n).astype(int).tolist()
    if len(nz) <= n:
        return nz.tolist()
    return nz[np.linspace(0, len(nz) - 1, n).astype(int)].tolist()


def get_slice(vol, axis, i):
    sl = [slice(None)] * 3
    sl[axis] = i
    return vol[tuple(sl)]


def render(image_path, mask_path, out_png, axis_name="axial", n=12):
    img_nii, msk_nii = nib.load(str(image_path)), nib.load(str(mask_path))
    img = np.asarray(img_nii.get_fdata(), dtype=np.float32)
    mask = np.asarray(msk_nii.get_fdata()) > 0
    if img.shape[:3] != mask.shape[:3]:
        raise ValueError(f"shape mismatch: image {img.shape[:3]} vs mask {mask.shape[:3]}")
    if not np.allclose(img_nii.affine, msk_nii.affine, atol=1e-3):
        print(f"[warn] {Path(mask_path).name}: image and mask affines differ")

    axis = AXES[axis_name]
    voxel_ml = float(np.prod(msk_nii.header.get_zooms()[:3])) / 1000.0
    _, n_les = ndi.label(mask, structure=np.ones((3, 3, 3)))
    stats = f"{n_les} lesions | {mask.sum() * voxel_ml:.2f} ml | {int(mask.sum())} voxels"

    brain = img[img > 0]
    lo, hi = (np.percentile(brain, 1), np.percentile(brain, 99.5)) if brain.size else (0, 1)

    idxs = pick_slices(mask, axis, n)
    cols = min(len(idxs), 6)
    rows = int(np.ceil(len(idxs) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows), squeeze=False)
    for a in axs.ravel():
        a.axis("off")
    for a, i in zip(axs.ravel(), idxs):
        s, m = get_slice(img, axis, i), get_slice(mask, axis, i)
        a.imshow(s.T, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        if m.any():
            a.imshow(np.ma.masked_where(~m, m).T, cmap=ListedColormap(["red"]), alpha=0.55, origin="lower", vmin=0, vmax=1)
        a.set_title(f"{axis_name} {i}", fontsize=8)
    fig.suptitle(f"{Path(mask_path).name.split('_mask')[0]}  |  {stats}", fontsize=11)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--mask")
    ap.add_argument("--out", default="mask_preview.png")
    ap.add_argument("--manifest")
    ap.add_argument("--modality", default="flair", choices=["flair", "t1", "t2", "pd"])
    ap.add_argument("--masks_dir")
    ap.add_argument("--out_dir", default="previews")
    ap.add_argument("--name_pattern", default="{acq}_mask.nii")
    ap.add_argument("--axis", default="axial", choices=list(AXES))
    ap.add_argument("--n", type=int, default=12, help="slices per montage")
    a = ap.parse_args()

    if a.manifest:
        manifest = json.loads(Path(a.manifest).read_text())
        for acq, info in manifest.items():
            mp = Path(a.masks_dir) / a.name_pattern.format(acq=acq)
            if not mp.exists():
                print(f"[skip] {acq}: {mp} not found")
                continue
            out = Path(a.out_dir) / f"{acq}.png"
            print(f"{acq}: {render(info['nii'][a.modality], mp, out, a.axis, a.n)}  ->  {out}")
    else:
        if not (a.image and a.mask):
            raise SystemExit("Give --image and --mask, or --manifest and --masks_dir")
        print(render(a.image, a.mask, a.out, a.axis, a.n), "->", a.out)