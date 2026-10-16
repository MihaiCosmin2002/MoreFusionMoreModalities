"""
Stages 2-4: predict the test set with every trained model and export 3D NIfTI masks.

1) python slice_test.py                                  -> test_manifest.json

2) python predict_test.py predict --manifest test_manifest.json \
       --models_dir trained_models --out_dir test_predictions
   Every model predicts all three views (sagittal, coronal, axial), as it was trained on all three;
   the probabilities are averaged in 3D, thresholded, cleaned and saved.
   -> test_predictions/<model_tag>/<acq>_mask.nii.gz  and  test_predictions/<model_tag>.zip
   Finished models are skipped on a re-run.

3) python predict_test.py gather --out_dir test_predictions --manifest test_manifest.json
   -> all_predictions.zip (every model's masks) + test_predictions/index.csv, and a completeness check.
   Upload the per-model zips from step 2 for validation; all_predictions.zip is for archiving/transfer.
"""
import argparse
import gc
import json
import re
import zipfile
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from torch.amp import autocast
from tqdm import tqdm

import architecture

T = 256
VIEWS = {"sagittal": 0, "coronal": 1, "axial": 2}       # view -> axis that slicer.py slices along
SLICE_RE = re.compile(r"_(sagittal|coronal|axial)_slice(\d+)$")
# run folder looks like  <exp>_<fusion>[-nopre]_<mod1>-<mod2>...   (RUN_NAME in train.py)
RUN_RE = re.compile(
    r"(early|intermediate|multibranch)(?:-nopre)?_"
    r"((?:flair|t1|t2|pd)(?:-(?:flair|t1|t2|pd))*)(?=$|_)")


# ----------------------------------------------------------------------------
# slice <-> volume mapping (exact inverse of slicer.py)
# ----------------------------------------------------------------------------
def view_files(folder, acq, view):
    """{slice index: path} for one acquisition and one view (names: <acq>_<view>_slice<idx>.png)."""
    out = {}
    for p in Path(folder).glob(f"{acq}_{view}_slice*.png"):
        m = SLICE_RE.search(p.stem)
        if m:
            out[int(m.group(2))] = p
    return out


def unslice(P, shape, idxs, axis):
    """P: (n, 256, 256) probabilities aligned with idxs. slicer.py took vol[i] along `axis` and did
    cv2.resize(.., (256, 256), INTER_LINEAR) with no flips/transposes, so we resize back to the
    original slice shape and put it at index i."""
    h0, w0 = [s for a, s in enumerate(shape) if a != axis]
    vol = np.zeros(shape, np.float32)
    for k, i in enumerate(idxs):
        s = cv2.resize(np.ascontiguousarray(P[k], dtype=np.float32), (w0, h0),
                       interpolation=cv2.INTER_LINEAR)               # dsize = (width, height)
        sl = [slice(None)] * 3
        sl[axis] = i
        vol[tuple(sl)] = s
    return vol


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------
def build_model(ckpt, device):
    """Rebuild the UNet from the shapes stored in the state dict."""
    sd = torch.load(ckpt, map_location="cpu")
    if "encoders.0.0.conv1.weight" not in sd:
        raise RuntimeError("checkpoint is not from the current UNet (old key names)")
    w = sd["encoders.0.0.conv1.weight"]
    filters, in_ch = w.shape[0], w.shape[1]
    n_branches = len({k.split(".")[1] for k in sd if k.startswith("encoders.")})
    if any(k.startswith("fusers.") for k in sd):
        fusion = "multibranch" if any(".paths." in k for k in sd) else "intermediate"
    else:
        fusion = "early"
    n_mod = in_ch if fusion == "early" else n_branches
    row_bn = "encoders.0.0.bn2.beta" in sd
    model = architecture.UNet(filters, n_modalities=n_mod, fusion=fusion,
                              img_size=T, tf_row_bn=row_bn)
    model.load_state_dict(sd)
    return model.to(device).eval(), fusion, n_mod


def parse_modalities(rel_path):
    for part in reversed(rel_path.parts):
        m = RUN_RE.search(part)
        if m:
            return m.group(2).split("-")
    return None


@torch.no_grad()
def predict_probs(model, X, device, bs=16):
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(xb)
        out.append(torch.sigmoid(logits.float())[:, 0].cpu().numpy())
    return np.concatenate(out)                      # (n, 256, 256)


def remove_small(mask, min_size):
    """Drop 3D connected components smaller than min_size voxels (26-connectivity)."""
    if min_size <= 1 or not mask.any():
        return mask
    lab, n = ndi.label(mask, structure=np.ones((3, 3, 3)))
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.concatenate([[False], sizes >= min_size])
    return keep[lab]


# ----------------------------------------------------------------------------
# prediction
# ----------------------------------------------------------------------------
def predict(args):
    manifest = json.loads(Path(args.manifest).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir, out_root = Path(args.models_dir), Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(models_dir.rglob("BestModel.pth"))
    print(f"{len(ckpts)} models x {len(manifest)} volumes | views: {args.views}")

    for ck in ckpts:
        rel = ck.parent.relative_to(models_dir)
        tag = "__".join(rel.parts) or "model"
        zip_path = out_root / f"{tag}.zip"
        if zip_path.exists():
            print(f"[skip] {tag} (already done)")
            continue

        try:
            model, fusion, n_mod = build_model(ck, device)
        except Exception as e:
            print(f"[skip] {tag}: {e}")
            continue
        mods = parse_modalities(rel) or args.modalities
        if mods is None or len(mods) != n_mod:
            print(f"[skip] {tag}: model expects {n_mod} modalities but I got {mods}. "
                  f"Pass --modalities (in training order).")
            del model
            continue
        print(f"\n{tag} | fusion={fusion} | modalities={mods}")

        # every modality the model was trained on must exist for every test volume
        absent = sorted({(acq, m) for acq, info in manifest.items() for m in mods
                         if m not in info["nii"] or m not in info["png"]})
        if absent:
            print(f"[skip] {tag}: missing modalities, e.g. {absent[:3]}")
            del model
            continue

        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "model_info.json").write_text(json.dumps(
            dict(tag=tag, checkpoint=str(ck), fusion=fusion, modalities=mods, views=args.views), indent=2))

        for acq, info in tqdm(manifest.items(), desc=tag):
            # geometry comes from a modality THIS model was trained on (not necessarily FLAIR)
            ref = nib.load(info["nii"][mods[0]])
            shape = ref.shape[:3]
            for m in mods[1:]:
                other = nib.load(info["nii"][m])
                if other.shape[:3] != shape or not np.allclose(other.affine, ref.affine, atol=1e-3):
                    raise RuntimeError(f"{acq}: '{m}' and '{mods[0]}' are not in the same space")

            acc = np.zeros(shape, np.float32)
            for view in args.views:
                axis = VIEWS[view]
                per_mod = [view_files(info["png"][m], acq, view) for m in mods]
                idxs = sorted(per_mod[0])
                if not idxs or any(sorted(d) != idxs for d in per_mod):
                    raise RuntimeError(f"{acq} [{view}]: missing PNGs, or modality folders differ")
                if len(idxs) != shape[axis]:
                    tqdm.write(f"[warn] {acq} [{view}]: {len(idxs)} PNGs for {shape[axis]} slices "
                               f"(missing slices count as empty)")

                X = np.empty((len(idxs), len(mods), T, T), np.float32)
                for c, d in enumerate(per_mod):
                    for k, i in enumerate(idxs):
                        X[k, c] = np.asarray(Image.open(d[i]).convert("L"), dtype=np.float32) / 255.0

                acc += unslice(predict_probs(model, X, device), shape, idxs, axis)

            vol = (acc / len(args.views)) > args.threshold
            vol = remove_small(vol, args.min_size)

            img = nib.Nifti1Image(vol.astype(np.uint8), ref.affine, ref.header)
            img.set_data_dtype(np.uint8)
            nib.save(img, str(out_dir / args.name_pattern.format(acq=acq)))

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out_dir.glob("*.nii*")):
                z.write(f, arcname=f.name)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def gather(args):
    """Collect every model's predicted masks into ONE zip + index.csv, and check completeness."""
    import csv
    out_root = Path(args.out_dir)
    n_expected = len(json.loads(Path(args.manifest).read_text()))
    rows, incomplete = [], []

    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
        for d in sorted(p for p in out_root.iterdir() if p.is_dir()):
            info_p = d / "model_info.json"
            info = json.loads(info_p.read_text()) if info_p.exists() else {}
            files = sorted(d.glob("*.nii*"))
            for f in files:
                z.write(f, arcname=f"{d.name}/{f.name}")          # <model_tag>/<acq>_mask.nii.gz
            rows.append([d.name, info.get("fusion", "?"), "-".join(info.get("modalities", [])),
                         "-".join(info.get("views", [])), len(files), info.get("checkpoint", "")])
            if len(files) != n_expected:
                incomplete.append((d.name, len(files)))

    with open(out_root / "index.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model_tag", "fusion", "modalities", "views", "n_masks", "checkpoint"])
        w.writerows(rows)

    print(f"{len(rows)} models gathered into {args.zip}  (index: {out_root / 'index.csv'})")
    if incomplete:
        print(f"WARNING: expected {n_expected} masks per model, but:")
        for tag, n in incomplete:
            print(f"   {tag}: {n}")
    else:
        print(f"OK: every model has {n_expected} masks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict")
    p.add_argument("--manifest", default="test_manifest.json")
    p.add_argument("--models_dir", default="trained_models")
    p.add_argument("--out_dir", default="test_predictions")
    p.add_argument("--views", nargs="+", choices=list(VIEWS), default=list(VIEWS),
                   help="views to predict and average (default: all three, as in training)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min_size", type=int, default=3,
                   help="remove predicted components smaller than this many voxels (1 = off)")
    p.add_argument("--name_pattern", default="{acq}_mask.nii.gz",
                   help="output file name; check the challenge page for the required naming")
    p.add_argument("--modalities", nargs="+", default=None,
                   help="fallback if the modalities can't be parsed from the run folder name")

    g = sub.add_parser("gather")
    g.add_argument("--out_dir", default="test_predictions")
    g.add_argument("--manifest", default="test_manifest.json")
    g.add_argument("--zip", default="all_predictions.zip")

    a = ap.parse_args()
    {"predict": predict, "gather": gather}[a.cmd](a)