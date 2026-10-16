"""
Stage 1: slice the ISBI test volumes with YOUR slicer.py (same per-volume normalisation,
same three views, same 256x256 resize) and write test_manifest.json for predict_test.py.

The test set has no masks, so no mask file is read. An all-zero label volume is passed to the
slicer only because its signature needs one (blank PNGs end up in the 'mask' folder; ignore them).
train=False keeps EVERY slice of every view.

Run from the folder that holds slicer.py / folder_structure.py:   python slice_test.py
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np

import folder_structure
import slicer

IMG_SIZE = (256, 256)
access_rights = 0o777

REPO_ROOT = Path(__file__).resolve().parent.parent
base_data = REPO_ROOT / "testdata"      # testdata/test01/preprocessed/test01_01_flair_pp.nii ...

# order = order of the folders returned by create_paths: [flair, t1, t2, pd, mask]
NII_SUFFIX = {"flair": "flair_pp", "t1": "mprage_pp", "t2": "t2_pp", "pd": "pd_pp"}


def get_dest(subj):
    try:
        return folder_structure.create_paths(str(base_data), "Test", subj, access_rights, False)
    except FileExistsError:                       # re-run: the folders already exist
        return folder_structure.get_paths(str(base_data), "Test", subj, False)


subjects = sorted(p.name for p in base_data.iterdir()
                  if p.is_dir() and (p / "preprocessed").is_dir())
print(f"Subjects found: {subjects}")

manifest = {}
for subj in subjects:
    pre = base_data / subj / "preprocessed"
    acqs = sorted({f.name.split("_flair_pp")[0] for f in pre.glob("*_flair_pp.nii*")})
    if not acqs:
        continue
    dest = get_dest(subj)

    for acq in acqs:                              # e.g. test01_01 (timepoints are discovered, not assumed)
        files = {}
        for key, suf in NII_SUFFIX.items():
            found = sorted(pre.glob(f"{acq}_{suf}.nii*"))
            if not found:
                raise SystemExit(f"{acq}: file for modality '{key}' ({acq}_{suf}.nii*) not found in {pre}")
            files[key] = found[0]

        shape = nib.load(str(files["flair"])).shape[:3]
        expected = sum(shape)                     # slicer saves shape[0] + shape[1] + shape[2] slices per modality
        have = len(list(Path(dest[0]).glob(f"{acq}_*_slice*.png")))

        if have == expected:
            print(f"{acq}: already sliced ({have} PNGs per modality), skipping")
        else:
            vols = {k: slicer.read_img_vol(str(v), True) for k, v in files.items()}   # min-max normalised
            blank = np.zeros(vols["flair"].shape, np.float32)                          # stands in for the mask
            slicer.slice_and_save_vol_img(IMG_SIZE, blank, vols["flair"], vols["t1"], vols["t2"], vols["pd"],
                                          f"{acq}_", dest[0], dest[1], dest[2], dest[3], dest[4], False)
            have = len(list(Path(dest[0]).glob(f"{acq}_*_slice*.png")))
            print(f"\n{acq}: volume {shape} -> {have} PNGs per modality (expected {expected})")
            del vols

        manifest[acq] = {"nii": {k: str(v) for k, v in files.items()},
                         "png": {k: str(dest[i]) for i, k in enumerate(NII_SUFFIX)}}

Path("test_manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"\nWrote test_manifest.json with {len(manifest)} volumes")