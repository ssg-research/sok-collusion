"""Shared utilities for the GIFD-based image-AIA / image-DIA pipelines.

Loads UTKFace and CelebA at 64x64 with a consistent (image, y, z) interface where
`y` is the target task and `z` is the sensitive attribute that AIA infers.

Datasets:
    UTKFace: y / z chosen from {age, race, sex} (see load_utkface). Default is
             y = race-binary (White vs non-White), z = sex - the pair the AIA
             pipeline ships with. Aligned faces at 48x48 in the CSV; resized to 64x64.
    CelebA:  y = Smiling (binary), z = Male (binary). Center-cropped & resized 64x64
             cache already prepared by prep_celeba.py.

Both expose .tensor (uint8, [N, 3, 64, 64]) and .y / .z (long arrays).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# Face data lives under ./data (override with $PARTC_DATA). The 64x64 CelebA
# tensor + attribute archive are produced once by prep_celeba.py; UTKFace is a
# single csv. See README C.2 / C.4 for the one-time preparation step.
_DATA = Path(os.environ.get("PARTC_DATA", Path(__file__).resolve().parent / "data"))
_UTKCSV = _DATA / "utkface" / "utkface.csv"
_CELEBA_T = _DATA / "celeba" / "celeba_64.pt"
_CELEBA_A = _DATA / "celeba" / "celeba_attr.npz"


@dataclass
class FaceData:
    name: str
    images: torch.Tensor  # uint8 [N, 3, 64, 64]
    y: np.ndarray  # binary task label
    z: np.ndarray  # binary sensitive attribute


def _resize_64(t: torch.Tensor) -> torch.Tensor:
    """Resize a uint8 image tensor [N, 3, H, W] to 64x64 with bilinear interp."""
    return F.interpolate(t.float(), size=64, mode="bilinear", align_corners=False).round().clamp(0, 255).to(torch.uint8)


UTK_ATTRS = ("age", "race", "sex")


def _utk_attr(df: pd.DataFrame, name: str) -> np.ndarray:
    """Binarize one UTKFace label column. age → ≥ median; race → non-White; sex → as-is."""
    if name == "age":
        age = df["age"].to_numpy()
        return (age >= np.median(age)).astype(np.int64)
    if name == "race":
        # 0 = White in UTKFace's ethnicity convention → binary non-White flag.
        return (df["ethnicity"].to_numpy() != 0).astype(np.int64)
    if name == "sex":
        return df["gender"].to_numpy().astype(np.int64)
    raise ValueError(f"unknown UTKFace attribute {name!r}; expected one of {UTK_ATTRS}")


def load_utkface(y_attr: str = "race", z_attr: str = "sex") -> FaceData:
    """UTKFace at 64x64 with a chosen (target task, sensitive attribute) pair.

    Both `y_attr` and `z_attr` are drawn from {age, race, sex}. The default
    y = race-binary, z = sex is the pair the AIA/DIA pipelines ship with: race
    classification has well-documented gender bias on UTKFace, so member features
    encode z. Other pairs are screened by screen_utkface_aia.py.
    """
    if y_attr == z_attr:
        raise ValueError(f"y_attr and z_attr must differ (both {y_attr!r})")
    df = pd.read_csv(_UTKCSV)
    pixels = np.stack([np.fromstring(p, sep=" ", dtype=np.uint8) for p in df["pixels"].to_numpy()])
    pixels = pixels.reshape(-1, 48, 48)
    pixels_rgb = np.repeat(pixels[:, None, :, :], 3, axis=1)
    imgs48 = torch.from_numpy(np.ascontiguousarray(pixels_rgb))
    imgs = _resize_64(imgs48)
    return FaceData(name="utkface", images=imgs, y=_utk_attr(df, y_attr), z=_utk_attr(df, z_attr))


def load_celeba() -> FaceData:
    imgs = torch.load(_CELEBA_T, map_location="cpu", weights_only=False)
    attrs = np.load(_CELEBA_A)
    smiling = ((attrs["Smiling"] + 1) // 2).astype(np.int64) if attrs["Smiling"].min() < 0 else attrs["Smiling"].astype(np.int64)
    male = ((attrs["Male"] + 1) // 2).astype(np.int64) if attrs["Male"].min() < 0 else attrs["Male"].astype(np.int64)
    return FaceData(name="celeba", images=imgs, y=smiling, z=male)


def split_indices(n_total: int, ns: dict[str, int], seed: int) -> dict[str, np.ndarray]:
    """Deterministic disjoint splits. Keys: d_train (members), d_out (probe aux),
    target_test_members (held-out members), target_test_non (held-out non-members),
    shadow_pool (DIA shadow), recon_pool (records to reconstruct)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    out: dict[str, np.ndarray] = {}
    start = 0
    for k, count in ns.items():
        if start + count > n_total:
            raise ValueError(f"split overflow at {k}: need {count}, have {n_total - start}")
        out[k] = perm[start:start + count]
        start += count
    return out


# Reusable split sizes for our experiments.
# Key invariant: members of the target model = d_train_full = recon_pool ∪ test_members.
# Both subsets were trained on. The adversary reconstructs records from recon_pool
# (recovering imperfect copies of those records' gradients) and the AIA / DIA
# evaluation queries the model on test_members (held out from the adversary).
UTK_SPLITS = {
    "recon_pool": 500,          # half of d_train_full; adversary tries to reconstruct these
    "test_members": 500,        # held-out half of d_train_full; used for evaluation
    "d_out": 1500,              # auxiliary non-members for probe training
    "non_member_test": 500,     # additional non-members for evaluation
    "shadow_pool": 3000,        # DIA shadow models draw from here
}

CELEBA_SPLITS = {
    "recon_pool": 500,
    "test_members": 500,
    "d_out": 1500,
    "non_member_test": 500,
    "shadow_pool": 6000,
}


def get_splits(face: FaceData, seed: int) -> dict[str, np.ndarray]:
    splits_def = UTK_SPLITS if face.name == "utkface" else CELEBA_SPLITS
    s = split_indices(len(face.images), splits_def, seed=seed)
    # The target was trained on the union of recon_pool + test_members (= d_train_full).
    s["d_train_full"] = np.concatenate([s["recon_pool"], s["test_members"]])
    return s


# Standard normalization: scale uint8 [0,255] -> [-1, 1] (matches StyleGAN2 output range)
def to_model_input(img_u8: torch.Tensor) -> torch.Tensor:
    return (img_u8.float() / 255.0 - 0.5) * 2


def from_model_input(img_pm1: torch.Tensor) -> torch.Tensor:
    return ((img_pm1.detach().cpu() + 1) / 2 * 255.0).round().clamp(0, 255).to(torch.uint8)
