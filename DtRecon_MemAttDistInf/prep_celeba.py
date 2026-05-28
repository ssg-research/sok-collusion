"""Prepare a 64x64 CelebA cache for AIA/DIA + GIFD pipelines.

Writes under $PARTC_DATA (default ./data), matching where face_common.py reads:
    data/celeba/celeba_64.pt    tensor of shape (N, 3, 64, 64) uint8
    data/celeba/celeba_attr.npz arrays: Male, Smiling, Attractive, ...
"""

import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torchvision import transforms

_HERE = Path(__file__).resolve().parent
_DATA = Path(os.environ.get("PARTC_DATA", _HERE / "data"))
_OUT = _DATA / "celeba"
_OUT.mkdir(parents=True, exist_ok=True)

N_KEEP = 15000  # plenty for D_train + D_out + adv shadow + headroom
ATTRS = ["Male", "Smiling", "Attractive", "Heavy_Makeup", "Young", "Wearing_Lipstick"]

tx = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(64),
    transforms.PILToTensor(),  # uint8 (0..255), CHW
])


def main():
    ds = load_dataset("flwrlabs/celeba", split="train", streaming=True)
    imgs = torch.empty((N_KEEP, 3, 64, 64), dtype=torch.uint8)
    attrs = {a: np.empty(N_KEEP, dtype=np.int64) for a in ATTRS}
    n = 0
    for sample in ds:
        if n >= N_KEEP:
            break
        try:
            imgs[n] = tx(sample["image"].convert("RGB"))
        except Exception:
            continue
        for a in ATTRS:
            attrs[a][n] = int(sample[a])
        n += 1
        if n % 1000 == 0:
            print(f"  loaded {n}/{N_KEEP}")
    imgs = imgs[:n]
    for a in ATTRS:
        attrs[a] = attrs[a][:n]
    torch.save(imgs, _OUT / "celeba_64.pt")
    np.savez(_OUT / "celeba_attr.npz", **attrs)
    print(f"wrote {_OUT / 'celeba_64.pt'}  shape={tuple(imgs.shape)}")
    print(f"wrote {_OUT / 'celeba_attr.npz'}  fields={list(attrs.keys())}")


if __name__ == "__main__":
    main()
