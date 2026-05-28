"""Train and load ResNet-18 target classifiers for UTKFace and CelebA at 64x64.

The target is overfit on a small N_TRAIN to amplify the member/non-member
overconfidence gap that downstream AIA/DIA attacks exploit. Architecture is a
64x64-adapted ResNet-18 (3x3 stride-1 stem, no initial maxpool) matching GIFD's
expected FL model family.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from face_common import FaceData, get_splits, load_celeba, load_utkface, to_model_input


TARGET_DIR = _HERE / "models" / "face_targets"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 2


def build_resnet18_64(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet-18 adapted for 64x64 input."""
    m = resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def target_path(dataset: str, seed: int, task: str | None = None) -> Path:
    """Checkpoint path for a target model. `task` tags UTKFace targets by their
    target attribute (age/race/sex) so the (y,z)-pair screen doesn't collide;
    `task=None` keeps the original untagged path."""
    suffix = f"_{task}" if task else ""
    return TARGET_DIR / f"{dataset}{suffix}_resnet18_64_seed{seed}.pt"


def train_target(face: FaceData, seed: int, task: str | None = None,
                  epochs: int = 60, lr: float = 0.01,
                  n_train: int | None = None, weight_decay: float = 5e-4,
                  n_recon_train: int | None = None,
                  n_test_train: int | None = None) -> Path:
    """Train a target ResNet-18.

    Three composition modes:
      (default)                       : train on the full d_train_full (1000 records).
      n_train=K                       : train on the first K of d_train_full.
      n_recon_train=A, n_test_train=B : train on recon_pool[:A] + test_members[:B]
                                        (use this to keep ALL gifd reconstructions in
                                        training while pushing the target to drastic
                                        overfit by capping the total train size).

    Saves trimmed test_members so downstream AIA evaluation only sees real members.
    """
    p = target_path(face.name, seed, task)
    if p.exists():
        return p
    torch.manual_seed(seed)
    np.random.seed(seed)
    splits = get_splits(face, seed=seed)
    full = np.asarray(splits["d_train_full"])
    recon = np.asarray(splits["recon_pool"])
    test = np.asarray(splits["test_members"])
    if n_recon_train is not None or n_test_train is not None:
        a = n_recon_train if n_recon_train is not None else len(recon)
        b = n_test_train if n_test_train is not None else len(test)
        train_idx = np.concatenate([recon[:a], test[:b]])
        splits = {**splits, "d_train_full": train_idx, "test_members": test[:b]}
    elif n_train is not None:
        assert n_train >= len(recon), f"n_train={n_train} must include all of recon_pool({len(recon)})"
        train_idx = full[:n_train]
        train_set = set(train_idx.tolist())
        splits = {**splits,
                  "d_train_full": train_idx,
                  "test_members": np.asarray([i for i in splits["test_members"] if i in train_set])}
    else:
        train_idx = full
    x_train = to_model_input(face.images[train_idx]).cuda()
    y_train = torch.from_numpy(face.y[train_idx]).long().cuda()

    model = build_resnet18_64().cuda()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)

    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        if (ep + 1) % 20 == 0:
            with torch.no_grad():
                model.eval()
                train_acc = (model(x_train).argmax(1) == y_train).float().mean().item()
                model.train()
            print(f"  ep {ep + 1}/{epochs} train_acc={train_acc:.3f}")

    model.eval()
    test_idx = splits["non_member_test"]
    with torch.no_grad():
        x_test = to_model_input(face.images[test_idx]).cuda()
        y_test = torch.from_numpy(face.y[test_idx]).long().cuda()
        test_acc = (model(x_test).argmax(1) == y_test).float().mean().item()
        x_mem = to_model_input(face.images[train_idx]).cuda()
        y_mem = torch.from_numpy(face.y[train_idx]).long().cuda()
        train_acc = (model(x_mem).argmax(1) == y_mem).float().mean().item()
    print(f"[{face.name} seed{seed} n_train={len(train_idx)}] member_acc={train_acc:.3f} "
          f"non_member_acc={test_acc:.3f} overfit_gap={train_acc - test_acc:.3f}")

    torch.save({"state_dict": model.state_dict(),
                "splits": {k: np.asarray(v).tolist() for k, v in splits.items()},
                "seed": seed, "n_train": len(train_idx),
                "train_acc": train_acc, "test_acc": test_acc}, p)
    return p


def load_target(dataset: str, seed: int, task: str | None = None) -> tuple[nn.Module, dict[str, np.ndarray]]:
    p = target_path(dataset, seed, task)
    ckpt = torch.load(p, map_location="cuda", weights_only=False)
    model = build_resnet18_64().cuda()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    splits = {k: np.asarray(v) for k, v in ckpt["splits"].items()}
    return model, splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["utkface", "celeba"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--y_attr", default=None,
                        help="UTKFace target attribute (age/race/sex); also tags the checkpoint.")
    parser.add_argument("--z_attr", default=None, help="UTKFace sensitive attribute (age/race/sex).")
    parser.add_argument("--task", default=None,
                        help="Checkpoint tag override. The AIA headline targets use 'of600' "
                             "(CelebA) / 'sex_of600' (UTKFace). Default: y_attr tag (UTKFace) / none (CelebA).")
    parser.add_argument("--n_train", type=int, default=None,
                        help="Train on the first n_train records of D_train. The overfit headline uses 600.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="SGD weight decay. The overfit headline uses 0.")
    args = parser.parse_args()

    if args.dataset == "utkface":
        y_attr = args.y_attr or "race"
        z_attr = args.z_attr or ("sex" if y_attr != "sex" else "race")
        face = load_utkface(y_attr=y_attr, z_attr=z_attr)
        default_task = args.y_attr  # tag only when explicitly requested
    else:
        face = load_celeba()
        default_task = None
    task = args.task if args.task is not None else default_task
    p = train_target(face, args.seed, task=task, epochs=args.epochs,
                     n_train=args.n_train, weight_decay=args.weight_decay)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
