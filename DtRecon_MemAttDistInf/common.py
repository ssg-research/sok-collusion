"""
Shared utilities for datarecon -> {MIA, AIA, DIA} overlap sweep experiments.

Key public API:
    build_hybrid_overlap_image() - legacy Fredrikson + oracle padding (kept for back-compat)
    run_fredrikson_recon()       - run (or load cached) FredriksonCCS2015 reconstructions
    save_result_row()            - append one CSV row to a results file
    set_seeds()                  - global deterministic seeding

The "oracle" recon source samples p% of the adversary's auxiliary-data input directly
from D_train, simulating the upper bound of a perfect data reconstruction attack (a
validation source). The "gifd" recon source instead loads cached records recovered by
gradient inversion (Geiping NeurIPS 2020 / GIFD ICCV 2023), the realistic attack.
"""

import csv
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset, Subset, TensorDataset

# amuletml is installed via `uv sync` (see ../pyproject.toml); import directly.
from amulet.data_reconstruction.attacks import FredriksonCCS2015

OVERLAP_LEVELS = [0.0, 0.25, 0.50, 0.75]
ALPHA2_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]


# ── Seeding ─────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_member_indices(full_train_size: int, seed: int, pkeep: float = 0.5) -> dict:
    """Deterministic member-set partition shared by the MIA pipeline
    (mia_overlap.py) and the realistic-source reconstruction pool
    (cifar_recon.py). Uses an isolated RandomState so the result depends
    only on (full_train_size, seed, pkeep) - never on global RNG state.

    Returns:
        d_train_idx:   original-dataset indices in target's candidate pool (first half of shuffle).
        d_out_idx:     original-dataset indices in adversary's "out" pool (second half).
        keep_local:    sorted positions within d_train that the target actually trained on.
        keep_full_idx: original-dataset indices of actual target members (= d_train_idx[keep_local]).
    """
    rng = np.random.RandomState(seed)
    all_idx = np.arange(full_train_size)
    rng.shuffle(all_idx)
    target_size = full_train_size // 2
    d_train_idx = all_idx[:target_size]
    d_out_idx = all_idx[target_size:]
    keep_local = np.sort(rng.choice(target_size, size=int(pkeep * target_size), replace=False))
    return {
        "d_train_idx": d_train_idx,
        "d_out_idx": d_out_idx,
        "keep_local": keep_local,
        "keep_full_idx": d_train_idx[keep_local],
    }


# ── Fredrikson reconstruction cache ─────────────────────────────────────────

def _recon_cache_path(cache_dir: Path, model_path: Path, num_classes: int) -> Path:
    key = hashlib.md5(f"{model_path}_{num_classes}".encode()).hexdigest()[:12]
    return cache_dir / f"fredrikson_{key}_nc{num_classes}.pkl"


def run_fredrikson_recon(
    target_model: nn.Module,
    input_size: tuple,
    num_classes: int,
    device: str,
    cache_dir: Path,
    model_path: Path,
    alpha: int = 3000,
) -> list[torch.Tensor]:
    """
    Run FredriksonCCS2015 or load from cache.

    Returns a list of `num_classes` tensors (class prototypes), each shaped `input_size[1:]`
    (without the leading batch dimension). `input_size` must include a batch dim of 1,
    e.g. `(1, 3, 32, 32)` for CIFAR-10 or `(1, num_features)` for tabular.

    Results are cached keyed on model file path + num_classes so the same target reuses them
    across all overlap levels.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _recon_cache_path(cache_dir, model_path, num_classes)

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # FredriksonCCS2015 expects input_size to include the batch dim
    attack = FredriksonCCS2015(
        target_model=target_model,
        input_size=input_size,
        output_size=num_classes,
        device=device,
        alpha=alpha,
    )
    reconstructed = attack.attack()  # list of num_classes tensors, shape = input_size (with batch)

    # Strip leading batch dimension so each prototype is shaped input_size[1:]
    reconstructed_cpu = [t.detach().cpu().squeeze(0) for t in reconstructed]
    with open(cache_path, "wb") as f:
        pickle.dump(reconstructed_cpu, f)

    return reconstructed_cpu


# ── Legacy Fredrikson + oracle-padding pools (kept for back-compat) ─────────

class _LabeledTensorDataset(Dataset):
    """Wraps (x_tensor, y_tensor) pairs, supporting a mix of dtypes."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx].item()


def _augment_cifar(x: torch.Tensor, rng: np.random.RandomState) -> torch.Tensor:
    """Standard CIFAR shadow-training augmentation on a [0,1] CHW tensor:
    reflect-pad-4 random crop + random hflip + small Gaussian pixel noise.
    Used to expand a small recon set into a diverse member sample."""
    # reflect pad 4, random 32-crop
    xp = F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode="reflect")[0]
    top = int(rng.randint(0, 9))
    left = int(rng.randint(0, 9))
    x = xp[:, top:top + x.shape[1], left:left + x.shape[2]]
    if rng.rand() < 0.5:
        x = torch.flip(x, dims=[2])
    x = x + torch.from_numpy(rng.normal(0, 0.02, size=tuple(x.shape)).astype("float32"))
    return x.clamp(0, 1)


def build_gifd_pool_image(
    recon_dir: Path,
    d_out: Dataset,
    p: float,
    n_a: int,
    seed: int,
    recon_budget: int | None = None,
    reserved_out_idx: np.ndarray | None = None,
    augment: bool = False,
    pool_mode: str = "budget",
) -> tuple[Dataset, np.ndarray, dict]:
    """Realistic-source adversary auxiliary pool for image MIA / DIA.

    `pool_mode` sets how the overlap p maps to the number of injected members:
      "budget"   (default; reproduces the paper's reported MIA numbers): the
                 adversary injects the p-fraction of its fixed reconstruction
                 budget it has recovered, n_in = min(p · recon_budget, |pool|),
                 mixed into an n_a-sized pool of non-members. Recons are a small,
                 distinct minority of the shadow pool.
      "fraction" (follow-up diversity study): the pool's member fraction equals
                 the overlap, n_in = int(p · n_a), filled from distinct recons
                 (replicated, optionally augmented, if the budget is too small).
        n_out = n_a - n_in    - distribution-matched non-members from `d_out`

    The n_in member slots are filled with the adversary's distinct Geiping
    reconstructions. When the adversary holds enough distinct recons
    (n_distinct ≥ n_in) they are sampled WITHOUT replacement and used RAW - the
    faithful realistic analogue of the oracle pool (distinct member records, no
    augmentation), so pool members and pool non-members share the same raw image
    statistics as the challenge set (no augmentation asymmetry to bias shadows).
    Only when n_distinct < n_in (recon budget too small for the requested member
    fraction) do we fall back to with-replacement sampling + augmentation, and the
    composition_log flags it via `replicated`.

    Recon .pt files are stored in [-1,1] (Geiping clamp); converted to [0,1] to
    match the CIFAR ToTensor pipeline LiRA's shadow training expects.
    """
    rng = np.random.RandomState(seed)
    files = sorted(Path(recon_dir).glob("rec_*.pt"))
    assert len(files), f"no rec_*.pt files in {recon_dir}"
    meta = json.loads((Path(recon_dir) / "meta.json").read_text())["records"]
    pool_size = len(files)
    if recon_budget is None:
        recon_budget = pool_size
    n_distinct = min(recon_budget, pool_size)  # distinct recons the adversary has

    if pool_mode == "budget":
        n_in = min(int(p * recon_budget), pool_size)
    elif pool_mode == "fraction":
        n_in = int(p * n_a)
    else:
        raise ValueError(f"pool_mode must be 'budget' or 'fraction', got {pool_mode!r}")
    n_out = max(n_a - n_in, 0)

    parts: list[Dataset] = []
    pool_in_indices = np.arange(n_in, dtype=int)
    replicated = n_in > n_distinct

    if n_in > 0:
        # Preload the distinct recons once (convert [-1,1] -> [0,1]).
        base_imgs, base_labels = [], []
        for i in range(n_distinct):
            t = torch.load(files[i], map_location="cpu", weights_only=False)
            base_imgs.append(((t[0] + 1) / 2).clamp(0, 1))
            base_labels.append(int(meta[i]["y"]))
        if not replicated:
            # Enough distinct recons: sample WITHOUT replacement, use RAW (oracle-faithful).
            pick = rng.choice(n_distinct, size=n_in, replace=False)
            imgs = [base_imgs[d].clone() for d in pick]
            labels = [base_labels[d] for d in pick]
        else:
            # Recon budget too small: replicate with augmentation (degraded fallback).
            draws = rng.randint(0, n_distinct, size=n_in)
            imgs = [(_augment_cifar(base_imgs[d], rng) if augment else base_imgs[d].clone())
                    for d in draws]
            labels = [base_labels[d] for d in draws]
        x_t = torch.stack(imgs).float()
        y_t = torch.tensor(labels, dtype=torch.long)
        parts.append(_LabeledTensorDataset(x_t, y_t))

    if n_out > 0:
        out_size = len(d_out)
        if reserved_out_idx is not None and len(reserved_out_idx) > 0:
            available = np.setdiff1d(np.arange(out_size), reserved_out_idx)
            out_idx = rng.choice(available, size=min(n_out, len(available)), replace=False)
        else:
            out_idx = rng.choice(out_size, size=min(n_out, out_size), replace=False)
        parts.append(Subset(d_out, out_idx.tolist()))

    pool = ConcatDataset(parts) if len(parts) > 1 else parts[0]
    composition_log = {
        "p": p, "n_a": n_a, "n_in": n_in, "n_out": n_out, "seed": seed,
        "recon_budget": recon_budget, "n_distinct": n_distinct, "pool_mode": pool_mode,
        "n_recon": n_in, "n_oracle": 0, "replicated": replicated,
        "augment": augment if replicated else False,
    }
    return pool, pool_in_indices, composition_log


def build_hybrid_overlap_image(
    target_model: nn.Module,
    d_train: Dataset,
    d_out: Dataset,
    p: float,
    n_a: int,
    seed: int,
    num_classes: int,
    input_size: tuple,
    device: str,
    cache_dir: Path,
    model_path: Path,
    reserved_train_idx: np.ndarray | None = None,
    reserved_out_idx: np.ndarray | None = None,
) -> tuple[Dataset, np.ndarray, dict]:
    """
    Build adversary auxiliary pool for image tasks (MIA / DIA).

    Pool of size n_a:
        n_recon  = min(p*n_a, num_classes)  - Fredrikson reconstructed class prototypes
        n_oracle = p*n_a - n_recon          - oracle samples from d_train (proxy padding)
        n_out    = (1-p)*n_a               - disjoint samples from d_out

    `reserved_train_idx` / `reserved_out_idx` (optional): indices within d_train / d_out
    that MUST NOT be sampled into the pool. Used by MIA to reserve records for the
    fixed challenge set so pool records and challenge records are physically disjoint
    (otherwise the same underlying record appears at two positions with independent
    shadow in/out labels, corrupting LiRA's mean_in/mean_out estimates).

    Returns (pool, pool_in_indices, composition_log) where pool_in_indices are the
    indices *within pool* of records that come from D_train (recon + oracle). LiRA's
    `in_data` should be set to pool_in_indices to correctly label pool members.
    """
    rng = np.random.RandomState(seed)

    n_in = int(p * n_a)
    n_recon = min(n_in, num_classes)
    n_oracle = n_in - n_recon
    n_out = n_a - n_in

    parts: list[Dataset] = []
    pool_in_indices = np.arange(n_in, dtype=int)
    oracle_idx = np.array([], dtype=int)

    if n_recon > 0:
        recon_tensors = run_fredrikson_recon(
            target_model, input_size, num_classes, device,
            cache_dir, model_path,
        )
        xs = torch.stack([recon_tensors[c] for c in range(n_recon)])
        ys = torch.arange(n_recon, dtype=torch.long)
        parts.append(_LabeledTensorDataset(xs, ys))

    if n_oracle > 0:
        train_size = len(d_train)  # type: ignore[arg-type]
        if reserved_train_idx is not None and len(reserved_train_idx) > 0:
            available = np.setdiff1d(np.arange(train_size), reserved_train_idx)
            if len(available) < n_oracle:
                raise ValueError(
                    f"Not enough train records after reservation: "
                    f"need {n_oracle}, have {len(available)} "
                    f"(train_size={train_size}, reserved={len(reserved_train_idx)})"
                )
            oracle_idx = rng.choice(available, size=n_oracle, replace=False)
        else:
            oracle_idx = rng.choice(train_size, size=n_oracle, replace=False)
        parts.append(Subset(d_train, oracle_idx.tolist()))

    out_selected_idx = np.array([], dtype=int)
    if n_out > 0:
        out_size = len(d_out)  # type: ignore[arg-type]
        if reserved_out_idx is not None and len(reserved_out_idx) > 0:
            available = np.setdiff1d(np.arange(out_size), reserved_out_idx)
            if len(available) < n_out:
                raise ValueError(
                    f"Not enough out records after reservation: "
                    f"need {n_out}, have {len(available)}"
                )
            out_selected_idx = rng.choice(available, size=n_out, replace=False)
        else:
            out_selected_idx = rng.choice(out_size, size=min(n_out, out_size), replace=False)
        parts.append(Subset(d_out, out_selected_idx.tolist()))

    pool = ConcatDataset(parts) if len(parts) > 1 else parts[0]

    composition_log = {
        "p": p,
        "n_a": n_a,
        "n_recon": n_recon,
        "n_oracle": n_oracle,
        "n_out": n_out,
        "seed": seed,
        "oracle_idx": oracle_idx,
        "out_selected_idx": out_selected_idx,
    }
    return pool, pool_in_indices, composition_log


# ── CSV result persistence ───────────────────────────────────────────────────

RESULT_COLUMNS = [
    "table", "dataset", "filter_prop", "alpha2", "overlap_p",
    "seed", "metric", "value",
    "n_recon", "n_oracle", "n_out", "extra",
    "task", "alpha_off", "recon_source", "target_seed",
]


def save_result_row(results_csv: Path, row: dict[str, Any]) -> None:
    """Append a result row to a CSV file; writes header if the file is new."""
    write_header = not results_csv.exists()
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        # Fill missing fields with empty string
        full_row = {k: row.get(k, "") for k in RESULT_COLUMNS}
        writer.writerow(full_row)


def load_results(results_csv: Path) -> list[dict]:
    if not results_csv.exists():
        return []
    with open(results_csv, newline="") as f:
        return list(csv.DictReader(f))


def result_exists(results_csv: Path, match: dict) -> bool:
    """Return True if a row matching all keys in `match` already exists."""
    for row in load_results(results_csv):
        if all(str(row.get(k)) == str(v) for k, v in match.items()):
            return True
    return False
