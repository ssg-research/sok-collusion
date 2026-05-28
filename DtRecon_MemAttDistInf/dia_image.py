"""T-DIA on image targets (UTKFace + CelebA) with data-reconstruction collusion.

Implements the black-box distribution-inference attack of Suri et al. (Dissecting
Distribution Inference, SaTML 2023); the alpha_1 vs alpha_2 ratio choices follow
Suri & Evans (PETS 2022). The adversary
trains shadow models on data with two distinct sensitive-attribute ratios
(\\alpha_1 vs \\alpha_2) and trains a meta-classifier that distinguishes the
ratios from model outputs on a fixed query set. Collusion injects reconstructed
members into the adversary's \\alpha_1-side shadow training data (the side that
matches the victim's distribution).

Design notes (post-diagnosis of the flat-trend issue):
  * Injection is *no-replacement* from a fixed `recon_budget` (= the number of
    records the adversary targeted for recovery). p is the fraction of that
    budget actually injected, so p ∈ {0,.25,.5,.75} → {0,25,50,75} unique
    records - no duplication. recon_budget is identical for oracle and gifd so
    the two sources' p-axes are directly comparable.
  * `n_train_per_shadow` is small (200) so the injected records are a meaningful
    fraction of each shadow's training set (mirrors the AIA n_dout fix).
  * Each shadow draws its own data + injected subset from an *independent*
    per-shadow RNG, so the collusion treatment carries genuine within-group
    variance the meta-classifier can learn distributionally.
  * Shadow feature = per-z-subgroup sorted-loss vector ‖ 4 scalar summaries
    (mean loss / mean acc per z group). The query set is z-balanced so the
    Male-vs-non-Male shift that collusion induces is preserved (plain global
    sorting destroys it).
  * cuDNN determinism on; shadow training is a pure function of (seed, k).

Two binary tasks per dataset:
    "lo": alpha_ref=0.5 vs alpha_2=0.1
    "hi": alpha_ref=0.5 vs alpha_2=0.9

Usage:
    python dia_image.py --dataset celeba --task lo --target_seed 0 --seed 0 \
        --recon_source gifd --recon_dir recon_pools/celeba_seed0_geiping --overlap 0.5
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from common import OVERLAP_LEVELS, save_result_row, set_seeds
from face_common import get_splits, load_celeba, load_utkface, to_model_input
from face_targets import build_resnet18_64

RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ALPHA1_LOOKUP = {"lo": 0.1, "hi": 0.9}  # the alpha_2 (off-balance) side
ALPHA_REF = 0.5                          # the reference distribution = victim's

# Defaults chosen so the collusion treatment is a meaningful, un-duplicated
# fraction of each shadow's training data and the meta-acc estimate is tight.
N_TRAIN_PER_SHADOW = 200
N_SHADOW = 64
N_QUERY_PER_Z = 250        # query set is 2 * N_QUERY_PER_Z, z-balanced
RECON_BUDGET = 100         # records the adversary targeted for recovery
SHADOW_EPOCHS = 12

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _sample_with_ratio(z_all: np.ndarray, pool_idx: np.ndarray, n_target: int,
                       alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Sample n_target indices from pool_idx with z=1 proportion = alpha."""
    z = z_all[pool_idx]
    pos = pool_idx[z == 1]
    neg = pool_idx[z == 0]
    n_pos = int(round(alpha * n_target))
    n_neg = n_target - n_pos
    pos_pick = rng.choice(pos, size=n_pos, replace=(len(pos) < n_pos))
    neg_pick = rng.choice(neg, size=n_neg, replace=(len(neg) < n_neg))
    chosen = np.concatenate([pos_pick, neg_pick])
    rng.shuffle(chosen)
    return chosen


def _train_shadow(x_train: torch.Tensor, y_train: torch.Tensor, model_seed: int,
                  epochs: int = SHADOW_EPOCHS) -> nn.Module:
    """Train one shadow ResNet-18. Pure function of (x_train, y_train, model_seed)."""
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    g = torch.Generator()
    g.manual_seed(model_seed)
    model = build_resnet18_64().cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64,
                        shuffle=True, generator=g)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            F.cross_entropy(model(xb.cuda()), yb.cuda()).backward()
            opt.step()
    model.eval()
    return model


def _shadow_features(model: nn.Module, query_x: torch.Tensor, query_y: torch.Tensor,
                     query_z: np.ndarray) -> np.ndarray:
    """Per-z-subgroup sorted-loss vector ‖ 4 scalar summaries.

    The query set is z-balanced; computing the order statistics *within* each z
    group preserves the subgroup-specific shift that collusion induces (a global
    sort would mix and cancel it). The 4 scalars (mean loss / mean acc per z
    group) are low-variance anchors for the meta-classifier.
    """
    with torch.no_grad():
        logits = model(query_x.cuda())
        losses = F.cross_entropy(logits, query_y.cuda(), reduction="none").cpu().numpy()
        correct = (logits.argmax(1).cpu().numpy() == query_y.cpu().numpy()).astype(np.float64)
    feats = []
    scalars = []
    for zval in (0, 1):
        m = query_z == zval
        feats.append(np.sort(losses[m]))
        scalars.append(losses[m].mean())
        scalars.append(correct[m].mean())
    return np.concatenate(feats + [np.asarray(scalars)])


def _load_gifd_pool(recon_dir: Path, z_all: np.ndarray):
    """Load recovered images + their (true) z labels. Returns (imgs[-1,1], y, z)."""
    import json
    files = sorted(recon_dir.glob("rec_*.pt"))
    meta = json.loads((recon_dir / "meta.json").read_text())["records"]
    imgs = torch.stack([torch.load(f, map_location="cpu", weights_only=False).squeeze(0)
                        for f in files])
    rec_ids = np.asarray([r["rec_id"] for r in meta], dtype=np.int64)
    ys = np.asarray([r["y"] for r in meta], dtype=np.int64)
    zs = z_all[rec_ids]  # re-derive z against current z_attr
    return imgs, ys, zs


def run_one(dataset: str, alpha_off: float, overlap_p: float, target_seed: int, seed: int,
            recon_source: str, recon_dir: Path | None,
            n_shadow: int = N_SHADOW, n_train_per_shadow: int = N_TRAIN_PER_SHADOW,
            n_query_per_z: int = N_QUERY_PER_Z, recon_budget: int = RECON_BUDGET):
    set_seeds(seed)
    face = load_utkface() if dataset == "utkface" else load_celeba()
    z_all = face.z

    # Splits are a deterministic function of (dataset size, seed), so derive them
    # directly rather than loading a target checkpoint: DIA trains its own victim and
    # shadow models, so it never needs the pretrained target, only its index splits.
    splits = get_splits(face, target_seed)
    shadow_pool = np.asarray(splits["shadow_pool"])

    # Fixed z-balanced query set (disjoint from per-shadow training draws is not
    # required for the BB attack, but we draw it from shadow_pool for consistency).
    qrng = np.random.default_rng(seed * 7 + 1)
    q_pos = shadow_pool[z_all[shadow_pool] == 1]
    q_neg = shadow_pool[z_all[shadow_pool] == 0]
    query_idx = np.concatenate([
        qrng.choice(q_pos, size=n_query_per_z, replace=(len(q_pos) < n_query_per_z)),
        qrng.choice(q_neg, size=n_query_per_z, replace=(len(q_neg) < n_query_per_z)),
    ])
    query_x = to_model_input(face.images[query_idx]).cuda()
    query_y = torch.from_numpy(face.y[query_idx]).long()
    query_z = z_all[query_idx]

    # Reconstruction pool. Injection is no-replacement from a fixed recon_budget.
    n_inject = int(round(overlap_p * recon_budget))
    if n_inject > 0:
        if recon_source == "oracle":
            pool_idx = np.asarray(splits["recon_pool"])[:recon_budget]
            recon_imgs = to_model_input(face.images[pool_idx])
            recon_y = face.y[pool_idx]
            recon_z = z_all[pool_idx]
        elif recon_source == "gifd":
            assert recon_dir is not None and Path(recon_dir).exists()
            recon_imgs, recon_y, recon_z = _load_gifd_pool(Path(recon_dir), z_all)
            recon_imgs = recon_imgs[:recon_budget]
            recon_y = recon_y[:recon_budget]
            recon_z = recon_z[:recon_budget]
        else:
            raise ValueError(recon_source)
    else:
        recon_imgs = recon_y = recon_z = None

    def shadow_train_set(alpha: float, k: int, inject: bool):
        """Build one shadow's training tensors using an independent per-shadow RNG."""
        srng = np.random.default_rng(seed * 100_000 + k * 2 + (0 if alpha == ALPHA_REF else 1))
        idx = _sample_with_ratio(z_all, shadow_pool, n_train_per_shadow, alpha, srng)
        x = to_model_input(face.images[idx])
        y = torch.from_numpy(face.y[idx]).long()
        if inject and n_inject > 0:
            # Draw n_inject reconstructed records WITHOUT replacement, at the
            # shadow's own z-ratio, using the per-shadow RNG.
            pick = _sample_with_ratio(recon_z, np.arange(len(recon_z)),
                                      n_inject, alpha, srng)
            x = torch.cat([x, recon_imgs[pick]], dim=0)
            y = torch.cat([y, torch.from_numpy(recon_y[pick]).long()], dim=0)
        return x, y

    # Asymmetric injection: only the alpha_ref shadows (matching the victim's
    # distribution) receive reconstructed members.
    feats_ref, feats_off = [], []
    for k in range(n_shadow):
        xr, yr = shadow_train_set(ALPHA_REF, k, inject=True)
        xo, yo = shadow_train_set(alpha_off, k, inject=False)
        m_ref = _train_shadow(xr, yr, model_seed=seed * 1_000_000 + k * 2)
        m_off = _train_shadow(xo, yo, model_seed=seed * 1_000_000 + k * 2 + 1)
        feats_ref.append(_shadow_features(m_ref, query_x, query_y, query_z))
        feats_off.append(_shadow_features(m_off, query_x, query_y, query_z))
        del m_ref, m_off
        torch.cuda.empty_cache()

    X = np.concatenate([np.stack(feats_ref), np.stack(feats_off)], axis=0)
    y = np.concatenate([np.zeros(len(feats_ref)), np.ones(len(feats_off))]).astype(np.int64)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    preds = cross_val_predict(clf, X, y, cv=5)
    meta_acc = float((preds == y).mean()) * 100.0
    return {"meta_acc": meta_acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["utkface", "celeba"], required=True)
    ap.add_argument("--task", choices=["lo", "hi"], required=True)
    ap.add_argument("--target_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--overlap", type=float, default=None)
    ap.add_argument("--recon_source", choices=["oracle", "gifd"], required=True)
    ap.add_argument("--recon_dir", type=str, default=None)
    ap.add_argument("--n_shadow", type=int, default=N_SHADOW)
    ap.add_argument("--n_train_per_shadow", type=int, default=N_TRAIN_PER_SHADOW)
    ap.add_argument("--recon_budget", type=int, default=RECON_BUDGET)
    ap.add_argument("--results_csv", type=str, default="dia_image.csv")
    args = ap.parse_args()

    recon_dir = Path(args.recon_dir) if args.recon_dir else None
    alpha_off = ALPHA1_LOOKUP[args.task]
    levels = OVERLAP_LEVELS if args.overlap is None else [args.overlap]
    # Bare filename -> under results/; an explicit path -> used as given (relative
    # to CWD). Matches mia_overlap.py so all three attacks resolve --results_csv the
    # same way and never double-nest results/results/.
    _csv = Path(args.results_csv)
    csv_path = _csv if (_csv.is_absolute() or _csv.parent != Path(".")) else RESULTS_DIR / _csv

    for p in levels:
        result = run_one(args.dataset, alpha_off, p, args.target_seed, args.seed,
                         args.recon_source, recon_dir,
                         n_shadow=args.n_shadow, n_train_per_shadow=args.n_train_per_shadow,
                         recon_budget=args.recon_budget)
        for metric, value in result.items():
            row = {
                "dataset": args.dataset, "task": args.task, "alpha_off": alpha_off,
                "target_seed": args.target_seed, "recon_source": args.recon_source,
                "overlap_p": p, "seed": args.seed, "metric": metric, "value": value,
                "extra": f"n_shadow={args.n_shadow},n_train={args.n_train_per_shadow},"
                         f"recon_budget={args.recon_budget}",
            }
            save_result_row(csv_path, row)
            print(f"[{args.dataset} task={args.task} src={args.recon_source} "
                  f"p={p:.2f} seed={args.seed}] {metric}={value:.2f}")


if __name__ == "__main__":
    main()
