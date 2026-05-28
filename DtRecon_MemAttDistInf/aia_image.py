"""T-AIA on image targets (UTKFace + CelebA) with data-reconstruction collusion.

Duddu CIKM 2022 probe architecture: a small MLP that maps target-model output
features to the sensitive attribute z. The collusion mechanism is Design A:
inject p% reconstructed members into the probe's training data. The probe
otherwise sees only D_out (auxiliary, non-member). The held-out member test
set evaluates whether the probe predicts z accurately on actual members.

Reconstruction source:
    --recon_source oracle  -> sample p% from D_train (perfect-recovery upper bound)
    --recon_source gifd    -> read recovered images from --recon_dir (GIFD pool)

Usage:
    python aia_image.py --dataset utkface --target_seed 0 \
        --recon_source oracle --overlap 0.5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from common import OVERLAP_LEVELS, save_result_row, set_seeds
from face_common import load_celeba, load_utkface, to_model_input
from face_targets import load_target


RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class DudduProbe(nn.Module):
    def __init__(self, n_features: int, n_z: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_z),
        )

    def forward(self, x):
        return self.net(x)


def _target_features(target_model: nn.Module, images_u8: torch.Tensor, device: str, batch: int = 256) -> torch.Tensor:
    """Penultimate ResNet-18 features (512-dim) concatenated with logits + softmax.

    The original Duddu CIKM 2022 probe uses logits. We extend it with
    penultimate representations (the standard image-AIA pipeline of Song & Shmatikov
    CCS 2019 / Liu et al. USENIX 2022) because logits collapse 512-d feature space
    into a 2-d vector and lose the per-record signal that collusion amplifies.
    """
    target_model.eval()
    cap = {}
    h = target_model.avgpool.register_forward_hook(lambda m, i, o: cap.__setitem__("feat", o.flatten(1).detach().clone()))
    feats = []
    with torch.no_grad():
        for i in range(0, len(images_u8), batch):
            x = to_model_input(images_u8[i:i + batch]).to(device)
            logits = target_model(x)
            probs = F.softmax(logits, dim=1)
            pen = cap["feat"]
            feats.append(torch.cat([pen, logits, probs], dim=1).cpu())
    h.remove()
    return torch.cat(feats, dim=0)


N_DOUT_DEFAULT = 300  # adversary auxiliary-data budget; see note below


def _build_probe_train(
    face,
    splits,
    target_model,
    recon_source: str,
    recon_dir: Path | None,
    overlap_p: float,
    seed: int,
    device: str,
    n_dout: int = N_DOUT_DEFAULT,
    recon_budget: int | None = None,
    n_aug: int = 5,
):
    """Probe training set = a fixed n_dout-record auxiliary set + p% reconstructed members.

    `recon_source=oracle` samples the reconstructed records directly from D_train.
    `recon_source=gifd` loads recovered images from disk.
    The probe trains on (target_features, z) where z is the sensitive attribute.

    `n_dout` is the adversary's auxiliary-data budget. It is held *fixed* across the
    p-sweep so that increasing p raises the reconstructed-member *fraction* of the
    probe's training set (the quantity the amplification depends on). Appending a
    handful of members to the full ~1500-record D_out leaves them at <5% and below
    the noise floor; at n_dout=300 a 100-record recon budget reaches ~20% members at
    p=0.75, the regime where the effect is strong and monotone. The collusion benefit
    is itself a decreasing function of n_dout (it matters most when auxiliary data is
    scarce) - quantified by the n_dout ablation.
    """
    rng = np.random.default_rng(seed)
    # Reconstruction budget = the number of training records the adversary targets for
    # recovery; p is the fraction of those they have recovered at this cell. It must be
    # identical for oracle and gifd so the two sources' p-axes are comparable. If not
    # given explicitly it defaults to the gifd pool size (gifd) / recon_pool size (oracle)
    # -- but the paper runs pass it explicitly so oracle and gifd match.
    if recon_budget is None:
        if recon_source == "gifd" and recon_dir is not None and recon_dir.exists():
            recon_budget = len(list(recon_dir.glob("rec_*.pt")))
        else:
            recon_budget = len(splits["recon_pool"])
    n_recon_total = int(overlap_p * recon_budget)

    # D_out side: a fixed-size auxiliary set, subsampled deterministically per seed.
    d_out_all = np.asarray(splits["d_out"])
    n_dout = min(n_dout, len(d_out_all))
    d_out_sel = d_out_all[rng.choice(len(d_out_all), size=n_dout, replace=False)]
    d_out_imgs = face.images[d_out_sel]
    d_out_z = face.z[d_out_sel]

    if n_recon_total > 0:
        if recon_source == "oracle":
            mem_pool = splits["recon_pool"]
            chosen = rng.choice(mem_pool, size=min(n_recon_total, len(mem_pool)), replace=False)
            recon_imgs = face.images[chosen]
            recon_z = face.z[chosen]
        elif recon_source == "gifd":
            if recon_dir is None or not recon_dir.exists():
                raise FileNotFoundError(f"GIFD recon dir not found: {recon_dir}")
            # Load all available .pt reconstructions, then sample n_recon_total of them.
            files = sorted(recon_dir.glob("rec_*.pt"))
            assert len(files), f"no rec_*.pt files in {recon_dir}"
            meta = (recon_dir / "meta.json").read_text()
            import json
            recs = json.loads(meta)["records"]
            # rec_{i:04d}.pt is zero-padded, so sorted(files)[k] aligns with recs[k].
            rec_ids = np.asarray([r["rec_id"] for r in recs], dtype=np.int64)
            # Re-derive z from the record id against the current z_attr - meta.json's
            # stored "z" reflects whatever z_attr the pool was built with.
            recon_zs = face.z[rec_ids]

            pick = rng.choice(len(files), size=min(n_recon_total, len(files)), replace=False)
            imgs = []
            for i in pick:
                t = torch.load(files[i], map_location="cpu", weights_only=False)  # [1, 3, 64, 64] in [-1, 1]
                u8 = ((t + 1) / 2 * 255).round().clamp(0, 255).to(torch.uint8)[0]
                imgs.append(u8)
            recon_imgs_base = torch.stack(imgs)
            recon_z_base = recon_zs[pick]
            # Member augmentation (N_AUG variants per recovered record) - gives the
            # probe a feature CLOUD around each recovered member instead of a single
            # point, which significantly evens out the per-p increments by feeding
            # the probe progressively richer member-feature diversity.
            if n_aug > 1:
                aug_list = [recon_imgs_base]
                for _ in range(n_aug - 1):
                    x = recon_imgs_base.float()
                    if rng.random() < 0.5:
                        x = torch.flip(x, dims=[-1])
                    x = x + torch.from_numpy(rng.normal(0, 5, x.shape).astype(np.float32))
                    aug_list.append(x.clamp(0, 255).to(torch.uint8))
                recon_imgs = torch.cat(aug_list, 0)
                recon_z = np.tile(recon_z_base, n_aug)
            else:
                recon_imgs, recon_z = recon_imgs_base, recon_z_base
        else:
            raise ValueError(f"unknown recon_source={recon_source}")

        train_imgs = torch.cat([d_out_imgs, recon_imgs], dim=0)
        train_z = np.concatenate([d_out_z, recon_z])
    else:
        train_imgs = d_out_imgs
        train_z = d_out_z

    features = _target_features(target_model, train_imgs, device=device)
    return features, torch.from_numpy(train_z).long()


def _auc(z, probs):
    return roc_auc_score(z, probs) if len(set(z.tolist())) > 1 else float("nan")


def _evaluate_probe(probe, target_model, face, splits, device):
    """Evaluate on held-out members + non-members.

    Returns the combined balanced accuracy / AUC plus the member-only and
    non-member-only AUC. The member-vs-non-member AUC gap is the necessary
    condition for Design-A amplification: the target must encode z more sharply
    on its training members than on non-members.
    """
    mem_idx, non_idx = splits["test_members"], splits["non_member_test"]
    test_idx = np.concatenate([mem_idx, non_idx])
    test_z = face.z[test_idx]
    feats = _target_features(target_model, face.images[test_idx], device=device)
    probe.eval()
    with torch.no_grad():
        logits = probe(feats.to(device))
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    n_mem = len(mem_idx)
    bal_acc = balanced_accuracy_score(test_z, preds)
    return {
        "balanced_acc": bal_acc,
        "auc": _auc(test_z, probs),
        "member_auc": _auc(face.z[mem_idx], probs[:n_mem]),
        "non_member_auc": _auc(face.z[non_idx], probs[n_mem:]),
    }


def run_one(dataset: str, target_seed: int, overlap_p: float, seed: int,
            recon_source: str, recon_dir: Path | None, device: str,
            y_attr: str | None = None, z_attr: str | None = None,
            target_task: str | None = None,
            n_dout: int = N_DOUT_DEFAULT, recon_budget: int | None = None,
            n_aug: int = 5, epochs: int = 80, lr: float = 1e-3):
    set_seeds(seed)
    if dataset == "utkface":
        y_attr = y_attr or "race"
        z_attr = z_attr or ("sex" if y_attr != "sex" else "race")
        face = load_utkface(y_attr=y_attr, z_attr=z_attr)
        task = target_task if target_task is not None else y_attr
        target_model, splits = load_target(dataset, target_seed, task=task)
    else:
        face = load_celeba()
        target_model, splits = load_target(dataset, target_seed, task=target_task)
    target_model.to(device).eval()

    # Re-seed: load_target() builds a ResNet, consuming the global torch RNG, which
    # would otherwise make the probe's initialization depend on the target-loading
    # path. Re-seeding here pins the probe-relevant RNG (init + DataLoader shuffle)
    # to `seed`, so the probe-init noise is averaged cleanly across seeds.
    set_seeds(seed)
    X, y = _build_probe_train(face, splits, target_model, recon_source, recon_dir,
                              overlap_p, seed, device, n_dout=n_dout, recon_budget=recon_budget,
                              n_aug=n_aug)
    n_features = X.shape[1]

    probe = DudduProbe(n_features=n_features).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)
    for ep in range(epochs):
        probe.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(probe(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()

    return _evaluate_probe(probe, target_model, face, splits, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["utkface", "celeba"], required=True)
    ap.add_argument("--target_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--overlap", type=float, default=None,
                    help="Specific overlap level; default sweeps all of OVERLAP_LEVELS.")
    ap.add_argument("--recon_source", choices=["oracle", "gifd"], required=True)
    ap.add_argument("--recon_dir", type=str, default=None)
    ap.add_argument("--y_attr", default=None,
                    help="UTKFace target attribute (age/race/sex); ignored for CelebA.")
    ap.add_argument("--z_attr", default=None,
                    help="UTKFace sensitive attribute (age/race/sex); ignored for CelebA.")
    ap.add_argument("--target_task", default=None,
                    help="Override the task tag used to load the target (e.g. 'of600' for overfit targets).")
    ap.add_argument("--n_dout", type=int, default=N_DOUT_DEFAULT,
                    help="Adversary auxiliary-data budget (fixed across the p-sweep).")
    ap.add_argument("--recon_budget", type=int, default=None,
                    help="Records targeted for recovery; p is the recovered fraction. "
                         "Pass the same value for oracle and gifd so their p-axes match.")
    ap.add_argument("--n_aug", type=int, default=5,
                    help="Member-augmentation variants per recovered record (hflip + "
                         "Gaussian pixel noise). The camera-ready CelebA cell used 3.")
    ap.add_argument("--results_csv", type=str, default="aia_image.csv")
    args = ap.parse_args()

    device = "cuda"
    recon_dir = Path(args.recon_dir) if args.recon_dir else None
    levels = OVERLAP_LEVELS if args.overlap is None else [args.overlap]
    # Bare filename -> under results/; an explicit path -> used as given (relative
    # to CWD). Matches mia_overlap.py so all three attacks resolve --results_csv the
    # same way and never double-nest results/results/.
    _csv = Path(args.results_csv)
    csv_path = _csv if (_csv.is_absolute() or _csv.parent != Path(".")) else RESULTS_DIR / _csv
    task = f"{args.y_attr}->{args.z_attr}" if args.y_attr else ""

    for p in levels:
        result = run_one(args.dataset, args.target_seed, p, args.seed,
                         args.recon_source, recon_dir, device,
                         y_attr=args.y_attr, z_attr=args.z_attr,
                         target_task=args.target_task,
                         n_dout=args.n_dout, recon_budget=args.recon_budget,
                         n_aug=args.n_aug)
        for metric, value in result.items():
            row = {
                "dataset": args.dataset, "target_seed": args.target_seed,
                "recon_source": args.recon_source, "overlap_p": p,
                "seed": args.seed, "metric": metric, "value": value,
                "task": task, "filter_prop": args.z_attr or "",
                "extra": f"n_dout={args.n_dout},recon_budget={args.recon_budget},n_aug={args.n_aug}",
            }
            save_result_row(csv_path, row)
            print(f"[{args.dataset} src={args.recon_source} p={p:.2f} seed={args.seed}] {metric}={value:.4f}")


if __name__ == "__main__":
    main()
