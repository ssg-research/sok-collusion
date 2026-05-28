"""
T-MIA: LiRA membership inference with hybrid auxiliary overlap sweep.

Usage:
    python mia_overlap.py --dataset cifar10 --overlap_p 0.0 --num_shadow 8 --epochs 30 --seed 0
    python mia_overlap.py --dataset cifar10 --overlap_p 0.75 --num_shadow 8 --epochs 30 --seed 0

Rows:    overlap p ∈ {0, 0.25, 0.50, 0.75}
Columns: CIFAR-10, CIFAR-100
Metric:  LiRA AUC + TPR@FPR=1e-2 (→ also reports 1e-3 when available)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

# ── Path setup ───────────────────────────────────────────────────────────────
# amuletml is installed via `uv sync`; only this dir needs to be importable.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from amulet.membership_inference.attacks import LiRA
from amulet.membership_inference.metrics import compute_mi_metrics
from amulet.utils import (
    create_dir,
    get_accuracy,
    initialize_model,
    load_data,
    train_classifier,
)

from common import (
    OVERLAP_LEVELS,
    build_gifd_pool_image,
    build_hybrid_overlap_image,
    compute_member_indices,
    result_exists,
    save_result_row,
    set_seeds,
)

# ── N_A: adversary auxiliary pool size ─────────────────────────────────────
# Full train=50k; target uses pkeep*full_train/2 = 12500 members.
# After reserving N_CHAL/2=1000 records for the challenge member set, only
# 11500 training records remain for oracle sampling. At p=0.75 the pool
# requires n_oracle = 0.75*N_A - num_classes records, so we need N_A such
# that 0.75*N_A ≤ 11500 + num_classes → N_A ≤ 15346. Use 15000 for margin.
N_A_CIFAR10 = 15_000
N_A_CIFAR100 = 15_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T-MIA overlap sweep (LiRA)")
    parser.add_argument("--root", type=str, default=str(_HERE),
                        help="Working root; data/ and saved_models/ are created here.")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--model_capacity", type=str, default="m1")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Shadow-model training epochs (LiRA).")
    parser.add_argument("--target_epochs", type=int, default=None,
                        help="Target-model training epochs (default = --epochs). "
                             "Raise (with --target_wd 0) to overfit the target.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_shadow", type=int, default=64)
    parser.add_argument("--target_wd", type=float, default=5e-4,
                        help="Target weight decay. Set 0 (with more epochs) to overfit "
                             "the target, widening the member/non-member gap so the "
                             "overlap effect is more visible.")
    parser.add_argument("--pkeep", type=float, default=0.5)
    parser.add_argument("--n_a", type=int, default=None,
                        help="Override adversary pool size (default 15000). Smaller n_a "
                             "makes a limited recon set a larger member fraction with "
                             "milder replication.")
    parser.add_argument("--challenge_size", type=int, default=2000,
                        help="Fixed challenge set size (members + non-members, 50/50)")
    parser.add_argument("--overlap_p", type=float, default=0.0,
                        choices=OVERLAP_LEVELS)
    parser.add_argument("--recon_source", choices=["fredrikson", "gifd"],
                        default="fredrikson",
                        help="fredrikson = legacy Fredrikson prototypes + oracle padding; "
                             "gifd = realistic Geiping reconstructions from --recon_dir.")
    parser.add_argument("--recon_dir", type=str, default=None,
                        help="Pool of rec_*.pt + meta.json (required when --recon_source gifd).")
    parser.add_argument("--recon_budget", type=int, default=None,
                        help="Records the adversary targets to recover. p is the fraction "
                             "recovered. Defaults to the pool size.")
    parser.add_argument("--pool_mode", choices=["budget", "fraction"], default="budget",
                        help="budget = inject p*recon_budget recons (reproduces the paper's "
                             "MIA numbers); fraction = member fraction of the pool equals p.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Adversary RNG seed: challenge sampling, recon-pool draw, "
                             "shadow training. Varied to get n>1 per cell.")
    parser.add_argument("--target_seed", type=int, default=None,
                        help="Target-model seed: member partition + target training. "
                             "Fixed across adversary seeds so one realistic recon pool "
                             "(built for this target's members) stays valid. "
                             "Defaults to --seed (single-seed backward-compatible mode).")
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results_dir", type=str,
                        default=str(_HERE / "results"))
    parser.add_argument("--results_csv", type=str, default=None,
                        help="Explicit results CSV path (default results_dir/mia_overlap.csv). "
                             "Use a per-process path to run adversary seeds in parallel "
                             "without append races, then merge.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    set_seeds(args.seed)
    target_seed = args.target_seed if args.target_seed is not None else args.seed

    root_dir = Path(args.root)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_csv = (Path(args.results_csv) if args.results_csv
                   else results_dir / "mia_overlap.csv")
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    match_key = {
        "table": "T-MIA",
        "dataset": args.dataset,
        "overlap_p": args.overlap_p,
        "seed": args.seed,
        "target_seed": target_seed,
        "recon_source": args.recon_source,
        "metric": "lira_online_auc",
    }
    if result_exists(results_csv, match_key):
        print(f"[T-MIA] Already computed: {match_key}. Skipping.")
        return

    log_dir = root_dir / "logs"
    create_dir(log_dir)
    logging.basicConfig(
        level=logging.INFO,
        filename=log_dir / "mia_overlap.log",
        filemode="a",
    )
    log = logging.getLogger("mia_overlap")
    log.addHandler(logging.StreamHandler())

    # ── Load full dataset ────────────────────────────────────────────────────
    data = load_data(root_dir, args.dataset, training_size=1.0, log=log, exp_id=args.seed)
    full_train_size = len(data.train_set)  # type: ignore[arg-type]

    # ── Target model: train on half of D_train ───────────────────────────────
    models_path = root_dir / "saved_models" / "mia_overlap"
    target_model_path = models_path / "target"
    create_dir(target_model_path)

    n_a = args.n_a if args.n_a is not None else (
        N_A_CIFAR10 if args.dataset == "cifar10" else N_A_CIFAR100)
    # Member-set partition is keyed by target_seed and shared with cifar_recover.py
    # so the realistic-source recon pool (built for this target's members) is valid
    # for every adversary seed attacking the same target.
    idx = compute_member_indices(full_train_size, target_seed, args.pkeep)
    d_train_idx = idx["d_train_idx"]
    d_out_idx = idx["d_out_idx"]
    keep = idx["keep_local"]
    target_size = len(d_train_idx)

    d_train = Subset(data.train_set, d_train_idx.tolist())
    d_out = Subset(data.train_set, d_out_idx.tolist())
    target_subset = Subset(d_train, keep.tolist())

    target_epochs = args.target_epochs if args.target_epochs is not None else args.epochs
    filename = (
        f"{args.dataset}_{args.model}_{args.model_capacity}"
        f"_ep{target_epochs}_wd{args.target_wd:g}_seed{target_seed}.pt"
    )
    target_model_file = target_model_path / filename

    criterion = torch.nn.CrossEntropyLoss()

    if target_model_file.exists():
        log.info("Loading target model from %s", target_model_file)
        target_model = torch.load(target_model_file, map_location=args.device)
    else:
        log.info("Training target model (%s %s, %d epochs)",
                 args.model, args.model_capacity, args.epochs)
        set_seeds(target_seed)  # target weights deterministic in target_seed only
        target_model = initialize_model(
            args.model, args.model_capacity,
            data.num_features, data.num_classes, log,
        ).to(args.device)
        optimizer = torch.optim.SGD(
            target_model.parameters(), lr=0.01, momentum=0.9, weight_decay=args.target_wd
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, target_epochs)
        train_loader = DataLoader(target_subset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4)
        train_classifier(target_model, train_loader, criterion, optimizer,
                         target_epochs, args.device)
        scheduler.step()
        torch.save(target_model, target_model_file)
        log.info("Target model saved.")

    set_seeds(args.seed)  # adversary RNG governs challenge/pool/shadows downstream

    test_loader = DataLoader(data.test_set, batch_size=args.batch_size, shuffle=False)
    test_acc = get_accuracy(target_model, test_loader, args.device)
    log.info("Target model test accuracy: %.4f", test_acc)

    # ── Reserve challenge set FIRST, then build pool disjoint from reservations.
    # Challenge records (members from target_subset, non-members from d_out) must
    # be physically disjoint from pool records; otherwise the same underlying
    # record appears twice with independent shadow in/out labels, corrupting
    # LiRA's mean_in / mean_out estimates (OUT distribution gets polluted by
    # member-like scores from records the shadow was actually trained on).
    N_CHAL = args.challenge_size
    chal_rng = np.random.RandomState(1234 + args.seed)
    chal_mem_choice = chal_rng.choice(
        len(target_subset), size=N_CHAL // 2, replace=False
    )
    chal_non_idx_d_out = chal_rng.choice(
        len(d_out), size=N_CHAL // 2, replace=False
    )

    input_size = (1, 3, 32, 32)
    cache_dir = models_path / "fredrikson_cache"

    if args.recon_source == "gifd":
        assert args.recon_dir is not None, "--recon_dir required for --recon_source gifd"
        adv_pool, pool_in_indices, comp_log = build_gifd_pool_image(
            recon_dir=Path(args.recon_dir),
            d_out=d_out,
            p=args.overlap_p,
            n_a=n_a,
            seed=args.seed,
            recon_budget=args.recon_budget,
            reserved_out_idx=chal_non_idx_d_out,
            pool_mode=args.pool_mode,
        )
    else:
        adv_pool, pool_in_indices, comp_log = build_hybrid_overlap_image(
            target_model=target_model,
            d_train=target_subset,
            d_out=d_out,
            p=args.overlap_p,
            n_a=n_a,
            seed=args.seed,
            num_classes=data.num_classes,
            input_size=input_size,
            device=args.device,
            cache_dir=cache_dir,
            model_path=target_model_file,
            reserved_train_idx=chal_mem_choice,
            reserved_out_idx=chal_non_idx_d_out,
        )
    log.info("Overlap pool: %s", comp_log)

    challenge_mem = Subset(target_subset, chal_mem_choice.tolist())
    challenge_non = Subset(d_out, chal_non_idx_d_out.tolist())
    # train_set = adv_pool ∪ challenge_mem ∪ challenge_non. LiRA will train shadow
    # models on random pkeep fraction of this; its `in_data` param marks which
    # positions the TARGET was trained on (adv_pool members + challenge_mem).
    train_set = ConcatDataset([adv_pool, challenge_mem, challenge_non])
    pool_size = len(adv_pool)  # type: ignore[arg-type]
    chal_mem_pos = np.arange(pool_size, pool_size + len(challenge_mem))
    in_data = np.concatenate([pool_in_indices, chal_mem_pos]).astype(int)

    shadow_model_dir = models_path / "shadow_models" / args.dataset / f"p{args.overlap_p}_seed{args.seed}"
    create_dir(shadow_model_dir)

    lira = LiRA(
        target_model=target_model,
        in_data=in_data,
        shadow_architecture=args.model,
        shadow_capacity=args.model_capacity,
        train_set=train_set,
        dataset=args.dataset,
        num_features=data.num_features,
        num_classes=data.num_classes,
        batch_size=args.batch_size,
        pkeep=args.pkeep,
        criterion=criterion,
        num_shadow=args.num_shadow,
        epochs=args.epochs,
        device=args.device,
        models_dir=shadow_model_dir,
        exp_id=args.seed,
    )

    results = lira.attack()
    # Restrict metrics to the challenge set positions (first pool_size are pool,
    # next N_CHAL/2 are members, next N_CHAL/2 are non-members). This ensures
    # AUC is well-defined at every p.
    chal_start = pool_size
    chal_end = pool_size + len(challenge_mem) + len(challenge_non)
    chal_preds_on = results["lira_online_preds"][chal_start:chal_end]
    chal_preds_off = results["lira_offline_preds"][chal_start:chal_end]
    chal_labels = results["true_labels"][chal_start:chal_end]
    online_metrics = compute_mi_metrics(chal_preds_on, chal_labels)
    offline_metrics = compute_mi_metrics(chal_preds_off, chal_labels)

    log.info("[T-MIA] dataset=%s p=%.2f seed=%d online=%s offline=%s",
             args.dataset, args.overlap_p, args.seed, online_metrics, offline_metrics)
    print(f"\n[T-MIA] dataset={args.dataset} p={args.overlap_p} seed={args.seed}")
    print(f"  online:  {online_metrics}")
    print(f"  offline: {offline_metrics}")

    base_row = dict(
        table="T-MIA",
        dataset=args.dataset,
        filter_prop="",
        alpha2="",
        overlap_p=args.overlap_p,
        seed=args.seed,
        target_seed=target_seed,
        recon_source=args.recon_source,
        n_recon=comp_log["n_recon"],
        n_oracle=comp_log["n_oracle"],
        n_out=comp_log["n_out"],
        extra=f"num_shadow={args.num_shadow},epochs={args.epochs},test_acc={test_acc:.4f}",
    )
    for variant, metrics in [("online", online_metrics), ("offline", offline_metrics)]:
        for metric_name, value in metrics.items():
            save_result_row(results_csv, {
                **base_row,
                "metric": f"lira_{variant}_{metric_name}",
                "value": f"{value:.6f}",
            })

    print(f"[T-MIA] Results saved to {results_csv}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
