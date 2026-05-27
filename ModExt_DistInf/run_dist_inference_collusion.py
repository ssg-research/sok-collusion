"""
Distribution inference x model extraction collusion experiment.

Compares three ways of producing adversary shadow populations:

  --setting 1  shadow arch != target arch, trained from scratch (baseline)
  --setting 2  shadow arch != target arch, extracted from victims via distillation
  --setting 3  shadow arch == target arch, extracted from victims (gold standard)

Hypothesis: Setting 1 <= Setting 2 <= Setting 3.

See outputs/di_collusion_experiment_plan.md for the full experimental design.
"""

import argparse
import csv
import hashlib
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from amulet.datasets import (
    AmuletDataset,
    load_celeba,
    load_utkface,
)
from amulet.distribution_inference.attacks import SuriEvans2022
from amulet.distribution_inference.dataset_utils import prepare_distribution_splits
from amulet.distribution_inference.metrics import evaluate_distinguishing_accuracy
from amulet.unauth_model_ownership.attacks import ModelExtraction
from amulet.utils import get_accuracy, get_fidelity, initialize_model, train_classifier

EXPERIMENTS_DIR = Path(__file__).parent
MODELS_DIR = EXPERIMENTS_DIR / "models"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
LOGS_DIR = EXPERIMENTS_DIR / "logs"
DATA_DIR = EXPERIMENTS_DIR / "data"
RESULTS_CSV = RESULTS_DIR / "collusion_results.csv"

CSV_FIELDNAMES = [
    "timestamp",
    "setting",
    "dataset",
    "target_attribute",
    "target_arch",
    "target_capacity",
    "shadow_arch",
    "shadow_capacity",
    "num_models",
    "epochs",
    "extraction_epochs",
    "extraction_loss_type",
    "batch_size",
    "ratio1",
    "ratio2",
    "filter_column",
    "filter_value",
    "train_subsample",
    "test_subsample",
    "exp_id",
    "victim_acc_mean",
    "victim_acc_std",
    "shadow_acc_mean",
    "shadow_acc_std",
    "fidelity_mean",
    "fidelity_std",
    "distinguishing_accuracy",
    "auc_score",
]


# ---------------------------------------------------------------------------
# Dataset dispatch
# ---------------------------------------------------------------------------


def _load_celeba(args: argparse.Namespace) -> tuple[AmuletDataset, str]:
    target_attribute = args.target_attribute or "Mouth_Slightly_Open"
    data = load_celeba(
        path=Path(args.data_root) / "celeba",
        target_attribute=target_attribute,
        random_seed=args.exp_id,
    )
    return data, target_attribute


def _load_utkface(args: argparse.Namespace) -> tuple[AmuletDataset, str]:
    target = args.target_attribute or "gender"
    data = load_utkface(
        path=Path(args.data_root) / "utkface",
        target=target,
        attribute_1=args.utkface_attr1,
        attribute_2=args.utkface_attr2,
        age_bins=args.utkface_age_bins,
        random_seed=args.exp_id,
    )
    return data, target


_DATASET_LOADERS: dict[
    str, Callable[[argparse.Namespace], tuple[AmuletDataset, str]]
] = {
    "celeba": _load_celeba,
    "utkface": _load_utkface,
}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _config_hash(config: dict) -> str:  # type: ignore[type-arg]
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def _checkpoint_path(models_dir: Path, config: dict, model_id: int) -> Path:  # type: ignore[type-arg]
    return models_dir / f"{_config_hash(config)}_{model_id}.pth"


# ---------------------------------------------------------------------------
# Population training / extraction
# ---------------------------------------------------------------------------


def _train_population(
    loader: DataLoader,  # type: ignore[type-arg]
    *,
    config: dict,  # type: ignore[type-arg]
    arch: str,
    capacity: str,
    num_features: int,
    num_classes: int,
    num_models: int,
    epochs: int,
    device: str,
    models_dir: Path,
) -> list[nn.Module]:
    """Train or load a population of models with hash-based checkpointing."""
    models_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    models: list[nn.Module] = []

    for model_id in range(num_models):
        ckpt_path = _checkpoint_path(models_dir, config, model_id)
        model = initialize_model(arch, capacity, num_features, num_classes).to(device)

        if ckpt_path.exists():
            saved = torch.load(ckpt_path, weights_only=False, map_location=device)
            model.load_state_dict(saved["state_dict"])
        else:
            label = (
                f"{config['role']}/{config['dist']} model {model_id + 1}/{num_models}"
            )
            print(f"Training {label}")
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )
            model = train_classifier(
                model, loader, criterion, optimizer, epochs, device, scheduler=scheduler
            )
            torch.save({"state_dict": model.state_dict(), "config": config}, ckpt_path)

        model.eval()
        model.requires_grad_(False)
        models.append(model)

    return models


def _extract_population(
    victims: list[nn.Module],
    query_loader: DataLoader,  # type: ignore[type-arg]
    *,
    config: dict,  # type: ignore[type-arg]
    arch: str,
    capacity: str,
    num_features: int,
    num_classes: int,
    extraction_epochs: int,
    extraction_loss_type: str,
    device: str,
    models_dir: Path,
) -> list[nn.Module]:
    """Distill one shadow per victim. Shadow i is extracted from victims[i]."""
    models_dir.mkdir(parents=True, exist_ok=True)
    shadows: list[nn.Module] = []

    for model_id, victim in enumerate(victims):
        ckpt_path = _checkpoint_path(models_dir, config, model_id)
        shadow = initialize_model(arch, capacity, num_features, num_classes).to(device)

        if ckpt_path.exists():
            saved = torch.load(ckpt_path, weights_only=False, map_location=device)
            shadow.load_state_dict(saved["state_dict"])
        else:
            label = (
                f"{config['role']}/{config['dist']} shadow "
                f"{model_id + 1}/{len(victims)}"
            )
            print(f"Extracting {label}")
            victim = victim.to(device)
            victim.eval()
            optimizer = torch.optim.Adam(shadow.parameters(), lr=1e-3)
            extraction = ModelExtraction(
                target_model=victim,
                attack_model=shadow,
                optimizer=optimizer,
                train_loader=query_loader,
                device=device,
                epochs=extraction_epochs,
                loss_type=extraction_loss_type,
            )
            shadow = extraction.attack()
            torch.save({"state_dict": shadow.state_dict(), "config": config}, ckpt_path)

        shadow.eval()
        shadow.requires_grad_(False)
        shadows.append(shadow)

    return shadows


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _make_full_test_loader(x: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:  # type: ignore[type-arg]
    """Build a DataLoader over the full test set (no ratio subsampling)."""
    ds = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(y.ravel(), dtype=np.int64)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _pop_accuracy(
    models: list[nn.Module],
    loader: DataLoader,
    device: str,  # type: ignore[type-arg]
) -> tuple[float, float]:
    accs = [get_accuracy(m.to(device), loader, device) for m in models]
    return float(np.mean(accs)), float(np.std(accs))


# ---------------------------------------------------------------------------
# CSV reporting
# ---------------------------------------------------------------------------


def _append_csv(row: dict, output: Path) -> None:  # type: ignore[type-arg]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists()
    with output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--setting",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="1=scratch baseline, 2=extracted smaller arch, 3=extracted matched arch",
    )
    parser.add_argument("--data_root", type=str, default=str(DATA_DIR))
    parser.add_argument(
        "--dataset",
        type=str,
        default="celeba",
        choices=sorted(_DATASET_LOADERS.keys()),
        help="Dataset to load. Determines which loader is used and which "
        "sensitive attributes are exposed for --filter_column.",
    )

    # Target (victim) architecture
    parser.add_argument("--target_arch", type=str, default="resnet")
    parser.add_argument("--target_capacity", type=str, default="m1")

    # Shadow architecture (Setting 3 ignores these and uses target arch/capacity)
    parser.add_argument("--shadow_arch", type=str, default="vgg")
    parser.add_argument("--shadow_capacity", type=str, default="m1")

    # Training
    parser.add_argument("--num_models", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--extraction_epochs", type=int, default=5)
    parser.add_argument(
        "--extraction_loss_type",
        type=str,
        default="kl",
        choices=["kl", "mse", "ce"],
    )
    parser.add_argument("--batch_size", type=int, default=64)

    # Distribution inference task
    parser.add_argument("--ratio1", type=float, default=0.45)
    parser.add_argument("--ratio2", type=float, default=0.55)
    parser.add_argument("--filter_column", type=str, default="Male")
    parser.add_argument("--filter_value", type=int, default=0)
    parser.add_argument("--train_subsample", type=int, default=10000)
    parser.add_argument("--test_subsample", type=int, default=10000)

    # Dataset-specific config
    parser.add_argument(
        "--target_attribute",
        type=str,
        default="Mouth_Slightly_Open",
        help="Classification target. CelebA: any binary attribute "
        "(default 'Mouth_Slightly_Open'). UTKFace: one of 'age', 'gender', "
        "'race' (default 'gender').",
    )
    parser.add_argument(
        "--utkface_attr1",
        type=str,
        default="race",
        choices=["age", "gender", "race"],
        help="UTKFace first sensitive attribute.",
    )
    parser.add_argument(
        "--utkface_attr2",
        type=str,
        default="age",
        choices=["age", "gender", "race"],
        help="UTKFace second sensitive attribute.",
    )
    parser.add_argument(
        "--utkface_age_bins",
        type=int,
        nargs="*",
        default=[30],
        help="Bin edges for discretizing UTKFace age. Default [30] produces a "
        "binary split (0-29, 30+). Pass multiple edges for more groups.",
    )

    # Misc
    parser.add_argument("--exp_id", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_CSV),
        help="Path to the CSV file where results are appended.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                LOGS_DIR / f"di_collusion_s{args.setting}_{args.exp_id}.log",
                mode="w",
            ),
            logging.StreamHandler(),
        ],
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("di_collusion")

    torch.manual_seed(args.exp_id)
    torch.cuda.manual_seed_all(args.exp_id)
    np.random.seed(args.exp_id)

    # Setting 3 uses target arch for shadows (extraction, matched arch).
    shadow_arch = args.target_arch if args.setting == 3 else args.shadow_arch
    shadow_capacity = (
        args.target_capacity if args.setting == 3 else args.shadow_capacity
    )

    log.info(
        "Setting %d | target=%s/%s shadow=%s/%s | device=%s | dataset=%s",
        args.setting,
        args.target_arch,
        args.target_capacity,
        shadow_arch,
        shadow_capacity,
        args.device,
        args.dataset,
    )

    data, target_attribute = _DATASET_LOADERS[args.dataset](args)
    if any(
        v is None
        for v in (
            data.x_train,
            data.y_train,
            data.z_train,
            data.x_test,
            data.y_test,
            data.z_test,
            data.sensitive_columns,
        )
    ):
        raise RuntimeError(f"{args.dataset} loader did not populate required arrays.")

    assert data.sensitive_columns is not None
    if args.filter_column not in data.sensitive_columns:
        raise ValueError(
            f"--filter_column {args.filter_column!r} is not a sensitive attribute "
            f"for dataset {args.dataset!r}. Available: {data.sensitive_columns}"
        )

    # Type-narrowed references after None check.
    assert data.x_train is not None
    assert data.y_train is not None
    assert data.z_train is not None
    assert data.x_test is not None
    assert data.y_test is not None
    assert data.z_test is not None

    splits = prepare_distribution_splits(
        data.x_train,
        data.y_train,
        data.z_train,
        data.x_test,
        data.y_test,
        data.z_test,
        sensitive_columns=data.sensitive_columns,
        filter_column=args.filter_column,
        ratio1=args.ratio1,
        ratio2=args.ratio2,
        train_subsample=args.train_subsample,
        test_subsample=args.test_subsample,
        filter_value=args.filter_value,
        batch_size=args.batch_size,
        seed=args.exp_id,
    )

    # Full test set loader for per-model accuracy and fidelity measurement.
    fidelity_loader = _make_full_test_loader(data.x_test, data.y_test, args.batch_size)

    # Base config fields shared by all populations in this run.
    # These are hashed to produce deterministic checkpoint filenames.
    base_cfg: dict = {  # type: ignore[type-arg]
        "dataset": args.dataset,
        "target_attribute": target_attribute,
        "setting": args.setting,
        "target_arch": args.target_arch,
        "target_capacity": args.target_capacity,
        "shadow_arch": shadow_arch,
        "shadow_capacity": shadow_capacity,
        "num_models": args.num_models,
        "epochs": args.epochs,
        "exp_id": args.exp_id,
        "ratio1": args.ratio1,
        "ratio2": args.ratio2,
        "filter_column": args.filter_column,
        "filter_value": args.filter_value,
    }

    shared_kw: dict = {  # type: ignore[type-arg]
        "num_features": data.num_features,
        "num_classes": data.num_classes,
        "device": args.device,
        "models_dir": MODELS_DIR,
    }

    # ----- Victim populations (target arch, trained from scratch) -----
    models_vic_1 = _train_population(
        splits.vic_trainloader_1,
        config={**base_cfg, "role": "vic", "dist": "1"},
        arch=args.target_arch,
        capacity=args.target_capacity,
        num_models=args.num_models,
        epochs=args.epochs,
        **shared_kw,
    )
    models_vic_2 = _train_population(
        splits.vic_trainloader_2,
        config={**base_cfg, "role": "vic", "dist": "2"},
        arch=args.target_arch,
        capacity=args.target_capacity,
        num_models=args.num_models,
        epochs=args.epochs,
        **shared_kw,
    )

    # ----- Adversary (shadow) populations -----
    if args.setting == 1:
        # Scratch shadows with a smaller arch than the victim.
        models_adv_1 = _train_population(
            splits.adv_trainloader_1,
            config={**base_cfg, "role": "adv", "dist": "1"},
            arch=shadow_arch,
            capacity=shadow_capacity,
            num_models=args.num_models,
            epochs=args.extraction_epochs,
            **shared_kw,
        )
        models_adv_2 = _train_population(
            splits.adv_trainloader_2,
            config={**base_cfg, "role": "adv", "dist": "2"},
            arch=shadow_arch,
            capacity=shadow_capacity,
            num_models=args.num_models,
            epochs=args.extraction_epochs,
            **shared_kw,
        )
    else:
        # Settings 2 and 3: extract shadow i from victim i using the adversary
        # split as query data (D1 ratio encoded in the query distribution).
        extract_cfg = {**base_cfg, "extraction_epochs": args.extraction_epochs}
        extract_kw = {
            **shared_kw,
            "extraction_epochs": args.extraction_epochs,
            "extraction_loss_type": args.extraction_loss_type,
        }
        models_adv_1 = _extract_population(
            models_vic_1,
            splits.adv_trainloader_1,
            config={**extract_cfg, "role": "adv", "dist": "1"},
            arch=shadow_arch,
            capacity=shadow_capacity,
            **extract_kw,
        )
        models_adv_2 = _extract_population(
            models_vic_2,
            splits.adv_trainloader_2,
            config={**extract_cfg, "role": "adv", "dist": "2"},
            arch=shadow_arch,
            capacity=shadow_capacity,
            **extract_kw,
        )

    # ----- Run the KL distinguishing test -----
    attack = SuriEvans2022(
        x_train=data.x_train,
        y_train=data.y_train,
        z_train=data.z_train,
        x_test=data.x_test,
        y_test=data.y_test,
        z_test=data.z_test,
        sensitive_columns=data.sensitive_columns,
        filter_column=args.filter_column,
        ratio1=args.ratio1,
        ratio2=args.ratio2,
        model_arch=args.target_arch,
        model_capacity=args.target_capacity,
        num_features=data.num_features,
        num_classes=data.num_classes,
        num_models=args.num_models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        models_dir=MODELS_DIR,
        dataset=args.dataset,
        exp_id=args.exp_id,
        filter_value=args.filter_value,
        train_subsample=args.train_subsample,
        test_subsample=args.test_subsample,
    )

    di_results = attack.attack(
        models_adv_1=models_adv_1,
        models_adv_2=models_adv_2,
        models_vic_1=models_vic_1,
        models_vic_2=models_vic_2,
        test_loader_1=splits.test_loader_1,
        test_loader_2=splits.test_loader_2,
    )
    di_metrics = evaluate_distinguishing_accuracy(
        di_results["predictions"], di_results["ground_truth"]
    )

    # ----- Per-model metrics -----
    all_victims = models_vic_1 + models_vic_2
    all_adv_d1 = models_adv_1
    all_adv_d2 = models_adv_2

    vic_acc_mean, vic_acc_std = _pop_accuracy(all_victims, fidelity_loader, args.device)
    sha_acc_mean, sha_acc_std = _pop_accuracy(
        all_adv_d1 + all_adv_d2, fidelity_loader, args.device
    )

    # Fidelity: only meaningful for extraction settings (2 and 3).
    # Shadow i of D1 is paired with victim i of D1, shadow i of D2 with victim i of D2.
    if args.setting in (2, 3):
        all_pairs = list(zip(models_vic_1, models_adv_1, strict=False)) + list(
            zip(models_vic_2, models_adv_2, strict=False)
        )
        all_fids = np.array([
            get_fidelity(
                v.to(args.device), s.to(args.device), fidelity_loader, args.device
            )
            for v, s in all_pairs
        ])
        fid_mean: float | None = float(np.mean(all_fids))
        fid_std: float | None = float(np.std(all_fids))
    else:
        fid_mean = None
        fid_std = None

    log.info(
        "victim_acc=%.2f±%.2f  shadow_acc=%.2f±%.2f  fidelity=%s  "
        "dist_acc=%.4f  auc=%.4f",
        vic_acc_mean,
        vic_acc_std,
        sha_acc_mean,
        sha_acc_std,
        f"{fid_mean:.2f}±{fid_std:.2f}" if fid_mean is not None else "N/A",
        di_metrics["distinguishing_accuracy"],
        di_metrics["auc_score"],
    )
    print(
        f"\nSetting {args.setting} results:\n"
        f"  victim acc:   {vic_acc_mean:.2f} ± {vic_acc_std:.2f}\n"
        f"  shadow acc:   {sha_acc_mean:.2f} ± {sha_acc_std:.2f}\n"
        f"  fidelity:     "
        + (
            f"{fid_mean:.2f} ± {fid_std:.2f}"
            if fid_mean is not None
            else "N/A (Setting 1)"
        )
        + f"\n"
        f"  dist. acc:    {di_metrics['distinguishing_accuracy']:.4f}\n"
        f"  AUC:          {di_metrics['auc_score']:.4f}\n"
    )

    _append_csv({
        "timestamp": datetime.now(UTC).isoformat(),
        "setting": args.setting,
        "dataset": args.dataset,
        "target_attribute": target_attribute,
        "target_arch": args.target_arch,
        "target_capacity": args.target_capacity,
        "shadow_arch": shadow_arch,
        "shadow_capacity": shadow_capacity,
        "num_models": args.num_models,
        "epochs": args.epochs,
        "extraction_epochs": args.extraction_epochs,
        "extraction_loss_type": args.extraction_loss_type,
        "batch_size": args.batch_size,
        "ratio1": args.ratio1,
        "ratio2": args.ratio2,
        "filter_column": args.filter_column,
        "filter_value": args.filter_value,
        "train_subsample": args.train_subsample,
        "test_subsample": args.test_subsample,
        "exp_id": args.exp_id,
        "victim_acc_mean": round(vic_acc_mean, 4),
        "victim_acc_std": round(vic_acc_std, 4),
        "shadow_acc_mean": round(sha_acc_mean, 4),
        "shadow_acc_std": round(sha_acc_std, 4),
        "fidelity_mean": round(fid_mean, 4) if fid_mean is not None else "",
        "fidelity_std": round(fid_std, 4) if fid_std is not None else "",
        "distinguishing_accuracy": round(di_metrics["distinguishing_accuracy"], 4),
        "auc_score": round(di_metrics["auc_score"], 4),
    }, Path(args.output))


if __name__ == "__main__":
    main(parse_args())
