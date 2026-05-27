"""
Poisoning x model extraction collusion experiment (paper Section sec:trteeval).

For a fixed (dataset, exp_id), the script:

  1. Trains a ResNet34 target on D_train. If --poisoned_portion > 0, D_train is
     first passed through amulet's BadNets poisoner so the target carries a
     trigger-bearing backdoor.
  2. Steals a ResNet34 surrogate from the target via amulet's
     KnockoffNets-style ModelExtraction, using --query_size of D_aux as the
     query pool.
  3. Reports clean-test accuracy, trigger-test accuracy, fidelity, and
     correct-fidelity for both the target and the surrogate.

Hypothesis (validated as negative in the paper): increasing --poisoned_portion
does NOT improve the surrogate's accuracy or fidelity over the unpoisoned
baseline. This confirms there is no collusion potential between Poison and
Model Extraction.
"""

import argparse
import csv
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from amulet.poisoning.attacks import BadNets
from amulet.unauth_model_ownership.attacks import ModelExtraction
from amulet.unauth_model_ownership.metrics import evaluate_extraction
from amulet.utils import (
    create_dir,
    get_accuracy,
    initialize_model,
    load_data,
    train_classifier,
)

EXPERIMENTS_DIR = Path(__file__).parent
MODELS_DIR = EXPERIMENTS_DIR / "models"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
LOGS_DIR = EXPERIMENTS_DIR / "logs"
DATA_DIR = EXPERIMENTS_DIR / "data"

CSV_FIELDNAMES = [
    "timestamp",
    "dataset",
    "target_arch",
    "target_capacity",
    "surrogate_arch",
    "surrogate_capacity",
    "epochs",
    "extraction_loss_type",
    "batch_size",
    "training_size",
    "adv_train_fraction",
    "trigger_label",
    "poisoned_portion",
    "query_size_fraction",
    "query_size_records",
    "exp_id",
    "target_acc_test",
    "target_acc_poisoned",
    "stolen_acc_test",
    "stolen_acc_poisoned",
    "fidelity",
    "correct_fidelity",
]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _target_ckpt_path(args: argparse.Namespace) -> Path:
    """Target checkpoint path keyed on the fields that determine target
    training (everything except the extraction-query budget)."""
    return (
        MODELS_DIR
        / "target_poisoned"
        / f"adv_train_fraction_{args.adv_train_fraction}"
        / f"poisoned_portion_{args.poisoned_portion}"
        / f"{args.dataset}_{args.model}_{args.model_capacity}"
        f"_{args.training_size * 100:.0f}_{args.batch_size}_{args.epochs}"
        f"_{args.exp_id}.pt"
    )


def _surrogate_ckpt_path(args: argparse.Namespace) -> Path:
    """Surrogate checkpoint path additionally keyed on the query budget."""
    return (
        MODELS_DIR
        / "surrogate"
        / f"adv_train_fraction_{args.adv_train_fraction}"
        / f"poisoned_portion_{args.poisoned_portion}"
        / f"{args.dataset}_{args.model}_{args.model_capacity}"
        f"_q{args.query_size * 100:.0f}_{args.batch_size}_{args.epochs}"
        f"_{args.exp_id}.pt"
    )


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
        "--root",
        type=str,
        default=str(EXPERIMENTS_DIR),
        help="Root directory for cached datasets, models, and logs. "
        "Defaults to this script's directory.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset name passed to amulet.utils.load_data. Tested with "
        "'cifar10' and 'cifar100' for paper Table tab:trteeval.",
    )

    # Target / surrogate architecture (single arch in this experiment).
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--model_capacity", type=str, default="m1")

    # Training
    parser.add_argument("--training_size", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--extraction_loss_type",
        type=str,
        default="mse",
        choices=["mse", "kl", "ce"],
        help="Loss used by amulet.ModelExtraction. Reference numbers were "
        "produced with the amulet default 'mse'.",
    )

    # Poisoning / extraction knobs
    parser.add_argument("--adv_train_fraction", type=float, default=0.5)
    parser.add_argument("--poisoned_portion", type=float, default=0.1)
    parser.add_argument("--trigger_label", type=int, default=1)
    parser.add_argument(
        "--query_size",
        type=float,
        default=1.0,
        help="Fraction of D_aux used as the surrogate's query pool.",
    )

    # Misc
    parser.add_argument("--exp_id", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV path to append a result row to. Defaults to "
        "'{root}/results/pois_modext_results_{dataset}.csv'.",
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
    root_dir = Path(args.root)
    logs_dir = root_dir / "logs"
    create_dir(logs_dir)
    log_path = (
        logs_dir
        / f"pois_modext_{args.dataset}_p{args.poisoned_portion}"
        f"_q{args.query_size}_exp{args.exp_id}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("pois_modext")

    torch.manual_seed(args.exp_id)
    torch.cuda.manual_seed_all(args.exp_id)
    np.random.seed(args.exp_id)

    log.info(
        "dataset=%s | model=%s/%s | poison=%s | query=%s | device=%s",
        args.dataset,
        args.model,
        args.model_capacity,
        args.poisoned_portion,
        args.query_size,
        args.device,
    )

    # ----- Data -----
    data = load_data(root_dir, args.dataset, args.training_size, log)

    adv_train_size = int(args.adv_train_fraction * len(data.train_set))  # type: ignore[reportArgumentType]
    target_train_size = len(data.train_set) - adv_train_size  # type: ignore[reportArgumentType]
    generator = torch.Generator().manual_seed(args.exp_id)
    target_train_set, adv_train_set = random_split(
        data.train_set, [target_train_size, adv_train_size], generator=generator
    )

    adv_query_size = int(args.query_size * len(adv_train_set))
    adv_query_set, _ = random_split(
        adv_train_set,
        [adv_query_size, len(adv_train_set) - adv_query_size],
        generator=generator,
    )

    target_train_loader = DataLoader(
        dataset=target_train_set, batch_size=args.batch_size, shuffle=False
    )
    adv_train_loader = DataLoader(
        dataset=adv_query_set, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        dataset=data.test_set, batch_size=args.batch_size, shuffle=False
    )

    # ----- Backdoor poisoner -----
    if args.dataset in ["census", "lfw"]:
        dataset_type = "tabular"
    else:
        dataset_type = "image"

    poisoning = BadNets(
        args.trigger_label,
        args.poisoned_portion,
        args.exp_id,
        dataset_type,
    )

    # ----- Train or load target model -----
    target_ckpt = _target_ckpt_path(args)
    criterion = torch.nn.CrossEntropyLoss()

    if target_ckpt.exists():
        log.info("Target model loaded from %s", target_ckpt)
        target_model = torch.load(target_ckpt, weights_only=False)
    else:
        target_model = initialize_model(
            args.model,
            args.model_capacity,
            data.num_features,
            data.num_classes,
            log,
            resnet_replace_first=True,
        ).to(args.device)
        optimizer = torch.optim.SGD(
            target_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)

        if args.poisoned_portion == 0:
            log.info("Training clean target")
            train_loader = target_train_loader
        else:
            log.info("Training poisoned target via BadNets")
            poisoned_train_set = poisoning.poison_train(target_train_set)
            train_loader = DataLoader(
                dataset=poisoned_train_set,
                batch_size=args.batch_size,
                shuffle=False,
            )

        target_model = train_classifier(
            target_model,
            train_loader,
            criterion,
            optimizer,
            args.epochs,
            args.device,
            scheduler=scheduler,
        )
        create_dir(target_ckpt.parent, log)
        torch.save(target_model, target_ckpt)

    # ----- Train or load surrogate via Model Extraction -----
    surrogate_ckpt = _surrogate_ckpt_path(args)
    if surrogate_ckpt.exists():
        log.info("Surrogate model loaded from %s", surrogate_ckpt)
        surrogate_model = torch.load(surrogate_ckpt, weights_only=False)
    else:
        log.info("Running Model Extraction")
        surrogate_model = initialize_model(
            args.model,
            args.model_capacity,
            data.num_features,
            data.num_classes,
            log,
            resnet_replace_first=True,
        ).to(args.device)
        optimizer = torch.optim.Adam(
            surrogate_model.parameters(), lr=1e-3, weight_decay=5e-4
        )
        extraction = ModelExtraction(
            target_model=target_model,
            attack_model=surrogate_model,
            optimizer=optimizer,
            train_loader=adv_train_loader,
            device=args.device,
            epochs=args.epochs,
            loss_type=args.extraction_loss_type,
        )
        surrogate_model = extraction.attack()
        create_dir(surrogate_ckpt.parent, log)
        torch.save(surrogate_model, surrogate_ckpt)

    # ----- Evaluate -----
    poisoned_test_set = poisoning.poison_test(data.test_set)
    poisoned_test_loader = DataLoader(
        dataset=poisoned_test_set, batch_size=args.batch_size, shuffle=False
    )

    target_acc_test = get_accuracy(target_model, test_loader, args.device)
    target_acc_poisoned = get_accuracy(target_model, poisoned_test_loader, args.device)
    stolen_acc_test = get_accuracy(surrogate_model, test_loader, args.device)
    stolen_acc_poisoned = get_accuracy(
        surrogate_model, poisoned_test_loader, args.device
    )
    eval_results = evaluate_extraction(
        target_model, surrogate_model, test_loader, args.device
    )

    log.info(
        "target_test=%.2f target_trig=%.2f | stolen_test=%.2f stolen_trig=%.2f | "
        "fid=%.2f corr_fid=%.2f",
        target_acc_test,
        target_acc_poisoned,
        stolen_acc_test,
        stolen_acc_poisoned,
        eval_results["fidelity"],
        eval_results["correct_fidelity"],
    )
    print(
        f"\nPoisoning={args.poisoned_portion} | query={args.query_size} results:\n"
        f"  target acc (clean / trigger): "
        f"{target_acc_test:.2f} / {target_acc_poisoned:.2f}\n"
        f"  stolen acc (clean / trigger): "
        f"{stolen_acc_test:.2f} / {stolen_acc_poisoned:.2f}\n"
        f"  fidelity:                     {eval_results['fidelity']:.2f}\n"
        f"  correct fidelity:             {eval_results['correct_fidelity']:.2f}\n"
    )

    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = (
            root_dir / "results" / f"pois_modext_results_{args.dataset}.csv"
        )

    _append_csv(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "dataset": args.dataset,
            "target_arch": args.model,
            "target_capacity": args.model_capacity,
            "surrogate_arch": args.model,
            "surrogate_capacity": args.model_capacity,
            "epochs": args.epochs,
            "extraction_loss_type": args.extraction_loss_type,
            "batch_size": args.batch_size,
            "training_size": args.training_size,
            "adv_train_fraction": args.adv_train_fraction,
            "trigger_label": args.trigger_label,
            "poisoned_portion": args.poisoned_portion,
            "query_size_fraction": args.query_size,
            "query_size_records": adv_query_size,
            "exp_id": args.exp_id,
            "target_acc_test": round(target_acc_test, 4),
            "target_acc_poisoned": round(target_acc_poisoned, 4),
            "stolen_acc_test": round(stolen_acc_test, 4),
            "stolen_acc_poisoned": round(stolen_acc_poisoned, 4),
            "fidelity": round(eval_results["fidelity"], 4),
            "correct_fidelity": round(eval_results["correct_fidelity"], 4),
        },
        output_path,
    )


if __name__ == "__main__":
    main(parse_args())
