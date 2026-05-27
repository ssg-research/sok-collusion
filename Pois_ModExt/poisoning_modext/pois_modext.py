import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from amulet.unauth_model_ownership.attacks import ModelExtraction
from amulet.unauth_model_ownership.metrics import evaluate_extraction
from amulet.poisoning.attacks import BadNets
from amulet.utils import (
    load_data,
    initialize_model,
    train_classifier,
    create_dir,
    get_accuracy,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default=str(SCRIPT_DIR),
        help="Root directory of models and datasets. Defaults to this script's directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV path to append a result row to. "
             "Defaults to '{root}/pois_modext_results_{dataset}.csv'.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        help="Options: cifar10, fmnist, lfw, census, celeba.",
    )
    parser.add_argument(
        "--model", type=str, default="resnet", help="Options: vgg, linearnet."
    )
    parser.add_argument(
        "--model_capacity",
        type=str,
        default="m1",
        help="Size of the model to use. Options: m1, m2, m3, m4, where m1 is the smallest.",
    )
    parser.add_argument(
        "--training_size", type=float, default=1, help="Fraction of dataset to use."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size of input data."
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs for training."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=torch.device(
            "cuda:{0}".format(0) if torch.cuda.is_available() else "cpu"
        ),
        help="Device on which to run PyTorch",
    )
    parser.add_argument(
        "--exp_id", type=int, default=0, help="Used as a random seed for experiments."
    )
    parser.add_argument(
        "--adv_train_fraction",
        type=float,
        default=0.5,
        help="Fraction of trianing data used by the adversary.",
    )
    parser.add_argument(
        "--poisoned_portion",
        type=float,
        default=0.1,
        help="posioning portion (float, range from 0 to 1, default: 0.1)",
    )
    parser.add_argument(
        "--trigger_label",
        type=int,
        default=1,
        help="The NO. of trigger label (int, range from 0 to 10, default: 0)",
    )
    parser.add_argument(
        "--query_size", type=float, default=1.0, help="Fraction of dataset to use."
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Setup logger
    root_dir = Path(args.root)
    log_dir = root_dir / "logs"
    create_dir(log_dir)
    logging.basicConfig(
        level=logging.INFO, filename=log_dir / "poisoning_model_extraction.log", filemode="w"
    )
    log = logging.getLogger("All")
    log.addHandler(logging.StreamHandler())

    # Set random seeds for reproducibility
    torch.manual_seed(args.exp_id)

    # Load dataset and split train data for adversary
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
        generator=generator
    )

    # Create data loaders
    adv_train_loader = DataLoader(
        dataset=adv_query_set, batch_size=args.batch_size, shuffle=False
    )
    target_train_loader = DataLoader(
        dataset=target_train_set, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        dataset=data.test_set, batch_size=args.batch_size, shuffle=False
    )

    # Set up filename and directories to save/load models
    models_path = root_dir / "saved_models"
    filename = f"{args.dataset}_{args.model}_{args.model_capacity}_{args.training_size*100}_{args.batch_size}_{args.epochs}_{args.exp_id}.pt"

    # Train or Load Target Model
    target_model_path = (
        models_path
        / "targetForExtraction_poisoned"
        / f"adv_train_fraction_{args.adv_train_fraction}"
        / f"poisoned_portion_{args.poisoned_portion}"
    )
    target_model_filename = target_model_path / filename
    criterion = torch.nn.CrossEntropyLoss()

    # Train poisoned Model
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
    if target_model_filename.exists():
        log.info("Target model loaded from %s", target_model_filename)
        target_model = torch.load(target_model_filename, weights_only=False)
    else:
        target_model = initialize_model(
            args.model, args.model_capacity, data.num_features, data.num_classes, log, resnet_replace_first=True,
        ).to(args.device)
        optimizer = torch.optim.SGD(target_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)

        if args.poisoned_portion == 0:
            target_model = train_classifier(
                target_model,
                target_train_loader,
                criterion,
                optimizer,
                args.epochs,
                args.device,
                scheduler=scheduler
            )
            log.info("Target model trained")

            # Save model
            create_dir(target_model_path, log)
            torch.save(target_model, target_model_filename)
        else:
            log.info("Running Model Poisoning attack")
            poisoned_train_set = poisoning.attack(target_train_set)
            poisoned_train_loader = DataLoader(
                dataset=poisoned_train_set, batch_size=args.batch_size, shuffle=False
            )
            target_model = train_classifier(
                target_model,
                poisoned_train_loader,
                criterion,
                optimizer,
                args.epochs,
                args.device,
                scheduler=scheduler
            )

            # Save model
            create_dir(target_model_path, log)
            torch.save(target_model, target_model_filename)

    # Train or Load model for Model Extraction
    attack_model_path = (
        models_path
        / "modelExtraction_poisoned"
        / f"adv_train_fraction_{args.adv_train_fraction}"
        / f"poisoned_portion_{args.poisoned_portion}"
    )

    filename = f"{args.dataset}_{args.model}_{args.model_capacity}_{args.query_size*100}_{args.batch_size}_{args.epochs}_{args.exp_id}.pt"
    attack_model_filename = attack_model_path / filename

    if attack_model_filename.exists():
        log.info("Attack model loaded from %s", attack_model_filename)
        attack_model = torch.load(attack_model_filename, weights_only=False)
    else:
        log.info("Running Model Extraction attack")
        attack_model = initialize_model(
            args.model, args.model_capacity, data.num_features, data.num_classes, log, resnet_replace_first=True
        ).to(args.device)
        optimizer = torch.optim.Adam(attack_model.parameters(), lr=1e-3, weight_decay=5e-4)
        model_extraction = ModelExtraction(
            target_model,
            attack_model,
            optimizer,
            adv_train_loader,
            args.device,
            args.epochs,
        )
        attack_model = model_extraction.attack()

        # Save model
        create_dir(attack_model_path, log)
        torch.save(attack_model, attack_model_filename)

    # Evaluate
    poisoned_test_set = poisoning.attack(data.test_set, mode="test")
    poisoned_test_loader = DataLoader(
        dataset=poisoned_test_set, batch_size=args.batch_size, shuffle=False
    )

    target_test_accuracy = get_accuracy(target_model, test_loader, args.device)
    target_poisoned_accuracy = get_accuracy(target_model, poisoned_test_loader, args.device)
    stolen_test_accuracy = get_accuracy(attack_model, test_loader, args.device)
    stolen_poisoned_accuracy = get_accuracy(attack_model, poisoned_test_loader, args.device)

    evaluation_results = evaluate_extraction(
        target_model, attack_model, test_loader, args.device
    )

    log.info("Target Model test accuracy: %s", target_test_accuracy)
    log.info("Target Model poisoned accuracy: %s", target_poisoned_accuracy)
    log.info("Stolen Model test accuracy: %s", stolen_test_accuracy)
    log.info("Stolen Model poisoned accuracy: %s", stolen_poisoned_accuracy)
    log.info("Fidelity: %s", evaluation_results["fidelity"])
    log.info("Correct Fidelity: %s", evaluation_results["correct_fidelity"])

    headers = [
        'exp_id',
        'target_acc_test',
        'poisoning_percentage',
        'target_acc_poisoned',
        'query_size',
        'stolen_acc_test',
        'fidelity',
        'correct_fidelity',
        'stolen_acc_poisoned',
    ]

    results = [
        args.exp_id,
        target_test_accuracy,
        args.poisoned_portion,
        target_poisoned_accuracy,
        adv_query_size,
        stolen_test_accuracy,
        evaluation_results['fidelity'],
        evaluation_results['correct_fidelity'],
        stolen_poisoned_accuracy,
    ]

    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = root_dir / f"pois_modext_results_{args.dataset}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not output_path.exists():
        with open(output_path, 'w') as f:
            f.write(','.join(headers) + '\n')

    with open(output_path, 'a') as f:
        f.write(','.join(f"{x}" for x in results) + '\n')



if __name__ == "__main__":
    args = parse_args()
    main(args)
