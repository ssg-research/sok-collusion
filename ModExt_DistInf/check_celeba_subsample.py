"""
Check the maximum viable train_subsample / test_subsample for CelebA
given an arbitrary sensitive-attribute ratio.

The binding constraint on `subsample` (= cwise_sample in _heuristic_sample) is:
  - need `class_imbalance * subsample` items of y=0 from the ratio-filtered pool
  - need `subsample` items of y=1 from the ratio-filtered pool

so max_subsample = min(n_y0_in_pool // class_imbalance, n_y1_in_pool).

Usage:
    uv run python experiments/check_celeba_subsample.py --ratio 0.45
    uv run python experiments/check_celeba_subsample.py --ratio1 0.45 --ratio2 0.55
    uv run python experiments/check_celeba_subsample.py --ratio1 0.4 --ratio2 0.6
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from amulet.datasets import load_celeba
from amulet.distribution_inference.dataset_utils import _stratify_key


def _max_subsample(
    y: np.ndarray,
    z_col: np.ndarray,
    ratio: float,
    filter_value: int,
    class_imbalance: float,
    seed: int = 0,
) -> tuple[int, float]:
    """Return (max_cwise_sample, achieved_ratio) for one split at the given ratio.

    Args:
        y: 1-D label array for this split.
        z_col: 1-D sensitive-attribute column for this split.
        ratio: Target fraction of filter_value items in the pool.
        filter_value: The sensitive-attribute value being filtered on.
        class_imbalance: Majority/minority y-label ratio (matches prepare_distribution_splits).
        seed: NumPy random seed for the pool shuffle.
    """
    rng = np.random.default_rng(seed)
    filter_mask = z_col == filter_value

    # Replicate _filter_by_ratio with a controlled seed.
    qualify = np.nonzero(filter_mask)[0]
    notqualify = np.nonzero(~filter_mask)[0]
    current_ratio = len(qualify) / (len(qualify) + len(notqualify))

    if current_ratio <= ratio:
        rng.shuffle(notqualify)
        if ratio < 1:
            nqi = notqualify[: int(((1 - ratio) * len(qualify)) / ratio)]
            pool = np.concatenate([qualify, nqi])
        else:
            pool = qualify
    else:
        rng.shuffle(qualify)
        if ratio > 0:
            qi = qualify[: int((ratio * len(notqualify)) / (1 - ratio))]
            pool = np.concatenate([qi, notqualify])
        else:
            pool = notqualify

    y_pool = y[pool]
    n_y0 = int((y_pool == 0).sum())
    n_y1 = int((y_pool == 1).sum())
    achieved_ratio = float(filter_mask[pool].mean())
    max_sub = min(int(n_y0 / class_imbalance), n_y1)
    return max_sub, achieved_ratio


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Single ratio to check (sets both ratio1 and ratio2 to this value).",
    )
    parser.add_argument(
        "--ratio1", type=float, default=0.45, help="First ratio (default 0.45)."
    )
    parser.add_argument(
        "--ratio2", type=float, default=0.55, help="Second ratio (default 0.55)."
    )
    parser.add_argument(
        "--filter_column",
        type=str,
        default="Male",
        help="Sensitive column to filter on (default 'Male').",
    )
    parser.add_argument(
        "--filter_value",
        type=int,
        default=0,
        help="Value of filter_column that is 'True' in the filter mask (default 0 = Female).",
    )
    parser.add_argument(
        "--class_imbalance",
        type=float,
        default=None,
        help="Majority/minority y-label ratio. Defaults to the actual imbalance in the dataset, "
        "matching prepare_distribution_splits behavior.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(Path(__file__).parent / "data" / "celeba"),
    )
    args = parser.parse_args()

    ratios: list[float] = (
        [args.ratio] if args.ratio is not None else [args.ratio1, args.ratio2]
    )

    data = load_celeba(path=args.data_root)

    assert data.x_train is not None
    assert data.y_train is not None
    assert data.z_train is not None
    assert data.x_test is not None
    assert data.y_test is not None
    assert data.z_test is not None
    assert data.sensitive_columns is not None

    y_train = data.y_train.ravel()
    z_train = data.z_train
    y_test = data.y_test.ravel()
    z_test = data.z_test

    class_imbalance = args.class_imbalance
    if class_imbalance is None:
        n_majority = int((y_train == 0).sum())
        n_minority = int((y_train == 1).sum())
        class_imbalance = n_majority / n_minority

    filter_col_idx = data.sensitive_columns.index(args.filter_column)
    z_train_col = z_train[:, filter_col_idx]
    z_test_col = z_test[:, filter_col_idx]

    n_true_train = int((z_train_col == args.filter_value).sum())
    n_true_test = int((z_test_col == args.filter_value).sum())
    print(f"Total train: {len(y_train):,}  |  Total test: {len(y_test):,}")
    print(
        f"Train {args.filter_column}={args.filter_value}: {n_true_train:,}"
        f" ({n_true_train / len(y_train):.1%})"
    )
    print(
        f"Test  {args.filter_column}={args.filter_value}: {n_true_test:,}"
        f" ({n_true_test / len(y_test):.1%})"
    )

    # 50/50 stratified split — matches prepare_distribution_splits seed=0
    strat_train = _stratify_key(y_train, z_train)
    vic_idx, adv_idx = next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0).split(
            y_train, strat_train
        )
    )

    strat_test = _stratify_key(y_test, z_test)
    tv_idx, ta_idx = next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0).split(
            y_test, strat_test
        )
    )

    splits = [
        ("victim train", vic_idx, y_train, z_train_col),
        ("adv    train", adv_idx, y_train, z_train_col),
        ("victim test", tv_idx, y_test, z_test_col),
        ("adv    test", ta_idx, y_test, z_test_col),
    ]

    print(f"class_imbalance: {class_imbalance:.3f} (y=0/y=1 in full train set)")

    for ratio in ratios:
        print(f"\n=== ratio={ratio}  class_imbalance={class_imbalance:.3f} ===")
        binding: list[int] = []
        for split_name, idx, y_all, z_col_all in splits:
            y_split = y_all[idx]
            z_split = z_col_all[idx]
            n_total = len(idx)
            n_true = int((z_split == args.filter_value).sum())
            max_sub, actual_ratio = _max_subsample(
                y_split,
                z_split,
                ratio,
                args.filter_value,
                class_imbalance,
            )
            print(
                f"  {split_name:15s} | n={n_total:6,} |"
                f" {args.filter_column}={args.filter_value}: {n_true:5,} ({n_true / n_total:.1%}) |"
                f" pool_ratio={actual_ratio:.3f} | max_subsample={max_sub:,}"
            )
            binding.append(max_sub)
        print(f"  => Binding max_subsample = {min(binding):,}  (all splits)")


if __name__ == "__main__":
    main()
