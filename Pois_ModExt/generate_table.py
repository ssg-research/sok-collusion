"""
Generate the LaTeX for the paper's Table 5 (§5.2) from poisoning x model
extraction result CSVs.

Reads two result files written by pois_modext.py:
  - results/pois_modext_results_cifar10.csv
  - results/pois_modext_results_cifar100.csv

Rows are grouped into per-poison-rate sections {0, 5, 10, 15, 20}% (the 0%
section is the baseline). Each section has four query-budget rows
{2500, 6250, 12500, 25000} and reports surrogate accuracy and surrogate
fidelity (mean and sample std over seeds) per dataset. A per-section
"Target Acc." line reports the (poisoned) target's clean-test accuracy.

For a given budget, non-baseline cells are colored relative to the 0%
baseline at the same budget, dataset, and metric:
  green  if mean exceeds baseline mean + baseline std,
  red    if mean is below baseline mean - baseline std,
  orange otherwise (within the baseline std band).

Usage:
    uv run python generate_table.py
    uv run python generate_table.py --output table.tex
"""

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
DATASET_CSV = {
    "cifar10": RESULTS_DIR / "pois_modext_results_cifar10.csv",
    "cifar100": RESULTS_DIR / "pois_modext_results_cifar100.csv",
}

DATASETS = ["cifar10", "cifar100"]
DATASET_HEADER = {"cifar10": r"\textbf{CIFAR10}", "cifar100": r"\textbf{CIFAR100}"}
POISON_RATES = [0.0, 0.05, 0.1, 0.15, 0.2]
BUDGETS = [2500, 6250, 12500, 25000]
# (column metric, header) pairs shown per dataset.
METRICS = [("stolen_acc_test", r"\textbf{Surr. Acc.}"), ("fidelity", r"\textbf{Surr. Fid.}")]


def _load(dataset: str) -> pd.DataFrame:
    csv = DATASET_CSV[dataset]
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found. Run run_all.sh first (or migrate the reference CSV)."
        )
    return pd.read_csv(csv)


def _cell_values(
    df: pd.DataFrame, poison: float, budget: int, metric: str
) -> pd.Series:
    mask = (
        (df["poisoned_portion"].round(4) == round(poison, 4))
        & (df["query_size_records"] == budget)
    )
    return df.loc[mask, metric].dropna()


def _target_acc(df: pd.DataFrame, poison: float) -> tuple[float, float]:
    """Mean and sample std of target clean-test accuracy over unique seeds.

    target_acc_test is constant across the four budget rows of a given
    (poison, seed), so de-duplicate on exp_id before aggregating to avoid
    shrinking the std with repeated values.
    """
    sub = df[df["poisoned_portion"].round(4) == round(poison, 4)]
    per_seed = sub.groupby("exp_id")["target_acc_test"].first()
    return float(per_seed.mean()), float(per_seed.std(ddof=1))


def _fmt(mean: float, std: float) -> str:
    return f"${mean:.2f}_{{\\pm {std:.2f}}}$"


def _color(cell: str, mean: float, base_mean: float, base_std: float) -> str:
    if mean > base_mean + base_std:
        return r"\cellcolor{green!15}" + cell
    if mean < base_mean - base_std:
        return r"\cellcolor{red!15}" + cell
    return r"\cellcolor{orange!15}" + cell


def build_table(frames: dict[str, pd.DataFrame]) -> str:  # type: ignore[type-arg]
    # Baseline (0% poison) stats per (dataset, budget, metric) for coloring.
    baseline: dict[tuple[str, int, str], tuple[float, float]] = {}
    for ds in DATASETS:
        for b in BUDGETS:
            for m, _ in METRICS:
                vals = _cell_values(frames[ds], 0.0, b, m)
                if not vals.empty:
                    baseline[(ds, b, m)] = (
                        float(vals.mean()),
                        float(vals.std(ddof=1)),
                    )

    lines: list[str] = []
    lines.append(r"\begin{table}[!htb]")
    lines.append(
        r"\caption{\textbf{\poison} $\rightarrow$ \textbf{\modelext}: Surrogate "
        r"model's accuracy (``Surr. Acc.'') and fidelity (``Surr. Fid.'') for 0\% "
        r"poison are the baseline. For a given budget, \colorbox{green!15}{green} "
        r"$\rightarrow$ higher than baseline, \colorbox{orange!15}{orange} "
        r"$\rightarrow$ similar to baseline (within std. dev.), and "
        r"\colorbox{red!15}{red} $\rightarrow$ lower than baseline. We report "
        r"target accuracy to show that \modelext is effective for different "
        r"poisoning rates.}"
    )
    lines.append(r"\begin{center}")
    lines.append(r"\resizebox{0.8\columnwidth}{!}{")
    lines.append(r"\begin{tabular}{l|c|c|c|c}")
    lines.append(r"\bottomrule")
    lines.append("")
    lines.append(r"\toprule")
    lines.append(
        r"& \multicolumn{2}{c|}{" + DATASET_HEADER["cifar10"] + r"} "
        r"& \multicolumn{2}{c}{" + DATASET_HEADER["cifar100"] + r"}\\"
    )
    metric_headers = " & ".join(h for _, h in METRICS)
    lines.append(r"\textbf{Budget} &" + metric_headers + " &  " + metric_headers + r"\\")
    lines.append(r"\bottomrule")
    lines.append("")
    lines.append(r"\toprule")

    for pi, poison in enumerate(POISON_RATES):
        pct = int(round(poison * 100))
        label = (
            r"\textbf{0\% Poison (Baseline)}"
            if poison == 0.0
            else rf"\textbf{{{pct}\% Poison}}"
        )
        lines.append(r"\multicolumn{5}{c}{" + label + r"}\\")

        ta10_m, ta10_s = _target_acc(frames["cifar10"], poison)
        ta100_m, ta100_s = _target_acc(frames["cifar100"], poison)
        lines.append(
            r"& \multicolumn{2}{c|}{\textbf{Target Acc.}: " + _fmt(ta10_m, ta10_s) + r"} "
            r"& \multicolumn{2}{c}{\textbf{Target Acc.}: " + _fmt(ta100_m, ta100_s) + r"}\\"
        )
        lines.append(r"\midrule")

        for b in BUDGETS:
            cells: list[str] = []
            for ds in DATASETS:
                for m, _ in METRICS:
                    vals = _cell_values(frames[ds], poison, b, m)
                    cell = _fmt(float(vals.mean()), float(vals.std(ddof=1)))
                    if poison != 0.0:
                        base = baseline.get((ds, b, m))
                        if base is not None:
                            cell = _color(cell, float(vals.mean()), *base)
                    cells.append(cell)
            lines.append(f"{b} & " + " & ".join(cells) + r"\\")

        # Section separator: midrule between sections, closing rules at the end.
        if pi < len(POISON_RATES) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append("")
    lines.append(r"\toprule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{center}")
    lines.append(r"\label{tab:trteeval}")
    lines.append(r"\vspace{-0.75cm}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write LaTeX to this path. If omitted, print to stdout.",
    )
    args = parser.parse_args()

    frames = {ds: _load(ds) for ds in DATASETS}
    table = build_table(frames)
    if args.output:
        Path(args.output).write_text(table + "\n")
        print(f"Wrote table to {args.output}")
    else:
        print(table)


if __name__ == "__main__":
    main()
