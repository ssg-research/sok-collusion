"""
Generate the LaTeX table tab:modextDIA from collusion experiment CSVs.

Reads three result files written by modext_distinf.py:
  - collusion_results_045.csv     (CelebA, ratio 0.45/0.55)
  - collusion_results_0475.csv    (CelebA, ratio 0.475/0.525)
  - collusion_results_utkface.csv (UTKFace, both ratios)

Rows are the three settings (Baseline, Cross-Arch, Same-Arch).
Columns are dataset x ratio pair (CelebA 0.45, CelebA 0.475, UTKFace 0.45,
UTKFace 0.475). Cells display mean and std of the chosen metric in percent.
Non-baseline cells are colored relative to the baseline row at the same column:
  green if mean exceeds baseline mean + std,
  red if mean is below baseline mean - std,
  orange if within the baseline std band.

Missing cells (experiments not yet run) render as '--' with no coloring.

Usage:
    uv run python generate_table.py
    uv run python generate_table.py --metric auc_score
    uv run python generate_table.py --output table.tex
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
CELEBA_045_CSV = RESULTS_DIR / "collusion_results_045.csv"
CELEBA_0475_CSV = RESULTS_DIR / "collusion_results_0475.csv"
UTKFACE_CSV = RESULTS_DIR / "collusion_results_utkface.csv"

# Row order. Setting 1 is the baseline used for coloring the other two rows.
SETTING_LABELS = {
    1: r"\textbf{Baseline.}",
    2: r"\textbf{Cross-Arch}",
    3: r"\textbf{Same-Arch}",
}
SETTINGS = [1, 2, 3]

# Column order. Each entry is (dataset key in CSV, (ratio1, ratio2)).
COLUMNS: list[tuple[str, tuple[float, float]]] = [
    ("celeba", (0.45, 0.55)),
    ("celeba", (0.475, 0.525)),
    ("utkface", (0.45, 0.55)),
    ("utkface", (0.475, 0.525)),
]
DATASET_HEADER = {"celeba": r"\textbf{CELEBA}", "utkface": r"\textbf{UTKFACE}"}


def _load_results() -> pd.DataFrame:
    frames = []
    for csv in (CELEBA_045_CSV, CELEBA_0475_CSV, UTKFACE_CSV):
        if csv.exists():
            frames.append(pd.read_csv(csv))
    if not frames:
        raise FileNotFoundError(
            f"No result CSVs found in {RESULTS_DIR}. "
            "Run the experiment shell scripts first."
        )
    return pd.concat(frames, ignore_index=True)


def _filter(
    df: pd.DataFrame,
    setting: int,
    dataset: str,
    ratio1: float,
    ratio2: float,
    metric: str,
) -> pd.Series:
    mask = (
        (df["setting"] == setting)
        & (df["dataset"] == dataset)
        & (df["ratio1"].round(4) == round(ratio1, 4))
        & (df["ratio2"].round(4) == round(ratio2, 4))
    )
    return df.loc[mask, metric].dropna()


def _fmt_cell(values: pd.Series) -> str:
    if values.empty or values.isna().all():
        return "--"
    mean = values.mean() * 100
    std = values.std(ddof=0) * 100
    if np.isnan(std) or len(values) < 2:
        return f"${mean:.2f}$"
    return f"${mean:.2f}_{{\\pm {std:.2f}}}$"


def _color_cell(
    cell: str, values: pd.Series, base_mean: float, base_std: float
) -> str:
    if cell == "--" or values.empty:
        return cell
    mean = values.mean()
    if mean > base_mean + base_std:
        return r"\cellcolor{green!15}" + cell
    if mean < base_mean - base_std:
        return r"\cellcolor{red!15}" + cell
    return r"\cellcolor{orange!15}" + cell


def build_table(df: pd.DataFrame, metric: str) -> str:
    baseline: dict[tuple[str, float, float], tuple[float, float]] = {}
    for dataset, (r1, r2) in COLUMNS:
        vals = _filter(df, 1, dataset, r1, r2, metric)
        if not vals.empty:
            baseline[(dataset, r1, r2)] = (vals.mean(), vals.std(ddof=0))

    lines: list[str] = []
    lines.append(r"\begin{table}[!htb]")
    lines.append(
        r"\caption{\textbf{\modelext $\rightarrow$ \dia:} "
        r"\emph{Baseline} $\rightarrow$ shadow models trained independently "
        r"without querying $\model$ (target); "
        r"\emph{Cross-Arch} $\rightarrow$ VGG11 shadow model extracted from "
        r"$\model$; "
        r"\emph{Same-Arch} $\rightarrow$ ResNet34 shadow model extracted from "
        r"$\model$. "
        r"\colorbox{green!15}{Green} $\rightarrow$ higher than baseline, "
        r"\colorbox{orange!15}{orange} $\rightarrow$ similar to baseline "
        r"(within std.\ dev.), "
        r"\colorbox{red!15}{red} $\rightarrow$ lower than baseline.}"
    )
    lines.append(r"\label{tab:modextDIA}")
    lines.append(r"\vspace{0.25cm}")
    lines.append(r"\resizebox{\columnwidth}{!}{")
    lines.append(
        r"\begin{tabular}{ @{\hspace{6pt}} l @{\hspace{4pt}} | "
        r"@{\hspace{4pt}} c @{\hspace{4pt}}  @{\hspace{4pt}} c @{\hspace{4pt}} | "
        r"@{\hspace{4pt}} c @{\hspace{4pt}}  @{\hspace{4pt}} c @{\hspace{4pt}}  }"
    )
    lines.append(r"\bottomrule")
    lines.append("")
    lines.append(r"\toprule")

    # Header row 1: dataset spans 2 ratio columns each
    header1 = r"\multirow{3}{*}{\textbf{Setting}}"
    for dataset in ("celeba", "utkface"):
        header1 += f" & \\multicolumn{{2}}{{c}}{{{DATASET_HEADER[dataset]}}}"
    lines.append(header1 + r" \\")

    # Header row 2: alpha_1
    header2 = "".join(
        f" & $\\alpha_1={r1}$" for _, (r1, _) in COLUMNS
    )
    lines.append(header2 + r" \\")

    # Header row 3: alpha_2
    header3 = "".join(
        f" & $\\alpha_2={r2}$" for _, (_, r2) in COLUMNS
    )
    lines.append(header3 + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for setting in SETTINGS:
        row = SETTING_LABELS[setting]
        for dataset, (r1, r2) in COLUMNS:
            vals = _filter(df, setting, dataset, r1, r2, metric)
            cell = _fmt_cell(vals)
            if setting != 1:
                base = baseline.get((dataset, r1, r2))
                if base is not None:
                    cell = _color_cell(cell, vals, *base)
            row += f" & {cell}"
        lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append("")
    lines.append(r"\toprule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\vspace{-0.25cm}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--metric",
        default="distinguishing_accuracy",
        choices=["distinguishing_accuracy", "auc_score"],
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write LaTeX to this path. If omitted, print to stdout.",
    )
    args = parser.parse_args()

    df = _load_results()
    table = build_table(df, args.metric)
    if args.output:
        Path(args.output).write_text(table + "\n")
        print(f"Wrote table to {args.output}")
    else:
        print(table)


if __name__ == "__main__":
    main()
