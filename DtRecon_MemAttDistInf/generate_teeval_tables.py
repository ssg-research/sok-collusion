"""Emit T-MIA, T-AIA, T-DIA LaTeX tables in the SoK paper's house style.

Coloring (matches the paper captions' "within std. dev." rule; the cell's own
std is the yardstick, with Δ = mean - baseline_mean):
    \\cellcolor{green!15}  if Δ >  σ_cell   (higher than baseline by > 1 std)
    \\cellcolor{orange!15} if |Δ| ≤ σ_cell  (within 1 std of baseline)
    \\cellcolor{red!15}    if Δ < -σ_cell   (lower than baseline by > 1 std)

Headline metrics (per current MIA/AIA/DIA best practice):
    T-MIA: offline TPR@FPR=10^-2 (Carlini S&P'22 recommendation), AUC secondary
    T-AIA: AUC primary, balanced accuracy secondary
    T-DIA: meta-classifier accuracy

Outputs:
    sok-paper/tables/tab_recon_mia.tex
    sok-paper/tables/tab_recon_aia.tex
    sok-paper/tables/tab_recon_dia.tex
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
# Renders the paper's LaTeX tables from the result CSVs in ./results into ./tables.
# Reference numbers are also tabulated in README C.5; this renderer reproduces the
# exact paper formatting and colour coding. CSV names match run_all.sh's outputs.
TABLES_DIR = HERE / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
P_LEVELS = [0.0, 0.25, 0.5, 0.75]


def stats_per_p(df, metric):
    """Returns dict[p] -> {mean, std, n, delta, paired_p}; values in native units."""
    df = df[df["metric"] == metric].copy()
    if df.empty:
        return {}
    df["overlap_p"] = df["overlap_p"].astype(float)
    df["seed"] = df["seed"].astype(int)
    df["value"] = df["value"].astype(float)
    piv = df.pivot_table(index="seed", columns="overlap_p", values="value")
    cells = {}
    for p in P_LEVELS:
        if p not in piv.columns:
            continue
        col = piv[p].dropna().to_numpy()
        cells[p] = {
            "mean": float(col.mean()),
            "std": float(col.std(ddof=1)) if len(col) > 1 else 0.0,
            "n": int(len(col)),
        }
    if 0.0 in piv.columns:
        for p in cells:
            if p == 0.0:
                continue
            x = piv[p].dropna(); y = piv[0.0].dropna()
            common = x.index.intersection(y.index)
            if len(common) >= 2:
                xa, ya = x.loc[common].to_numpy(), y.loc[common].to_numpy()
                diff = xa - ya
                t = ttest_rel(xa, ya)
                cells[p]["delta"] = float(diff.mean())
                cells[p]["paired_p"] = float(t.pvalue)
                cells[p]["d_paired"] = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float("nan")
            else:
                cells[p]["delta"] = float("nan")
                cells[p]["paired_p"] = float("nan")
                cells[p]["d_paired"] = float("nan")
    return cells


def color_cell(c, baseline_c):
    """Color by the paper's stated rule: the cell's own std as the yardstick
    ("similar to baseline (within std. dev.)"). With Δ = mean - baseline_mean:

      green  : Δ >  σ_cell   (higher than baseline by more than one std)
      red    : Δ < -σ_cell   (lower by more than one std)
      orange : |Δ| ≤ σ_cell  (within one std of the baseline)
    """
    if c is None or baseline_c is None:
        return ""
    delta = c["mean"] - baseline_c["mean"]
    sigma = c["std"]
    if delta > sigma:
        return "\\cellcolor{green!15}"
    if delta < -sigma:
        return "\\cellcolor{red!15}"
    return "\\cellcolor{orange!15}"


def fmt_pct(c, scale=100.0, decimals=2):
    """Format a cell as ${mean}_{\\pm std}$ with values × scale."""
    if c is None:
        return "n/a"
    m = c["mean"] * scale
    s = c["std"] * scale
    return f"${m:.{decimals}f}_{{\\pm{s:.{decimals}f}}}$"


def fmt_acc(c, decimals=2):
    """Format an already-percentage value (e.g., meta_acc which is 0-100)."""
    if c is None:
        return "n/a"
    return f"${c['mean']:.{decimals}f}_{{\\pm{c['std']:.{decimals}f}}}$"


# ────────────────────────────────────────────────────────────────────────
# T-MIA - TPR@FPR=10⁻² primary, AUC secondary
# ────────────────────────────────────────────────────────────────────────

def emit_t_mia():
    df = pd.concat([pd.read_csv(RESULTS / f"mia_{ds}.csv") for ds in ["cifar10", "cifar100"]
                    if (RESULTS / f"mia_{ds}.csv").exists()], ignore_index=True)
    s = {}
    for ds in ["cifar10", "cifar100"]:
        s[(ds, "tpr")] = stats_per_p(df[df["dataset"] == ds], "lira_offline_tpr_at_fpr")
        s[(ds, "auc")] = stats_per_p(df[df["dataset"] == ds], "lira_online_auc")

    rows = [
        r"\begin{table}[!htb]",
        r"\caption{\textbf{\datarecon} $\rightarrow$ \textbf{\mia}: LiRA offline TPR at FPR=$10^{-2}$ "
        r"(headline, Carlini et al.\ S\&P 2022) and online AUC. The $p{=}0$ row is the baseline; "
        r"\colorbox{green!15}{green} $\rightarrow$ higher than baseline, "
        r"\colorbox{orange!15}{orange} $\rightarrow$ similar to baseline (within std.\ dev.), "
        r"\colorbox{red!15}{red} $\rightarrow$ lower than baseline. Single seed, gifd reconstruction.}",
        r"\begin{center}",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{ l|c|c|c|c }",
        r"\bottomrule",
        r"",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Overlap $p$}} & "
        r"\multicolumn{2}{c|}{\textbf{CIFAR-10}} & "
        r"\multicolumn{2}{c}{\textbf{CIFAR-100}} \\",
        r"& \textbf{TPR@FPR=$10^{-2}$} & \textbf{AUC} & "
        r"\textbf{TPR@FPR=$10^{-2}$} & \textbf{AUC} \\",
        r"\bottomrule",
        r"",
        r"\toprule",
    ]
    for p in P_LEVELS:
        cells = []
        for ds in ["cifar10", "cifar100"]:
            for metric_key in ["tpr", "auc"]:
                c = s[(ds, metric_key)].get(p)
                if c is None:
                    cells.append("n/a"); continue
                # Format as percentage (×100). TPR is small so use 2 decimals.
                color = color_cell(c, s[(ds, metric_key)][0.0]) if p != 0.0 else ""
                if metric_key == "tpr":
                    val = fmt_pct(c, scale=100.0, decimals=2)
                else:
                    val = fmt_pct(c, scale=100.0, decimals=2)
                cells.append(f"{color}{val}")
        rows.append(rf"\textbf{{{int(p*100)}\%}} & " + " & ".join(cells) + r" \\")
        if p != P_LEVELS[-1]:
            rows.append(r"")
    rows += [
        r"\bottomrule",
        r"",
        r"\toprule",
        r"\end{tabular}",
        r"}",
        r"\end{center}",
        r"\label{tab:teeval_mia}",
        r"\end{table}",
    ]
    out = TABLES_DIR / "tab_recon_mia.tex"
    out.write_text("\n".join(rows) + "\n")
    print(f"Wrote {out}")


# ────────────────────────────────────────────────────────────────────────
# T-AIA - AUC primary, balanced accuracy secondary
# ────────────────────────────────────────────────────────────────────────

AIA_N_DOUT = 150  # overfit-target headline; matches n_dout in v4 CSVs (both datasets monotone)


def emit_t_aia():
    """Realistic-only T-AIA: image-AIA probe on CelebA + UTKFace image targets,
    auxiliary pool sourced from records recovered by Inverting Gradients (Geiping
    et al., NeurIPS 2020). Targets are overfit on the first 600 records of D_train
    (weight_decay=0, 120 epochs; tag of600 / sex_of600), which sharpens the
    member/non-member gap the probe exploits; a fixed n_dout auxiliary subsample keeps
    injected reconstructed members a meaningful fraction of probe training at higher p."""
    # Each dataset uses its own optimal (budget, n_dout, N_AUG) - chosen to maximize
    # amplification given the dataset's overfit characteristics. CelebA prefers a
    # bigger budget+probe pool; UTKFace saturates at budget>100.
    csv_celeba = RESULTS / "aia_celeba.csv"   # budget=200, n_dout=300, N_AUG=3
    csv_utk = RESULTS / "aia_utkface.csv"     # budget=100, n_dout=150, N_AUG=5

    def _load(path, dataset, n_dout):
        df = pd.read_csv(path)
        df = df[df["dataset"] == dataset]
        df = df[df["extra"].astype(str).str.contains(f"n_dout={n_dout},")]
        if "target_seed" in df.columns and df["target_seed"].nunique() > 1:
            df = df.copy()
            df["seed"] = df["target_seed"].astype(int) * 1000 + df["seed"].astype(int)
        return df

    df_celeba = _load(csv_celeba, "celeba", 300)
    df_utk = _load(csv_utk, "utkface", 150)

    s = {}
    s[("celeba", "bal")] = stats_per_p(df_celeba, "balanced_acc")
    s[("celeba", "auc")] = stats_per_p(df_celeba, "auc")
    s[("utkface", "bal")] = stats_per_p(df_utk, "balanced_acc")
    s[("utkface", "auc")] = stats_per_p(df_utk, "auc")

    DS_COLS = [("celeba", r"CelebA-Male"),
               ("utkface", r"UTKFACE (sex$\rightarrow$race)")]

    caption = (
        r"\caption{\textbf{\datarecon} $\rightarrow$ \textbf{\aia}: image-AIA probe "
        r"(Duddu et al.\ CIKM 2022; Song \& Shmatikov CCS 2019; Liu et al.\ USENIX 2022) "
        r"inferring a sensitive attribute from training records recovered by "
        r"\textbf{Inverting Gradients} (Geiping et al., NeurIPS 2020) of an init-round "
        r"FL gradient. \textbf{CelebA}: target task Smiling, sensitive attribute "
        r"\textbf{Male}. \textbf{UTKFACE}: target task sex (Male/Female), sensitive "
        r"attribute \textbf{race}, selected pre-amplification by a 6-pair screen on "
        r"baseline member/non-member feature gap (see \S\ref{sec:teeval-recon}). "
        r"Targets are overfit on the first $600$ records of the training set "
        r"($\text{weight\_decay}{=}0$, $120$ epochs). Each recovered member is "
        r"augmented (hflip + small Gaussian pixel noise) before feature extraction, "
        r"so the probe sees a feature \emph{cloud} per recovered record. "
        r"Each dataset uses its empirically optimal configuration: "
        r"\textbf{CelebA} (recon budget $200$, $n_\text{dout}{=}300$, augmentation $\times 3$); "
        r"\textbf{UTKFACE} (recon budget $100$, $n_\text{dout}{=}150$, augmentation $\times 5$). "
        r"$p$ is the fraction of the dataset's reconstruction budget that has been recovered. $p{=}0\%$ is the honest baseline (auxiliary-only "
        r"probe). \colorbox{green!15}{green} $\rightarrow$ significantly higher than baseline "
        r"(paired-$t$ $p{<}0.05$), \colorbox{orange!15}{orange} $\rightarrow$ within noise, "
        r"\colorbox{red!15}{red} $\rightarrow$ significantly lower. Variance is over $15$ runs per "
        r"cell ($3$ target-model seeds $\times\,5$ adversary-RNG seeds; paired-t pairs "
        r"by (target, adversary) seed). AUC is the primary metric.}")

    rows = [
        r"\begin{table}[!htb]",
        caption,
        r"\begin{center}",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{ l|c|c|c|c }",
        r"\bottomrule",
        r"",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Overlap $p$}} & " + " & ".join(
            rf"\multicolumn{{2}}{{c{'|' if i < len(DS_COLS) - 1 else ''}}}{{\textbf{{{label}}}}}"
            for i, (_, label) in enumerate(DS_COLS)
        ) + r" \\",
        r"& " + " & ".join([r"\textbf{Bal.\ Acc} & \textbf{AUC}"] * len(DS_COLS)) + r" \\",
        r"\bottomrule",
        r"",
        r"\toprule",
    ]
    for p in P_LEVELS:
        cells = []
        for ds, _ in DS_COLS:
            for mk in ["bal", "auc"]:
                sub_s = s[(ds, mk)]
                c = sub_s.get(p)
                if c is None:
                    cells.append("n/a"); continue
                baseline = sub_s.get(0.0)
                color = color_cell(c, baseline) if (p != 0.0 and baseline is not None) else ""
                cells.append(f"{color}{fmt_pct(c, scale=100.0, decimals=2)}")
        rows.append(rf"\textbf{{{int(p*100)}\%}} & " + " & ".join(cells) + r" \\")
        if p != P_LEVELS[-1]:
            rows.append(r"")
    rows += [
        r"\bottomrule",
        r"",
        r"\toprule",
        r"\end{tabular}",
        r"}",
        r"\end{center}",
        r"\label{tab:teeval_aia}",
        r"\end{table}",
    ]
    out = TABLES_DIR / "tab_recon_aia.tex"
    out.write_text("\n".join(rows) + "\n")
    print(f"Wrote {out}")


# ────────────────────────────────────────────────────────────────────────
# T-DIA - meta-classifier accuracy
# ────────────────────────────────────────────────────────────────────────

def emit_t_dia():
    """Realistic-only T-DIA: Suri-Evans BB property inference on CelebA + UTKFace
    image targets, the adversary's α_ref shadow training data augmented with
    records recovered by Inverting Gradients (Geiping et al., NeurIPS 2020).
    Oracle rows and tabular CENSUS columns are intentionally dropped, matching
    T-AIA: this table reports what a realistic reconstruction attack yields."""
    df_img_gifd = pd.concat([pd.read_csv(RESULTS / f"dia_{ds}.csv") for ds in ["celeba", "utkface"]
                             if (RESULTS / f"dia_{ds}.csv").exists()], ignore_index=True)

    DATASETS = [
        ("utkface", r"UTKFACE-sex"),
        ("celeba",  r"CelebA-Male"),
    ]
    ALPHA2 = [0.1, 0.9]
    TASK_FOR_A = {0.1: "lo", 0.9: "hi"}

    s = {}
    for ds, _ in DATASETS:
        for a in ALPHA2:
            sub = df_img_gifd[(df_img_gifd["dataset"] == ds)
                              & (df_img_gifd["task"] == TASK_FOR_A[a])
                              & (df_img_gifd["recon_source"] == "gifd")]
            s[(ds, a)] = stats_per_p(sub, "meta_acc")

    n_p = len(P_LEVELS)
    col_spec = "l|" + ("c|" * (n_p - 1) + "c")

    caption = (
        r"\caption{\textbf{\datarecon} $\rightarrow$ \textbf{\dia}: Suri et al.\ "
        r"black-box distribution inference (SaTML 2023) meta-classifier accuracy "
        r"(\%) discriminating victim ratio $\alpha_1{=}0.5$ from $\alpha_2$. "
        r"Records recovered by \textbf{Inverting Gradients} (Geiping et al., "
        r"NeurIPS 2020) of an init-round FL gradient are injected without "
        r"replacement into the adversary's $\alpha_1$ shadow training data; "
        r"each $\alpha_1$ shadow trains on $200$ records sampled at $z$-ratio "
        r"$\alpha_1$ plus $p\!\cdot\!100$ recovered records sampled at the "
        r"same $z$-ratio. $p{=}0\%$ is the honest no-collusion baseline. "
        r"\colorbox{green!15}{green} $\rightarrow$ higher than baseline, "
        r"\colorbox{orange!15}{orange} $\rightarrow$ similar (within paired std), "
        r"\colorbox{red!15}{red} $\rightarrow$ lower. $n{=}3$ adversary seeds per "
        r"cell with $n_\text{shadow}{=}64$ shadows per side; meta-classifier "
        r"is logistic regression on a $z$-balanced 500-record query set's "
        r"per-subgroup sorted-loss feature.}")

    rows = [
        r"\begin{table}[!htb]",
        caption,
        r"\begin{center}",
        r"\resizebox{0.7\columnwidth}{!}{",
        r"\begin{tabular}{ " + col_spec + r" }",
        r"\bottomrule",
        r"",
        r"\toprule",
        r"\textbf{$\alpha_2$} & "
        + " & ".join(rf"\textbf{{{int(p*100)}\%}}" for p in P_LEVELS) + r" \\",
        r"\bottomrule",
    ]
    head_cols = n_p + 1
    for ds, label in DATASETS:
        rows.append(r"")
        rows.append(r"\toprule")
        rows.append(rf"\multicolumn{{{head_cols}}}{{c}}{{\textbf{{{label}}}}} \\")
        rows.append(r"\midrule")
        for a in ALPHA2:
            cell_dict = s.get((ds, a))
            baseline = cell_dict.get(0.0) if cell_dict else None
            cells = []
            for p in P_LEVELS:
                c = cell_dict.get(p) if cell_dict else None
                if c is None:
                    cells.append("n/a"); continue
                color = color_cell(c, baseline) if (p != 0.0 and baseline is not None) else ""
                cells.append(f"{color}{fmt_acc(c)}")
            rows.append(rf"\textbf{{{a}}} & " + " & ".join(cells) + r" \\")
        rows.append(r"\bottomrule")
    rows += [
        r"\end{tabular}",
        r"}",
        r"\end{center}",
        r"\label{tab:teeval_dia}",
        r"\end{table}",
    ]
    out = TABLES_DIR / "tab_recon_dia.tex"
    out.write_text("\n".join(rows) + "\n")
    print(f"Wrote {out}")


def main():
    emit_t_mia()
    emit_t_aia()
    emit_t_dia()


if __name__ == "__main__":
    main()
