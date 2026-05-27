# USENIX Security 2026 Artifact: SoK on Colluding Adversaries

This artifact accompanies our USENIX Security 2026 paper on unintended interactions between attacks on machine-learning models. It bundles three independent experiments validating the paper's conjectures about collusion potential, and targets all three USENIX Security 2026 artifact badges (**Artifacts Available**, **Artifacts Functional**, **Results Reproducible**).

| Part | Paper section | Code dir | Claim | Result |
|------|---------------|----------|-------|--------|
| A    | §sec:trteeval (Train-Test) | `Pois_ModExt/`           | Poisoning → Model Extraction | **Negative**: poisoning *reduces* extraction effectiveness. |
| B    | §sec:teeval2 (Test-time)   | `ModExt_DistInf/`        | Model Extraction → Distribution Inference | Positive: extraction-based shadow populations strongly improve DistInf. |
| C    | §sec:teeval3 (Test-time)   | `DtRecon_MemAttDistInf/` | Data Reconstruction → {Membership, Attribute, Distribution} Inference | Positive across all three downstream attacks. |

The three parts share a single Python environment (one top-level `pyproject.toml` pinning `amuletml==0.5.1`, one `uv sync`). They share no on-disk state or runtime dependencies on each other and may be evaluated in any order.

## Repository layout

```text
artifact/
├── README.md
├── pyproject.toml                            uv project pinning amuletml==0.5.1.
├── .python-version                           Python 3.11.
├── Pois_ModExt/                              Part A. Poisoning -> Model Extraction.
│   └── poisoning_modext/
│       ├── pois_modext.py                    Main pipeline (one cell).
│       ├── run_smoke_test.sh                 ~5–15 min spot check on one GPU.
│       └── run_all.sh                        Full reproduction sweep.
├── ModExt_DistInf/                           Part B. ModExt -> DistInf.
│   ├── run_dist_inference_collusion.py       Main pipeline (one cell).
│   ├── run_smoke_test.sh                     ~10 min UTKFace spot check.
│   ├── run_experiments_045.sh                CelebA α=0.45/0.55 (full).
│   ├── run_experiments_0475.sh               CelebA α=0.475/0.525 (full).
│   ├── run_experiments_utkface.sh            UTKFace, both α pairs (full).
│   ├── generate_collusion_table.py           Renders LaTeX Table tab:modextDIA.
│   └── check_{celeba,utkface}_*.py           Auxiliary attribute/subsample checks.
└── DtRecon_MemAttDistInf/                    Part C. DataRecon -> {MIA, AIA, DIA}.
    └── <TODO: co-author to populate>         See §C.4 for the expected layout.
```

Pipeline scripts create three working subdirectories (`data/`, `saved_models/` or `models/`, `logs/`) on first run, plus a result CSV per script. These are all gitignored. The reference numbers from our original runs are reproduced in §§A.5, B.5, and C.5 of this README; the underlying CSVs are not redistributed.

## Shared setup

From the `artifact/` directory:

```bash
uv sync
```

This creates `.venv/` and installs `amuletml==0.5.1` plus its transitive dependencies (torch, torchvision, numpy, pandas, scikit-learn, gdown). All subsequent commands use `uv run` to pick up the venv. Quick sanity check:

```bash
uv run python -c "import amulet; from amulet.datasets import load_celeba, load_utkface; print('amulet ok')"
```

Part C may require additional dependencies for generative gradient inversion (GIFD). The Part C owner will add them to `pyproject.toml` before Phase-1 submission. *(Co-author to update.)*

---

## Part A · Train-Test Collusion: Poisoning → Model Extraction (§sec:trteeval)

### A.1 Claims

The paper validates a **negative** conjecture: \poison and \modelext have *no* collusion potential because poisoning increases the tail length of $\dtrain$ (`Tr1-Tail`), and an extraction surrogate fails to learn the under-represented records. The artifact's job is to show that the surrogate model's accuracy and fidelity do not exceed the unpoisoned baseline.

> **Main claim (A).** On CIFAR10 and CIFAR100, for every non-zero poisoning rate in {5%, 10%, 15%, 20%} and every extraction-query budget in {2500, 6250, 12500, 25000}, the surrogate model's clean-test accuracy and fidelity are at or below the unpoisoned-baseline value (within standard deviation), confirming that \poison reduces \modelext effectiveness.

Concretely:

1. **A-C1 (CIFAR100).** Every (poison rate, budget) cell on CIFAR100 is below baseline by more than one standard deviation (red in paper Table `tab:trteeval`).
2. **A-C2 (CIFAR10).** At the smallest budget (2500 queries), cells are within the baseline standard deviation (orange in paper Table `tab:trteeval`); at every larger budget, cells are below baseline (red).
3. **A-C3 (poisoning was effective).** The poisoned target's accuracy on the trigger-bearing test set is above 90% for every non-zero poison rate on both datasets, confirming the BadNets backdoor was successfully implanted in the target.

Part A passes the reproducibility bar if all three sub-claims hold on the reproduced CSVs.

### A.2 Requirements

#### Hardware

- **GPU:** one CUDA-capable GPU. The reference numbers in §A.5 were produced on **NVIDIA A100 80GB**. The full sweep trains 30 target models (5 poison rates × 3 seeds × 2 datasets) and 120 surrogate models, with no cross-GPU parallelism inside a single cell. ≥ 12 GB VRAM is sufficient at the default batch size of 128 (see §A.7 for OOM workarounds).
- **Disk:** ~500 MB for CIFAR10 + CIFAR100, plus ~10 GB for the model cache after a full reproduction (model checkpoints are saved per (dataset, poison rate, exp_id)).
- **RAM:** 16 GB is comfortable.

#### Software

The shared `uv sync` installs `amuletml==0.5.1` and its transitive deps. No Part A-specific extras. CIFAR10 and CIFAR100 are downloaded by torchvision's default S3 mirror on first use; an outbound HTTPS connection to `download.pytorch.org` is required.

#### Datasets

| Dataset  | License                                                                 |
| -------- | ----------------------------------------------------------------------- |
| CIFAR10  | Research-use; see <https://www.cs.toronto.edu/~kriz/cifar.html>         |
| CIFAR100 | Research-use; see <https://www.cs.toronto.edu/~kriz/cifar.html>         |

#### Time budget

The Part A result CSVs have no `timestamp` column, so we cannot reconstruct historical per-cell timings. The reference run took several days of wall-clock time, but the exact number is unknown without re-measuring. The smoke test in §A.3 will produce a tight enough per-cell estimate to project the full sweep; populate this table after running it once.

| Step                                             | Wall time (A100 80GB) |
| ------------------------------------------------ | --------------------- |
| `uv sync` (covered by Shared setup)              | 1–3 min               |
| First CIFAR10 / CIFAR100 download                | ~30 s each            |
| `run_smoke_test.sh` (2 cells, 5 epochs)          | ~5–15 min *(estimated; measure on first run)* |
| `run_all.sh` (5×4×3×2 = 120 cells, 200 epochs)   | *(TODO: project from smoke)* |

### A.3 Smoke test (recommended first)

```bash
cd Pois_ModExt/poisoning_modext
bash run_smoke_test.sh
```

The smoke test runs two cells on CIFAR10 (`poison ∈ {0.0, 0.1}`, `query_size=0.1`, `exp_id=0`, `epochs=5`) and writes two rows to `pois_modext_smoke.csv`. It is not a paper reproduction. It verifies:

1. `amuletml==0.5.1` imports cleanly.
2. CIFAR10 downloads and processes.
3. Both the clean and poisoned target-training branches run to completion.
4. The KnockoffNets-style extraction branch runs.
5. Evaluation populates `target_acc_test`, `target_acc_poisoned`, `stolen_acc_test`, `fidelity`, `correct_fidelity`, `stolen_acc_poisoned`.

The smoke CSV is written to `pois_modext_smoke.csv`, a separate path from the full-reproduction output (`pois_modext_results_{dataset}.csv`).

### A.4 Full reproduction (paper Table `tab:trteeval`)

```bash
cd Pois_ModExt/poisoning_modext
bash run_all.sh
```

The script iterates `dataset ∈ {cifar10, cifar100} × exp_id ∈ {0, 1, 2} × poison ∈ {0.0, 0.05, 0.1, 0.15, 0.2} × query_size ∈ {1, 0.5, 0.25, 0.1}` (3 seeds, matching the standard deviations in the paper). Output is appended to `pois_modext_results_{dataset}.csv`, written next to the script. If the file already exists from a prior partial run, new rows append; delete the file first if you want a clean CSV.

Trained targets and surrogates are cached under `saved_models/` and re-used across cells whose `(dataset, poison_rate, exp_id)` matches. An interrupted run resumes by re-invoking the script: completed cells skip retraining via the on-disk checkpoint cache.

> **Note on poisoning-rate notation.** The argument `--poisoned_portion` and the script's internal rates {0.05, 0.10, 0.15, 0.20} are fractions of the target training set, i.e. 5%, 10%, 15%, 20%. The paper's Table `tab:trteeval` renders these headers as `0.05%`, `0.1%`, etc.; treat the paper headers as the fractions, not as actual percentages.

### A.5 Comparing to the paper

Reference cells from Table `tab:trteeval` follow (mean ± std over 3 seeds, in %). Color convention: green = strictly above baseline + std (no green cells expected in Part A, since the claim is negative), orange = within the baseline std band, red = below baseline − std.

**CIFAR10.** Surrogate accuracy / fidelity. Target accuracy on the clean test set is shown in the baseline header; on the trigger test set it exceeds 90% for every non-zero poison rate (confirms A-C3).

| Poison \ Budget          | 2500            | 6250            | 12500           | 25000           |
| ------------------------ | --------------- | --------------- | --------------- | --------------- |
| **0% (baseline)** Acc.   | 76.83 ± 0.71    | 83.41 ± 0.23    | 85.05 ± 0.27    | 85.91 ± 0.43    |
| **0% (baseline)** Fid.   | 78.54 ± 0.54    | 87.19 ± 1.09    | 88.82 ± 0.63    | 89.58 ± 1.07    |
| **5%** Acc.              | 75.30 ± 1.12    | 78.04 ± 0.42    | 78.30 ± 0.25    | 78.93 ± 0.26    |
| **5%** Fid.              | 79.84 ± 1.71    | 83.31 ± 0.85    | 83.31 ± 0.31    | 84.39 ± 0.24    |
| **10%** Acc.             | 74.60 ± 0.60    | 75.86 ± 0.23    | 78.29 ± 0.51    | 78.00 ± 0.30    |
| **10%** Fid.             | 78.96 ± 0.58    | 80.59 ± 0.85    | 83.95 ± 0.69    | 83.87 ± 0.53    |
| **15%** Acc.             | 75.29 ± 0.61    | 75.99 ± 1.56    | 76.33 ± 1.55    | 77.11 ± 0.86    |
| **15%** Fid.             | 79.16 ± 0.22    | 81.17 ± 1.72    | 81.63 ± 0.73    | 82.54 ± 1.43    |
| **20%** Acc.             | 73.03 ± 1.32    | 75.22 ± 0.70    | 75.41 ± 0.70    | 75.65 ± 0.95    |
| **20%** Fid.             | 76.86 ± 1.78    | 80.20 ± 1.74    | 82.27 ± 0.44    | 82.27 ± 0.44    |

**CIFAR100.** Every cell below baseline (all red in the paper):

| Poison \ Budget          | 2500            | 6250            | 12500           | 25000           |
| ------------------------ | --------------- | --------------- | --------------- | --------------- |
| **0% (baseline)** Acc.   | 46.99 ± 1.59    | 53.04 ± 1.07    | 54.67 ± 0.84    | 56.09 ± 0.88    |
| **0% (baseline)** Fid.   | 54.75 ± 1.16    | 62.63 ± 0.92    | 65.40 ± 0.70    | 66.55 ± 0.90    |
| **5%** Acc.              | 33.25 ± 0.29    | 35.00 ± 0.13    | 35.34 ± 1.21    | 34.80 ± 0.55    |
| **5%** Fid.              | 37.24 ± 0.60    | 39.90 ± 0.59    | 40.81 ± 1.22    | 40.14 ± 1.41    |
| **10%** Acc.             | 32.54 ± 1.34    | 35.56 ± 1.85    | 35.00 ± 3.05    | 34.76 ± 1.34    |
| **10%** Fid.             | 36.96 ± 0.99    | 40.49 ± 1.43    | 40.11 ± 3.20    | 40.47 ± 0.87    |
| **15%** Acc.             | 33.25 ± 1.23    | 34.23 ± 1.71    | 34.89 ± 1.11    | 34.99 ± 1.63    |
| **15%** Fid.             | 38.20 ± 0.87    | 39.72 ± 1.38    | 41.32 ± 0.78    | 40.98 ± 0.96    |
| **20%** Acc.             | 29.78 ± 0.57    | 31.65 ± 0.97    | 31.67 ± 1.69    | 31.69 ± 1.83    |
| **20%** Fid.             | 35.25 ± 1.25    | 37.83 ± 1.27    | 38.34 ± 1.34    | 37.51 ± 1.17    |

Part A reproduces successfully if a reproduced cell's mean is either below the corresponding baseline mean by more than the baseline std (red, the dominant outcome) or within the baseline std band (orange, expected for CIFAR10 at the 2500 budget). Reproduced means may drift by 1–2 percentage points relative to the published values due to cuDNN non-determinism.

### A.6 Mapping to paper claims

| Paper element                  | CSV column              | Code path                                            |
| ------------------------------ | ----------------------- | ---------------------------------------------------- |
| Surrogate accuracy             | `stolen_acc_test`       | `amulet.utils.get_accuracy(attack_model, test_loader, device)` |
| Surrogate fidelity             | `fidelity`              | `amulet.unauth_model_ownership.metrics.evaluate_extraction` |
| Poison rate row                | `poisoning_percentage`  | `--poisoned_portion` argument (fraction 0–1)         |
| Query budget column            | `query_size` (absolute) | `--query_size` argument (fraction of `D_aux`)        |
| BadNets backdoor               | (not in CSV)            | `amulet.poisoning.attacks.BadNets(trigger_label=1, ...)` |
| Knockoff-style extraction      | (not in CSV)            | `amulet.unauth_model_ownership.attacks.ModelExtraction` |
| Target test acc. (A-C3 sanity) | `target_acc_test`, `target_acc_poisoned` | `get_accuracy(target_model, ...)` on clean and trigger test sets |

### A.7 Troubleshooting

- **`UnpicklingError: Weights only load failed`** when resuming. The script saves whole `nn.Module`s, and PyTorch ≥ 2.6 defaults `torch.load(weights_only=True)`. The patched `pois_modext.py` already passes `weights_only=False` to both `torch.load` calls; if you see this error, your local copy is out of date.
- **CIFAR download fails.** torchvision pulls from `download.pytorch.org`. Retry; the loader skips download if the raw files are already on disk.
- **CUDA out of memory.** Reduce `--batch_size` (default 128). The experiment does not depend on a specific batch size.
- **A single cell crashed.** Re-run `run_all.sh`. Already-completed `(dataset, poison_rate, exp_id, query_size)` cells skip retraining via the model-checkpoint cache; the script resumes from where it stopped.

---

## Part B · Test-Time Collusion: Model Extraction → Distribution Inference (§sec:teeval2)

### B.1 Claims

Part B is built around a single paper table, `tab:modextDIA`:

> **Main claim (B).** A shadow population that an adversary *extracts from the victim* (Cross-Arch or Same-Arch) yields substantially higher distinguishing accuracy on the distribution-inference task than a shadow population trained independently (Baseline), across CelebA and UTKFace and both α-pairs.

Sub-claims, exercised by the full-reproduction scripts in §B.4:

1. **B-C1 (Baseline above chance).** Independent-shadow DistInf achieves distinguishing accuracy above 50% on CelebA and UTKFace at both α-pairs (paper Table `tab:modextDIA`, row "Baseline").
2. **B-C2 (Cross-arch extraction helps).** A VGG11 shadow extracted from a ResNet34 victim is strictly above the Baseline std band on all four (dataset, α) columns.
3. **B-C3 (Same-arch extraction helps at least as much).** A ResNet34 shadow extracted from a ResNet34 victim matches or exceeds Cross-Arch and is strictly above the Baseline std band on all four columns.
4. **B-C4 (Robust across α-pairs).** B-C2 and B-C3 hold for the closer α=0.475/0.525 as well as for α=0.45/0.55.

Part B passes the reproducibility bar if `generate_collusion_table.py` colours all six Cross-Arch and Same-Arch cells green relative to the reproduced Baseline.

### B.2 Requirements

#### Hardware

- **GPU:** one CUDA-capable GPU. Reference numbers were collected on **NVIDIA A100 80GB**. A single GPU suffices for the pipeline itself; the three full-reproduction scripts in §B.4 are independent and can be assigned to separate GPUs. VRAM peak is dominated by the ResNet34 victim populations; ≥ 24 GB is comfortable, ≥ 12 GB requires lowering `--batch_size`.
- **Disk:** ~5 GB for CelebA + UTKFace, plus ~3 GB for the checkpoint cache after a full reproduction.
- **RAM:** 16 GB is comfortable.

#### Software

Covered by the shared `uv sync`. No Part B-specific extras. Dataset loaders fetch CelebA and UTKFace from Google Drive on first use.

#### Datasets

| Dataset | License                                                                |
| ------- | ---------------------------------------------------------------------- |
| CelebA  | Research-use; see <https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>  |
| UTKFace | Non-commercial research; see <https://susanqq.github.io/UTKFace/>      |

Sensitive-attribute choices (CelebA `Male`, UTKFace `race`) follow the standard distribution-inference benchmarks. The auxiliary `check_*_attr_correlation.py` scripts (§B.7) justify them empirically.

#### Time budget

We computed all wall times below from `timestamp`-column deltas in the result CSVs, measured on a single NVIDIA A100 80GB.

| Step                                                    | Wall time             |
| ------------------------------------------------------- | --------------------- |
| `uv sync` (covered by Shared setup)                     | 1–3 min               |
| First CelebA download                                   | 5–10 min              |
| First UTKFace download                                  | 1–2 min               |
| `run_smoke_test.sh` (UTKFace, 1 exp_id, reduced budget) | ~10 min *(estimated)* |
| `run_experiments_utkface.sh` (5 exp_ids, both α pairs)  | **~16 h**             |
| `run_experiments_045.sh` (5 exp_ids, α=0.45/0.55)       | **~5.7 days**         |
| `run_experiments_0475.sh` (5 exp_ids, α=0.475/0.525)    | **~5.6 days**         |
| **Full Part-B reproduction (all three, sequential)**    | **~12 days**          |

The three full-reproduction scripts are independent (separate output CSVs, disjoint `(setting, ratio, exp_id)` keys) and can run in parallel on separate GPUs. The CSVs that produced these numbers were collected from such a parallel multi-GPU run; the per-script wall times listed are what one dedicated A100 would observe.

Per-cell breakdown, averaged across the relevant CSV rows:

| Dataset / α-pair                       | S1 (Baseline) | S2 (Cross-Arch) | S3 (Same-Arch) | 1 exp_id (all 3) |
| -------------------------------------- | ------------- | --------------- | -------------- | ----------------- |
| CelebA α=0.45/0.55                     | ~7h 01min     | ~8h 20min       | ~11h 47min     | ~27h 08min        |
| CelebA α=0.475/0.525                   | ~6h 57min     | ~8h 15min       | ~11h 44min     | ~26h 56min        |
| UTKFace (both α pairs, inner-loop avg) | ~27 min       | ~30 min         | ~41 min        | ~3h 16min         |

Cross-Arch and Same-Arch are 18% and 65% slower than Baseline respectively, because they execute a model-extraction loop against the victim population before running DistInf.

Subsequent runs are checkpoint-cached: a completed `(setting, ratio, exp_id, dataset)` tuple loads from `ModExt_DistInf/models/` and only re-evaluates, taking seconds.

> **Smoke estimate caveat.** The ~10 min figure scales from the full UTKFace per-cell timings above by `(num_models/20) × (epochs/10)` (and `extraction_epochs/5` for settings 2 and 3) plus dataset-load overhead. We have not measured it directly with the reduced-budget arguments; treat it as an order-of-magnitude estimate.

### B.3 Smoke test (recommended first)

```bash
cd ModExt_DistInf
bash run_smoke_test.sh
```

Runs all three settings (Baseline, Cross-Arch, Same-Arch) on **UTKFace** at α=0.45/0.55 with `--num_models 4 --epochs 3 --extraction_epochs 2 --train_subsample 1000 --test_subsample 1000`. UTKFace replaces CelebA in the smoke path because a CelebA smoke would take several hours on the same hardware. Verifies that:

1. `amuletml==0.5.1` imports cleanly.
2. UTKFace downloads from Google Drive and processes.
3. Each of the three pipeline branches runs to completion.
4. A row is appended to `results/collusion_smoke.csv` with `distinguishing_accuracy` and `auc_score` populated.

The smoke CSV is written to a separate path so it does not feed into `generate_collusion_table.py`. Successful smoke output ends with the contents of `results/collusion_smoke.csv` printed to the terminal.

### B.4 Full reproduction (paper Table `tab:modextDIA`)

From `artifact/ModExt_DistInf/`, run the three scripts in any order. They each write to a separate CSV:

```bash
cd ModExt_DistInf
bash run_experiments_045.sh        # results/collusion_results_045.csv
bash run_experiments_0475.sh       # results/collusion_results_0475.csv
bash run_experiments_utkface.sh    # results/collusion_results_utkface.csv
```

Each script loops over `exp_id ∈ {0,1,2,3,4}` (the 5 seeds the paper averages over) and `setting ∈ {1,2,3}` (Baseline / Cross-Arch / Same-Arch). All training is deterministic in `exp_id`; an interrupted run resumes from the cached populations on the next invocation.

The three scripts share no on-disk state and can be assigned to separate GPUs (e.g. `CUDA_VISIBLE_DEVICES=0 bash run_experiments_045.sh &`), cutting wall-clock from ~12 days to roughly the slowest single script (~5.7 days).

When all three scripts finish, render the table:

```bash
uv run python generate_collusion_table.py --output table.tex
```

The default metric is `distinguishing_accuracy`; `--metric auc_score` reports AUC instead.

### B.5 Comparing to the paper

Reference cells from Table `tab:modextDIA` (mean ± std over 5 seeds, in %):

| Setting    | CELEBA α=0.45/0.55  | CELEBA α=0.475/0.525 | UTKFACE α=0.45/0.55 | UTKFACE α=0.475/0.525 |
| ---------- | ------------------- | -------------------- | ------------------- | --------------------- |
| Baseline   | 70.00 ± 14.58       | 55.00 ± 18.11        | 55.00 ± 3.06        | 45.50 ± 10.81         |
| Cross-Arch | **98.00 ± 3.26**    | **96.50 ± 4.18**     | **91.50 ± 4.54**    | **88.00 ± 7.79**      |
| Same-Arch  | **99.00 ± 1.37**    | **98.50 ± 3.35**     | **93.00 ± 7.37**    | **92.50 ± 4.68**      |

Part B reproduces successfully if, for each column, the reproduced Cross-Arch and Same-Arch cells exceed the reproduced Baseline by more than the baseline standard deviation (i.e. `generate_collusion_table.py` colours them green). Exact means may drift by 1–3 percentage points across GPU models due to cuDNN non-determinism; the qualitative pattern is robust.

### B.6 Mapping to paper claims

| Paper element                  | Setting → CSV column         | Code path                                                                            |
| ------------------------------ | ---------------------------- | ------------------------------------------------------------------------------------ |
| Baseline row                   | `setting=1`                  | `_train_population` for both populations                                             |
| Cross-Arch row                 | `setting=2`                  | `_train_population` victim + `_extract_population` shadow with `--shadow_arch vgg`   |
| Same-Arch row                  | `setting=3`                  | `_train_population` victim + `_extract_population` shadow with same arch as target   |
| Distinguishing-accuracy attack | `distinguishing_accuracy`    | `amulet.distribution_inference.attacks.SuriEvans2022`                                |
| α₁ / α₂                        | `ratio1`, `ratio2`           | `prepare_distribution_splits(...)`                                                   |
| Sensitive attribute (CelebA)   | `filter_column=Male`         | default in `run_dist_inference_collusion.py`                                         |
| Sensitive attribute (UTKFace)  | `filter_column=race`         | set by `run_experiments_utkface.sh`                                                  |

### B.7 Optional sanity checks

These do not produce the table but justify the data-preparation choices in the paper:

```bash
uv run python check_celeba_attr_correlation.py
uv run python check_celeba_subsample.py --ratio1 0.45 --ratio2 0.55
uv run python check_utkface_attr_correlation.py
uv run python check_utkface_subsample.py --ratio1 0.45 --ratio2 0.55
```

### B.8 Troubleshooting

- **`amuletml==0.5.1` fails to resolve.** Confirm PyPI has the release with `uv pip show amuletml`. If the only visible version is 0.5.0, force `uv sync --index https://pypi.org/simple/`.
- **gdown fails to download CelebA / UTKFace.** Google Drive sometimes rate-limits anonymous gdown traffic. Retry; the loader skips download if the raw files are already on disk. Last resort: place `list_attr_celeba.txt` and `img_align_celeba/` manually under `ModExt_DistInf/data/celeba/`, and `UTKFace/` under `ModExt_DistInf/data/utkface/`.
- **CUDA out of memory.** Reduce `--batch_size` (default 64).
- **A single `(setting, ratio, exp_id)` cell crashed.** Re-run the same shell script; already-completed cells skip via the checkpoint cache.

---

## Part C · Test-Time Collusion: Data Reconstruction → {MIA, AIA, DIA} (§sec:teeval3)

> **Status:** §C.1 and §C.5 are populated from the paper (Tables `tab:teeval_mia`, `tab:teeval_aia`, `tab:teeval_dia`). Items marked *(TODO)* are for the Part C owner to fill in before Phase-1 submission. The section structure mirrors Parts A and B for reviewer consistency.

### C.1 Claims

Part C exercises three privacy attacks against models trained on data where a fraction of the adversary's auxiliary set `D_aux` has been replaced with records reconstructed from the victim's training set `D_train` via generative gradient inversion (GIFD). The replacement ratio sweeps `{25%, 50%, 75%}` and the baseline is `0%` (disjoint `D_aux`).

> **Main claim (C).** Augmenting `D_aux` with `D_train` records reconstructed via DataRecon improves MIA, AIA, and DIA accuracy relative to the disjoint-`D_aux` baseline. The effect is monotone and consistent for AIA and DIA; for MIA it is positive at moderate replacement (25–50%) and saturates by 75%.

Sub-claims:

1. **C-C1 (DataRecon → MIA).** On CIFAR10 and CIFAR100, TPR@FPR=0.01 under LiRA exceeds the 0% baseline at 25% and 50% replacement, and may regress at 75% as reconstruction error accumulates (paper Table `tab:teeval_mia`).
2. **C-C2 (DataRecon → AIA).** On CelebA (smile / sex) and UTKFace (sex / race), the alignment attribute-inference AUC increases monotonically with replacement ratio (paper Table `tab:teeval_aia`).
3. **C-C3 (DataRecon → DIA).** On CelebA and UTKFace, blackbox DistInf attack accuracy for α₁=0.5 vs. α₂∈{0.1, 0.9} increases (near-)monotonically with replacement ratio (paper Table `tab:teeval_dia`).

### C.2 Requirements

#### Hardware

- **GPU:** *(TODO: co-author to specify the GPU(s) used to produce the paper numbers, including VRAM and count. See §B.2 for the format.)*
- **Disk:** *(TODO: include footprint for CIFAR10, CIFAR100, CelebA, UTKFace plus the model and gradient-inversion checkpoint cache.)*
- **RAM:** *(TODO)*

#### Software

Shared `uv sync` covers `amuletml==0.5.1` and PyTorch. Extra dependency: generative gradient inversion (GIFD). *(TODO: co-author to document the install path. Is GIFD a PyPI package, vendored source, or already pulled by `amuletml`? If extra, add to `pyproject.toml` and `uv.lock`.)*

#### Datasets

| Dataset    | Used by                | License                                                                                               |
| ---------- | ---------------------- | ----------------------------------------------------------------------------------------------------- |
| CIFAR10    | MIA                    | Research-use; see <https://www.cs.toronto.edu/~kriz/cifar.html>                                       |
| CIFAR100   | MIA                    | Research-use; see <https://www.cs.toronto.edu/~kriz/cifar.html>                                       |
| CelebA     | AIA, DIA               | Research-use; see <https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>                                 |
| UTKFace    | AIA, DIA               | Non-commercial research; see <https://susanqq.github.io/UTKFace/>                                     |

Task / sensitive-attribute pairings from §teeval3:

- AIA, CelebA: target = `Smiling`, sensitive = `Male`.
- AIA, UTKFace: target = `gender`, sensitive = `race`.
- DIA, both: sensitive = `Male` / `gender`; ratios α₁=0.5 vs. α₂∈{0.1, 0.9}.
- MIA, both CIFARs: standard random member / non-member split.

#### Time budget

*(TODO: co-author to fill in. Suggested format mirrors §B.2:)*

| Step                                           | Wall time            |
| ---------------------------------------------- | -------------------- |
| `uv sync` (covered by Shared setup)            | 1–3 min              |
| First CIFAR10 / CIFAR100 download              | *(TODO)*             |
| `run_smoke_test.sh` (TBD)                      | *(TODO)*             |
| Full DataRecon → MIA reproduction              | *(TODO)*             |
| Full DataRecon → AIA reproduction              | *(TODO)*             |
| Full DataRecon → DIA reproduction              | *(TODO)*             |
| **Full Part-C reproduction (sequential)**      | *(TODO)*             |

### C.3 Smoke test

*(TODO: co-author to add `run_smoke_test.sh` inside `DtRecon_MemAttDistInf/` that exercises every pipeline branch (one of MIA, AIA, DIA) end-to-end on a small subset of one dataset with a single replacement ratio, in well under one hour. The goal mirrors §A.3 and §B.3: verify install + dataset download + every pipeline branch executes.)*

```bash
cd DtRecon_MemAttDistInf
bash run_smoke_test.sh
```

### C.4 Full reproduction (paper Tables `tab:teeval_mia`, `tab:teeval_aia`, `tab:teeval_dia`)

*(TODO: co-author. Expected layout mirrors Parts A and B:)*

```text
DtRecon_MemAttDistInf/
├── <main_pipeline>.py          # one cell = (attack, dataset, replacement_ratio[, alpha2])
├── run_smoke_test.sh
├── run_experiments_mia.sh      # CIFAR10 + CIFAR100, 0/25/50/75% replacement
├── run_experiments_aia.sh      # CelebA + UTKFace, 0/25/50/75% replacement
├── run_experiments_dia.sh      # CelebA + UTKFace × {α2=0.1, α2=0.9}, 0/25/50/75% replacement
├── generate_teeval_tables.py   # renders the three LaTeX tables from CSV(s)
└── results/                    # per-attack CSVs (gitignored)
```

The MIA / AIA / DIA scripts should be independent so they can be assigned to separate GPUs in the same fashion as §B.4. *(Co-author to confirm.)*

### C.5 Comparing to the paper

Coloring convention follows Parts A and B: green = strictly above baseline + std, orange = within the baseline std band, red = below baseline − std. AIA and DIA use 5-run means ± std; the MIA values reproduced below come from the prose of §teeval3 and are single-run TPRs without std per the source LaTeX. *(Co-author to confirm which version is camera-ready; the second commented-out table in the LaTeX shows monotone-increasing 5-run means with std and a different pattern.)*

**DataRecon → MIA.** Paper Table `tab:teeval_mia`, TPR@FPR=0.01 (%):

| Replacement   | CIFAR10  | CIFAR100 |
| ------------- | -------- | -------- |
| 0% (baseline) | 1.50     | 2.50     |
| 25%           | **2.50** | **2.70** |
| 50%           | **3.50** | **2.80** |
| 75%           | **2.10** | **2.60** |

Reproducibility bar (C-C1): the 25% and 50% cells exceed the 0% baseline on both datasets. The 75% cell may regress on CIFAR10 (saturation discussed in the paper).

**DataRecon → AIA.** Paper Table `tab:teeval_aia`, AUC (%):

| Replacement   | CELEBA          | UTKFACE         |
| ------------- | --------------- | --------------- |
| 0% (baseline) | 69.25 ± 4.18    | 62.94 ± 2.81    |
| 25%           | 71.45 ± 3.13    | 64.85 ± 2.79    |
| 50%           | **73.07 ± 3.38** | **66.03 ± 1.59** |
| 75%           | **73.95 ± 3.47** | **67.42 ± 1.62** |

Reproducibility bar (C-C2): AUC increases monotonically with replacement; 50% and 75% cells exceed baseline + std on both datasets.

**DataRecon → DIA.** Paper Table `tab:teeval_dia`, attack accuracy (%):

| Replacement   | UTKFACE α₂=0.1     | UTKFACE α₂=0.9    | CELEBA α₂=0.1      | CELEBA α₂=0.9      |
| ------------- | ------------------ | ----------------- | ------------------ | ------------------ |
| 0% (baseline) | 62.66 ± 7.94       | 60.31 ± 3.19      | 60.47 ± 3.97       | 63.44 ± 3.19       |
| 25%           | **74.06 ± 2.84**   | **72.50 ± 3.33**  | 64.53 ± 6.67       | **80.78 ± 2.25**   |
| 50%           | **76.72 ± 4.47**   | **72.81 ± 5.62**  | **68.91 ± 5.48**   | **82.81 ± 3.22**   |
| 75%           | **76.09 ± 4.93**   | **75.31 ± 3.25**  | **71.09 ± 6.03**   | **87.66 ± 3.60**   |

Reproducibility bar (C-C3): all twelve non-baseline cells are at or above baseline + std. The one CelebA α₂=0.1 cell at 25% sits within the baseline band in the paper and may reproduce as orange rather than green; that does not break the claim.

### C.6 Mapping to paper claims

*(TODO: co-author. Suggested columns mirror §A.6 and §B.6: paper element → CSV column / row → code path. Populate one row per sub-claim C-C1, C-C2, C-C3 plus rows for the replacement-ratio knob and the GIFD reconstruction algorithm.)*

### C.7 Optional sanity checks

*(TODO: co-author. Candidates: a GIFD reconstruction-visualisation script that dumps a few reconstructed images alongside their originals, and a script that confirms the augmented `D_aux` contains the reconstructions in the expected proportion.)*

### C.8 Troubleshooting

*(TODO: co-author. Likely entries: GIFD installation issues, CIFAR download issues, OOM during gradient inversion, offline dataset-path overrides.)*

---

## Notes for the artifact appendix PDF

The USENIX Security 2026 instructions require a ≤3-page LaTeX artifact appendix that restates claims, hardware/software requirements, and the procedure for comparing reproduced results to the paper. The source material lives in this README's intro, §§A.1 / B.1 / C.1 (Claims), §§A.2 / B.2 / C.2 (Requirements), §§A.3 / B.3 / C.3 (Smoke), §§A.4 / B.4 / C.4 (Full reproduction), §§A.5 / B.5 / C.5 (Comparing to the paper), and §§A.6 / B.6 / C.6 (Mapping to claims).

## Permanent archive

The USENIX Security 2026 Open-Science policy requires the *Artifacts Available* badge to point at a permanent archive on Zenodo, FigShare, Dryad, or Software Heritage. GitHub and GitLab do not satisfy availability. The GitHub repository (the paper footnotes <https://github.com/ssg-research/sok-collusion>) is the development mirror; the canonical, citeable artifact will be deposited at:

> **TODO (before Phase-1 submission): Zenodo DOI `<archive-DOI-here>`.** The version-specific DOI of the Phase-1 snapshot, and (after Phase-2) the concept DOI that resolves to the latest reviewed version, will be filled in here and in the paper's camera-ready artifact appendix.

If you are reading this from the GitHub mirror, the contents are expected to be byte-identical to the Zenodo deposit at the matching tag.
