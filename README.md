# USENIX Security 2026 Artifact: SoK on Colluding Adversaries

Three independent experiments validating the paper's collusion conjectures. Targets all three USENIX Security 2026 badges (Artifacts Available, Functional, Results Reproducible).

| Part | Paper § | Code dir | Result |
|------|---------|----------|--------|
| A | §5.2, Table 5    | `Pois_ModExt/`           | **Negative**: poisoning reduces extraction. |
| B | §5.3, Table 6    | `ModExt_DistInf/`        | Positive: extraction-based shadows improve DistInf. |
| C | §5.4, prose-only | `DtRecon_MemAttDistInf/` | Positive across MIA, AIA, DIA. |

One `pyproject.toml`, one `setup.sh`. The three parts share no state and run in any order.

## Repository layout

```text
artifact/
├── README.md
├── setup.sh                                  One-shot env + data setup (`uv sync` + Part C face data).
├── pyproject.toml                            uv project pinning amuletml==0.5.1.
├── .python-version                           Python 3.11.
├── Pois_ModExt/                              Part A.
│   ├── pois_modext.py                        Main pipeline.
│   ├── run_smoke_test.sh                     ~3 min spot check.
│   ├── run_minimal_full.sh                   Reduced-budget config sweep (~34 min).
│   └── run_all.sh                            Full paper reproduction.
├── ModExt_DistInf/                           Part B.
│   ├── modext_distinf.py                     Main pipeline.
│   ├── generate_table.py                     Renders LaTeX for Table 6.
│   ├── run_smoke_test.sh                     ~10 min UTKFace spot check.
│   ├── run_minimal_full.sh                   Reduced-budget config sweep (~1.5 h).
│   ├── run_all_celeba_ratio045.sh            CelebA α=0.45/0.55.
│   ├── run_all_celeba_ratio0475.sh           CelebA α=0.475/0.525.
│   ├── run_all_utkface.sh                    UTKFace, both α pairs.
│   └── extra_scripts/                        Attribute / subsample sanity checks.
└── DtRecon_MemAttDistInf/                    Part C.
    ├── cifar_recon.py                        Batched init-round Geiping inversion (MIA pool).
    ├── face_recon.py                         Face reconstruction at 64x64 (Geiping default; --method gifd uses StyleGAN2).
    ├── prep_celeba.py                        One-time CelebA -> 64x64 tensor preparation.
    ├── common.py, face_common.py, face_targets.py   Shared pool/data/target utilities.
    ├── mia_overlap.py                        DataRecon -> MIA (LiRA).
    ├── aia_image.py                          DataRecon -> AIA (Duddu-style probe).
    ├── dia_image.py                          DataRecon -> DIA (Suri et al. SaTML 2023 black-box).
    ├── generate_teeval_tables.py             Renders the paper LaTeX tables from results/.
    ├── run_smoke_test.sh                     ~3-5 min MIA spot check (no GIFD/faces).
    ├── run_minimal.sh                        Reduced-budget sweep over every setting.
    └── run_all.sh                            Full paper reproduction (section 5.4).
```

Each pipeline creates `data/`, `models/`, `logs/`, `results/` on first run. All gitignored; reference numbers live in §§A.5, B.5, C.5, not on disk.

## Shared setup

```bash
bash setup.sh
```

`setup.sh` sets up all three parts in one shot: it runs `uv sync` (installing `amuletml==0.5.1` and the pinned torch from the single `pyproject.toml`), verifies the amulet import, and prepares Part C's AIA/DIA face data — it builds the CelebA 64×64 cache and downloads the UTKFace CSV. CIFAR (Parts A/B and Part C's MIA) auto-downloads when those experiments first run. The reproduction reconstructs with the GAN-free geiping method, so **no GIFD or StyleGAN2 checkpoint is required**; that is only for the optional `--method gifd` (§C.2).

---

## Part A · Train-Test Collusion: Poisoning → Model Extraction

**TL;DR.** `bash Pois_ModExt/run_all.sh` reproduces §5.2, Table 5, ~3 days on one A100. Run `bash Pois_ModExt/run_smoke_test.sh` first (~3 min) to confirm the setup works.

### A.1 Claims

> **Main claim (A).** On CIFAR10 and CIFAR100, for every non-zero poisoning rate {5%, 10%, 15%, 20%} and every query budget {2500, 6250, 12500, 25000}, surrogate accuracy and fidelity are at or below the unpoisoned baseline (within std). Poisoning reduces extraction effectiveness.

1. **A-C1.** Every cell on CIFAR100 is below baseline by >1 std (red in paper Table 5).
2. **A-C2.** On CIFAR10, the 2500-query column is within the baseline std band (orange); larger budgets are below (red).
3. **A-C3.** The poisoned target's trigger-test accuracy exceeds 90% at every non-zero poison rate, confirming BadNets implanted the backdoor.

### A.2 Requirements

| Resource | Need |
|----------|------|
| GPU      | One CUDA GPU, ≥12 GB VRAM at batch 128. Reference runs on A100 80 GB. |
| Disk     | ~500 MB datasets + ~10 GB model cache. |
| RAM      | 16 GB. |

`uv sync` covers everything; CIFAR{10,100} pulled from `download.pytorch.org` on first use. Both datasets are research-use only (<https://www.cs.toronto.edu/~kriz/cifar.html>).

#### Time budget (A100 80 GB)

| Step                                                 | Wall time             |
|------------------------------------------------------|-----------------------|
| First CIFAR download (each)                          | ~10 s                 |
| `run_smoke_test.sh`                                  | ~2.5 min              |
| `run_minimal_full.sh`                                | ~34 min               |
| `run_all.sh`, CIFAR10                                | ~1.2 days             |
| `run_all.sh`, CIFAR100                               | ~1.8 days             |
| `run_all.sh`, full                                   | **~3 days**           |

The full-sweep rows are extrapolated from `run_minimal_full.sh` (16 cells at 10 epochs, 34 min). Target training and extraction both scale linearly with `--epochs`, so the 200-epoch full sweep is ~20× per cell; cost is dominated by extraction at the 25k-query budget (~45 min/cell on CIFAR10, ~80 min on CIFAR100). Per `(dataset, seed, poison)` group the target is trained once and reused across the four query budgets.

### A.3 Smoke test

```bash
cd Pois_ModExt
bash run_smoke_test.sh        # ~3 min: confirms the setup works.
bash run_minimal_full.sh      # ~34 min: may hint at trends, too short to reproduce the table.
```

### A.4 Full reproduction (paper §5.2, Table 5)

```bash
cd Pois_ModExt
bash run_all.sh
```

Iterates `dataset ∈ {cifar10, cifar100} × exp_id ∈ {0, 1, 2} × poison ∈ {0.0, 0.05, 0.1, 0.15, 0.2} × query_size ∈ {1, 0.5, 0.25, 0.1}`. Models cache to `models/`; an interrupted run resumes from the cache. Output appends to `results/pois_modext_results_{dataset}.csv`. The `--poisoned_portion` argument and the paper headers (`0.05%`, `0.1%`, ...) are fractions of the training set (5%, 10%, 15%, 20%).

### A.5 Comparing to the paper

Reference cells from Table 5, mean ± std over 3 seeds, in %. CIFAR100 non-baseline cells are red (below baseline − std); CIFAR10 non-baseline cells are red except the 2500-query column, which is orange (within std). Reproduced means may drift 1–2 pp from cuDNN non-determinism.

**CIFAR10** (acc. / fid.; trigger-test accuracy exceeds 90% at every non-zero poison rate, confirming A-C3):

| Poison \ Budget        | 2500          | 6250          | 12500         | 25000         |
|------------------------|---------------|---------------|---------------|---------------|
| **0% (baseline)** Acc. | 76.83 ± 0.71  | 83.41 ± 0.23  | 85.05 ± 0.27  | 85.91 ± 0.43  |
| **0% (baseline)** Fid. | 78.54 ± 0.54  | 87.19 ± 1.09  | 88.82 ± 0.63  | 89.58 ± 1.07  |
| **5%** Acc.            | 75.30 ± 1.12  | 78.04 ± 0.42  | 78.30 ± 0.25  | 78.93 ± 0.26  |
| **5%** Fid.            | 79.84 ± 1.71  | 83.31 ± 0.85  | 83.31 ± 0.31  | 84.39 ± 0.24  |
| **10%** Acc.           | 74.60 ± 0.60  | 75.86 ± 0.23  | 78.29 ± 0.51  | 78.00 ± 0.30  |
| **10%** Fid.           | 78.96 ± 0.58  | 80.59 ± 0.85  | 83.95 ± 0.69  | 83.87 ± 0.53  |
| **15%** Acc.           | 75.29 ± 0.61  | 75.99 ± 1.56  | 76.33 ± 1.55  | 77.11 ± 0.86  |
| **15%** Fid.           | 79.16 ± 0.22  | 81.17 ± 1.72  | 81.63 ± 0.73  | 82.54 ± 1.43  |
| **20%** Acc.           | 73.03 ± 1.32  | 75.22 ± 0.70  | 75.41 ± 0.70  | 75.65 ± 0.95  |
| **20%** Fid.           | 76.86 ± 1.78  | 80.20 ± 1.74  | 82.27 ± 0.44  | 82.27 ± 0.44  |

**CIFAR100** (every non-baseline cell below baseline − std):

| Poison \ Budget        | 2500          | 6250          | 12500         | 25000         |
|------------------------|---------------|---------------|---------------|---------------|
| **0% (baseline)** Acc. | 46.99 ± 1.59  | 53.04 ± 1.07  | 54.67 ± 0.84  | 56.09 ± 0.88  |
| **0% (baseline)** Fid. | 54.75 ± 1.16  | 62.63 ± 0.92  | 65.40 ± 0.70  | 66.55 ± 0.90  |
| **5%** Acc.            | 33.25 ± 0.29  | 35.00 ± 0.13  | 35.34 ± 1.21  | 34.80 ± 0.55  |
| **5%** Fid.            | 37.24 ± 0.60  | 39.90 ± 0.59  | 40.81 ± 1.22  | 40.14 ± 1.41  |
| **10%** Acc.           | 32.54 ± 1.34  | 35.56 ± 1.85  | 35.00 ± 3.05  | 34.76 ± 1.34  |
| **10%** Fid.           | 36.96 ± 0.99  | 40.49 ± 1.43  | 40.11 ± 3.20  | 40.47 ± 0.87  |
| **15%** Acc.           | 33.25 ± 1.23  | 34.23 ± 1.71  | 34.89 ± 1.11  | 34.99 ± 1.63  |
| **15%** Fid.           | 38.20 ± 0.87  | 39.72 ± 1.38  | 41.32 ± 0.78  | 40.98 ± 0.96  |
| **20%** Acc.           | 29.78 ± 0.57  | 31.65 ± 0.97  | 31.67 ± 1.69  | 31.69 ± 1.83  |
| **20%** Fid.           | 35.25 ± 1.25  | 37.83 ± 1.27  | 38.34 ± 1.34  | 37.51 ± 1.17  |

### A.6 Mapping to paper claims

| Paper element                  | CSV column              | Code path |
|--------------------------------|-------------------------|-----------|
| Surrogate accuracy             | `stolen_acc_test`       | `amulet.utils.get_accuracy(attack_model, ...)` |
| Surrogate fidelity             | `fidelity`              | `amulet.unauth_model_ownership.metrics.evaluate_extraction` |
| Poison rate row                | `poisoned_portion`  | `--poisoned_portion` |
| Query budget column            | `query_size_records`    | `--query_size` |
| BadNets backdoor               | (not in CSV)            | `amulet.poisoning.attacks.BadNets` |
| KnockoffNets-style extraction  | (not in CSV)            | `amulet.unauth_model_ownership.attacks.ModelExtraction` |
| Target trigger acc. (A-C3)     | `target_acc_poisoned`   | `get_accuracy(target_model, poisoned_test_loader, ...)` |

---

## Part B · Test-Time Collusion: Model Extraction → Distribution Inference

**TL;DR.** The three `bash ModExt_DistInf/run_all_*.sh` scripts reproduce §5.3, Table 6, ~12 days on one A100 or ~5.7 days across three GPUs. Run `bash ModExt_DistInf/run_smoke_test.sh` first (~10 min) to confirm the setup works.

### B.1 Claims

> **Main claim (B).** Shadow populations extracted from the victim (Cross-Arch or Same-Arch) yield substantially higher distinguishing accuracy on DistInf than independently trained shadows, across CelebA and UTKFace and both α-pairs.

1. **B-C1.** Independent-shadow DistInf is above chance (>50%) on all four (dataset, α) columns.
2. **B-C2.** VGG11 shadow extracted from a ResNet34 victim is strictly above baseline + std on all four columns.
3. **B-C3.** ResNet34 shadow extracted from a ResNet34 victim matches or exceeds Cross-Arch and is strictly above baseline + std on all four columns.
4. **B-C4.** B-C2 and B-C3 hold at both α=0.45/0.55 and α=0.475/0.525.

### B.2 Requirements

| Resource | Need |
|----------|------|
| GPU  | One CUDA GPU, ≥24 GB VRAM comfortable / ≥12 GB requires smaller batch. Reference runs on A100 80 GB. |
| Disk | ~5 GB datasets + ~3 GB model cache. |
| RAM  | 16 GB. |

`uv sync` covers everything; CelebA + UTKFace pulled from Google Drive via gdown on first use. CelebA (<https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>) and UTKFace (<https://susanqq.github.io/UTKFace/>) are both research-use only. Sensitive-attribute choices (CelebA `Male`, UTKFace `race`) follow the DistInf benchmark.

#### Time budget (A100 80 GB)

| Step                                            | Wall time             |
|-------------------------------------------------|-----------------------|
| First CelebA download                           | 5–10 min              |
| First UTKFace download                          | 1–2 min               |
| `run_smoke_test.sh`                             | ~10 min               |
| `run_minimal_full.sh`                           | ~1.5 h                |
| `run_all_utkface.sh` (5 exp_ids)                | ~16 h                 |
| `run_all_celeba_ratio045.sh` (5 exp_ids)        | ~5.7 days             |
| `run_all_celeba_ratio0475.sh` (5 exp_ids)       | ~5.6 days             |
| **All three, sequential**                       | **~12 days**          |

The three scripts can run on separate GPUs, cutting wall-clock to the slowest single script (~5.7 days). Per-cell averages (one cell = one `(setting, ratio, exp_id)`); S2 and S3 are slower than S1 because they additionally extract from the victim:

| Dataset / α-pair                       | S1 (Baseline) | S2 (Cross-Arch) | S3 (Same-Arch) | 1 exp_id (all 3) |
|----------------------------------------|---------------|-----------------|----------------|------------------|
| CelebA α=0.45/0.55                     | ~7h 01min     | ~8h 20min       | ~11h 47min     | ~27h 08min       |
| CelebA α=0.475/0.525                   | ~6h 57min     | ~8h 15min       | ~11h 44min     | ~26h 56min       |
| UTKFace (both α pairs, inner-loop avg) | ~27 min       | ~30 min         | ~41 min        | ~3h 16min        |

### B.3 Smoke test

```bash
cd ModExt_DistInf
bash run_smoke_test.sh        # ~10 min: confirms the setup works.
bash run_minimal_full.sh      # ~1.5 h: may hint at trends, too short to reproduce the table.
```

### B.4 Full reproduction (paper §5.3, Table 6)

```bash
cd ModExt_DistInf
bash run_all_celeba_ratio045.sh    # results/collusion_results_045.csv
bash run_all_celeba_ratio0475.sh   # results/collusion_results_0475.csv
bash run_all_utkface.sh            # results/collusion_results_utkface.csv
```

Each script loops `exp_id ∈ {0,1,2,3,4} × setting ∈ {1,2,3}`. Training is deterministic in `exp_id`; the three scripts can run in parallel (e.g. `CUDA_VISIBLE_DEVICES=0 bash run_all_celeba_ratio045.sh &`). When all three finish, render the table:

```bash
uv run python generate_table.py --output table.tex   # --metric auc_score for AUC
```

### B.5 Comparing to the paper

Reference cells from Table 6, mean ± std over 5 seeds, in %. Bold = above baseline + std (green in paper). Reproduced means may drift by 1–3 pp from cuDNN non-determinism; the qualitative pattern (six green cells) is robust.

| Setting    | CELEBA α=0.45/0.55  | CELEBA α=0.475/0.525 | UTKFACE α=0.45/0.55 | UTKFACE α=0.475/0.525 |
|------------|---------------------|----------------------|---------------------|-----------------------|
| Baseline   | 70.00 ± 14.58       | 55.00 ± 18.11        | 55.00 ± 3.06        | 45.50 ± 10.81         |
| Cross-Arch | **98.00 ± 3.26**    | **96.50 ± 4.18**     | **91.50 ± 4.54**    | **88.00 ± 7.79**      |
| Same-Arch  | **99.00 ± 1.37**    | **98.50 ± 3.35**     | **93.00 ± 7.37**    | **92.50 ± 4.68**      |

### B.6 Mapping to paper claims

| Paper element             | Setting → CSV column      | Code path |
|---------------------------|---------------------------|-----------|
| Baseline                  | `setting=1`               | `_train_population` for both pops |
| Cross-Arch                | `setting=2`               | `_train_population` victim + `_extract_population` VGG11 shadow |
| Same-Arch                 | `setting=3`               | `_train_population` victim + `_extract_population` ResNet34 shadow |
| Distinguishing accuracy   | `distinguishing_accuracy` | `amulet.distribution_inference.attacks.SuriEvans2022` |
| α₁ / α₂                   | `ratio1`, `ratio2`        | `prepare_distribution_splits` |
| Sensitive attr. (CelebA)  | `filter_column=Male`      | default in `modext_distinf.py` |
| Sensitive attr. (UTKFace) | `filter_column=race`      | set by `run_all_utkface.sh` |

---

## Part C · Test-Time Collusion: Data Reconstruction → {MIA, AIA, DIA}

**TL;DR.** `bash DtRecon_MemAttDistInf/run_all.sh` reproduce the §5.4 results (reported as prose in the paper, no numbered tables). Run `bash DtRecon_MemAttDistInf/run_smoke_test.sh` first (~3-5 min, MIA only) to confirm the setup. MIA needs only CIFAR; AIA and DIA additionally need the face data that `setup.sh` prepares (§C.2). The reproduction reconstructs with the GAN-free geiping method, so GIFD/StyleGAN2 is not required.

### C.1 Claims

> **Main claim (C).** Augmenting `D_aux` with `D_train` records reconstructed via DataRecon (Geiping NeurIPS 2020 gradient inversion; the `--method gifd` StyleGAN2 variant of Fang et al. ICCV 2023 is optional) improves MIA, AIA, and DIA accuracy over the disjoint-`D_aux` baseline. The effect is monotone for AIA and DIA; for MIA it is positive at 25-50% replacement and saturates by 75%.

1. **C-C1 (DataRecon → MIA).** On CIFAR10 and CIFAR100, LiRA TPR@FPR=0.01 exceeds baseline at 25% and 50% replacement; may regress at 75% as reconstruction error accumulates.
2. **C-C2 (DataRecon → AIA).** On CelebA and UTKFace, attribute-inference AUC increases monotonically with replacement ratio.
3. **C-C3 (DataRecon → DIA).** On CelebA and UTKFace, blackbox DistInf accuracy for α₁=0.5 vs. α₂∈{0.1, 0.9} increases (near-)monotonically with replacement ratio.

### C.2 Requirements

| Resource | Need |
|----------|------|
| GPU  | One CUDA GPU. MIA (ResNet-34 LiRA) and batched CIFAR reconstruction run in ≥16 GB; the 64×64 face reconstruction (geiping) is the heaviest step and wants ≥24 GB (lower the recon batch size to fit less). Reference runs on A100 80 GB. |
| Disk | ~2 GB datasets (CIFAR + CelebA + UTKFace) + ~5 GB reconstruction pools and model cache. ~7 GB total. |
| RAM  | 16 GB. |

`uv sync` (run by `setup.sh`) installs everything the reproduction needs: torch and amuletml (all MIA needs) plus `datasets` (for the CelebA cache used by AIA/DIA). The reproduction reconstructs with the GAN-free geiping method (`run_all.sh` passes `--method geiping`), so it needs **no GIFD and no StyleGAN2 checkpoint**.

GIFD (Fang et al., ICCV 2023) is required *only* for the optional `--method gifd`, which `run_all.sh` does not use. To enable it, install the extra and set up the external checkout:

1. `uv sync --extra gifd` — adds the `setuptools` + `ninja` build backend.
2. Clone GIFD into `DtRecon_MemAttDistInf/GIFD` (or point `$GIFD_ROOT` at an existing checkout): `git clone https://github.com/ffhibnese/GIFD GIFD`.
3. Download the StyleGAN2-FFHQ generator `stylegan2-ffhq-config-f.pt` into `GIFD/inversefed/genmodels/stylegan2_io/`.
4. On first `--method gifd` import the StyleGAN2 ops JIT-compile two CUDA kernels (`fused`, `upfirdn2d`) via `torch.utils.cpp_extension`, which additionally needs a system CUDA toolkit (`nvcc`, on torch's CUDA 12.4 line) and a C++ compiler.

Face data: `setup.sh` prepares this for AIA/DIA — it builds the CelebA 64×64 cache (`prep_celeba.py` → `data/celeba/celeba_64.pt` + `celeba_attr.npz`) and downloads `data/utkface/utkface.csv`. To do it by hand instead, run `uv run python prep_celeba.py` and place the UTKFace CSV at `data/utkface/utkface.csv`. Override the data root with `$PARTC_DATA`.

Face targets: the AIA target is a ResNet-18 trained per `(dataset, seed)` on `D_train` (§5.4). `run_all.sh` and `run_minimal.sh` train it automatically before the sweep, so no manual step is needed.

Datasets: CIFAR{10,100} (MIA), CelebA + UTKFace (AIA, DIA). Same licenses as Parts A and B.

Task / sensitive-attribute pairings (§5.4):

- AIA, CelebA: target=`Smiling`, sensitive=`Male`.
- AIA, UTKFace: target=`sex`, sensitive=`race`.
- DIA, sensitive: `Male` (CelebA) / `sex` (UTKFace); α₁=0.5 vs α₂∈{0.1, 0.9}.
- MIA, CIFARs: standard random member / non-member split.

#### Time budget (A100 80 GB)

MIA and CIFAR-reconstruction timings are measured; AIA/DIA and face-reconstruction timings are estimated. The three attack blocks are independent and can run on separate GPUs.

| Step                                                       | Wall time            |
|------------------------------------------------------------|----------------------|
| `run_smoke_test.sh`                                        | ~3-5 min             |
| `run_minimal.sh`                                           | ~30-60 min           |
| CIFAR reconstruction, per dataset (100 records, batched)   | ~10-15 min           |
| Full DataRecon → MIA (n=1 seed; 4 overlaps × 2 datasets)   | ~8 h (≈1 h/cell) |
| Face reconstruction (geiping@64²), per 100-record pool     | several hours |
| Full DataRecon → AIA (6 recon pools + 120 probe cells)     | ~1-3 days, recon-dominated |
| Full DataRecon → DIA (reuses target-0 pools + 48 cells)    | ~half a day |
| **Full Part-C reproduction (sequential)**                  | **~3-5 days** |

### C.3 Smoke test

`run_smoke_test.sh` exercises the cheapest end-to-end path: reconstruct 16 CIFAR10 records, then run one MIA cell (overlap 0.5, 2 shadows, 2 epochs). It touches the whole MIA stack (amulet import, GIFD-free reconstruction, LiRA, metrics) in ~3-5 min and needs no GIFD or face data. It does not reproduce any paper number.

```bash
cd DtRecon_MemAttDistInf
bash run_smoke_test.sh        # ~3-5 min: confirms the setup works (MIA path).
bash run_minimal.sh           # ~30-60 min: reduced budget over every setting; hints at trends.
```

### C.4 Full reproduction (paper §5.4)

```bash
cd DtRecon_MemAttDistInf
bash run_all.sh                                  # all three attacks, sequential
uv run python generate_teeval_tables.py          # render the paper LaTeX tables from results/
```

`run_all.sh` runs three independent blocks, each reconstructing then sweeping; split them across GPUs by running the blocks separately. One cell = `(attack, dataset, overlap[, α₂][, seed])`. Layout:

```text
DtRecon_MemAttDistInf/
├── cifar_recon.py / face_recon.py / prep_celeba.py   # reconstruction + face-data prep
├── common.py / face_common.py / face_targets.py      # shared pool, data, target utilities
├── mia_overlap.py / aia_image.py / dia_image.py       # one pipeline per attack
├── generate_teeval_tables.py                          # renders the paper tables
├── run_smoke_test.sh / run_minimal.sh / run_all.sh
├── recon_pools/                                       # reconstructions (created on first run)
└── results/                                           # per-attack CSVs (created on first run)
```

Each attack replaces an overlap-fraction `p ∈ {0, 0.25, 0.5, 0.75}` of the adversary's auxiliary data (`D_aux`) with `D_train` records recovered by DataRecon, then trains its shadow/probe models on the mixed set (§5.4). At `p=0` the adversary's data is disjoint from `D_train`.

The three attack blocks in `run_all.sh` are independent and can run on separate GPUs: the MIA and AIA blocks each reconstruct their own pools, and the DIA block rebuilds the seed-0 reconstruction pool itself if the AIA block has not already produced it (cf. §B.4).

### C.5 Comparing to the paper

Reference cells from §5.4. Bold = above baseline (green in the paper). AIA and DIA are means ± std over five and three runs respectively; MIA reports the §5.4 single-run TPRs. Reproduced means may drift slightly from cuDNN nondeterminism.

**DataRecon → MIA** (TPR@FPR=0.01, %):

| Replacement   | CIFAR10  | CIFAR100 |
|---------------|----------|----------|
| 0% (baseline) | 1.50     | 2.50     |
| 25%           | **2.50** | **2.70** |
| 50%           | **3.50** | **2.80** |
| 75%           | **2.10** | **2.60** |

C-C1: 25% and 50% exceed baseline on both datasets; 75% may regress on CIFAR10.

**DataRecon → AIA** (AUC, %):

| Replacement   | CELEBA           | UTKFACE          |
|---------------|------------------|------------------|
| 0% (baseline) | 69.25 ± 4.18     | 62.94 ± 2.81     |
| 25%           | 71.45 ± 3.13     | 64.85 ± 2.79     |
| 50%           | **73.07 ± 3.38** | **66.03 ± 1.59** |
| 75%           | **73.95 ± 3.47** | **67.42 ± 1.62** |

C-C2: monotone increase; 50% and 75% cells exceed baseline + std on both datasets.

**DataRecon → DIA** (attack accuracy, %):

| Replacement   | UTKFACE α₂=0.1   | UTKFACE α₂=0.9   | CELEBA α₂=0.1    | CELEBA α₂=0.9    |
|---------------|------------------|------------------|------------------|------------------|
| 0% (baseline) | 62.66 ± 7.94     | 60.31 ± 3.19     | 60.47 ± 3.97     | 63.44 ± 3.19     |
| 25%           | **74.06 ± 2.84** | **72.50 ± 3.33** | 64.53 ± 6.67     | **80.78 ± 2.25** |
| 50%           | **76.72 ± 4.47** | **72.81 ± 5.62** | **68.91 ± 5.48** | **82.81 ± 3.22** |
| 75%           | **76.09 ± 4.93** | **75.31 ± 3.25** | **71.09 ± 6.03** | **87.66 ± 3.60** |

C-C3: all twelve non-baseline cells at or above baseline + std. The CelebA α₂=0.1 cell at 25% sits within the baseline std band (orange); that does not break the claim.

### C.6 Mapping to paper claims

| Paper element                          | Where to read it                          | Code path |
|----------------------------------------|-------------------------------------------|-----------|
| DataRecon reconstruction (CIFAR)       | `recon_pools/<ds>_seed0/rec_*.pt`         | `cifar_recon.py` (batched init-round Geiping inversion) |
| DataRecon reconstruction (faces)       | `recon_pools/<ds>_seed*_geiping/rec_*.pt` | `face_recon.py` (geiping inverting-gradients at 64²; `--method gifd` StyleGAN2 optional) |
| Replacement / overlap ratio `p`        | `overlap_p` column / `--overlap[_p]`      | injected into shadow data by `build_gifd_pool_image` (MIA) and the per-attack pool builders (AIA/DIA) |
| **C-C1** MIA TPR@FPR=0.01              | `metric=lira_offline_tpr_at_fpr`          | `amulet.membership_inference.attacks.LiRA` + `compute_mi_metrics` in `mia_overlap.py` |
| MIA AUC                                | `metric=lira_online_auc`                  | same |
| **C-C2** AIA AUC                       | `auc` column                              | Duddu CIKM 2022 probe extended with penultimate features (Song & Shmatikov CCS 2019, Liu et al. USENIX 2022) in `aia_image.py`; `sklearn.roc_auc_score` |
| AIA target / sensitive attribute       | `--target_task` / `--z_attr`              | `face_common.load_celeba` (Smiling/Male), `load_utkface` (sex/race) |
| **C-C3** DIA distinguishing accuracy  | `metric=meta_acc`                         | logistic-regression meta-classifier on per-subgroup sorted losses in `dia_image.py` (Suri et al. SaTML 2023 black-box; α-ratios per Suri & Evans PETS 2022) |
| DIA α₁ (victim) vs α₂                  | `--task lo` (α₂=0.1) / `--task hi` (α₂=0.9)| `ALPHA_REF=0.5`, `ALPHA1_LOOKUP` in `dia_image.py` |

The shared collusion knob across all three attacks is the overlap `p`: at `p=0` the adversary's shadow/probe data is disjoint from `D_train`; at `p>0` an overlap-fraction is replaced by reconstructed `D_train` records. C-C1 holds where the moderate-overlap cells exceed baseline (MIA, with the 75% regression discussed in §5.4); C-C2 and C-C3 hold where the bolded cells in §C.5 exceed baseline + std.

---

## Artifact appendix PDF

USENIX requires a ≤3-page LaTeX appendix restating claims, requirements, and the comparison procedure. Source: this README's intro, §§*.1, §§*.2, §§*.3, §§*.4, §§*.5, §§*.6.

## Permanent archive

USENIX's Open-Science policy requires the *Artifacts Available* badge to point at Zenodo, FigShare, Dryad, or Software Heritage (not GitHub). The GitHub repo (paper footnote: <https://github.com/ssg-research/sok-collusion>) is the development mirror.

> **TODO (before Phase-1):** Zenodo DOI `<archive-DOI-here>`. Version-specific DOI of the Phase-1 snapshot; concept DOI added after Phase-2.

The GitHub mirror and the Zenodo deposit are byte-identical at matching tags.
