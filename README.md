# Higgs Boson Signal Detection

Particle event classification with expert features, physics-derived features and neural networks:
a calibrated, statistically tested benchmark on the UCI
[HIGGS](https://archive.ics.uci.edu/dataset/280/higgs) dataset (11M simulated collision events),
following Baldi, Sadowski & Whiteson (2014).

**Questions.**
1. How much do the 7 physicist-derived masses help classical ML (Phase 1) and a neural network (Phase 2) at a fixed data budget?
2. Can features derived only from a physical symmetry replace them?
3. When does deep learning beat tuned trees?
4. Does a justified ML+DL hybrid beat both?

## Headline results

Test ROC AUC on the same 500k held-out events. At 250k events each value is the mean of 5 seeds
(sd ≤ 0.0011). Every difference discussed below is significant in every seed under a paired DeLong test.

| Features | LogReg | XGBoost (tuned) | MLP | Stack (XGB+MLP) |
|---|---|---|---|---|
| `low`: 21 raw | 0.596 | 0.738 | 0.760 | 0.766 |
| `geo`: 40 geometric, no masses | 0.630 | 0.801 | 0.826 | 0.831 |
| `high`: 7 expert masses | 0.645 | 0.787 | 0.793 | 0.793 |
| `all`: raw + expert (the paper's 28) | 0.684 | 0.824 | 0.831 | 0.835 |
| `geo_high`: geometric + expert (47) | 0.692 | 0.838 | 0.842 | **0.851** |

- **Symmetry beats expert features.** `geo` uses only pairwise angles and ΔR, and no masses. It
  outperforms the 7 expert masses for both XGBoost (+0.014) and the MLP (+0.033). Combined with the
  masses, it beats the paper's feature set for every model.
- **DL vs trees depends on data.** At 50k events XGBoost wins on the richer feature sets. From 250k
  upwards the MLP wins on all of them. Its lead on raw features grows from +0.007 at 50k to +0.023 at
  250k and about +0.05 at 1M–10.5M. On the full pool the MLP on 21 raw features (0.846) nearly matches
  XGBoost given the expert masses (0.851), which reproduces Baldi et al.
- **Calibration.** XGBoost is well calibrated (ECE ≈ 0.005). The MLP is 5–7× worse until Platt
  scaling (≈ 0.003). Logistic regression is miscalibrated where it is misspecified, which only
  isotonic regression fixes.
- **Hybrids.**
  - Stacking (a logistic meta-learner on both models' logits, fit on a held-out split) beats both of
    its base models at every budget, with ECE 0.004 and no post-hoc step. Its weights shift towards
    the MLP as data grow.
  - The physics-informed wide & deep network lost to its matched control: an informative negative
    result.
- **Ablations.**
  - Depth and dropout matter most. Dropout 0.3 beats our 0.1 baseline.
  - BatchNorm helps modestly.
  - Glorot vs default initialisation makes no difference once BatchNorm is used.

Full tables: [`results/summary/`](results/summary/). Write-ups: [`report/`](report/).

## The physics in brief

Each row is one **simulated** proton–proton collision at LHC energies, labelled signal or background.
- **Signal:** a hypothetical heavy Higgs H⁰ → W∓H± → W∓W±h⁰ → W∓W±bb̄.
- **Background:** top-antitop (tt̄) production. It leaves the same visible particles: one charged
  lepton, a neutrino (seen only as missing transverse momentum) and four jets, some of them b-tagged.

| Group | Columns | Meaning |
|---|---|---|
| Low-level (21) | pT, η, φ for the lepton and 4 jets; missing-energy magnitude and φ; b-tag per jet | Reconstructed detector quantities |
| High-level (7) | m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb | Invariant masses designed by physicists; they peak at a parent particle's mass |
| Geometric (40, ours) | pT, η, b-tags; \|Δφ\| for all 15 object pairs; ΔR for the 10 pairs with η | Rotation-invariant re-expression of the low-level features. The collision is symmetric about the beam, so absolute φ is dropped. No masses. |

The features are pre-scaled in the UCI release, so masses are not in GeV. A team primer is in
`docs/HIGGS_Physics_Overview_for_Team.docx`. **A high AUC means the two *simulated* classes are
separable. It is not evidence that the particle exists.**

## Quick start

```bash
make setup       # .venv + pinned requirements (Python 3.12)
make data        # get higgs.zip (UCI, ~2.8 GB) and convert once to data/processed/higgs_f32.npy (~1.3 GB)
                 # already have it? .venv/bin/python scripts/prepare_data.py --zip /path/to/higgs.zip
make test        # 19 unit tests on synthetic data, < 10 s, no download needed
make reproduce   # the whole study (~4 h on an 8 GB M3); step logs in results/logs/
```

Individual steps: `make eda`, `make tune`, `make bench`, `make ablations`, `make interpret` and
`make aggregate`. Then run `.venv/bin/python scripts/make_tables.py` to regenerate the report tables
and figures. Override the budget or scope from the command line:

```bash
.venv/bin/python scripts/run_benchmark.py --subset-size 1000000 --seed 1 --models xgboost mlp --feature-sets low all
```

## Experimental protocol

| Item | Choice |
|---|---|
| Source | `higgs.zip` from UCI (dataset 280). Its single member `HIGGS.csv.gz` (SHA-256 `ea302c18…b698`) is streamed without extraction. |
| Test set | The **last 500,000 rows**, as in Baldi et al. Identical for every model, seed and budget. |
| Subset | Stratified sample of `subset_size` events from rows 0 to 10,499,999 with an explicit seed. Indices and class counts are saved to `data/splits/`. |
| Splits | 80% train / 10% **val** (early stopping only) / 10% **cal** (post-hoc calibration and the stacking meta-learner only). |
| Leakage control | Labels are removed before any transform. The scaler, φ scale, calibrators and stacker are all fit on non-test splits. |
| Tuning | Stratified 3-fold CV on seed 0's training split. Logistic regression: grid over C. XGBoost: 12-configuration random search. Settings are frozen for all seeds and budgets (`results/tuning/`). |
| Models | L2 logistic regression; XGBoost (`hist`, early stopping); MLP (3×256, BatchNorm, ReLU, dropout 0.1, Glorot init, Adam, early stopping on val AUC); wide & deep hybrid; stacked XGB+MLP hybrid. |
| Metrics | ROC AUC with DeLong 95% CI, PR AUC, accuracy, log loss, Brier score and ECE (15 bins), each before and after Platt or isotonic calibration; train and inference time. |
| Statistics | Paired DeLong tests for every pair of models on the shared test set, repeated over seeds. |
| Budgets | 50k (3 seeds), **250k (5 seeds, headline)**, 1M (3 seeds), full 10.5M (1 seed; `low`/`all`). |

## Data quality and EDA (training split only)

These findings are for seed 0 (200k training events); `results/eda/` has the full report.
- **Clean data:** no NaN or infinite values and no constant columns. The signal fraction is 0.530.
- **Duplicates:** the UCI file contains exact duplicate events. 93 repeat within this training split,
  none with conflicting labels, and 446 of the 500k test events also appear in it. That's under
  0.1%, too few to move AUC, so the standard test set is kept for comparability.
- **Physics checks hold:**
  - Absolute φ is uninformative (single-feature AUC 0.500).
  - b-tags take 3 values.
  - m_bb, m_wbb and m_wwbb peak more sharply for signal; the W and top masses peak for both classes.
- **No single feature separates the classes:** the best, m_wwbb, reaches AUC 0.575 on its own.

## Repository layout

```
config.yaml                  every hyperparameter, seed, budget and split fraction
src/higgs/data.py            higgs.zip -> memmap, stratified subsets, train/val/cal/test, feature sets
src/higgs/physics.py         rotation-invariant geometric features (|Δφ|, ΔR)
src/higgs/models.py          logreg, XGBoost, MLP (ablation switches), wide & deep hybrid
src/higgs/hybrid.py          stacking hybrid (+ averaging control)
src/higgs/calibration.py     Platt and isotonic calibration
src/higgs/stats.py           fast DeLong AUC covariance, CIs and paired tests
src/higgs/metrics.py         ROC/PR AUC, Brier, log loss, ECE
src/higgs/eda.py             data-quality and physics sanity checks
src/higgs/plots.py           all figures (validated colour-blind-safe palette)
scripts/prepare_data.py      get and verify higgs.zip, convert once
scripts/run_eda.py           EDA and data-quality report
scripts/tune.py              CV hyperparameter selection on training data
scripts/run_benchmark.py     one seed x one budget: all models, calibration, hybrids, DeLong
scripts/run_ablations.py     MLP ablation study over seeds
scripts/interpret.py         coefficients, TreeSHAP, permutation importance, SHAP dependence
scripts/aggregate.py         mean ± sd over seeds, significance consistency, scaling
scripts/make_tables.py       LaTeX tables and figures for the report (no hand-typed numbers)
scripts/reproduce.sh         the whole study in order (make reproduce)
tests/                       19 fast synthetic-data tests
report/                      IEEE reports: phase1_ml.tex, phase2_dl.tex, references.bib
docs/                        physics primer, viva preparation
```

Each `results/subset<N>_seed<S>/` folder holds `metrics.csv`, `significance.csv`, the figures, and
`config.yaml`, `subset_manifest.json` and `run_info.json` (hardware, versions, epochs, stacking
weights). The large `test_predictions.npz` is git-ignored and regenerated by the run.

## Reports

`report/phase1_ml.tex` (machine learning) and `report/phase2_dl.tex` (deep learning and hybrid) are
IEEE conference papers. To compile, upload the `report/` folder to Overleaf, or run
`latexmk -pdf phase1_ml.tex` with a TeX installation. Tables are generated from `results/` by
`scripts/make_tables.py`.

## Known issue (macOS)

The xgboost wheel links Homebrew's `libomp`, and torch bundles its own copy. With both loaded,
multi-threaded torch CPU ops segfault or deadlock, including the CPU-side `randperm` on the MPS path.
On macOS, torch is therefore pinned to one intra-op thread and the MLP trains on MPS
(`device: auto`), and `models.py` imports xgboost before torch. Linux and CUDA machines are
unaffected. Training times above 250k events were measured under memory pressure on an 8 GB machine
and are not used for compute comparisons.

## References

Key references are listed below; the full bibliography is in `report/references.bib`.
- Baldi, Sadowski, Whiteson (2014). *Searching for Exotic Particles in High-Energy Physics with Deep Learning.* Nature Communications 5, 4308.
- Glorot, Bengio (2010). *Understanding the Difficulty of Training Deep Feedforward Neural Networks.* AISTATS.
- Chen, Guestrin (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
- DeLong, DeLong, Clarke-Pearson (1988). *Comparing the Areas Under Two or More Correlated ROC Curves.* Biometrics 44(3).
