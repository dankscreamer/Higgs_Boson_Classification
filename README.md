# Higgs Boson Signal Detection: Machine Learning (Phase 1)

Particle event classification with expert features and physics-derived features: a calibrated,
statistically tested machine-learning benchmark on the UCI
[HIGGS](https://archive.ics.uci.edu/dataset/280/higgs) dataset (11M simulated collision events),
following Baldi, Sadowski & Whiteson (2014).

This branch (`ml-only`) contains the **machine-learning part only**: logistic regression and
XGBoost. It has no neural networks or PyTorch.

**Questions.**
1. How much do the 7 physicist-derived masses help a linear model and a tree ensemble at a fixed data budget?
2. Can features derived only from a physical symmetry replace them?
3. What are the calibration and compute trade-offs?

## Headline results

Test ROC AUC on the same 500k held-out events. At 250k events each value is the mean of 5 seeds
(sd ≤ 0.0007). Every difference below is significant in every seed under a paired DeLong test.

| Features | Logistic regression | XGBoost (tuned) |
|---|---|---|
| `low`: 21 raw | 0.596 | 0.738 |
| `geo`: 40 geometric, no masses (ours) | 0.630 | 0.801 |
| `high`: 7 expert masses | 0.645 | 0.787 |
| `all`: raw + expert (the paper's 28) | 0.684 | 0.823 |
| `geo_high`: geometric + expert (47) | 0.692 | **0.838** |

- **Trees beat the linear model on every feature set**, by 0.14–0.17 AUC. A linear decision
  boundary cannot represent angle differences or masses.
- **Symmetry beats expert features.** `geo` uses only pairwise angles and ΔR, with no masses.
  - It improves XGBoost by +0.064 over raw features.
  - It outperforms the 7 expert masses (+0.014).
  - Combined with the masses, it beats the paper's feature set (+0.015).
- **Calibration.**
  - XGBoost is calibrated out of the box (ECE ≈ 0.005).
  - Logistic regression is miscalibrated where it is misspecified (ECE 0.048 on the masses). Platt
    scaling cannot fix that; isotonic regression can (0.006).
- **Compute.** Logistic regression trains in < 0.2 s and tuned XGBoost in 1.5–12 s on 200k events.
- **Scaling (50k → 10.5M events).** Logistic regression is flat, because it is bias-limited.
  XGBoost keeps improving, fastest on raw inputs (0.699 → 0.800).
- **Interpretability.** SHAP recovers the physics: m_bb in the h → bb̄ peak pushes towards signal,
  and m_jjj near the top-quark mass pushes towards background.

Full tables: [`results/summary/`](results/summary/). Report: [`report/phase1_ml.tex`](report/phase1_ml.tex).
Slides (Model Application + Calibration): [`presentation/phase1_ml/`](presentation/phase1_ml/).

## The physics in brief

Each row is one **simulated** proton–proton collision, labelled signal or background.
- **Signal:** a hypothetical heavy Higgs H⁰ → W∓H± → W∓W±h⁰ → W∓W±bb̄.
- **Background:** top-antitop (tt̄) production, which leaves the same visible particles: one lepton,
  missing transverse momentum from a neutrino, and four jets.

| Group | Columns | Meaning |
|---|---|---|
| Low-level (21) | pT, η, φ for the lepton and 4 jets; missing-energy magnitude and φ; b-tag per jet | Reconstructed detector quantities |
| High-level (7) | m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb | Invariant masses designed by physicists |
| Geometric (40, ours) | pT, η, b-tags; \|Δφ\| for all 15 object pairs; ΔR for the 10 pairs with η | Rotation-invariant: the collision is symmetric about the beam, so absolute φ is dropped |

The features are pre-scaled in the UCI release, so masses are not in GeV. A team primer is in
`docs/HIGGS_Physics_Overview_for_Team.docx`. **A high AUC means the simulated classes are separable.
It is not evidence that the particle exists.**

## Quick start

```bash
make setup       # .venv + pinned requirements (Python 3.12 tested; no PyTorch needed)
make data        # get higgs.zip (UCI, ~2.8 GB) and convert once to data/processed/higgs_f32.npy (~1.3 GB)
                 # already have it? .venv/bin/python scripts/prepare_data.py --zip /path/to/higgs.zip
make test        # 12 unit tests on synthetic data, < 10 s, no download needed
make reproduce   # the whole ML study (~1.5 h on an 8 GB M3); step logs in results/logs/
```

The individual steps are `make eda`, `make tune`, `make bench`, `make interpret` and `make aggregate`.

## Experimental protocol

| Item | Choice |
|---|---|
| Source | `higgs.zip` from UCI (dataset 280). Its single member `HIGGS.csv.gz` (SHA-256 `ea302c18…b698`) is streamed without extraction. |
| Test set | The **last 500,000 rows**, as in Baldi et al. Identical for every model, seed and budget. |
| Subset | Stratified sample from rows 0 to 10,499,999 with an explicit seed. Indices and class counts are saved to `data/splits/`. |
| Splits | 80% train / 10% **val** (XGBoost early stopping only) / 10% **cal** (post-hoc calibration only). |
| Leakage control | Labels are removed before any transform. The scaler, φ scale and calibrators are fit on non-test splits. |
| Tuning | Stratified 3-fold CV on seed 0's training split. Logistic regression: grid over C. XGBoost: 12-configuration random search. Settings are frozen for all seeds and budgets. |
| Models | L2 logistic regression; XGBoost (`hist` trees, early stopping on log loss). |
| Metrics | ROC AUC with DeLong 95% CI, PR AUC, accuracy, log loss, Brier score and ECE, each raw and after Platt or isotonic calibration; train and inference time. |
| Statistics | Paired DeLong tests for every pair of models on the shared test set, repeated over seeds. |
| Budgets | 50k (3 seeds), **250k (5 seeds, headline)**, 1M (3 seeds), full 10.5M (1 seed; `low`/`all`). |

## Data quality and EDA (training split only)

- **Clean data:** no NaN or infinite values and no constant columns. The signal fraction is 0.530.
- **Duplicates:** the UCI file contains exact duplicate events. 446 of the 500k test events also
  appear in seed 0's training split, under 0.1%, so the standard test set is kept.
- **Physics checks hold:**
  - Absolute φ is uninformative (AUC 0.500).
  - b-tags take 3 values.
  - The masses of the signal decay chain peak more sharply for signal.
- **No single feature separates the classes:** the best, m_wwbb, reaches AUC 0.575 on its own.

## Repository layout

```
config.yaml                  every hyperparameter, seed, budget and split fraction
src/higgs/data.py            higgs.zip -> memmap, stratified subsets, train/val/cal/test, feature sets
src/higgs/physics.py         rotation-invariant geometric features (|Δφ|, ΔR)
src/higgs/models.py          logistic regression, XGBoost
src/higgs/calibration.py     Platt and isotonic calibration
src/higgs/stats.py           fast DeLong AUC covariance, CIs and paired tests
src/higgs/metrics.py         ROC/PR AUC, Brier, log loss, ECE
src/higgs/eda.py             data-quality and physics sanity checks
src/higgs/plots.py           figures (validated colour-blind-safe palette)
scripts/prepare_data.py      get and verify higgs.zip, convert once
scripts/run_eda.py           EDA and data-quality report
scripts/tune.py              CV hyperparameter selection on training data
scripts/run_benchmark.py     one seed x one budget: both models, calibration, DeLong
scripts/interpret.py         LogReg coefficients, XGBoost TreeSHAP, SHAP dependence
scripts/aggregate.py         mean ± sd over seeds, significance consistency, scaling
scripts/make_tables.py       LaTeX tables and figures for the report (no hand-typed numbers)
scripts/make_presentation_figures.py   slide-ready figures
scripts/reproduce.sh         the whole study in order (make reproduce)
tests/                       12 fast synthetic-data tests
report/                      IEEE report phase1_ml.tex (+ phase1_ml_overleaf.zip for Overleaf)
presentation/phase1_ml/      slides, figures and speaker notes
docs/                        physics primer, viva preparation
```

The large `test_predictions.npz` files are git-ignored and regenerated by the runs. Training times
above 250k events were measured under memory pressure on an 8 GB machine and are not used for
compute comparisons.

## Key references

The full bibliography is in `report/references.bib`.
- Baldi, Sadowski, Whiteson (2014). *Searching for Exotic Particles in High-Energy Physics with Deep Learning.* Nature Communications 5, 4308.
- Chen, Guestrin (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
- Niculescu-Mizil, Caruana (2005). *Predicting Good Probabilities with Supervised Learning.* ICML.
- DeLong, DeLong, Clarke-Pearson (1988). *Comparing the Areas Under Two or More Correlated ROC Curves.* Biometrics 44(3).
- Lundberg et al. (2020). *From Local Explanations to Global Understanding with Explainable AI for Trees.* Nature Machine Intelligence 2.
