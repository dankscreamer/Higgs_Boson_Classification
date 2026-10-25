# Higgs Boson Signal Detection

Particle event classification with expert features and neural networks: a benchmark on the UCI
[HIGGS](https://archive.ics.uci.edu/dataset/280/higgs) dataset (11M simulated collision events).

**Question.** Linear models and tree ensembles do well on the 7 physicist-derived features (invariant
masses). Can a compact neural network learn equivalent interactions directly from the 21 raw
kinematic measurements, and what does that cost in calibration and compute?
(Baldi, Sadowski & Whiteson, 2014.)

## The physics in brief

Each row is one **simulated** proton-proton collision at LHC energies, labelled signal or background.
- **Signal:** a hypothetical heavy Higgs-like boson H decays through W bosons and a light Higgs h into
  W⁺W⁻bb̄.
- **Background:** top-antitop (tt̄) production, which leaves the same visible particles: one charged
  lepton, a neutrino (seen only as missing transverse momentum) and four jets, some of them b-tagged.

Seeing those particles is therefore not enough. The difference lies in how their momenta, angles and
masses combine.

| Group | Columns | Meaning |
|---|---|---|
| Low-level (21) | pT, η, φ for the lepton and 4 jets; missing-energy magnitude and φ; b-tag per jet | Reconstructed detector quantities |
| High-level (7) | m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb | Invariant masses of particle groups, designed by physicists. They cluster near a parent particle's mass when the group really came from that decay. |
| Geometric (40, ours) | pT, η, b-tags; \|Δφ\| for all 15 object pairs; ΔR for the 10 pairs with η | Rotation-invariant re-expression of the low-level features. The collision is symmetric about the beam, so absolute φ is dropped and only angles between objects are kept. No masses. |

The features in the UCI release are pre-scaled, so masses are not in GeV. A full team primer is in
`docs/HIGGS_Physics_Overview_for_Team.docx`.

**How to read the results:** a high AUC means the classifier separates the two *simulated* classes
as they are defined here. It is not evidence that this particle exists.

## Quick start

```bash
make setup      # .venv + pinned requirements
make data       # get higgs.zip (UCI, ~2.8 GB) and convert once to data/processed/higgs_f32.npy (~1.3 GB)
                # already have it? .venv/bin/python scripts/prepare_data.py --zip /path/to/higgs.zip
make test       # unit tests on synthetic data (no download needed)
make eda        # data-quality report + EDA on the training split -> results/eda/
make bench      # train all models x feature sets on the configured subset
```

## Data quality and EDA (training split only)

`make eda` writes `summary.md`, a quality report, per-feature AUCs and figures. Findings at 250k (seed 42):
- **Clean:** no NaN or infinite values and no constant columns. The signal fraction is 0.530, so we report
  PR AUC alongside accuracy.
- **Duplicates:** the UCI file contains exact duplicate events. There are 129 within the training
  split, none with conflicting labels, and 477 of the 500k test events also appear in training. That is
  too few to move the metrics, so we keep the standard test set to stay comparable with the paper.
- **Physics checks hold:**
  - Absolute φ is uninformative (single-feature AUC ≈ 0.500), as expected from rotational symmetry
    around the beam. Only angle *differences* can matter.
  - b-tags take 3 discrete values.
  - m_bb, m_wbb and m_wwbb are more sharply peaked for signal (the h→bb̄ and H decay chain), while
    m_jj, m_jjj, m_lv and m_jlv (W and top masses) peak more for the tt̄ background.
- **No single feature separates the classes:** the best, m_wwbb, reaches AUC 0.574 alone, while
  combined models exceed 0.83. That gap is the case for multivariate ML.

Override the budget or scope without editing the config:

```bash
.venv/bin/python scripts/run_benchmark.py --subset-size 1000000 --models xgboost mlp --feature-sets low all
```

## Experimental protocol

| Item | Choice |
|---|---|
| Source | `higgs.zip` from UCI (dataset 280). Its single member `HIGGS.csv.gz` has SHA-256 `ea302c18…b698` and is streamed without extraction. |
| Test set | The **last 500,000 rows**, the same held-out set Baldi et al. use. Identical for every model and budget. |
| Training pool | Rows 0 to 10,499,999. |
| Subset | Stratified sample of `subset_size` events (default 250,000) from the pool, with an explicit `seed`. Indices are saved to `data/splits/subset_<N>_seed<S>.npz`, and class counts to the matching `.json` manifest. |
| Validation | 10% of the subset, stratified. Used only for early stopping (XGBoost, MLP). |
| Leakage control | The label is split off before any transform. `StandardScaler` is fit on the training rows only, then applied to val and test. |
| Feature sets | `low`: 21 raw measurements. `geo`: 40 geometric features, no masses (`src/higgs/physics.py`). `high`: 7 expert masses. `all`: 28. Together they run from raw, to simple geometry, to expert masses. |
| `geo` details | UCI standardised every column. φ is uniform on [−π, π], so it is mapped back to radians with a factor fit on the training rows (π / max\|φ\|), and Δφ is wrapped to [0, π]. η's original scale cannot be recovered, but it is already close to unit scale, so ΔR is approximate along η. |
| Models | L2 logistic regression, XGBoost (`hist`, early stopping), MLP: 3x256, BatchNorm, ReLU, dropout 0.1, Glorot init, Adam, early stopping on val ROC AUC. |
| Metrics | ROC AUC, PR AUC (average precision), accuracy, log loss, Brier score, ECE (15 bins), training and inference wall-clock time. |

Every model sees the same rows at a given budget. Report the fixed-budget results first, then scale
`subset_size` (250k, then 1M, then the full 10.5M).

## Results so far: 250k budget, seed 42

All runs are scored on the same 500k test events. Test ROC AUC:

| Features | Logistic regression | XGBoost | MLP |
|---|---|---|---|
| `low`: 21 raw | 0.596 | 0.723 | 0.763 |
| `geo`: 40 geometric, no masses | 0.629 | 0.786 | **0.828** |
| `high`: 7 expert masses | 0.645 | 0.786 | 0.794 |
| `all`: raw + expert | 0.684 | 0.820 | **0.833** |

- **Re-expressing the raw data with the beam's rotational symmetry recovers most of the expert
  features' value, without any masses.** For XGBoost, `geo` matches `high` (0.786 vs 0.786). For the
  MLP, `geo` beats `high` (0.828 vs 0.794) and comes within 0.005 of `all`.
- The MLP gains the most from raw-style inputs: +0.040 over XGBoost on `low` and +0.042 on `geo`.
  That fits Baldi et al.'s claim that networks learn the useful interactions themselves.
- XGBoost is the best calibrated (ECE 0.003–0.008). The MLP's ECE is about 0.025 on every set except
  `high`, where it is 0.004.
- XGBoost trains 3–25× faster. Training times are from one M3 run; see the macOS note below.

These numbers come from a single seed at one budget. Error bars over seeds and the 1M budget come next.

## Outputs

`results/subset<N>_seed<S>/` contains:
- `metrics.csv` with one row per (model, feature set)
- `roc.png`, `reliability.png` and `auc_vs_train_time.png`
- `test_predictions.npz` with test-set probabilities for later analysis (calibration, SHAP)
- `config.yaml`, `subset_manifest.json` and `run_info.json` (hardware, epochs, best iteration) for reproducibility

## Layout

```
config.yaml              all hyperparameters, seed, budget
src/higgs/data.py        CSV to memmap conversion, stratified subset, splits
src/higgs/models.py      logreg / xgboost / MLP fitters
src/higgs/metrics.py     ROC/PR AUC, Brier, ECE
src/higgs/plots.py       report figures
src/higgs/eda.py         data-quality checks, physics sanity checks, EDA figures
src/higgs/physics.py     geometric (Δφ, ΔR) feature set
scripts/prepare_data.py  download and convert
scripts/run_eda.py       EDA on the training split
scripts/run_benchmark.py the experiment
tests/                   fast synthetic-data tests
```

## Known issue (macOS)

The xgboost wheel links Homebrew's `libomp`, and torch bundles its own copy. With both loaded,
multi-threaded torch CPU ops segfault or hang. This includes the CPU-side `randperm` on the MPS path.
On macOS, torch is therefore pinned to one intra-op thread, and the MLP trains on MPS
(`device: auto`). A `device: cpu` run on macOS would understate multi-threaded torch speed.
`models.py` imports xgboost before torch because the reverse order also crashes. Linux and CUDA
machines are unaffected.

## References

- Baldi, Sadowski, Whiteson (2014). *Searching for Exotic Particles in High-Energy Physics with Deep Learning.* Nature Communications 5, 4308.
- Glorot, Bengio (2010). *Understanding the Difficulty of Training Deep Feedforward Neural Networks.* AISTATS.
