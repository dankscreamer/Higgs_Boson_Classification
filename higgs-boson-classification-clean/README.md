# Higgs Boson Classification (Clean Rebuild)

A minimal, reproducible, and technically rigorous machine-learning implementation for binary classification on the UCI HIGGS dataset (signal: $H \to W^+ W^- \to l \nu j j$ vs. background: $t \bar{t}$).

## Project Objectives & Architecture
The goal of this project is to build a scientifically sound, understandable pipeline where every data validation check, exploratory plot, feature transformation, and classification model is mathematically justified.

```
higgs-boson-classification-clean/
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   └── README.md
├── notebooks/
│   └── higgs_analysis.ipynb
├── src/
│   └── higgs/
│       ├── __init__.py
│       ├── data.py          # Schema validation, streaming loader, quality checks, splitting
│       ├── physics.py       # Geometric kinematic transforms (Stage 3)
│       ├── models.py        # Logistic Regression, XGBoost, MLP (Stage 4)
│       └── evaluation.py    # ROC-AUC, threshold metrics, confusion matrices (Stage 6)
├── tests/
│   └── test_data.py         # Automated unit test suite
└── outputs/
    ├── figures/             # Exported publication-quality figures
    └── metrics/             # Exported performance evaluation tables
```

---

## Environment Setup

### 1. Python Environment
Python 3.10+ (tested through Python 3.14 on macOS arm64):

```bash
# Clone or navigate to the repository
cd higgs-boson-classification-clean

# (Optional) Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Acquisition
The UCI HIGGS dataset contains 11 million simulated particle collision events:
- Download `HIGGS.csv.gz` from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz).
- Place `HIGGS.csv.gz` (or `HIGGS.csv` / `higgs.zip`) inside the `data/` directory or project root.
- The loader automatically detects `.csv.gz`, `.csv`, or `.zip` files.

---

## Running Automated Tests

Run the test suite to verify schema definitions, streaming loaders, quality checks, and splitting logic:

```bash
PYTHONPATH=src pytest tests/ -v
```

---

## Interactive Notebook

Launch Jupyter to explore the notebook workflow:

```bash
jupyter lab notebooks/higgs_analysis.ipynb
# or
jupyter notebook notebooks/higgs_analysis.ipynb
```
