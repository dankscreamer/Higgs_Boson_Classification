# HIGGS Dataset Directory

This directory is intended to store the UCI HIGGS dataset files.

## Dataset Information
- **Source**: UC Irvine Machine Learning Repository
- **Reference**: Baldi, P., Sadowski, P., & Whiteson, D. (2014). *Searching for Exotic Particles in High-Energy Physics with Deep Learning*. Nature Communications, 5, 4308.
- **Direct Download URL**: https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
- **Total Instances**: 11,000,000 simulated particle collision events
- **Total Columns**: 29 (1 target label + 21 low-level kinematic features + 7 high-level invariant mass features)
- **Format**: Comma-separated values, no header row, float values in scientific notation.

## Expected Files
You can place any of the following supported formats in this directory:
1. `HIGGS.csv.gz` (recommended: compressed gzip format, ~2.8 GB)
2. `HIGGS.csv` (uncompressed CSV format, ~8.0 GB)
3. `higgs.zip` (zip archive containing `HIGGS.csv.gz`)

The dataset loading utility in `src/higgs/data.py` will automatically locate the dataset in `data/`, or in the parent project directory (`../higgs (1).zip` / `../HIGGS.csv.gz`), or at any explicitly supplied path.

*Note: Large raw dataset files are excluded from Git version control via `.gitignore`.*
