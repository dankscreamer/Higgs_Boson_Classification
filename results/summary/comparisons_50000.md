### Paired DeLong tests at 50,000 (per seed, α = 0.05)

| question | A | B | ΔAUC (A−B) mean ± sd | seeds A sig. better | seeds B sig. better |
|---|---|---|---|---|---|
| DL vs trees on raw measurements | mlp|low | xgboost|low | 0.0065 ± 0.0008 | 3/3 | 0/3 |
| DL vs trees on geometric features | mlp|geo | xgboost|geo | -0.0045 ± 0.0056 | 0/3 | 2/3 |
| DL vs trees on raw + expert | mlp|all | xgboost|all | -0.0216 ± 0.0011 | 0/3 | 3/3 |
| Feature engineering (geo) for trees | xgboost|geo | xgboost|low | 0.0632 ± 0.0012 | 3/3 | 0/3 |
| Feature engineering (geo) for the MLP | mlp|geo | mlp|low | 0.0522 ± 0.0043 | 3/3 | 0/3 |
| Geometry alone vs expert masses (MLP) | mlp|geo | mlp|high | -0.0250 ± 0.0020 | 0/3 | 3/3 |
| Geometry alone vs expert masses (XGB) | xgboost|geo | xgboost|high | -0.0173 ± 0.0025 | 0/3 | 3/3 |
| Geometry alone vs raw + expert (MLP) | mlp|geo | mlp|all | -0.0285 ± 0.0044 | 0/3 | 3/3 |
| geo+expert vs raw+expert (MLP) | mlp|geo_high | mlp|all | 0.0051 ± 0.0002 | 3/3 | 0/3 |
| Hybrid 1: wide&deep vs plain MLP, same inputs | widedeep|geo_high | mlp|geo_high | -0.0079 ± 0.0031 | 0/3 | 3/3 |
| Hybrid 2: stacking vs MLP | stack|geo_high | mlp|geo_high | 0.0323 ± 0.0005 | 3/3 | 0/3 |
| Hybrid 2: stacking vs XGB | stack|geo_high | xgboost|geo_high | 0.0052 ± 0.0006 | 3/3 | 0/3 |
| Hybrid 2: learned stacking vs naive average | stack|geo_high | average|geo_high | 0.0044 ± 0.0001 | 3/3 | 0/3 |
| Hybrid 2 on the paper's features: stacking vs MLP | stack|all | mlp|all | 0.0252 ± 0.0009 | 3/3 | 0/3 |
