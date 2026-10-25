### Paired DeLong tests at 1,000,000 (per seed, α = 0.05)

| question | A | B | ΔAUC (A−B) mean ± sd | seeds A sig. better | seeds B sig. better |
|---|---|---|---|---|---|
| DL vs trees on raw measurements | mlp|low | xgboost|low | 0.0522 ± 0.0016 | 3/3 | 0/3 |
| DL vs trees on geometric features | mlp|geo | xgboost|geo | 0.0364 ± 0.0005 | 3/3 | 0/3 |
| DL vs trees on raw + expert | mlp|all | xgboost|all | 0.0153 ± 0.0000 | 3/3 | 0/3 |
| Feature engineering (geo) for trees | xgboost|geo | xgboost|low | 0.0582 ± 0.0010 | 3/3 | 0/3 |
| Feature engineering (geo) for the MLP | mlp|geo | mlp|low | 0.0424 ± 0.0004 | 3/3 | 0/3 |
| Geometry alone vs expert masses (MLP) | mlp|geo | mlp|high | 0.0620 ± 0.0003 | 3/3 | 0/3 |
| Geometry alone vs expert masses (XGB) | xgboost|geo | xgboost|high | 0.0312 ± 0.0003 | 3/3 | 0/3 |
| Geometry alone vs raw + expert (MLP) | mlp|geo | mlp|all | 0.0076 ± 0.0002 | 3/3 | 0/3 |
| geo+expert vs raw+expert (MLP) | mlp|geo_high | mlp|all | 0.0170 ± 0.0000 | 3/3 | 0/3 |
| Hybrid 1: wide&deep vs plain MLP, same inputs | widedeep|geo_high | mlp|geo_high | -0.0057 ± 0.0007 | 0/3 | 3/3 |
| Hybrid 2: stacking vs MLP | stack|geo_high | mlp|geo_high | 0.0023 ± 0.0001 | 3/3 | 0/3 |
| Hybrid 2: stacking vs XGB | stack|geo_high | xgboost|geo_high | 0.0184 ± 0.0003 | 3/3 | 0/3 |
| Hybrid 2: learned stacking vs naive average | stack|geo_high | average|geo_high | 0.0017 ± 0.0000 | 3/3 | 0/3 |
| Hybrid 2 on the paper's features: stacking vs MLP | stack|all | mlp|all | 0.0018 ± 0.0001 | 3/3 | 0/3 |
