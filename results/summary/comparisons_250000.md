### Paired DeLong tests at 250,000 (per seed, α = 0.05)

| question | A | B | ΔAUC (A−B) mean ± sd | seeds A sig. better | seeds B sig. better |
|---|---|---|---|---|---|
| DL vs trees on raw measurements | mlp|low | xgboost|low | 0.0226 ± 0.0011 | 5/5 | 0/5 |
| DL vs trees on geometric features | mlp|geo | xgboost|geo | 0.0244 ± 0.0006 | 5/5 | 0/5 |
| DL vs trees on raw + expert | mlp|all | xgboost|all | 0.0075 ± 0.0005 | 5/5 | 0/5 |
| Feature engineering (geo) for trees | xgboost|geo | xgboost|low | 0.0635 ± 0.0008 | 5/5 | 0/5 |
| Feature engineering (geo) for the MLP | mlp|geo | mlp|low | 0.0654 ± 0.0010 | 5/5 | 0/5 |
| Geometry alone vs expert masses (MLP) | mlp|geo | mlp|high | 0.0325 ± 0.0011 | 5/5 | 0/5 |
| Geometry alone vs expert masses (XGB) | xgboost|geo | xgboost|high | 0.0139 ± 0.0004 | 5/5 | 0/5 |
| Geometry alone vs raw + expert (MLP) | mlp|geo | mlp|all | -0.0055 ± 0.0011 | 0/5 | 5/5 |
| geo+expert vs raw+expert (MLP) | mlp|geo_high | mlp|all | 0.0112 ± 0.0012 | 5/5 | 0/5 |
| Hybrid 1: wide&deep vs plain MLP, same inputs | widedeep|geo_high | mlp|geo_high | -0.0092 ± 0.0008 | 0/5 | 5/5 |
| Hybrid 2: stacking vs MLP | stack|geo_high | mlp|geo_high | 0.0084 ± 0.0008 | 5/5 | 0/5 |
| Hybrid 2: stacking vs XGB | stack|geo_high | xgboost|geo_high | 0.0124 ± 0.0001 | 5/5 | 0/5 |
| Hybrid 2: learned stacking vs naive average | stack|geo_high | average|geo_high | 0.0002 ± 0.0000 | 5/5 | 0/5 |
| Hybrid 2 on the paper's features: stacking vs MLP | stack|all | mlp|all | 0.0044 ± 0.0005 | 5/5 | 0/5 |
