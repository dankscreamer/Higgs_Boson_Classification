### Paired DeLong tests at 10,500,000 (per seed, α = 0.05)

| question | A | B | ΔAUC (A−B) mean ± sd | seeds A sig. better | seeds B sig. better |
|---|---|---|---|---|---|
| DL vs trees on raw measurements | mlp|low | xgboost|low | 0.0463 | 1/1 | 0/1 |
| DL vs trees on raw + expert | mlp|all | xgboost|all | 0.0173 | 1/1 | 0/1 |
| Hybrid 2 on the paper's features: stacking vs MLP | stack|all | mlp|all | 0.0003 | 1/1 | 0/1 |
