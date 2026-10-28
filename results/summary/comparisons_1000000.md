### Paired DeLong tests at 1,000,000 (per seed, α = 0.05)

| question | A | B | ΔAUC (A−B) mean ± sd | seeds A sig. better | seeds B sig. better |
|---|---|---|---|---|---|
| Trees vs linear on raw measurements | xgboost|low | logreg|low | 0.1689 ± 0.0012 | 3/3 | 0/3 |
| Trees vs linear on raw + expert | xgboost|all | logreg|all | 0.1523 ± 0.0002 | 3/3 | 0/3 |
| Feature engineering (geo) for trees | xgboost|geo | xgboost|low | 0.0582 ± 0.0010 | 3/3 | 0/3 |
| Feature engineering (geo) for the linear model | logreg|geo | logreg|low | 0.0339 ± 0.0001 | 3/3 | 0/3 |
| Geometry alone vs expert masses (XGB) | xgboost|geo | xgboost|high | 0.0312 ± 0.0003 | 3/3 | 0/3 |
| Adding raw features to the masses (XGB) | xgboost|all | xgboost|high | 0.0448 ± 0.0001 | 3/3 | 0/3 |
| geo+expert vs raw+expert (XGB) | xgboost|geo_high | xgboost|all | 0.0162 ± 0.0004 | 3/3 | 0/3 |
| geo+expert vs raw+expert (LogReg) | logreg|geo_high | logreg|all | 0.0076 ± 0.0001 | 3/3 | 0/3 |
