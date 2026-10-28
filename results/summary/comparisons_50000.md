### Paired DeLong tests at 50,000 (per seed, α = 0.05)

| question | A | B | ΔAUC (A−B) mean ± sd | seeds A sig. better | seeds B sig. better |
|---|---|---|---|---|---|
| Trees vs linear on raw measurements | xgboost|low | logreg|low | 0.1041 ± 0.0013 | 3/3 | 0/3 |
| Trees vs linear on raw + expert | xgboost|all | logreg|all | 0.1239 ± 0.0012 | 3/3 | 0/3 |
| Feature engineering (geo) for trees | xgboost|geo | xgboost|low | 0.0632 ± 0.0012 | 3/3 | 0/3 |
| Feature engineering (geo) for the linear model | logreg|geo | logreg|low | 0.0341 ± 0.0002 | 3/3 | 0/3 |
| Geometry alone vs expert masses (XGB) | xgboost|geo | xgboost|high | -0.0173 ± 0.0025 | 0/3 | 3/3 |
| Adding raw features to the masses (XGB) | xgboost|all | xgboost|high | 0.0282 ± 0.0008 | 3/3 | 0/3 |
| geo+expert vs raw+expert (XGB) | xgboost|geo_high | xgboost|all | 0.0106 ± 0.0009 | 3/3 | 0/3 |
| geo+expert vs raw+expert (LogReg) | logreg|geo_high | logreg|all | 0.0073 ± 0.0003 | 3/3 | 0/3 |
