### Paired DeLong tests at 250,000 (per seed, α = 0.05)

| question | A | B | ΔAUC (A−B) mean ± sd | seeds A sig. better | seeds B sig. better |
|---|---|---|---|---|---|
| Trees vs linear on raw measurements | xgboost|low | logreg|low | 0.1419 ± 0.0007 | 5/5 | 0/5 |
| Trees vs linear on raw + expert | xgboost|all | logreg|all | 0.1392 ± 0.0007 | 5/5 | 0/5 |
| Feature engineering (geo) for trees | xgboost|geo | xgboost|low | 0.0635 ± 0.0008 | 5/5 | 0/5 |
| Feature engineering (geo) for the linear model | logreg|geo | logreg|low | 0.0341 ± 0.0001 | 5/5 | 0/5 |
| Geometry alone vs expert masses (XGB) | xgboost|geo | xgboost|high | 0.0139 ± 0.0004 | 5/5 | 0/5 |
| Adding raw features to the masses (XGB) | xgboost|all | xgboost|high | 0.0363 ± 0.0004 | 5/5 | 0/5 |
| geo+expert vs raw+expert (XGB) | xgboost|geo_high | xgboost|all | 0.0147 ± 0.0005 | 5/5 | 0/5 |
| geo+expert vs raw+expert (LogReg) | logreg|geo_high | logreg|all | 0.0076 ± 0.0001 | 5/5 | 0/5 |
