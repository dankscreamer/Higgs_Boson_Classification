# Phase 1 talk: Model Application + Calibration & Compute

Speaker: Ananya. About 2.5–3 minutes in total. All figures are in `presentation/phase1_ml/figures/`
(regenerate them with `.venv/bin/python scripts/make_presentation_figures.py`).
All numbers: 250k-event budget, mean of 5 seeds, scored on the same 500k test events.

---

## Slide A: Model application, results (≈1 min)

**On the slide:** figure `1_auc_by_feature_set.png` (full width), plus at most these 3 bullets:
- Two models: L2 logistic regression and tuned XGBoost
- 5 feature sets: raw → geometric (ours) → expert masses → combinations
- Best: **XGBoost on geo + high, 0.838**

**Say:**
> "We applied two models. Logistic regression is our linear, interpretable baseline. XGBoost is a
> gradient-boosted tree ensemble, the standard in particle physics. We trained both on five feature
> sets, at the same 250k-event budget, scored on the same 500k test events, and repeated everything
> over five random seeds.
>
> Three things stand out. First, XGBoost beats logistic regression on every feature set, by 0.14 to
> 0.17 AUC. A straight-line model simply can't represent this problem.
> Second, look at our geometric features: 0.801, higher than the physicists' 7 expert masses at
> 0.787, without using any masses at all.
> Third, combining our geometry with the masses gives the best model, 0.838. That beats the
> feature set used in the original paper (0.823)."

**If asked "why does logistic regression fail?":** "Its log-odds are linear in the inputs. The signal
lives in angle differences and masses, which are non-linear functions of the raw measurements. A
linear function of two angles can't express their difference."

---

## Slide B: Model application, rigour (≈45 s)

**On the slide:** two figures side by side: `2_significance.png` (left) and `3_tuning.png` (right).
If space is tight, use only `2_significance.png` and put the tuning settings as one bullet.

**Say:**
> "How do we know these differences are real? All models are scored on the same test events, so
> their AUCs are correlated. A naive comparison would be wrong. We use DeLong's paired test, which
> accounts for that correlation. Every improvement here, for example +0.064 from our geometric
> features, is significant in all five seeds. Seed-to-seed variation is under 0.001.
>
> Hyperparameters were chosen fairly: stratified 3-fold cross-validation on training data only, so
> the test set was never touched. A random search over 12 configurations picked deep trees (depth 8
> to 10), learning rate 0.05 and min_child_weight 20. Deep trees make sense, because the signal is a
> high-order interaction of the raw inputs."

---

## Slide C: Calibration & compute (≈1 min)

**On the slide:** `4_reliability.png` (top or left) and `6_auc_vs_training_time.png` (bottom or right).
Backup slide if asked: `5_calibration_before_after.png`.

**Say (calibration, ~35 s):**
> "A good AUC means the model *ranks* events well. Calibration asks something different: when the
> model says 70% signal, is it right 70% of the time? This plot compares predicted probability with
> the observed signal rate, so a perfect model sits on the diagonal.
>
> XGBoost sits right on it: expected calibration error about 0.005, with no correction needed.
> Logistic regression bends away, with ECE 0.048 on the expert masses. That's because its linear
> assumption is wrong, not because of noise. We fit two standard fixes on a separate calibration split.
> Platt scaling only stretches the curve, so it can't fix the shape and leaves ECE at 0.048. Isotonic
> regression can follow any monotone shape and brings it down to 0.006."

**Say (compute, ~25 s):**
> "On compute: logistic regression trains in under a fifth of a second. XGBoost takes 1.5 to 12
> seconds on 200k events. Those few seconds buy 0.14 to 0.17 AUC. The best value is XGBoost on the
> paper's features, 4.8 seconds for 0.823. The best overall is geometry plus masses, 0.838 in about
> 10 seconds."

**Transition to the next speaker:** "So XGBoost is accurate, calibrated and cheap. Next, [name] will
explain what the model actually learned."

---

## Numbers cheat sheet

| Features | LogReg AUC | XGBoost AUC | LogReg ECE (raw → isotonic) | XGBoost ECE | XGB train time |
|---|---|---|---|---|---|
| low (21 raw) | 0.596 | 0.738 | 0.016 → 0.005 | 0.005 | 5.1 s |
| geo (40, ours) | 0.630 | 0.801 | 0.008 → 0.005 | 0.004 | 11.6 s |
| high (7 masses) | 0.645 | 0.787 | 0.048 → 0.006 | 0.006 | 1.5 s |
| all (paper's 28) | 0.684 | 0.823 | 0.024 → 0.006 | 0.005 | 4.8 s |
| geo + high (47) | 0.692 | **0.838** | 0.018 → 0.006 | 0.005 | 10.4 s |

Key differences (XGBoost, all significant in 5/5 seeds, paired DeLong):
geo vs low **+0.064** · geo vs high **+0.014** · all vs high **+0.036** · geo+high vs all **+0.015**.
LogReg geo vs low **+0.034**.

Setup: 250k events per seed → 200k train / 25k val (early stopping) / 25k cal (calibration).
Test: last 500k events, identical for every model. LogReg trains in < 0.2 s for every feature set.

---

## Likely questions on your part

**Why XGBoost and not random forest / LightGBM?**
XGBoost has the second-order objective with explicit regularisation (λ, min_child_weight), handles
early stopping natively, and comes from the 2014 Higgs ML Challenge. LightGBM would be similar. Our
point is a strong, tuned tree baseline, not a tree-library comparison.

**What is ROC AUC, in one sentence?**
The probability that a randomly chosen signal event gets a higher score than a randomly chosen
background event. 0.5 is guessing, 1.0 is perfect.

**Why not report accuracy?**
Accuracy depends on a threshold and hides the trade-off. AUC is threshold-free, and we report PR AUC
too, because signal is 53%. Accuracy is still in our tables (XGBoost geo+high: 75.5%).

**Isn't a difference of 0.014 tiny?**
It is consistent in all 5 seeds, over 30× its seed-to-seed sd (0.0004), and the DeLong 95% CI excludes zero.
In physics, a small AUC gain translates into more signal kept at the same background level.

**What is ECE exactly?**
Split predictions into 15 equal-width bins, and in each bin compare the average predicted probability
with the actual signal fraction. ECE is the bin-size-weighted average of those gaps.

**Why does Platt fail but isotonic work?**
Platt fits σ(a·logit(p) + b): it can only stretch and shift the curve. Logistic regression's
calibration curve is *bent* (it's misspecified), so only a flexible monotone map, isotonic, can fix it.

**Why is XGBoost well calibrated when the literature says boosting isn't?**
Older results were for boosted stumps with exponential loss. We train with log loss (a proper scoring
rule) and early-stop on log loss, which keeps probabilities honest.

**Why fit calibration on a separate split?**
If we calibrated on validation data, which already steered early stopping, or on test data, we'd get
optimistic results. The cal split is never used for anything else.

**Are the times fair?**
All at 250k events on the same Apple M3 machine, with XGBoost using all CPU cores. We do not compare
timings from the 1M+ runs, because those ran under memory pressure.

**Why do you call the AUC "significant" with 500k test events? Isn't everything significant?**
Yes, the standard error is only 5–8 × 10⁻⁴. That's why we also report effect size, the seed-to-seed
sd and the number of seeds (5/5), and only interpret consistent differences well above the noise.
