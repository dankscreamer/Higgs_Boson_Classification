# Viva preparation: HIGGS signal detection

Short answers to the questions an examiner is likely to ask, each tied to what this project actually
did. Numbers come from `results/summary/` (generated, not hand-typed). Before the viva, make sure
every team member can answer these without notes.

---

## 1. Problem and physics

**Q: What exactly are you classifying?**
Simulated proton–proton collisions. Signal: a hypothetical heavy neutral Higgs H⁰ → W∓H± → W∓W±h⁰ →
W∓W±bb̄. Background: top-pair production tt̄ → W∓b W±b̄. Both leave one charged lepton, a neutrino (seen
only as missing transverse momentum) and four jets. The label says which process generated the event.

**Q: Why is this hard if the particles are the same?**
The visible objects are identical. The difference is in *how their momenta combine*: in signal, the
two b-jets come from a 125-GeV h⁰, so their invariant mass m_bb peaks; in tt̄ they come from two
different top quarks. No single feature separates the classes well (best single-feature AUC ≈ 0.57).

**Q: Does a high AUC mean you found a Higgs boson?**
No. The data are simulated and labelled by construction. A high AUC means the two *simulated
classes* are separable; it says nothing about the existence of this particle.

**Q: What are η, φ, pT, ΔR, invariant mass?**
pT = momentum transverse to the beam. η = pseudorapidity, a function of the polar angle (direction
along the beam). φ = azimuthal angle around the beam. ΔR = √(Δη² + Δφ²), angular distance.
Invariant mass m² = (ΣE)² − |Σp|², the same in every reference frame; for decay products it equals
the parent's mass.

## 2. Data and EDA

**Q: Why not use all 11M events from the start?**
The brief asks for a fixed-budget comparison before scaling, and it is good practice: we can run five
seeds and full ablations cheaply at 250k, then show how the conclusions change at 1M and 10.5M.

**Q: How did you sample the subset? Why stratified?**
Stratified random sampling (scikit-learn `train_test_split(stratify=y)`) with an explicit seed from
the first 10.5M rows. Stratification keeps the 53% signal fraction identical in every split, so
differences between seeds come from the events, not the class balance. Indices and class counts are
saved per seed (`data/splits/*.json`).

**Q: Why four splits (train/val/cal/test)?**
train fits models. val is used *only* for early stopping. cal is used *only* for post-hoc calibration
and the stacking meta-learner, so the calibrators never see data that also steered training. test
(the last 500k rows, as in Baldi et al.) is touched once per model, at the end.

**Q: What data-quality problems did you find?**
No NaN or inf values and no constant columns. The UCI file contains exact duplicate events: about
90 repeat inside a 200k training split (never with conflicting labels; 93 for seed 0), and about 450 test events (446 for seed 0)
also appear in that training split (<0.1%). Too small to move AUC; we kept the standard test set
for comparability and documented it.

**Q: How did EDA connect to physics?**
Three predictions checked: φ is uniform and uninformative alone (rotational symmetry; AUC 0.500),
b-tags are discrete (3 values), and the masses of the signal decay chain (m_bb, m_Wbb, m_WWbb) are
narrower for signal while W/top masses (m_jj, m_jjj, m_lν, m_jlν) peak for both classes.

**Q: Why are the masses not in GeV?**
The UCI release standardised every column. That is also why we had to recover φ's scale (π / max|φ|)
before computing angle differences.

## 3. Feature engineering

**Q: What is the `geo` feature set and why?**
Because the physics is invariant under rotating the whole event about the beam, absolute φ is noise;
only angle *differences* can matter. `geo` drops the 5 absolute φ's and adds |Δφ| for all 15 object
pairs and ΔR for the 10 pairs with η: 40 rotation-invariant features with no masses. A unit test
rotates every event and checks all 40 features are unchanged.

**Q: How do you wrap Δφ?**
((φa − φb + π) mod 2π) − π, then absolute value: result in [0, π]. Tested at the ±π boundary.

**Q: Isn't ΔR wrong if η's scale is unknown?**
Partly. φ's scale is recoverable (uniform on [−π, π]); η's is not, but its standardised scale is
close to 1, so ΔR is exact in φ and approximate in η. Stated as a limitation.

**Q: Why scale features, and why only on training data?**
Logistic regression's L2 penalty and the MLP's optimisation assume comparable scales. Fitting the
scaler on val/test would leak their statistics into training. Trees are scale-invariant.

## 4. ML theory (Phase 1)

**Q: Why does logistic regression do so badly on raw features?**
Its log-odds are linear in the inputs. The signal lives in non-linear quantities (angle differences,
masses). A linear function of φa and φb cannot represent a dependence on |φa − φb|. Give it those
quantities (geo, high) and it improves.

**Q: Explain XGBoost's objective.**
Additive trees; each new tree minimises a second-order Taylor expansion of the loss plus a penalty
γT + ½λ‖w‖². Optimal leaf weight w* = −G/(H + λ); split gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) −
(G_L+G_R)²/(H_L+H_R+λ)] − γ.

**Q: Why might trees struggle with raw angles?**
Splits are axis-aligned. A smooth function of φa − φb needs a staircase of many splits; the
geometric features hand it the difference directly.

**Q: How did you tune? Isn't tuning on CV optimistic?**
Stratified 3-fold CV on seed 0's training split only; the test set is never used. XGBoost uses a
seeded random search (Bergstra & Bengio: random beats grid for few important dimensions). Early
stopping on the held-out fold adds the same mild optimism to every configuration, so the ranking is
fair. Chosen settings are frozen for all seeds and budgets.

**Q: The LogReg C hit the edge of the grid.**
Yes for some sets, but the CV AUC differences across C are tiny: with 200k rows and at most 47
features, the L2 penalty barely matters. It does not change any conclusion.

## 5. Evaluation and statistics

**Q: Why ROC AUC and not accuracy?**
Accuracy needs a threshold and hides the trade-off; AUC = P(score(signal) > score(background)),
threshold-free. PR AUC focuses on the signal-rich region. Log loss and Brier are proper scoring rules
(they reward calibration too). ECE isolates calibration.

**Q: What is ECE?**
Bin predictions into 15 equal-width bins; ECE = Σ (bin weight) × |observed signal rate − mean
predicted probability|.

**Q: Platt vs isotonic?**
Platt fits σ(a·logit p + c): 2 parameters, monotone, AUC unchanged. Isotonic fits any monotone step
function: more flexible, needs more data, can create ties. Both fit on the cal split.

**Q: How do you know a difference is real?**
DeLong's method gives the covariance of AUCs measured on the same test events; the paired z-test
uses Var(A−B) = Var(A) + Var(B) − 2Cov(A,B). We repeat each comparison over 5 seeds and report in how
many seeds it is significant (α = 0.05). Our implementation is checked against sklearn's AUC and a
bootstrap variance in the unit tests.

**Q: With 500k test events everything is significant. So what?**
Correct: tiny differences become "significant". That's why we report the *effect size* (ΔAUC with
CI) and seed-to-seed sd, and only call a difference meaningful when it is consistent across seeds and
larger than the seed-to-seed noise.

## 6. Deep learning (Phase 2)

**Q: Justify the MLP architecture.**
3×256, BatchNorm, ReLU, dropout 0.1, Glorot init, Adam 1e-3, batch 1024, early stopping on val AUC
(patience 6) with best weights restored, LR halved on plateau. Big enough to compose pairwise
interactions into mass-like features; small enough to train in seconds at 250k. Every choice is
ablated.

**Q: Derive Glorot initialisation.**
Forward: Var[z] = n_in Var[W] Var[x]; backward: Var[∂x] = n_out Var[W] Var[∂z]. Keeping both at 1
needs n_in Var[W] = 1 and n_out Var[W] = 1; the compromise is Var[W] = 2/(n_in + n_out), i.e.
U[−√(6/(n_in+n_out)), +√(6/(n_in+n_out))]. Derived for tanh-like units without normalisation.

**Q: Did Glorot init matter in your network?**
Barely. With BatchNorm, Glorot vs PyTorch's default changes AUC by ≤ 0.002, because BatchNorm
re-standardises every layer. Without BatchNorm (ReLU), default init loses a bit more (−0.006 vs
−0.004). In Glorot & Bengio's own setting (tanh, no BN), Glorot was *not* better at depth 3 (−0.007
vs −0.005). Their effect shows up in deep nets trained with plain SGD; ours is 3 layers with Adam,
whose per-parameter step sizes compensate for a mis-scaled start. Honest answer: theory predicts a
depth-dependent effect, and at our depth it's within noise.

**Q: What does BatchNorm do?**
Standardises each layer's pre-activations with mini-batch mean/variance, then learns a scale γ and
shift β; running averages at test time. Smooths the loss landscape (Santurkar et al.), allows larger
learning rates, and adds mild noise (regularisation).

**Q: What does dropout do?**
Randomly zeroes units with probability p during training (rescaling by 1/(1−p)); approximates
averaging an ensemble of thinned networks and discourages co-adaptation.

**Q: Why is the MLP less calibrated than XGBoost?**
Neural networks trained to minimise cross-entropy with early stopping on AUC (not log loss) and with
batch-statistics noise often end up slightly over- or under-confident (Guo et al. 2017). Platt
scaling on the cal split fixes most of it.

**Q: Why not CNNs/Transformers?**
The input is a fixed-length vector of 21–47 physical quantities with no grid or sequence
structure; an MLP is the natural architecture and what Baldi et al. used. Set/graph networks over
particles would be the next step (Energy Flow Networks), noted as future work.

## 7. Hybrid approach

**Q: What is your hybrid and why is it justified (not just "combining models")?**
Two hybrids, each motivated by a measurement and each with a control:
1. *Physics-informed wide & deep*: the 7 expert masses bypass the deep tower and join at the last
   hidden layer; the tower sees only the 40 geometric features. Control: a plain MLP on the same 47
   inputs, so any difference is architectural.
2. *Stacking*: logistic regression on the XGBoost and MLP logits, fit on the cal split (unseen by
   either base model). Motivation: the two models have similar AUC but different inductive biases and
   different calibration. Control: plain averaging.

**Q: Why fit the stacker on the cal split and not on val?**
Both base models early-stop on val, so their val predictions are slightly optimistic; a stacker fit
there would over-trust them. cal is untouched by either base model.

**Q: Why combine in logit space?**
If each model's logit approximates the log-likelihood ratio plus noise, the optimal combination is a
weighted sum of logits; averaging probabilities fixes equal weights in the wrong space.

## 8. Engineering and reproducibility

**Q: How would someone reproduce your results?**
`make setup && make data && make reproduce`. Every hyperparameter lives in `config.yaml`; per-seed
subset indices are saved; results folders contain the config, manifest and run info (hardware,
versions). 19 unit tests (`make test`, under 10 s) cover splits, invariance, metrics, DeLong,
calibration and models.

**Q: Any platform issues?**
On macOS, xgboost and PyTorch each bundle OpenMP; with both loaded, multi-threaded PyTorch CPU ops
deadlock. We found it (stack sample showed a hang in `randperm`), pinned PyTorch to one CPU thread,
and trained the MLP on the Apple GPU (MPS). It only affects wall-clock comparisons.

## 9. Results you must know cold

All at 250k events, mean over 5 seeds, test = 500k fixed events (source: `results/summary/`).

| Features | LogReg | XGBoost | MLP | Stack |
|---|---|---|---|---|
| low (21 raw) | 0.596 | 0.738 | 0.760 | 0.766 |
| geo (40, no masses) | 0.630 | 0.801 | 0.826 | 0.831 |
| high (7 masses) | 0.645 | 0.787 | 0.793 | 0.793 |
| all (paper's 28) | 0.684 | 0.824 | 0.831 | 0.835 |
| geo + high (47) | 0.692 | 0.838 | 0.842 | **0.851** |

Full 10.5M pool (1 seed): MLP on low 0.846, XGBoost on all 0.851, MLP on all 0.868.

**Q: What is your single most important finding?**
Encoding a physical symmetry beats hand-built expert features. `geo` (pairwise angles and ΔR, no
masses) outperforms the 7 expert masses for both XGBoost (+0.014) and the MLP (+0.033), in every
seed; geometry + masses beats the paper's feature set for every model.

**Q: Does deep learning beat trees?**
It depends on data. At 50k events XGBoost wins on the richer feature sets; from 250k the MLP wins on
every feature set, and its lead on raw features grows from +0.007 (50k) to +0.023 (250k) to +0.05
(1M, 10.5M). At 10.5M the MLP on raw features nearly matches XGBoost given the expert masses
(0.846 vs 0.851), which reproduces Baldi et al.

**Q: Your wide & deep hybrid lost. Why include it?**
It tests a specific physics hypothesis (expert features help most as a shortcut to the output)
against a matched control (plain MLP, same 47 inputs). It lost by 0.009 in every seed, which tells us
the masses are most useful *mixed with* geometry inside the network. A negative result with a proper
control is evidence, not a failure.

**Q: Does stacking really help, or is it just averaging?**
Ranking: almost all of the gain over the base models comes from combining two different models;
plain averaging gets within 0.0002 at 250k (0.0017 at 1M). What the learned stacker adds is
calibration with no post-hoc step (ECE 0.004) and adaptive weights: XGBoost/MLP weights go from
≈0.5/0.45 at 250k to 0.30/0.68 at 1M to 0.15/0.91 at 10.5M, i.e. it learns to trust the network
as it improves. Its edge over the MLP shrinks to +0.0003 at 10.5M.

**Q: What would you change with more time?**
Dropout 0.3 (the ablation shows +0.002 to +0.004, and that run was still improving at the 50-epoch
cap); a longer epoch budget at 1M+ (MLPs stopped at the cap); more seeds at 10.5M; a set network
over the six physics objects (permutation invariance on top of rotation invariance).

**Q: Which features matter most?**
m_bb for XGBoost and the MLP, m_WWbb for logistic regression. The SHAP dependence plot shows m_bb in
its peak (h → bb̄) pushing towards signal and m_jjj near the top-quark mass pushing towards
background. The model recovers the physics without being told.

**Q: Why are your full-data numbers lower than Baldi et al. (0.885)?**
Smaller network (3×256 vs 5×300), 50-epoch cap (our runs were still improving), no pre-training or
extensive tuning at that scale, and one seed. Our claims are about *relative* behaviour under equal
budgets, which is what the brief asks for.
