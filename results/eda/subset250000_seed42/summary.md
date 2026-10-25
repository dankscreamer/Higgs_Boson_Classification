# EDA summary: subset 250,000, seed 42 (training split only)

## Data quality
- Events: 225,000; signal fraction 0.530 (mildly imbalanced, so we report PR AUC as well as accuracy)
- NaN values: 0; infinite values: 0; duplicate rows: 129
- Test events that also appear exactly in train: 477 of 500,000 (the UCI file contains duplicate events; this is too few to move the metrics, and we keep the standard test set so results stay comparable with Baldi et al.)
- Constant features: none
- Discrete features: ['jet1_btag', 'jet2_btag', 'jet3_btag', 'jet4_btag']
- Units: features are pre-scaled in the UCI release (for example, momenta have a mean near 1), so masses are not in GeV.

## Physics checks
- **phi is uninformative alone** (rotational symmetry about the beam): single-feature AUC 0.500 to 0.502.
- **b-tag is discrete**: number of distinct values per jet {'jet1_btag': 3, 'jet2_btag': 3, 'jet3_btag': 3, 'jet4_btag': 3}.
- **Mass peaks**: interquartile range of each invariant mass, signal vs background (narrower = more peaked):

| mass | signal IQR | background IQR | signal narrower? |
|---|---|---|---|
| m_jj | 0.240 | 0.228 | no |
| m_jjj | 0.276 | 0.192 | no |
| m_lv | 0.037 | 0.032 | no |
| m_jlv | 0.410 | 0.336 | no |
| m_bb | 0.298 | 0.697 | yes |
| m_wbb | 0.250 | 0.410 | yes |
| m_wwbb | 0.225 | 0.357 | yes |

- **Expert features are functions of the raw ones**: strongest |correlation| between expert and raw features: m_wbb~jet1_pt 0.48, m_wwbb~jet1_pt 0.45, m_wbb~jet2_pt 0.43, m_wwbb~jet2_pt 0.38, m_bb~jet2_pt 0.37

## Single-feature separating power (ROC AUC, 0.5 = none)

| rank | feature | group | AUC |
|---|---|---|---|
| 1 | m_wwbb | high | 0.574 |
| 2 | m_bb | high | 0.558 |
| 3 | missing_energy_magnitude | low | 0.545 |
| 4 | jet1_pt | low | 0.538 |
| 5 | jet2_btag | low | 0.525 |
| 6 | m_jjj | high | 0.524 |
| 7 | m_wbb | high | 0.522 |
| 8 | jet4_pt | low | 0.520 |
| 9 | jet2_pt | low | 0.515 |
| 10 | jet3_btag | low | 0.512 |

The best single feature reaches AUC 0.574, while models that combine features reach above 0.8 (see the benchmark). Separating these events needs combinations of features, which is the case for ML.

These are simulated events. Results describe separability on this benchmark, not a physical discovery.
