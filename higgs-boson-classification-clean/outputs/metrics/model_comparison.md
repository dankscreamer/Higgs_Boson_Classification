# Experimental Comparison Table: 3 Models x 5 Feature Representations

Evaluated on the independent test set ($N_{\text{test}} = 7,500$ events, $N_{\text{train}} = 35,000$ events, $N_{\text{val}} = 7,500$ events, Random Seed = 42).

| Rank | Model | Feature Set | Feature Count | ROC AUC | Accuracy | Precision | Recall | F1 Score |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **XGBoost** | `all` | 28 | **0.7886** | 0.7125 | 0.7241 | 0.7412 | 0.7326 |
| 2 | **MLP (PyTorch)** | `all` | 28 | **0.7800** | 0.7073 | 0.7318 | 0.7088 | 0.7201 |
| 3 | **XGBoost** | `geo_high` | 17 | **0.7775** | 0.6984 | 0.7060 | 0.7407 | 0.7229 |
| 4 | **MLP (PyTorch)** | `geo_high` | 17 | **0.7698** | 0.6952 | 0.7051 | 0.7327 | 0.7186 |
| 5 | **MLP (PyTorch)** | `high` | 7 | **0.7697** | 0.6920 | 0.7004 | 0.7342 | 0.7169 |
| 6 | **XGBoost** | `high` | 7 | **0.7696** | 0.6928 | 0.7012 | 0.7347 | 0.7176 |
| 7 | **MLP (PyTorch)** | `low` | 21 | **0.6944** | 0.6424 | 0.6551 | 0.6900 | 0.6721 |
| 8 | **XGBoost** | `low` | 21 | **0.6808** | 0.6321 | 0.6389 | 0.7071 | 0.6713 |
| 9 | **Logistic Regression** | `all` | 28 | **0.6734** | 0.6341 | 0.6328 | 0.7415 | 0.6828 |
| 10 | **Logistic Regression** | `geo_high` | 17 | **0.6456** | 0.6191 | 0.6136 | 0.7641 | 0.6806 |
| 11 | **Logistic Regression** | `high` | 7 | **0.6363** | 0.6201 | 0.6095 | 0.7932 | 0.6893 |
| 12 | **Logistic Regression** | `low` | 21 | **0.5970** | 0.5665 | 0.5741 | 0.7126 | 0.6359 |
| 13 | **MLP (PyTorch)** | `geo` | 10 | **0.5967** | 0.5773 | 0.5790 | 0.7487 | 0.6530 |
| 14 | **XGBoost** | `geo` | 10 | **0.5953** | 0.5741 | 0.5765 | 0.7475 | 0.6509 |
| 15 | **Logistic Regression** | `geo` | 10 | **0.5582** | 0.5483 | 0.5550 | 0.7550 | 0.6397 |

## Key Empirical Takeaways

1. **Non-Linear Dominance:** Non-linear models (`XGBoost`: 0.7886, `MLP`: 0.7800) substantially outperform the linear baseline (`Logistic Regression`: 0.6734) on identical features (+11.5 AUC points), confirming that collider kinematics cannot be modeled with monotonic hyperplanes.
2. **Impact of Geometric Engineering:** The engineered `geo_high` representation (17 features) achieves **0.7775 AUC** in XGBoost, surpassing the 7 mass features alone (0.7696) and approaching the 28-feature model (0.7886) with 40% fewer inputs.
3. **Representation Learning in MLPs:** On raw `low` detector features alone, `MLP` reaches **0.6944 AUC**, outperforming `XGBoost` (0.6808) and greatly exceeding `Logistic Regression` (0.5970), demonstrating neural feature abstraction from raw detector inputs.
