"""
Model application module for the UCI HIGGS dataset.

This module provides reproducible, self-contained implementations for:
1. Logistic Regression (Linear baseline with StandardScaler)
2. XGBoost (Gradient-Boosted Decision Trees)
3. Multilayer Perceptron (Feed-forward Deep Neural Network in PyTorch)

Strict Leakage Guarantee:
Preprocessing (e.g. StandardScaler) is strictly fit on training data only
and applied to validation/test sets without target or test data leakage.
"""

from __future__ import annotations

import copy
import os
import random
from typing import Any, Sequence

# Prevent OpenMP runtime conflict between PyTorch bundled libomp and Homebrew libomp on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


# ---------------------------------------------------------------------------
# 1. Logistic Regression Baseline
# ---------------------------------------------------------------------------

class LogisticRegressionModel:
    """
    Logistic Regression classifier with integrated StandardScaler pipeline.

    Purpose:
        Serves as a transparent linear baseline to benchmark the limit of
        linear separability across different feature representations.

    Underlying Learning Mechanism:
        Models log-odds as a linear combination of input features:
            log(p / (1 - p)) = w^T x + b
        Minimizes cross-entropy loss with L2 regularization via the L-BFGS optimizer.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        solver: str = "lbfgs",
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver
        self.pipeline: Pipeline | None = None
        self.feature_names: list[str] | None = None

    def fit(self, X_train: pd.DataFrame | np.ndarray, y_train: pd.Series | np.ndarray) -> LogisticRegressionModel:
        """Fits StandardScaler and Logistic Regression strictly on training data."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)

        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.C,
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                        solver=self.solver,
                    ),
                ),
            ]
        )
        self.pipeline.fit(X_train, y_train)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Returns predicted probability for the positive (signal) class."""
        if self.pipeline is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions at the specified decision threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.float64)

    def get_coefficients(self) -> pd.DataFrame:
        """
        Returns feature weights sorted by absolute magnitude.

        Caveat:
            Predictive weight magnitude reflects linear feature correlation in
            the presence of collinearity; it does not establish physical causation.
        """
        if self.pipeline is None:
            raise RuntimeError("Model must be fitted to extract coefficients.")
        clf: LogisticRegression = self.pipeline.named_steps["clf"]
        weights = clf.coef_[0]
        names = self.feature_names if self.feature_names is not None else [f"feat_{i}" for i in range(len(weights))]

        df_coef = pd.DataFrame({"Feature": names, "Coefficient": weights, "Abs_Weight": np.abs(weights)})
        return df_coef.sort_values(by="Abs_Weight", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. XGBoost (Gradient-Boosted Decision Trees)
# ---------------------------------------------------------------------------

class XGBoostModel:
    """
    Gradient-boosted decision trees using XGBoost.

    Purpose:
        Provides high-capacity non-linear modeling for structured tabular data,
        capable of discovering complex axis-aligned feature splits and interactions.

    Underlying Learning Mechanism:
        Ensemble of decision trees trained sequentially. Each tree fits the
        negative gradient (pseudo-residuals) of the logistic loss function.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 15,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.clf: xgb.XGBClassifier | None = None
        self.feature_names: list[str] | None = None

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
        verbose: bool = False,
    ) -> XGBoostModel:
        """Trains gradient-boosted trees with validation-based early stopping."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)

        self.clf = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric="auc",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=verbose,
        )
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Returns predicted probability for the positive (signal) class."""
        if self.clf is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions at the specified decision threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.float64)

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns gain-based feature importances."""
        if self.clf is None:
            raise RuntimeError("Model must be fitted to extract feature importances.")
        importances = self.clf.feature_importances_
        names = self.feature_names if self.feature_names is not None else [f"feat_{i}" for i in range(len(importances))]

        df_imp = pd.DataFrame({"Feature": names, "Importance": importances})
        return df_imp.sort_values(by="Importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Multilayer Perceptron (PyTorch Feed-Forward Neural Network)
# ---------------------------------------------------------------------------

class HiggsMLPModule(nn.Module):
    """PyTorch Feed-Forward Neural Network architecture."""

    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int] = (128, 64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = in_features

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        # Final projection to single scalar logit
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class MultilayerPerceptronModel:
    """
    Multilayer Perceptron classifier wrapper.

    Purpose:
        Provides deep representation learning with smooth non-linear decision
        boundaries, closely reflecting the neural network architectures explored
        in Baldi et al. (2014).

    Underlying Learning Mechanism:
        Fully connected feed-forward layers with Batch Normalization, ReLU activations,
        and Dropout regularisation, trained using Adam on Binary Cross-Entropy loss.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int] = (128, 64, 32),
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        epochs: int = 20,
        batch_size: int = 256,
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        self.hidden_dims = list(hidden_dims)
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        self.scaler = StandardScaler()
        self.model: HiggsMLPModule | None = None
        self.best_weights: dict[str, Any] | None = None
        self.history: list[dict[str, float]] = []

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
        verbose: bool = False,
    ) -> MultilayerPerceptronModel:
        """Fits StandardScaler on train set and trains MLP with early stopping on validation AUC."""
        # Set reproducible seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        # Preprocessing: Fit scaler STRICTLY on train data
        X_tr_arr = self.scaler.fit_transform(X_train)
        X_val_arr = self.scaler.transform(X_val)

        y_tr_arr = np.asarray(y_train, dtype=np.float32)
        y_val_arr = np.asarray(y_val, dtype=np.float32)

        # Prepare PyTorch datasets
        train_ds = TensorDataset(
            torch.tensor(X_tr_arr, dtype=torch.float32),
            torch.tensor(y_tr_arr, dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )

        val_tensor_x = torch.tensor(X_val_arr, dtype=torch.float32).to(self.device)

        # Initialize network, loss, optimizer
        in_features = X_tr_arr.shape[1]
        self.model = HiggsMLPModule(
            in_features=in_features,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_val_auc = -1.0
        self.history = []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            num_batches = 0

            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

                train_loss += float(loss.item())
                num_batches += 1

            train_loss /= max(num_batches, 1)

            # Validation evaluation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(val_tensor_x)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_auc = float(roc_auc_score(y_val_arr, val_probs))

            self.history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc}
            )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                self.best_weights = copy.deepcopy(self.model.state_dict())

            if verbose and (epoch % 5 == 0 or epoch == self.epochs):
                print(f"Epoch {epoch:2d}/{self.epochs} - Loss: {train_loss:.4f} - Val AUC: {val_auc:.4f}")

        # Restore weights from best validation AUC epoch
        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)

        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Returns predicted probability for the positive (signal) class."""
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")
        self.model.eval()
        X_arr = self.scaler.transform(X)
        X_tensor = torch.tensor(X_arr, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions at the specified decision threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.float64)
