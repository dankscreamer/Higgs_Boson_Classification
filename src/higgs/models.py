"""The three model families. Each `fit_*` takes standardized arrays and returns a fitted Model."""
from __future__ import annotations

import copy
import sys
from typing import Protocol

import numpy as np

# xgboost must load its libomp before torch loads its own copy; the reverse order segfaults on macOS.
from xgboost import XGBClassifier  # isort: skip

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import nn


class Model(Protocol):
    info: dict

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Signal probability for each row, shape (n,)."""


class SklearnLike:
    """Adapter so sklearn/xgboost estimators return a 1-D signal probability."""

    def __init__(self, est, info: dict):
        self.est = est
        self.info = info

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.est.predict_proba(X)[:, 1]


def fit_logreg(X_tr, y_tr, X_val, y_val, params: dict, seed: int) -> SklearnLike:
    """L2 logistic regression; the interpretable linear baseline."""
    est = LogisticRegression(C=params["C"], max_iter=params["max_iter"], random_state=seed)
    est.fit(X_tr, y_tr)
    return SklearnLike(est, {"n_iter": int(est.n_iter_[0])})


def fit_xgboost(X_tr, y_tr, X_val, y_val, params: dict, seed: int) -> SklearnLike:
    """Histogram gradient-boosted trees, early-stopped on validation log loss."""
    est = XGBClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        early_stopping_rounds=params["early_stopping_rounds"],
        n_jobs=params["n_jobs"],
        tree_method="hist",
        eval_metric="logloss",
        random_state=seed,
    )
    est.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return SklearnLike(est, {"best_iteration": int(est.best_iteration)})


class MLP(nn.Module):
    """Compact feed-forward net: [Linear -> BatchNorm -> ReLU -> Dropout] x depth -> Linear logit."""

    def __init__(self, n_in: int, hidden: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_in
        for h in hidden:
            layers += [nn.Linear(width, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):  # Glorot & Bengio (2010) initialisation
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def pick_device(name: str) -> torch.device:
    """Resolve 'auto' to cuda, then mps, then cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchModel:
    def __init__(self, net: MLP, device: torch.device, info: dict):
        self.net = net
        self.device = device
        self.info = info

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, batch_size: int = 65536) -> np.ndarray:
        self.net.eval()
        out = []
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size], dtype=np.float32))
            out.append(torch.sigmoid(self.net(xb.to(self.device))).cpu().numpy())
        return np.concatenate(out)


def fit_mlp(X_tr, y_tr, X_val, y_val, params: dict, seed: int, verbose: bool = True) -> TorchModel:
    """Adam + BCE, LR halved on plateau, early-stopped on validation ROC AUC (best weights restored)."""
    torch.manual_seed(seed)
    device = pick_device(params["device"])
    if sys.platform == "darwin":
        # xgboost's Homebrew libomp and torch's bundled libomp deadlock/segfault in multi-threaded
        # CPU ops on macOS (even randperm on the MPS path). One intra-op thread avoids OpenMP.
        torch.set_num_threads(1)
    net = MLP(X_tr.shape[1], params["hidden"], params["dropout"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    loss_fn = nn.BCEWithLogitsLoss()

    X = torch.from_numpy(np.ascontiguousarray(X_tr, dtype=np.float32)).to(device)
    y = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    gen = torch.Generator().manual_seed(seed)
    model = TorchModel(net, device, {})
    best_auc, best_state, best_epoch, history = -1.0, None, 0, []

    for epoch in range(1, params["max_epochs"] + 1):
        net.train()
        perm = torch.randperm(len(X), generator=gen).to(device)
        total = 0.0
        for start in range(0, len(X), params["batch_size"]):
            b = perm[start:start + params["batch_size"]]
            if len(b) < 2:  # BatchNorm needs more than one sample
                continue
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(X[b]), y[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        val_auc = roc_auc_score(y_val, model.predict_proba(X_val))
        sched.step(val_auc)
        history.append({"epoch": epoch, "train_loss": total / len(X), "val_auc": val_auc})
        if verbose:
            print(f"    epoch {epoch:3d}  train_loss {total / len(X):.4f}  val_auc {val_auc:.4f}", flush=True)
        if val_auc > best_auc:
            best_auc, best_state, best_epoch = val_auc, copy.deepcopy(net.state_dict()), epoch
        elif epoch - best_epoch >= params["patience"]:
            break

    net.load_state_dict(best_state)
    model.info = {"best_epoch": best_epoch, "epochs_run": len(history), "device": str(device),
                  "history": history}
    return model


FITTERS = {"logreg": fit_logreg, "xgboost": fit_xgboost, "mlp": fit_mlp}
