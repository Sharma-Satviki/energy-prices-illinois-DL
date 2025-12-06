# ml3_utils.py
import os, json, math, random, time
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# -----------------------
# Reproducibility
# -----------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------
# Data utilities
# -----------------------
def load_pickled_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_pickle(path)
    # ensure datetime index if available
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").set_index("timestamp")
    else:
        df = df.sort_index()
    return df

@dataclass
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15  # test is remaining
    lookback: int = 48       # hours/steps
    horizon: int = 1

def time_splits(n: int, train_ratio=0.7, val_ratio=0.15) -> Tuple[slice, slice, slice]:
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)

def fit_scalers(train_X: np.ndarray, y: np.ndarray) -> Tuple[StandardScaler, Optional[StandardScaler]]:
    x_scaler = StandardScaler().fit(train_X)
    y_scaler = None
    if y.ndim == 2:  # (N, 1) or (N, k)
        y_scaler = StandardScaler().fit(y)
    return x_scaler, y_scaler

def transform_with_scalers(X: np.ndarray, y: Optional[np.ndarray], x_scaler, y_scaler):
    Xs = x_scaler.transform(X)
    ys = y
    if y is not None and y_scaler is not None:
        ys = y_scaler.transform(y)
    return Xs, ys

def make_supervised(df: pd.DataFrame, feature_cols: List[str], label_col: str,
                    lookback: int, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Create rolling windows: X[t-lookback:t, features] -> y[t+horizon-1]"""
    arr_X = df[feature_cols].values.astype(np.float32)
    arr_y = df[[label_col]].values.astype(np.float32)
    X_list, y_list = [], []
    for t in range(lookback, len(df) - horizon + 1):
        X_list.append(arr_X[t - lookback:t, :])
        y_list.append(arr_y[t + horizon - 1, :])
    X = np.stack(X_list)                       # (N, lookback, F)
    y = np.stack(y_list)                       # (N, 1)
    return X, y

class EnergySeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)  # (N, T, F)
        self.y = torch.from_numpy(y)  # (N, 1)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def build_dataloaders(X: np.ndarray, y: np.ndarray, split_cfg: SplitConfig,
                      batch_size: int = 64, num_workers: int = 0) -> Dict[str, DataLoader]:
    n = X.shape[0]
    tr_s, va_s, te_s = time_splits(n, split_cfg.train_ratio, split_cfg.val_ratio)
    ds_train = EnergySeqDataset(X[tr_s], y[tr_s])
    ds_val   = EnergySeqDataset(X[va_s], y[va_s])
    ds_test  = EnergySeqDataset(X[te_s], y[te_s])
    loaders = {
        "train": DataLoader(ds_train, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False),
        "val":   DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False),
        "test":  DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False),
    }
    return loaders

# -----------------------
# Model
# -----------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_size, out_size // 2),
            nn.ReLU(),
            nn.Linear(out_size // 2, 1),
        )
    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)           # (B, T, H[×2])
        last = out[:, -1, :]            # (B, H[×2])
        return self.head(last)          # (B, 1)

# -----------------------
# Training / Evaluation
# -----------------------
@torch.no_grad()
def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    # y_* shape: (N, 1)
    mse = nn.functional.mse_loss(y_pred, y_true).item()
    mae = nn.functional.l1_loss(y_pred, y_true).item()
    # safe MAPE
    eps = 1e-6
    mape = (torch.abs((y_true - y_pred) / torch.clamp(y_true.abs(), min=eps))).mean().item()
    return {"mse": mse, "mae": mae, "mape": mape}

def step_epoch(model, loader, device, optimizer=None, criterion=nn.MSELoss()):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_obs = 0.0, 0
    all_y, all_p = [], []
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(Xb)
        loss = criterion(pred, yb)
        if is_train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        bs = Xb.size(0)
        total_loss += loss.item() * bs
        n_obs += bs
        all_y.append(yb.detach().cpu())
        all_p.append(pred.detach().cpu())
    y_all = torch.cat(all_y, dim=0)
    p_all = torch.cat(all_p, dim=0)
    metrics = compute_metrics(y_all, p_all)
    metrics["loss"] = total_loss / max(1, n_obs)
    return metrics, y_all, p_all

def train_loop(model, loaders, device, optimizer, scheduler=None,
               epochs: int = 50, patience: int = 8, verbose: bool = True):
    best_val = float("inf")
    best_state = None
    history = {"train": [], "val": []}
    patience_left = patience
    for ep in range(1, epochs + 1):
        tr_metrics, _, _ = step_epoch(model, loaders["train"], device, optimizer)
        va_metrics, _, _ = step_epoch(model, loaders["val"], device, optimizer=None)
        history["train"].append(tr_metrics)
        history["val"].append(va_metrics)
        if scheduler is not None:
            # Use val loss to step if ReduceLROnPlateau
            if hasattr(scheduler, "step") and "loss" in va_metrics:
                try:
                    scheduler.step(va_metrics["loss"])
                except TypeError:
                    scheduler.step()
        if verbose:
            print(f"Epoch {ep:03d} | "
                  f"train_loss={tr_metrics['loss']:.4f} val_loss={va_metrics['loss']:.4f} "
                  f"val_mse={va_metrics['mse']:.4f}")
        if va_metrics["mse"] < best_val:
            best_val = va_metrics["mse"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                if verbose:
                    print("Early stopping.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history

@torch.no_grad()
def evaluate_model(model, loader, device) -> Dict[str, float]:
    metrics, y, p = step_epoch(model, loader, device, optimizer=None)
    return {"metrics": metrics, "y": y.numpy(), "p": p.numpy()}

# -----------------------
# Orchestration helpers
# -----------------------
def prepare_Xy(df: pd.DataFrame, feature_cols: List[str], label_col: str,
               split_cfg: SplitConfig) -> Tuple[np.ndarray, np.ndarray, StandardScaler, Optional[StandardScaler]]:
    X_all, y_all = make_supervised(df, feature_cols, label_col, split_cfg.lookback, split_cfg.horizon)
    # flatten time dim for scaler fit (fit on train only to avoid leakage)
    n, T, F = X_all.shape
    X_flat = X_all.reshape(n, T * F)
    tr_s, va_s, te_s = time_splits(n, split_cfg.train_ratio, split_cfg.val_ratio)
    x_scaler, y_scaler = fit_scalers(X_flat[tr_s], y_all[tr_s])
    Xs, ys = transform_with_scalers(X_flat, y_all, x_scaler, y_scaler)
    Xs = Xs.reshape(n, T, F)
    return Xs, ys, x_scaler, y_scaler

def save_artifacts(model, x_scaler, y_scaler, out_dir="models", name="baseline"):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, f"{name}.pt"))
    import joblib
    joblib.dump(x_scaler, os.path.join(out_dir, f"{name}_xscaler.pkl"))
    joblib.dump(y_scaler, os.path.join(out_dir, f"{name}_yscaler.pkl"))