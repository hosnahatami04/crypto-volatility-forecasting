"""Training loop for VolLSTM: early stopping on a validation slice carved
from the END of the training window (never from the future).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.models.lstm import VolLSTM, set_seed

VALIDATION_FRACTION = 0.15
MAX_EPOCHS = 100
PATIENCE = 8
LEARNING_RATE = 1e-3


@dataclass(frozen=True)
class TrainResult:
    model: VolLSTM
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    best_val_loss: float
    n_epochs_trained: int


def train_test_val_split(
    X: np.ndarray, y: np.ndarray, val_fraction: float = VALIDATION_FRACTION
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carve a validation slice from the END of the training samples --
    i.e. the most recent samples in the training window, never any sample
    from beyond the training window (which would be the forecast future).
    """
    n = len(X)
    n_val = max(1, int(n * val_fraction))
    split = n - n_val
    return X[:split], y[:split], X[split:], y[split:]


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    seed: int = 42,
) -> TrainResult:
    """Train VolLSTM with early stopping. X_train/y_train are the full
    training-fold windows (pre-split internally into fit/validation).
    """
    set_seed(seed)

    X_fit, y_fit, X_val, y_val = train_test_val_split(X_train, y_train)

    X_fit_t = torch.tensor(X_fit, dtype=torch.float32)
    y_fit_t = torch.tensor(y_fit, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    n_channels = X_train.shape[-1]
    model = VolLSTM(n_channels=n_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    n_epochs_trained = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(X_fit_t)
        loss = loss_fn(pred, y_fit_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        n_epochs_trained = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(
        model=model,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        best_val_loss=best_val_loss,
        n_epochs_trained=n_epochs_trained,
    )


def predict(model: VolLSTM, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        pred = model(X_t)
    return pred.numpy()
