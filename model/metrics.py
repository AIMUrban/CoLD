"""Metrics computed independently on each fold, on the original flow scale."""

import numpy as np


def regression_metrics(predictions, targets):
    predictions = np.maximum(np.asarray(predictions, dtype=np.float64), 0.0)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape or targets.size == 0:
        raise ValueError("Predictions and targets must have matching nonempty shapes")
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("Predictions and targets must be finite")
    error = predictions - targets
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    cpc = 2 * np.minimum(predictions, targets).sum() / (predictions.sum() + targets.sum() + 1e-8)
    return {
        "MSE": float(mse),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(rmse),
        "NRMSE": float(rmse / (np.std(targets) + 1e-8)),
        "CPC": float(cpc),
    }
