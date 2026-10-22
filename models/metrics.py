"""Metrics used by the original fixed-split evaluation."""

import numpy as np


def regression_metrics(predictions, targets):
    predictions = np.maximum(np.asarray(predictions, dtype=np.float64), 0.0)
    targets = np.asarray(targets, dtype=np.float64)
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
