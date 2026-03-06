from __future__ import annotations

import numpy as np


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2:
        return 0.0
    r_true = _rank(y_true)
    r_pred = _rank(y_pred)
    corr = np.corrcoef(r_true, r_pred)[0, 1]
    return float(np.nan_to_num(corr))


def sharpe_ratio(returns: np.ndarray, eps: float = 1e-9) -> float:
    returns = np.asarray(returns, dtype=float)
    std = returns.std()
    if std < eps:
        return 0.0
    return float(returns.mean() / std)


def sharpe_loss(returns: np.ndarray) -> float:
    return -sharpe_ratio(returns)
