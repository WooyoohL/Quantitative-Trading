from __future__ import annotations

import numpy as np


import torch
import torch.nn as nn

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



class PearsonLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(PearsonLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        # 展平 Tensor
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # 计算均值
        mu_pred = torch.mean(y_pred)
        mu_true = torch.mean(y_true)

        # 中心化
        pred_diff = y_pred - mu_pred
        true_diff = y_true - mu_true

        # 计算 Pearson 相关系数 r
        # 公式: r = cov(x,y) / (std(x) * std(y))
        numerator = torch.sum(pred_diff * true_diff)
        denominator = torch.sqrt(torch.sum(pred_diff ** 2) * torch.sum(true_diff ** 2) + self.eps)

        corr = numerator / denominator

        # 我们希望 corr 越大越好（最大为 1），所以 Loss = 1 - corr
        return 1 - corr