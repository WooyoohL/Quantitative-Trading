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


def daily_rank_ic_mean(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    dates = np.asarray(dates)
    if len(y_true) < 2 or len(y_true) != len(y_pred) or len(y_true) != len(dates):
        return 0.0

    daily_values: list[float] = []
    for date_value in np.unique(dates):
        mask = dates == date_value
        if int(mask.sum()) < 2:
            continue
        daily_values.append(rank_ic(y_true[mask], y_pred[mask]))

    if not daily_values:
        return 0.0
    return float(np.nan_to_num(np.mean(daily_values)))


def head_daily_rank_ic_mean(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: np.ndarray,
    top_n: int,
) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    dates = np.asarray(dates)
    if len(y_true) < 2 or len(y_true) != len(y_pred) or len(y_true) != len(dates):
        return 0.0

    top_n = int(top_n)
    if top_n <= 1:
        return 0.0

    daily_values: list[float] = []
    for date_value in np.unique(dates):
        mask = dates == date_value
        if int(mask.sum()) < 2:
            continue
        day_true = y_true[mask]
        day_pred = y_pred[mask]
        order = np.argsort(-day_pred)
        head_order = order[: min(top_n, len(order))]
        if len(head_order) < 2:
            continue
        daily_values.append(rank_ic(day_true[head_order], day_pred[head_order]))

    if not daily_values:
        return 0.0
    return float(np.nan_to_num(np.mean(daily_values)))


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


def _torch_rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks


def soft_rank(values: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    values = values.view(-1)
    if values.numel() < 2:
        return torch.ones_like(values, dtype=torch.float32)
    temperature = max(float(tau), 1e-6)
    diff = (values.unsqueeze(0) - values.unsqueeze(1)) / temperature
    pairwise = torch.sigmoid(diff)
    pairwise = pairwise - torch.diag_embed(torch.diagonal(pairwise))
    return 1.0 + pairwise.sum(dim=1)


class SoftRankICLoss(nn.Module):
    def __init__(self, tau: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.tau = float(tau)
        self.eps = float(eps)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        if y_pred.numel() < 2:
            return torch.zeros((), device=y_pred.device, dtype=torch.float32)

        pred_soft_rank = soft_rank(y_pred, tau=self.tau)
        true_rank = _torch_rank(y_true.detach())

        pred_centered = pred_soft_rank - pred_soft_rank.mean()
        true_centered = true_rank - true_rank.mean()
        numerator = torch.sum(pred_centered * true_centered)
        denominator = torch.sqrt(
            torch.sum(pred_centered ** 2) * torch.sum(true_centered ** 2) + self.eps
        )
        corr = numerator / denominator
        return 1 - corr


class HeadWeightedPairwiseLoss(nn.Module):
    def __init__(
        self,
        tau: float = 1.0,
        eps: float = 1e-8,
        top_k_focus: int = 3,
        head_boost: float = 3.0,
        top_internal_boost: float = 1.5,
        tail_weight: float = 0.0,
    ):
        super().__init__()
        self.tau = float(tau)
        self.eps = float(eps)
        self.top_k_focus = int(top_k_focus)
        self.head_boost = float(head_boost)
        self.top_internal_boost = float(top_internal_boost)
        self.tail_weight = float(tail_weight)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        if y_pred.numel() < 2:
            return torch.zeros((), device=y_pred.device, dtype=torch.float32)

        pred_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
        true_diff = y_true.unsqueeze(1) - y_true.unsqueeze(0)
        sign = torch.sign(true_diff)
        weight = torch.abs(true_diff)

        order = torch.argsort(y_true, descending=True)
        ranks = torch.empty_like(order, dtype=torch.long)
        ranks[order] = torch.arange(y_true.numel(), device=y_true.device, dtype=torch.long)
        top_mask = ranks < max(self.top_k_focus, 1)
        top_i = top_mask.unsqueeze(1)
        top_j = top_mask.unsqueeze(0)
        top_any = top_i | top_j
        top_both = top_i & top_j
        top_vs_rest = top_any & (~top_both)

        pair_multiplier = torch.full_like(weight, fill_value=max(self.tail_weight, 0.0), dtype=torch.float32)
        pair_multiplier = torch.where(top_any, torch.ones_like(pair_multiplier), pair_multiplier)
        pair_multiplier = torch.where(top_both, pair_multiplier * float(self.top_internal_boost), pair_multiplier)
        pair_multiplier = torch.where(top_vs_rest, pair_multiplier * float(self.head_boost), pair_multiplier)
        weight = weight * pair_multiplier
        valid = sign != 0
        valid = valid & (weight > 0)
        if not torch.any(valid):
            return torch.zeros((), device=y_pred.device, dtype=torch.float32)

        logits = -sign[valid] * pred_diff[valid] / max(self.tau, self.eps)
        losses = torch.nn.functional.softplus(logits)
        weighted = weight[valid] * losses
        return weighted.sum() / weight[valid].sum().clamp_min(self.eps)
