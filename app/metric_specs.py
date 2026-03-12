from __future__ import annotations

from dataclasses import dataclass

import math


@dataclass(frozen=True)
class MetricSpec:
    label: str
    recommended_range: str
    percent: bool = False


TRAINING_SUMMARY_SPECS: dict[str, MetricSpec] = {
    "valid_excess_return": MetricSpec("日均超额", "> 0", percent=True),
    "valid_positive_excess_rate": MetricSpec("超额胜率", ">= 50%", percent=True),
    "valid_relative_return": MetricSpec("相对收益", "> 0", percent=True),
    "valid_max_drawdown": MetricSpec("最大回撤", "-10% ~ 0，越接近 0 越好", percent=True),
    "valid_daily_ic": MetricSpec("Daily IC", "> 0"),
    "valid_head_daily_ic": MetricSpec("Head IC", "> 0 更稳"),
    "test_excess_return": MetricSpec("日均超额", "> 0", percent=True),
    "test_positive_excess_rate": MetricSpec("超额胜率", ">= 50%", percent=True),
    "test_relative_return": MetricSpec("相对收益", "> 0", percent=True),
    "test_max_drawdown": MetricSpec("最大回撤", "-10% ~ 0，越接近 0 越好", percent=True),
    "test_daily_ic": MetricSpec("Daily IC", "> 0"),
    "test_head_daily_ic": MetricSpec("Head IC", "> 0 更稳"),
    "train_loss": MetricSpec("Train Loss", "越低越好"),
    "valid_loss": MetricSpec("Valid Loss", "越低越好"),
    "test_loss": MetricSpec("Test Loss", "越低越好"),
}


def format_metric_value(value: float | None, *, percent: bool) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if percent:
        return f"{value:.2%}"
    return f"{value:.4f}"


def format_metric_line(name: str, value: float | None, *, width: int = 10) -> str:
    spec = TRAINING_SUMMARY_SPECS[name]
    rendered = format_metric_value(value, percent=spec.percent)
    return f"{spec.label:<{width}} {rendered:>10}  ({spec.recommended_range})"
