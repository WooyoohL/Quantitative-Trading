from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecommendationDecision:
    label: str
    reasons: list[str]
    valid_pass: bool
    test_pass: bool


def decide_recommendation(
    *,
    valid_excess_mean_return: float | None,
    valid_positive_excess_rate: float | None,
    valid_daily_ic: float | None,
    valid_max_drawdown: float | None,
    test_relative_return: float | None,
    test_positive_excess_rate: float | None,
    test_daily_ic: float | None,
) -> RecommendationDecision:
    reasons: list[str] = []

    valid_pass = True
    if valid_excess_mean_return is None or float(valid_excess_mean_return) <= 0.0:
        valid_pass = False
        reasons.append("valid 超额收益未转正")
    if valid_positive_excess_rate is None or float(valid_positive_excess_rate) < 0.50:
        valid_pass = False
        reasons.append("valid 超额胜率不足 50%")
    if valid_daily_ic is None or float(valid_daily_ic) <= 0.0:
        valid_pass = False
        reasons.append("valid daily_ic 未转正")
    if valid_max_drawdown is None or float(valid_max_drawdown) < -0.10:
        valid_pass = False
        reasons.append("valid 回撤超过 -10%")

    if not valid_pass:
        return RecommendationDecision(
            label="不建议",
            reasons=reasons,
            valid_pass=False,
            test_pass=False,
        )

    test_pass = True
    if test_relative_return is None or float(test_relative_return) <= 0.0:
        test_pass = False
        reasons.append("test 相对收益未转正")
    if test_positive_excess_rate is None or float(test_positive_excess_rate) < 0.50:
        test_pass = False
        reasons.append("test 超额胜率不足 50%")
    if test_daily_ic is None or float(test_daily_ic) <= 0.0:
        test_pass = False
        reasons.append("test daily_ic 未转正")

    return RecommendationDecision(
        label="推荐" if test_pass else "观察",
        reasons=reasons,
        valid_pass=True,
        test_pass=test_pass,
    )
