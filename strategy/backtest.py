from __future__ import annotations

import math

import pandas as pd


def backtest_top_k(scored_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    output_columns = [
        "date",
        "strategy_return",
        "market_return",
        "excess_return",
        "equity_curve",
        "market_curve",
        "relative_curve",
    ]
    if scored_df.empty:
        print("[Backtest] no scored rows; return empty backtest report.")
        return pd.DataFrame(columns=output_columns)

    rows: list[dict] = []
    for date, day_df in scored_df.groupby("date"):
        picks = day_df.nlargest(top_k, "score")
        rows.append(
            {
                "date": date,
                "strategy_return": picks["label"].mean(),
                "market_return": day_df["label"].mean(),
            }
        )

    if not rows:
        print("[Backtest] no grouped dates; return empty backtest report.")
        return pd.DataFrame(columns=output_columns)

    report = pd.DataFrame(rows).sort_values("date")
    report["strategy_return"] = report["strategy_return"].fillna(0.0)
    report["market_return"] = report["market_return"].fillna(0.0)
    report["excess_return"] = report["strategy_return"] - report["market_return"]
    report["equity_curve"] = (1.0 + report["strategy_return"]).cumprod()
    report["market_curve"] = (1.0 + report["market_return"]).cumprod()
    report["relative_curve"] = report["equity_curve"] / report["market_curve"].replace(0.0, pd.NA)
    return report


def summarize_backtest(report: pd.DataFrame) -> dict[str, float | int | None]:
    if report.empty:
        return {
            "n_backtest_days": 0,
            "top_k_mean_return": None,
            "market_mean_return": None,
            "excess_mean_return": None,
            "win_rate": None,
            "positive_excess_rate": None,
            "cumulative_return": None,
            "market_cumulative_return": None,
            "relative_return": None,
            "max_drawdown": None,
            "sharpe_annualized": None,
            "information_ratio": None,
        }

    strategy_return = report["strategy_return"].astype(float)
    market_return = report["market_return"].astype(float)
    excess_return = report["excess_return"].astype(float)
    equity_curve = report["equity_curve"].astype(float)
    market_curve = report["market_curve"].astype(float)
    relative_curve = report["relative_curve"].astype(float)

    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak.replace(0.0, pd.NA) - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    strategy_std = float(strategy_return.std(ddof=1)) if len(strategy_return) > 1 else 0.0
    excess_std = float(excess_return.std(ddof=1)) if len(excess_return) > 1 else 0.0
    annualizer = math.sqrt(252.0)
    sharpe_annualized = float(strategy_return.mean() / strategy_std * annualizer) if strategy_std > 0 else 0.0
    information_ratio = float(excess_return.mean() / excess_std * annualizer) if excess_std > 0 else 0.0

    return {
        "n_backtest_days": int(len(report)),
        "top_k_mean_return": float(strategy_return.mean()),
        "market_mean_return": float(market_return.mean()),
        "excess_mean_return": float(excess_return.mean()),
        "win_rate": float((strategy_return > 0.0).mean()),
        "positive_excess_rate": float((excess_return > 0.0).mean()),
        "cumulative_return": float(equity_curve.iloc[-1] - 1.0),
        "market_cumulative_return": float(market_curve.iloc[-1] - 1.0),
        "relative_return": float(relative_curve.iloc[-1] - 1.0),
        "max_drawdown": max_drawdown,
        "sharpe_annualized": sharpe_annualized,
        "information_ratio": information_ratio,
    }


def backtest_metric_guidance() -> dict[str, dict[str, str]]:
    return {
        "n_backtest_days": {
            "recommended_range": ">= 20，>= 40 更稳",
            "notes": "回测天数太短时，大部分指标都会失真。",
        },
        "top_k_mean_return": {
            "recommended_range": "> 0",
            "notes": "策略日均收益至少应为正，否则没有交易价值。",
        },
        "market_mean_return": {
            "recommended_range": "仅供参考",
            "notes": "同期开盘到开盘的市场平均收益。",
        },
        "excess_mean_return": {
            "recommended_range": "> 0",
            "notes": "长期应跑赢同日市场平均，通常比单看策略收益更重要。",
        },
        "win_rate": {
            "recommended_range": "> 50%，> 55% 更好",
            "notes": "胜率偏低时，需要更高的单次收益来补偿。",
        },
        "positive_excess_rate": {
            "recommended_range": "> 50%，> 55% 更好",
            "notes": "按日跑赢市场的频率，比单纯胜率更关键。",
        },
        "cumulative_return": {
            "recommended_range": "> 0",
            "notes": "累计收益应为正，但必须结合回撤一起看。",
        },
        "market_cumulative_return": {
            "recommended_range": "仅供参考",
            "notes": "基准累计收益，用于判断策略是否只是顺着市场上涨。",
        },
        "relative_return": {
            "recommended_range": "> 0",
            "notes": "相对市场的累计超额，适合比较不同模型。",
        },
        "max_drawdown": {
            "recommended_range": "-10% ~ 0，越接近 0 越好",
            "notes": "回撤绝对值越小越好；短测试里如果已经很深，通常不健康。",
        },
        "sharpe_annualized": {
            "recommended_range": "> 0.5 可接受，> 1.0 较好，> 1.5 很强",
            "notes": "短样本下容易虚高，必须结合回测天数一起看。",
        },
        "information_ratio": {
            "recommended_range": "> 0.3 可接受，> 0.5 较好，> 1.0 很强",
            "notes": "衡量超额收益质量，短样本下同样容易被放大。",
        },
    }
