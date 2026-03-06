from __future__ import annotations

import pandas as pd


def backtest_top_k(scored_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    rows: list[dict] = []
    for date, day_df in scored_df.groupby("date"):
        # 每个交易日按预测分数选 top_k，收益使用该日样本对应的未来标签收益。
        picks = day_df.nlargest(top_k, "score")
        rows.append(
            {
                "date": date,
                "strategy_return": picks["label"].mean(),
                "market_return": day_df["label"].mean(),
            }
        )

    report = pd.DataFrame(rows).sort_values("date")
    report["strategy_return"] = report["strategy_return"].fillna(0.0)
    report["market_return"] = report["market_return"].fillna(0.0)
    report["equity_curve"] = (1.0 + report["strategy_return"]).cumprod()
    return report
