from __future__ import annotations

import pandas as pd


def simulate_rebalance(top_k_df: pd.DataFrame, cash: float, max_positions: int) -> pd.DataFrame:
    if top_k_df.empty:
        return pd.DataFrame(columns=["date", "symbol", "action", "target_notional"])

    n_pos = min(max_positions, len(top_k_df))
    orders = top_k_df.nlargest(n_pos, "score").copy()
    allocation = cash / max(1, n_pos)
    orders["action"] = "BUY"
    orders["target_notional"] = allocation
    return orders[["date", "symbol", "action", "target_notional", "score"]]
