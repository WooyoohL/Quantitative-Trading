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
    base_columns = ["date", "symbol", "action", "target_notional", "score"]
    optional_columns = [column for column in ["buy_price", "buy_price_basis", "entry_price_ref_close"] if column in orders.columns]
    return orders[base_columns + optional_columns]
