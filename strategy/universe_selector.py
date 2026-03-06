from __future__ import annotations

import pandas as pd


def _is_main_board_symbol(symbol: str) -> bool:
    """A-share main board only: SH(600/601/603/605), SZ(000/001/002)."""
    code = symbol.split(".")[0]
    return (
        code.startswith(("600", "601", "603", "605"))
        or code.startswith(("000", "001", "002"))
    )


def _rank_desc(series: pd.Series) -> pd.Series:
    return series.rank(method="average", ascending=False, pct=True)


def _rank_asc(series: pd.Series) -> pd.Series:
    return series.rank(method="average", ascending=True, pct=True)


def select_training_universe(raw_df: pd.DataFrame, config: dict) -> tuple[list[str], pd.DataFrame]:
    """Select symbols for small-capital ultra-short-term (1-3 day) training."""
    lookback_days = int(config.get("lookback_days", 60))
    min_symbols = int(config.get("min_symbols", 8))
    max_symbols = int(config.get("max_symbols", 20))

    filters = config.get("filters", {})
    min_turnover = float(filters.get("min_avg_turnover", 2e8))
    min_price = float(filters.get("min_median_price", 3.0))
    max_price = float(filters.get("max_median_price", 60.0))
    max_intraday_range = float(filters.get("max_avg_intraday_range", 0.09))
    max_abs_ret = float(filters.get("max_avg_abs_ret1", 0.05))
    main_board_only = bool(filters.get("main_board_only", True))

    # Explicit exclusions for ST/suspended/other risk flags.
    st_symbols = set(config.get("st_symbols", []))
    suspended_symbols = set(config.get("suspended_symbols", []))
    exclude_symbols = set(config.get("exclude_symbols", []))

    latest_dates = sorted(raw_df["date"].unique())
    if len(latest_dates) == 0:
        return [], pd.DataFrame()
    # 只使用最近 lookback_days 天做股票池评估，避免旧市场状态污染筛选结果。
    active_dates = set(latest_dates[-lookback_days:])
    recent = raw_df[raw_df["date"].isin(active_dates)].copy()
    recent = recent.sort_values(["symbol", "date"])

    recent["ret_1"] = recent.groupby("symbol")["close"].pct_change().fillna(0.0)
    recent["intraday_range"] = (recent["high"] - recent["low"]) / recent["close"].replace(0.0, pd.NA)
    recent["intraday_range"] = recent["intraday_range"].fillna(0.0)

    agg = (
        recent.groupby("symbol", as_index=False)
        .agg(
            avg_turnover=("turnover", "mean"),
            median_price=("close", "median"),
            avg_intraday_range=("intraday_range", "mean"),
            avg_abs_ret1=("ret_1", lambda x: x.abs().mean()),
            n_days=("date", "nunique"),
        )
        .sort_values("symbol")
        .reset_index(drop=True)
    )

    if main_board_only:
        agg = agg[agg["symbol"].map(_is_main_board_symbol)].copy()

    if st_symbols:
        agg = agg[~agg["symbol"].isin(st_symbols)].copy()
    if suspended_symbols:
        agg = agg[~agg["symbol"].isin(suspended_symbols)].copy()
    if exclude_symbols:
        agg = agg[~agg["symbol"].isin(exclude_symbols)].copy()

    # 硬过滤：先剔除明显不符合小资金快进快出约束的标的。
    mask = (
        (agg["avg_turnover"] >= min_turnover)
        & (agg["median_price"] >= min_price)
        & (agg["median_price"] <= max_price)
        & (agg["avg_intraday_range"] <= max_intraday_range)
        & (agg["avg_abs_ret1"] <= max_abs_ret)
    )
    filtered = agg[mask].copy()

    # 如果严格过滤后样本太少，则退化为按流动性补足最小股票数。
    if len(filtered) < min_symbols:
        filtered = agg.nlargest(min_symbols, "avg_turnover").copy()

    filtered["liq_rank"] = _rank_desc(filtered["avg_turnover"])
    # 价格越接近目标中枢(15元)得分应越高，因此这里使用降序 rank。
    filtered["price_rank"] = _rank_desc((filtered["median_price"] - 15.0).abs())
    filtered["range_rank"] = _rank_asc(filtered["avg_intraday_range"])
    filtered["ret_rank"] = _rank_asc(filtered["avg_abs_ret1"])

    # 综合打分：优先流动性，同时压制波动/日收益过于极端的标的。
    filtered["pool_score"] = (
        0.45 * filtered["liq_rank"]
        + 0.20 * filtered["range_rank"]
        + 0.20 * filtered["ret_rank"]
        + 0.15 * filtered["price_rank"]
    )

    selected = filtered.nlargest(max_symbols, "pool_score").copy()
    selected_symbols = selected["symbol"].tolist()
    return selected_symbols, selected
