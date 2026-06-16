from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HeatConfig:
    min_turnover: float = 300000000.0
    max_price: float | None = 40.0
    top_industry_pct: float = 0.20
    top_ret1_pct: float = 0.08
    top_ret3_pct: float = 0.10
    volume_breakout_ratio: float = 1.8
    volume_ratio_min: float = 1.5
    overheat_3d_return: float = 0.18
    overheat_limitup_days: int = 2
    limit_up_threshold: float = 0.095
    failed_limit_close_gap: float = 0.025
    intraday_fade_threshold: float = 0.05


def _pct_rank_desc(series: pd.Series) -> pd.Series:
    return series.rank(method="average", ascending=False, pct=True)


def _normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _load_config(config: dict | None) -> HeatConfig:
    raw = dict(config or {})
    allowed = set(HeatConfig.__dataclass_fields__.keys())
    return HeatConfig(**{key: value for key, value in raw.items() if key in allowed})


def build_market_heat_candidates(
    *,
    stock_df: pd.DataFrame,
    industry_map_df: pd.DataFrame,
    industry_daily_df: pd.DataFrame,
    signal_date: pd.Timestamp | str | None = None,
    config: dict | None = None,
    candidate_symbols: set[str] | None = None,
    limit: int = 30,
) -> pd.DataFrame:
    heat_cfg = _load_config(config)
    if stock_df.empty:
        return pd.DataFrame()

    frame = stock_df.copy()
    frame["date"] = _normalize_date(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)

    target_date = pd.Timestamp(signal_date).normalize() if signal_date is not None else frame["date"].max()
    scoped = frame[frame["date"] <= target_date].copy()
    if scoped.empty:
        return pd.DataFrame()

    latest = scoped.groupby("symbol", as_index=False).tail(1).copy()
    latest = latest[latest["date"] == target_date].copy()
    if latest.empty:
        return pd.DataFrame()

    if candidate_symbols:
        latest = latest[latest["symbol"].isin(candidate_symbols)].copy()
    if latest.empty:
        return pd.DataFrame()

    grouped = scoped.groupby("symbol", group_keys=False)
    scoped["prev_close"] = grouped["close"].shift(1)
    scoped["ret_1_calc"] = scoped["close"] / scoped["prev_close"].replace(0.0, np.nan) - 1.0
    scoped["ret_1_effective"] = pd.to_numeric(scoped["pct_chg"], errors="coerce").where(
        pd.to_numeric(scoped["pct_chg"], errors="coerce").notna(),
        scoped["ret_1_calc"],
    )
    scoped["ret_3"] = grouped["close"].pct_change(3)
    scoped["ret_5"] = grouped["close"].pct_change(5)
    scoped["intraday_ret"] = scoped["close"] / scoped["open"].replace(0.0, np.nan) - 1.0
    scoped["ma5"] = grouped["close"].transform(lambda value: value.rolling(5, min_periods=2).mean())
    scoped["ma_gap_5"] = scoped["close"] / scoped["ma5"].replace(0.0, np.nan) - 1.0
    scoped["volume_mean_5"] = grouped["volume"].transform(lambda value: value.rolling(5, min_periods=2).mean())
    scoped["volume_ratio_5"] = scoped["volume"] / scoped["volume_mean_5"].replace(0.0, np.nan)
    scoped["turnover_mean_5"] = grouped["turnover"].transform(lambda value: value.rolling(5, min_periods=2).mean())
    scoped["turnover_ratio_5"] = scoped["turnover"] / scoped["turnover_mean_5"].replace(0.0, np.nan)
    scoped["prev_20_high"] = grouped["high"].transform(lambda value: value.shift(1).rolling(20, min_periods=5).max())
    scoped["limitup_day"] = scoped["ret_1_effective"] >= float(heat_cfg.limit_up_threshold)
    scoped["limitup_days_3"] = grouped["limitup_day"].transform(lambda value: value.rolling(3, min_periods=1).sum())

    latest = latest.merge(
        scoped[
            [
                "date",
                "symbol",
                "prev_close",
                "ret_1_effective",
                "ret_3",
                "ret_5",
                "intraday_ret",
                "ma_gap_5",
                "volume_ratio_5",
                "turnover_ratio_5",
                "prev_20_high",
                "limitup_day",
                "limitup_days_3",
            ]
        ],
        on=["date", "symbol"],
        how="left",
    )

    if not industry_map_df.empty:
        mapping = industry_map_df[["symbol", "industry_name", "industry_code"]].copy()
        mapping["symbol"] = mapping["symbol"].astype(str)
        mapping = mapping.drop_duplicates(subset=["symbol"], keep="last")
        latest = latest.merge(mapping, on="symbol", how="left", suffixes=("", "_map"))
        if "industry_name_map" in latest.columns:
            if "industry_name" not in latest.columns:
                latest["industry_name"] = pd.NA
            latest["industry_name"] = latest["industry_name"].fillna(latest["industry_name_map"])
            latest = latest.drop(columns=["industry_name_map"])
        if "industry_code_map" in latest.columns:
            if "industry_code" not in latest.columns:
                latest["industry_code"] = pd.NA
            latest["industry_code"] = latest["industry_code"].fillna(latest["industry_code_map"])
            latest = latest.drop(columns=["industry_code_map"])
    else:
        latest["industry_name"] = ""
        latest["industry_code"] = ""

    industry_strength = pd.DataFrame(columns=["industry_code", "industry_ret_1", "industry_ret_3", "industry_rank_pct"])
    if not industry_daily_df.empty:
        industry_frame = industry_daily_df.copy()
        industry_frame["date"] = _normalize_date(industry_frame["date"])
        industry_frame = industry_frame[industry_frame["date"] <= target_date].sort_values(["industry_code", "date"])
        industry_frame["industry_ret_1"] = pd.to_numeric(industry_frame["pct_chg"], errors="coerce")
        industry_frame["industry_ret_3"] = industry_frame.groupby("industry_code")["close"].pct_change(3)
        latest_industry = industry_frame.groupby("industry_code", as_index=False).tail(1).copy()
        latest_industry = latest_industry[latest_industry["date"] == target_date].copy()
        if not latest_industry.empty:
            latest_industry["industry_rank_pct"] = _pct_rank_desc(latest_industry["industry_ret_1"].fillna(-np.inf))
            industry_strength = latest_industry[
                ["industry_code", "industry_ret_1", "industry_ret_3", "industry_rank_pct"]
            ].copy()

    latest = latest.merge(industry_strength, on="industry_code", how="left")
    latest["industry_rank_pct"] = latest["industry_rank_pct"].fillna(1.0)
    latest["ret1_rank_pct"] = _pct_rank_desc(latest["ret_1_effective"].fillna(-np.inf))
    latest["ret3_rank_pct"] = _pct_rank_desc(latest["ret_3"].fillna(-np.inf))
    latest["relative_strength_vs_industry"] = latest["ret_1_effective"].fillna(0.0) - latest["industry_ret_1"].fillna(0.0)

    latest["strong_industry"] = latest["industry_rank_pct"] <= float(heat_cfg.top_industry_pct)
    latest["top_ret1_not_overheated"] = (
        (latest["ret1_rank_pct"] <= float(heat_cfg.top_ret1_pct))
        & (latest["ret_3"].fillna(0.0) <= float(heat_cfg.overheat_3d_return))
        & (latest["limitup_days_3"].fillna(0).astype(float) < int(heat_cfg.overheat_limitup_days))
    )
    latest["volume_breakout"] = (
        (latest["close"] > latest["prev_20_high"])
        & (latest["volume_ratio_5"].fillna(0.0) >= float(heat_cfg.volume_breakout_ratio))
    )
    latest["limit_up"] = latest["ret_1_effective"].fillna(0.0) >= float(heat_cfg.limit_up_threshold)
    latest["failed_limit_or_fade"] = (
        (latest["high"] / latest["prev_close"].replace(0.0, np.nan) - 1.0 >= float(heat_cfg.limit_up_threshold))
        & ((latest["high"] - latest["close"]) / latest["high"].replace(0.0, np.nan) >= float(heat_cfg.failed_limit_close_gap))
    ) | (
        ((latest["high"] - latest["close"]) / latest["prev_close"].replace(0.0, np.nan) >= float(heat_cfg.intraday_fade_threshold))
    )
    latest["top_ret3"] = latest["ret3_rank_pct"] <= float(heat_cfg.top_ret3_pct)
    latest["industry_relative_strength"] = latest["relative_strength_vs_industry"] > 0.0
    latest["volume_active"] = latest["volume_ratio_5"].fillna(0.0) >= float(heat_cfg.volume_ratio_min)

    latest["liquidity_pass"] = pd.to_numeric(latest["turnover"], errors="coerce").fillna(0.0) >= float(heat_cfg.min_turnover)
    latest["price_pass"] = True
    if heat_cfg.max_price is not None:
        latest["price_pass"] = pd.to_numeric(latest["close"], errors="coerce").fillna(np.inf) <= float(heat_cfg.max_price)

    flag_columns = [
        "strong_industry",
        "top_ret1_not_overheated",
        "volume_breakout",
        "limit_up",
        "failed_limit_or_fade",
        "top_ret3",
        "industry_relative_strength",
        "volume_active",
    ]
    latest["heat_signal_count"] = latest[flag_columns].sum(axis=1).astype(int)
    latest["heat_score"] = (
        2.0 * latest["strong_industry"].astype(float)
        + 2.0 * latest["top_ret1_not_overheated"].astype(float)
        + 2.5 * latest["volume_breakout"].astype(float)
        + 1.5 * latest["limit_up"].astype(float)
        + 1.0 * latest["failed_limit_or_fade"].astype(float)
        + 1.5 * latest["top_ret3"].astype(float)
        + 1.0 * latest["industry_relative_strength"].astype(float)
        + 1.0 * latest["volume_active"].astype(float)
        + (1.0 - latest["ret1_rank_pct"].fillna(1.0))
        + (1.0 - latest["industry_rank_pct"].fillna(1.0))
    )
    latest["heat_reasons"] = latest[flag_columns].apply(
        lambda row: ",".join(column for column, enabled in row.items() if bool(enabled)),
        axis=1,
    )

    selected = latest[
        latest["liquidity_pass"]
        & latest["price_pass"]
        & (latest["heat_signal_count"] > 0)
    ].copy()
    if selected.empty:
        return pd.DataFrame()

    selected = selected.sort_values(["heat_score", "turnover", "symbol"], ascending=[False, False, True]).head(int(limit)).copy()
    selected["signal_date"] = target_date.date().isoformat()
    selected["heat_rank"] = range(1, len(selected) + 1)
    output_columns = [
        "signal_date",
        "symbol",
        "industry_name",
        "industry_code",
        "close",
        "ret_1_effective",
        "ret_3",
        "ret_5",
        "intraday_ret",
        "ma_gap_5",
        "volume_ratio_5",
        "turnover_ratio_5",
        "industry_ret_1",
        "industry_ret_3",
        "industry_rank_pct",
        "relative_strength_vs_industry",
        "heat_score",
        "heat_rank",
        "heat_signal_count",
        "heat_reasons",
    ]
    for column in output_columns:
        if column not in selected.columns:
            selected[column] = pd.NA
    return selected[output_columns].reset_index(drop=True)
