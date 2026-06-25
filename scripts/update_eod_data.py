from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config_values import optional_float
from app.interrupts import run_cli
from data.fetcher import (
    AkshareFetcher,
    FetchConfig,
    INDEX_EOD_COLUMNS,
    INDUSTRY_DAILY_COLUMNS,
    INDUSTRY_MAP_COLUMNS,
    STOCK_EOD_COLUMNS,
    load_local_index_data,
    load_local_industry_daily,
    load_local_industry_map,
    load_local_stock_data,
    merge_time_series_frames,
    merge_stock_turnover_columns,
    save_frame,
    write_json,
)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update local stock, index, and industry data.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--backfill-history",
        action="store_true",
        help="Backfill earlier history inside the retention window instead of only fetching the latest missing days.",
    )
    parser.add_argument(
        "--backfill-days",
        type=int,
        default=None,
        help="Optional lookback window used together with --backfill-history.",
    )
    return parser.parse_args(argv)


def build_fetcher(config: dict) -> AkshareFetcher:
    fetch_cfg = config.get("fetch", {})
    return AkshareFetcher(
        FetchConfig(
            seed=int(config.get("seed", 7)),
            use_real_data=bool(config.get("use_real_data", True)),
            fallback_to_synthetic=bool(config.get("fallback_to_synthetic", False)),
            max_workers=int(fetch_cfg.get("max_workers", 4)),
            request_timeout=float(fetch_cfg.get("request_timeout", 15)),
            show_progress=bool(fetch_cfg.get("show_progress", True)),
        )
    )


def fallback_universe_frame(config: dict) -> pd.DataFrame:
    fallback_symbols = sorted(set(config.get("fallback_universe_symbols", [])))
    if not fallback_symbols:
        return pd.DataFrame(columns=["symbol", "name"])
    return pd.DataFrame({"symbol": fallback_symbols, "name": ""})


def load_stale_symbols(path: Path) -> pd.DataFrame:
    columns = ["symbol", "latest_date", "stale_trade_days", "removed_at", "reason"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    df["symbol"] = df["symbol"].astype(str)
    return df[columns].drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)


def load_universe_snapshot(path: Path) -> pd.DataFrame:
    columns = ["symbol", "name"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    for column in columns:
        if column not in df.columns:
            df[column] = "" if column == "name" else pd.NA
    df["symbol"] = df["symbol"].astype(str)
    df["name"] = df["name"].fillna("").astype(str)
    return df[columns].drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)


def save_universe_snapshot(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if "symbol" not in out.columns:
        out["symbol"] = pd.NA
    if "name" not in out.columns:
        out["name"] = ""
    out[["symbol", "name"]].drop_duplicates(subset=["symbol"], keep="last").sort_values(["symbol"]).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def save_stale_symbols(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["symbol", "latest_date", "stale_trade_days", "removed_at", "reason"]
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    out[columns].drop_duplicates(subset=["symbol"], keep="last").sort_values(["symbol"]).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def build_universe_frame(
    fetcher: AkshareFetcher,
    config: dict,
    existing_df: pd.DataFrame,
    snapshot_path: Path,
    extra_exclude_symbols: set[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    universe_cfg = config.get("universe", {})
    try:
        universe = fetcher.fetch_stock_universe()
        save_universe_snapshot(universe, snapshot_path)
        source = "stock_info_a_code_name"
    except Exception as exc:
        snapshot_df = load_universe_snapshot(snapshot_path)
        if not snapshot_df.empty:
            print(f"[DataUpdate] Failed to fetch stock universe, fallback to universe snapshot. reason={exc}")
            universe = snapshot_df
            source = "universe_snapshot"
        elif not existing_df.empty:
            print(f"[DataUpdate] Failed to fetch stock universe, fallback to local existing symbols. reason={exc}")
            universe = existing_df[["symbol"]].drop_duplicates().copy()
            universe["name"] = ""
            source = "fallback_existing"
        else:
            print(f"[DataUpdate] Failed to fetch stock universe, fallback to config list. reason={exc}")
            universe = fallback_universe_frame(config)
            source = "fallback_config"

    if universe_cfg.get("filters", {}).get("main_board_only", True):
        universe = universe[universe["symbol"].map(fetcher.is_main_board_symbol)].copy()

    exclude_symbols = set(universe_cfg.get("exclude_symbols", []))
    if extra_exclude_symbols:
        exclude_symbols.update({str(symbol) for symbol in extra_exclude_symbols})
    if exclude_symbols:
        universe = universe[~universe["symbol"].isin(exclude_symbols)].copy()

    return universe.drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True), source


def build_candidate_frame_from_local_eod(
    raw_df: pd.DataFrame,
    universe_frame: pd.DataFrame,
    config: dict,
    target_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    filters = config.get("universe", {}).get("filters", {})
    candidate = universe_frame.copy()
    if candidate.empty or raw_df.empty:
        candidate["last_price"] = pd.NA
        return candidate

    latest_rows = (
        raw_df.sort_values(["symbol", "date"])
        .groupby("symbol", as_index=False)
        .tail(1)[["symbol", "close", "date"]]
        .rename(columns={"close": "last_price", "date": "latest_date"})
    )
    latest_rows["latest_date"] = pd.to_datetime(latest_rows["latest_date"]).dt.normalize()
    candidate = candidate.merge(latest_rows, on="symbol", how="left")

    if filters.get("exclude_st", True) and "name" in candidate.columns:
        candidate = candidate[~candidate["name"].astype(str).str.upper().str.contains("ST", na=False)].copy()

    min_latest_price = float(filters.get("min_latest_price", 2.0))
    max_latest_price = optional_float(filters.get("max_latest_price"))
    candidate = candidate[
        candidate["last_price"].notna()
        & (candidate["last_price"] >= min_latest_price)
    ].copy()
    if max_latest_price is not None:
        candidate = candidate[candidate["last_price"] <= max_latest_price].copy()

    # 候选池只保留已经补到目标交易日的股票，避免停牌或缺数据股票混入训练和推理。
    if target_end is not None:
        candidate = candidate[candidate["latest_date"] == pd.Timestamp(target_end).normalize()].copy()

    return candidate.sort_values("symbol").reset_index(drop=True)


def build_industry_daily_from_stock_data(stock_df: pd.DataFrame, industry_map_df: pd.DataFrame) -> pd.DataFrame:
    if stock_df.empty or industry_map_df.empty:
        return pd.DataFrame(columns=INDUSTRY_DAILY_COLUMNS)

    merged = stock_df.merge(
        industry_map_df[["symbol", "industry_name", "industry_code"]].drop_duplicates(subset=["symbol"], keep="last"),
        on="symbol",
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=INDUSTRY_DAILY_COLUMNS)

    merged["pct_chg_effective"] = merged["pct_chg"]
    merged["pct_chg_effective"] = merged["pct_chg_effective"].where(
        merged["pct_chg_effective"].notna(),
        merged.groupby("symbol")["close"].pct_change(),
    )
    merged["turnover_rate_effective"] = pd.to_numeric(merged["turnover_rate"], errors="coerce")

    industry_daily = (
        merged.groupby(["date", "industry_name", "industry_code"], as_index=False)
        .agg(
            pct_chg=("pct_chg_effective", "mean"),
            turnover_rate=("turnover_rate_effective", "mean"),
            turnover=("turnover", "sum"),
            volume=("volume", "sum"),
            amplitude=("amplitude", "mean"),
        )
        .sort_values(["industry_code", "date"])
        .reset_index(drop=True)
    )

    industry_daily["pct_chg"] = pd.to_numeric(industry_daily["pct_chg"], errors="coerce").fillna(0.0)
    industry_daily["close"] = industry_daily.groupby("industry_code")["pct_chg"].transform(
        lambda x: 100.0 * (1.0 + x).cumprod()
    )
    industry_daily["chg"] = industry_daily.groupby("industry_code")["close"].diff()
    prev_close = industry_daily.groupby("industry_code")["close"].shift(1)
    industry_daily["open"] = prev_close.fillna(
        industry_daily["close"] / (1.0 + industry_daily["pct_chg"]).replace(0.0, np.nan)
    )
    industry_daily["open"] = industry_daily["open"].fillna(industry_daily["close"])
    base_high = industry_daily[["open", "close"]].max(axis=1)
    base_low = industry_daily[["open", "close"]].min(axis=1)
    amplitude = pd.to_numeric(industry_daily["amplitude"], errors="coerce").fillna(industry_daily["pct_chg"].abs())
    industry_daily["high"] = base_high * (1.0 + amplitude.clip(lower=0.0) / 2.0)
    industry_daily["low"] = base_low * (1.0 - amplitude.clip(lower=0.0) / 2.0)
    industry_daily["source"] = "local_industry_aggregate"
    return industry_daily[INDUSTRY_DAILY_COLUMNS].reset_index(drop=True)


def resolve_update_calendar(fetcher: AkshareFetcher, target_end: pd.Timestamp, history_days: int) -> pd.DatetimeIndex:
    full_start, _ = fetcher.resolve_date_range(None, target_end.date().isoformat(), history_days=history_days)
    try:
        calendar = fetcher.get_trade_calendar()
        calendar = pd.DatetimeIndex(pd.to_datetime(calendar).normalize())
        return calendar[(calendar >= pd.Timestamp(full_start)) & (calendar <= target_end.normalize())]
    except Exception:
        return pd.bdate_range(start=full_start, end=target_end.normalize())


def find_stale_symbols(
    stock_df: pd.DataFrame,
    update_calendar: pd.DatetimeIndex,
    target_end: pd.Timestamp,
    max_stale_trade_days: int,
) -> pd.DataFrame:
    if stock_df.empty or len(update_calendar) == 0 or int(max_stale_trade_days) <= 0:
        return pd.DataFrame(columns=["symbol", "latest_date", "stale_trade_days"])

    calendar = pd.DatetimeIndex(pd.to_datetime(update_calendar).normalize())
    calendar = calendar[calendar <= pd.Timestamp(target_end).normalize()]
    if len(calendar) == 0:
        return pd.DataFrame(columns=["symbol", "latest_date", "stale_trade_days"])

    latest_rows = (
        stock_df[["symbol", "date"]]
        .assign(date=lambda df: pd.to_datetime(df["date"]).dt.normalize())
        .groupby("symbol", as_index=False)["date"]
        .max()
        .rename(columns={"date": "latest_date"})
    )
    latest_rows["stale_trade_days"] = latest_rows["latest_date"].apply(lambda value: int((calendar > value).sum()))
    stale = latest_rows[latest_rows["stale_trade_days"] > int(max_stale_trade_days)].copy()
    return stale.sort_values(["stale_trade_days", "symbol"], ascending=[False, True]).reset_index(drop=True)


def build_stale_symbol_registry(stale_symbol_report: pd.DataFrame) -> pd.DataFrame:
    if stale_symbol_report.empty:
        return pd.DataFrame(columns=["symbol", "latest_date", "stale_trade_days", "removed_at", "reason"])

    out = stale_symbol_report.copy()
    out["latest_date"] = pd.to_datetime(out["latest_date"]).dt.date.astype(str)
    out["stale_trade_days"] = out["stale_trade_days"].astype(int)
    out["removed_at"] = pd.Timestamp.now().isoformat()
    out["reason"] = out["stale_trade_days"].map(lambda value: f"stale_gt_{int(value)}_trade_days")
    return out[["symbol", "latest_date", "stale_trade_days", "removed_at", "reason"]]


def split_job_keys(keys: list[str], chunk_size: int | None) -> list[list[str]]:
    normalized = [str(key) for key in keys if str(key)]
    if not normalized:
        return []
    if chunk_size is None or int(chunk_size) <= 0:
        return [normalized]
    size = max(1, int(chunk_size))
    return [normalized[start : start + size] for start in range(0, len(normalized), size)]


def build_incremental_fetch_jobs(
    existing_df: pd.DataFrame,
    key_column: str,
    target_keys: list[str],
    update_calendar: pd.DatetimeIndex,
    target_end: pd.Timestamp,
    backfill_history: bool = False,
    max_keys_per_job: int | None = None,
) -> list[tuple[list[str], str, str]]:
    if len(update_calendar) == 0 or not target_keys:
        return []

    latest_dates: dict[str, pd.Timestamp] = {}
    earliest_dates: dict[str, pd.Timestamp] = {}
    if not existing_df.empty:
        temp = existing_df[[key_column, "date"]].copy()
        temp["date"] = pd.to_datetime(temp["date"]).dt.normalize()
        latest_dates = temp.groupby(key_column)["date"].max().to_dict()
        earliest_dates = temp.groupby(key_column)["date"].min().to_dict()

    desired_start = pd.Timestamp(update_calendar[0]).normalize()
    end_date = target_end.normalize().date().isoformat()
    grouped_jobs: dict[tuple[str, str], list[str]] = {}

    for key in target_keys:
        latest_date = latest_dates.get(key)
        earliest_date = earliest_dates.get(key)

        if latest_date is None:
            start_date = desired_start.date().isoformat()
        elif backfill_history and (earliest_date is None or earliest_date > desired_start):
            # 显式回补模式下，如果本地历史深度不够，就从保留窗口起点重抓。
            start_date = desired_start.date().isoformat()
        else:
            future_dates = [trade_date for trade_date in update_calendar if trade_date > latest_date]
            if not future_dates:
                continue
            start_date = pd.Timestamp(future_dates[0]).date().isoformat()

        grouped_jobs.setdefault((start_date, end_date), []).append(key)

    jobs: list[tuple[list[str], str, str]] = []
    for (start_date, end_date), keys in grouped_jobs.items():
        for chunk in split_job_keys(keys, max_keys_per_job):
            jobs.append((chunk, start_date, end_date))
    return jobs


def update_stock_data(
    fetcher: AkshareFetcher,
    stock_df: pd.DataFrame,
    universe_symbols: list[str],
    target_end: pd.Timestamp,
    history_days: int,
    backfill_history: bool = False,
    persist_path: Path | None = None,
    max_symbols_per_job: int | None = None,
) -> tuple[pd.DataFrame, int]:
    current_existing = stock_df[stock_df["symbol"].isin(universe_symbols)].copy() if not stock_df.empty else stock_df
    update_calendar = resolve_update_calendar(fetcher, target_end=target_end, history_days=history_days)
    fetch_jobs = build_incremental_fetch_jobs(
        existing_df=current_existing,
        key_column="symbol",
        target_keys=universe_symbols,
        update_calendar=update_calendar,
        target_end=target_end,
        backfill_history=backfill_history,
        max_keys_per_job=max_symbols_per_job,
    )

    retention_start = pd.Timestamp(update_calendar.min()) if len(update_calendar) > 0 else target_end.normalize()
    for symbols, start_date, end_date in fetch_jobs:
        if pd.Timestamp(start_date) > pd.Timestamp(end_date):
            continue
        try:
            fetched = fetcher.fetch_stock_daily_data(symbols=symbols, start_date=start_date, end_date=end_date)
        except RuntimeError as exc:
            if "no rows returned" in str(exc).lower():
                print(
                    "[DataUpdate] No stock rows returned for this batch, skip it. "
                    f"start={start_date} end={end_date} symbols={len(symbols)}"
                )
                continue
            raise
        stock_df = merge_time_series_frames(stock_df, fetched, key_columns=["symbol", "date"])
        if not stock_df.empty:
            stock_df = stock_df[pd.to_datetime(stock_df["date"]) >= retention_start].copy()
        if persist_path is not None and not fetched.empty:
            save_frame(stock_df, persist_path, STOCK_EOD_COLUMNS, date_columns=["date"])
            print(
                "[DataUpdate] Stock batch persisted. "
                f"start={start_date} end={end_date} symbols={len(symbols)} rows={len(stock_df)}"
            )

    if not stock_df.empty:
        stock_df = stock_df[pd.to_datetime(stock_df["date"]) >= retention_start].copy()
    return stock_df.sort_values(["symbol", "date"]).reset_index(drop=True), len(fetch_jobs)


def update_stock_turnover_data(
    fetcher: AkshareFetcher,
    stock_df: pd.DataFrame,
    universe_symbols: list[str],
    target_end: pd.Timestamp,
    history_days: int,
    backfill_history: bool = False,
    persist_path: Path | None = None,
    max_symbols_per_job: int | None = None,
) -> tuple[pd.DataFrame, int, int]:
    if stock_df.empty or not universe_symbols:
        return stock_df, 0, 0

    update_calendar = resolve_update_calendar(fetcher, target_end=target_end, history_days=history_days)
    if len(update_calendar) == 0:
        return stock_df, 0, 0

    retention_start = pd.Timestamp(update_calendar.min()).normalize()
    scope = stock_df[
        stock_df["symbol"].astype(str).isin(universe_symbols)
        & (pd.to_datetime(stock_df["date"]).dt.normalize() >= retention_start)
    ].copy()
    if scope.empty:
        return stock_df, 0, 0

    turnover_missing = (
        pd.to_numeric(scope["turnover_rate"], errors="coerce").isna()
        | pd.to_numeric(scope["outstanding_share"], errors="coerce").isna()
    )
    missing_scope = scope[turnover_missing].copy()
    if missing_scope.empty:
        return stock_df, 0, 0

    grouped_jobs: dict[tuple[str, str], list[str]] = {}
    for symbol, symbol_df in missing_scope.groupby("symbol", sort=False):
        symbol_dates = pd.to_datetime(symbol_df["date"]).dt.normalize()
        start_date = symbol_dates.min().date().isoformat()
        end_date = symbol_dates.max().date().isoformat()
        if backfill_history:
            start_date = retention_start.date().isoformat()
            end_date = target_end.normalize().date().isoformat()
        grouped_jobs.setdefault((start_date, end_date), []).append(str(symbol))

    fetch_jobs: list[tuple[list[str], str, str]] = []
    for (start_date, end_date), symbols in grouped_jobs.items():
        for chunk in split_job_keys(symbols, max_symbols_per_job):
            fetch_jobs.append((chunk, start_date, end_date))
    supplemented_rows = 0
    for symbols, start_date, end_date in fetch_jobs:
        if pd.Timestamp(start_date) > pd.Timestamp(end_date):
            continue
        try:
            turnover_df = fetcher.fetch_stock_turnover_data(symbols=symbols, start_date=start_date, end_date=end_date)
        except RuntimeError as exc:
            if "no rows returned" in str(exc).lower():
                print(
                    "[DataUpdate] No stock turnover rows returned for this batch, skip it. "
                    f"start={start_date} end={end_date} symbols={len(symbols)}"
                )
                continue
            raise
        supplemented_rows += int(len(turnover_df))
        stock_df = merge_stock_turnover_columns(stock_df, turnover_df)
        if not stock_df.empty:
            stock_df = stock_df[pd.to_datetime(stock_df["date"]) >= retention_start].copy()
        if persist_path is not None and not turnover_df.empty:
            save_frame(stock_df, persist_path, STOCK_EOD_COLUMNS, date_columns=["date"])
            print(
                "[DataUpdate] Stock turnover batch persisted. "
                f"start={start_date} end={end_date} symbols={len(symbols)} rows={len(stock_df)}"
            )

    if not stock_df.empty:
        stock_df = stock_df[pd.to_datetime(stock_df["date"]) >= retention_start].copy()
    return stock_df.sort_values(["symbol", "date"]).reset_index(drop=True), len(fetch_jobs), supplemented_rows


def update_index_data(
    fetcher: AkshareFetcher,
    index_df: pd.DataFrame,
    config: dict,
    target_end: pd.Timestamp,
    history_days: int,
    backfill_history: bool = False,
    persist_path: Path | None = None,
    max_index_keys_per_job: int | None = None,
) -> tuple[pd.DataFrame, int]:
    index_cfg = config.get("index", {})
    if not index_cfg.get("enabled", True):
        return index_df, 0

    index_symbols = {str(k): str(v) for k, v in index_cfg.get("symbols", {}).items()}
    if not index_symbols:
        return index_df, 0

    update_calendar = resolve_update_calendar(fetcher, target_end=target_end, history_days=history_days)
    fetch_jobs = build_incremental_fetch_jobs(
        existing_df=index_df,
        key_column="index_key",
        target_keys=list(index_symbols.keys()),
        update_calendar=update_calendar,
        target_end=target_end,
        backfill_history=backfill_history,
        max_keys_per_job=max_index_keys_per_job,
    )

    retention_start = pd.Timestamp(update_calendar.min()) if len(update_calendar) > 0 else target_end.normalize()
    for index_keys, start_date, end_date in fetch_jobs:
        selected_symbols = {index_key: index_symbols[index_key] for index_key in index_keys}
        try:
            fetched_index = fetcher.fetch_index_daily_data(
                index_symbols=selected_symbols,
                start_date=start_date,
                end_date=end_date,
            )
        except RuntimeError as exc:
            if "no rows returned" in str(exc).lower():
                print(
                    "[DataUpdate] No index rows returned for this batch, skip it. "
                    f"start={start_date} end={end_date} index_count={len(index_keys)}"
                )
                continue
            raise
        index_df = merge_time_series_frames(index_df, fetched_index, key_columns=["index_key", "date"])
        if not index_df.empty:
            index_df = index_df[pd.to_datetime(index_df["date"]) >= retention_start].copy()
        if persist_path is not None and not fetched_index.empty:
            save_frame(index_df, persist_path, INDEX_EOD_COLUMNS, date_columns=["date"])
            print(
                "[DataUpdate] Index batch persisted. "
                f"start={start_date} end={end_date} index_count={len(index_keys)} rows={len(index_df)}"
            )

    if not index_df.empty:
        index_df = index_df[pd.to_datetime(index_df["date"]) >= retention_start].copy()
    return index_df.sort_values(["index_key", "date"]).reset_index(drop=True), len(fetch_jobs)


def update_industry_data(
    fetcher: AkshareFetcher,
    stock_df: pd.DataFrame,
    universe_symbols: list[str],
    industry_map_df: pd.DataFrame,
    existing_industry_daily_df: pd.DataFrame,
    config: dict,
    target_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, str, int]:
    industry_cfg = config.get("industry", {})
    if not industry_cfg.get("enabled", True):
        fallback_daily = (
            existing_industry_daily_df
            if not existing_industry_daily_df.empty
            else pd.DataFrame(columns=INDUSTRY_DAILY_COLUMNS)
        )
        return industry_map_df, fallback_daily, "disabled", 0

    known_symbols = set(industry_map_df["symbol"].dropna().astype(str)) if not industry_map_df.empty else set()
    missing_symbols = sorted(set(universe_symbols) - known_symbols)
    fetched_missing_symbols = 0
    source_parts: list[str] = []

    # 行业映射变化频率远低于日线，因此只补新股票；行业日线完全由本地股票数据聚合。
    if missing_symbols:
        try:
            fetched_map = fetcher.fetch_industry_map(
                symbols_filter=set(missing_symbols),
                end_date=target_end.date().isoformat(),
            )
            if not fetched_map.empty:
                industry_map_df = pd.concat([industry_map_df, fetched_map], ignore_index=True)
                industry_map_df = industry_map_df.drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)
                fetched_missing_symbols = int(fetched_map["symbol"].dropna().astype(str).nunique())
            source_parts.append("cninfo_change")
        except Exception as exc:
            print(f"[DataUpdate] Industry map fetch failed, keep existing map. reason={exc}")
            source_parts.append(f"keep_existing_map_failed_fetch:{exc}")
    else:
        source_parts.append("no_missing_symbols")

    try:
        industry_daily_df = build_industry_daily_from_stock_data(stock_df, industry_map_df)
    except Exception as exc:
        print(f"[DataUpdate] Industry daily aggregation failed, keep existing daily file. reason={exc}")
        if not existing_industry_daily_df.empty:
            source_parts.append(f"keep_existing_daily_failed_aggregate:{exc}")
            return industry_map_df, existing_industry_daily_df, "|".join(source_parts), fetched_missing_symbols
        source_parts.append(f"failed_no_existing_daily:{exc}")
        return industry_map_df, pd.DataFrame(columns=INDUSTRY_DAILY_COLUMNS), "|".join(source_parts), fetched_missing_symbols

    if industry_daily_df.empty and not existing_industry_daily_df.empty:
        print("[DataUpdate] Industry daily aggregation returned empty result, keep existing daily file.")
        source_parts.append("keep_existing_daily_empty_aggregate")
        return industry_map_df, existing_industry_daily_df, "|".join(source_parts), fetched_missing_symbols

    source_parts.append("local_aggregate")
    return industry_map_df, industry_daily_df, "|".join(source_parts), fetched_missing_symbols


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    fetcher = build_fetcher(config)

    data_cfg = config.get("data", {})
    stock_path = Path(data_cfg.get("path", "data/eod_daily.csv"))
    stock_meta_path = Path(data_cfg.get("meta_path", "data/eod_daily_meta.json"))
    stale_symbols_path = Path(data_cfg.get("stale_symbols_path", "data/stale_symbols.csv"))
    universe_snapshot_path = Path(data_cfg.get("universe_snapshot_path", "data/universe_snapshot.csv"))
    base_candidate_path = Path(data_cfg.get("base_candidate_path", "data/base_candidates.csv"))
    candidate_path = Path(data_cfg.get("candidate_path", "data/current_candidates.csv"))
    index_path = Path(data_cfg.get("index_path", "data/index_daily.csv"))
    industry_map_path = Path(data_cfg.get("industry_map_path", "data/industry_map.csv"))
    industry_daily_path = Path(data_cfg.get("industry_daily_path", "data/industry_daily.csv"))

    stock_df = load_local_stock_data(stock_path)
    index_df = load_local_index_data(index_path)
    industry_map_df = load_local_industry_map(industry_map_path)
    industry_daily_df = load_local_industry_daily(industry_daily_path)
    stale_symbol_registry = load_stale_symbols(stale_symbols_path)
    persisted_stale_symbols = set(stale_symbol_registry["symbol"].astype(str)) if not stale_symbol_registry.empty else set()

    target_end = fetcher.latest_closed_trade_date()
    history_days = int(data_cfg.get("calendar_lookback_days", 420))
    effective_history_days = int(args.backfill_days) if args.backfill_days is not None else history_days
    if effective_history_days <= 0:
        raise ValueError("--backfill-days must be positive.")
    job_chunk_size = int(config.get("fetch", {}).get("job_chunk_size", 100))
    if job_chunk_size <= 0:
        raise ValueError("fetch.job_chunk_size must be positive.")
    update_calendar = resolve_update_calendar(fetcher, target_end=target_end, history_days=effective_history_days)

    stale_cfg = config.get("universe", {}).get("filters", {})
    max_stale_trade_days = int(stale_cfg.get("max_stale_trade_days", 20))
    if persisted_stale_symbols:
        print(
            "[DataUpdate] Stale registry will be refreshed after data fetch. "
            f"current_registry_count={len(persisted_stale_symbols)}"
        )

    universe_frame, universe_source = build_universe_frame(
        fetcher,
        config,
        existing_df=stock_df,
        snapshot_path=universe_snapshot_path,
        extra_exclude_symbols=None,
    )
    universe_symbols = universe_frame["symbol"].dropna().astype(str).tolist()
    if not universe_symbols:
        raise ValueError("No symbols available for data update. Please set fallback_universe_symbols in config.")

    if args.backfill_history:
        print(
            f"[DataUpdate] Backfill mode enabled. "
            f"effective_history_days={effective_history_days} target_end={target_end.date()}"
        )

    stock_df, stock_job_count = update_stock_data(
        fetcher=fetcher,
        stock_df=stock_df,
        universe_symbols=universe_symbols,
        target_end=target_end,
        history_days=effective_history_days,
        backfill_history=bool(args.backfill_history),
        persist_path=stock_path,
        max_symbols_per_job=job_chunk_size,
    )
    stock_df, stock_turnover_job_count, stock_turnover_rows = update_stock_turnover_data(
        fetcher=fetcher,
        stock_df=stock_df,
        universe_symbols=universe_symbols,
        target_end=target_end,
        history_days=effective_history_days,
        backfill_history=bool(args.backfill_history),
        persist_path=stock_path,
        max_symbols_per_job=job_chunk_size,
    )
    save_frame(stock_df, stock_path, STOCK_EOD_COLUMNS, date_columns=["date"])

    stale_symbol_report = find_stale_symbols(
        stock_df=stock_df,
        update_calendar=update_calendar,
        target_end=target_end,
        max_stale_trade_days=max_stale_trade_days,
    )
    newly_stale_symbols = set(stale_symbol_report["symbol"].astype(str)) - persisted_stale_symbols if not stale_symbol_report.empty else set()
    stale_symbols = set(stale_symbol_report["symbol"].astype(str)) if not stale_symbol_report.empty else set()
    stale_symbol_registry = build_stale_symbol_registry(stale_symbol_report)
    save_stale_symbols(stale_symbol_registry, stale_symbols_path)

    candidate_frame = build_candidate_frame_from_local_eod(
        raw_df=stock_df,
        universe_frame=universe_frame,
        config=config,
        target_end=target_end,
    )
    base_candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_frame.to_csv(base_candidate_path, index=False, encoding="utf-8-sig")
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_frame.to_csv(candidate_path, index=False, encoding="utf-8-sig")

    index_df, index_job_count = update_index_data(
        fetcher=fetcher,
        index_df=index_df,
        config=config,
        target_end=target_end,
        history_days=effective_history_days,
        backfill_history=bool(args.backfill_history),
        persist_path=index_path,
        max_index_keys_per_job=job_chunk_size,
    )
    save_frame(index_df, index_path, INDEX_EOD_COLUMNS, date_columns=["date"])

    industry_map_df, industry_daily_df, industry_source, industry_missing_symbols = update_industry_data(
        fetcher=fetcher,
        stock_df=stock_df,
        universe_symbols=universe_symbols,
        industry_map_df=industry_map_df,
        existing_industry_daily_df=industry_daily_df,
        config=config,
        target_end=target_end,
    )
    save_frame(industry_map_df, industry_map_path, INDUSTRY_MAP_COLUMNS)
    save_frame(industry_daily_df, industry_daily_path, INDUSTRY_DAILY_COLUMNS, date_columns=["date"])

    meta = {
        "updated_at": pd.Timestamp.now().isoformat(),
        "target_end_date": target_end.date().isoformat(),
        "backfill_history": bool(args.backfill_history),
        "effective_history_days": int(effective_history_days),
        "fetch_job_chunk_size": int(job_chunk_size),
        "stock_path": str(stock_path),
        "stale_symbols_path": str(stale_symbols_path),
        "universe_snapshot_path": str(universe_snapshot_path),
        "base_candidate_path": str(base_candidate_path),
        "candidate_path": str(candidate_path),
        "index_path": str(index_path),
        "industry_map_path": str(industry_map_path),
        "industry_daily_path": str(industry_daily_path),
        "universe_source": universe_source,
        "industry_source": industry_source,
        "universe_count": int(len(universe_symbols)),
        "candidate_count": int(len(candidate_frame)),
        "max_stale_trade_days": int(max_stale_trade_days),
        "stale_symbols_excluded": int(len(stale_symbols)),
        "persisted_stale_symbols": int(len(persisted_stale_symbols)),
        "newly_stale_symbols": int(len(newly_stale_symbols)),
        "stale_symbols_registry_count": int(len(stale_symbol_registry)),
        "stale_symbol_samples": stale_symbol_report["symbol"].head(10).tolist() if not stale_symbol_report.empty else [],
        "stock_symbol_count": int(stock_df["symbol"].nunique()) if not stock_df.empty else 0,
        "stock_rows": int(len(stock_df)),
        "index_rows": int(len(index_df)),
        "industry_map_rows": int(len(industry_map_df)),
        "industry_daily_rows": int(len(industry_daily_df)),
        "stock_fetch_jobs": int(stock_job_count),
        "stock_turnover_fetch_jobs": int(stock_turnover_job_count),
        "stock_turnover_rows": int(stock_turnover_rows),
        "stock_turnover_non_null": int(pd.to_numeric(stock_df["turnover_rate"], errors="coerce").notna().sum())
        if not stock_df.empty
        else 0,
        "index_fetch_jobs": int(index_job_count),
        "industry_missing_symbols_fetched": int(industry_missing_symbols),
        "latest_stock_date": str(pd.to_datetime(stock_df["date"]).max().date()) if not stock_df.empty else None,
        "latest_index_date": str(pd.to_datetime(index_df["date"]).max().date()) if not index_df.empty else None,
        "latest_industry_date": str(pd.to_datetime(industry_daily_df["date"]).max().date())
        if not industry_daily_df.empty
        else None,
        "source": fetcher.last_source,
    }
    write_json(stock_meta_path, meta)

    print(f"Universe source: {universe_source}")
    print(f"Universe symbols: {len(universe_symbols)}")
    print(f"Candidates after local-close filter: {len(candidate_frame)}")
    print(f"Target end date: {target_end.date()}")
    print(f"Max stale trade days: {max_stale_trade_days}")
    print(f"Stale symbols after data fetch: {len(stale_symbols)}")
    print(f"Persisted stale symbols: {len(persisted_stale_symbols)}")
    print(f"Newly stale symbols: {len(newly_stale_symbols)}")
    print(f"Backfill mode: {bool(args.backfill_history)}")
    print(f"Effective history days: {effective_history_days}")
    print(f"Stock fetch jobs: {stock_job_count}")
    print(f"Stock turnover fetch jobs: {stock_turnover_job_count}")
    print(f"Stock turnover supplemented rows: {stock_turnover_rows}")
    print(f"Index fetch jobs: {index_job_count}")
    print(f"Industry missing symbols fetched: {industry_missing_symbols}")
    print(f"Stock rows: {len(stock_df)}")
    print(f"Index rows: {len(index_df)}")
    print(f"Industry map rows: {len(industry_map_df)}")
    print(f"Industry daily rows: {len(industry_daily_df)}")
    print(f"Latest stock date: {meta['latest_stock_date']}")
    print(f"Latest index date: {meta['latest_index_date']}")
    print(f"Latest industry date: {meta['latest_industry_date']}")
    print(f"Saved stock data to: {stock_path.resolve()}")
    print(f"Saved base candidates to: {base_candidate_path.resolve()}")
    print(f"Saved active candidates to: {candidate_path.resolve()}")


if __name__ == "__main__":
    run_cli(
        main,
        label="DataUpdate",
        detail="User interrupted. Remaining fetch tasks have been stopped.",
    )
