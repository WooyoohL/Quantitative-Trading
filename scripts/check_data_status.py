from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.interrupts import run_cli
from data.fetcher import (
    AkshareFetcher,
    FetchConfig,
    load_local_index_data,
    load_local_industry_daily,
    load_local_stock_data,
)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local stock, index, and industry data status.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    return parser.parse_args(argv)


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_fetcher(config: dict) -> AkshareFetcher:
    fetch_cfg = config.get("fetch", {})
    return AkshareFetcher(
        FetchConfig(
            seed=int(config.get("seed", 7)),
            use_real_data=bool(config.get("use_real_data", True)),
            fallback_to_synthetic=bool(config.get("fallback_to_synthetic", False)),
            max_workers=int(fetch_cfg.get("max_workers", 4)),
            request_timeout=float(fetch_cfg.get("request_timeout", 15)),
            show_progress=False,
        )
    )


def latest_date_or_none(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    return pd.to_datetime(df["date"]).max().date().isoformat()


def count_stale_symbols(
    stock_df: pd.DataFrame,
    fetcher: AkshareFetcher,
    latest_closed_trade_date: str,
    max_stale_trade_days: int,
) -> int:
    if stock_df.empty or int(max_stale_trade_days) <= 0:
        return 0
    try:
        calendar = fetcher.get_trade_calendar()
        calendar = pd.DatetimeIndex(pd.to_datetime(calendar).normalize())
        calendar = calendar[calendar <= pd.Timestamp(latest_closed_trade_date).normalize()]
    except Exception:
        calendar = pd.bdate_range(end=pd.Timestamp(latest_closed_trade_date), periods=max(60, int(max_stale_trade_days) * 3))
    latest_rows = (
        stock_df[["symbol", "date"]]
        .assign(date=lambda df: pd.to_datetime(df["date"]).dt.normalize())
        .groupby("symbol", as_index=False)["date"]
        .max()
    )
    latest_rows["stale_trade_days"] = latest_rows["date"].apply(lambda value: int((calendar > value).sum()))
    return int((latest_rows["stale_trade_days"] > int(max_stale_trade_days)).sum())


def recent_stock_coverage(
    stock_df: pd.DataFrame,
    fetcher: AkshareFetcher,
    latest_closed_trade_date: str,
    lookback_days: int = 60,
) -> dict:
    if stock_df.empty:
        return {
            "lookback_days": int(lookback_days),
            "expected_trade_days": 0,
            "missing_trade_days": 0,
            "latest_day_symbol_count": 0,
            "min_daily_symbol_count": 0,
        }
    try:
        calendar = fetcher.get_trade_calendar()
        calendar = pd.DatetimeIndex(pd.to_datetime(calendar).normalize())
        calendar = calendar[calendar <= pd.Timestamp(latest_closed_trade_date).normalize()]
    except Exception:
        calendar = pd.bdate_range(end=pd.Timestamp(latest_closed_trade_date), periods=int(lookback_days))
    recent_dates = pd.DatetimeIndex(calendar[-int(lookback_days) :])
    counts = (
        stock_df[["symbol", "date"]]
        .assign(date=lambda df: pd.to_datetime(df["date"]).dt.normalize())
        .groupby("date")["symbol"]
        .nunique()
    )
    coverage = pd.DataFrame({"date": recent_dates})
    coverage["symbol_count"] = coverage["date"].map(counts).fillna(0).astype(int)
    latest_count = int(coverage["symbol_count"].iloc[-1]) if not coverage.empty else 0
    min_count = int(coverage["symbol_count"].min()) if not coverage.empty else 0
    missing_days = int((coverage["symbol_count"] == 0).sum()) if not coverage.empty else 0
    partial_days = int((coverage["symbol_count"] > 0).sum()) if not coverage.empty else 0
    return {
        "lookback_days": int(lookback_days),
        "expected_trade_days": int(len(recent_dates)),
        "missing_trade_days": missing_days,
        "non_empty_trade_days": partial_days,
        "latest_day_symbol_count": latest_count,
        "min_daily_symbol_count": min_count,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    data_cfg = config.get("data", {})

    stock_path = Path(data_cfg.get("path", "data/eod_daily.csv"))
    meta_path = Path(data_cfg.get("meta_path", "data/eod_daily_meta.json"))
    stale_symbols_path = Path(data_cfg.get("stale_symbols_path", "data/stale_symbols.csv"))
    universe_snapshot_path = Path(data_cfg.get("universe_snapshot_path", "data/universe_snapshot.csv"))
    base_candidate_path = Path(data_cfg.get("base_candidate_path", "data/base_candidates.csv"))
    candidate_path = Path(data_cfg.get("candidate_path", "data/current_candidates.csv"))
    index_path = Path(data_cfg.get("index_path", "data/index_daily.csv"))
    industry_daily_path = Path(data_cfg.get("industry_daily_path", "data/industry_daily.csv"))

    stock_df = load_local_stock_data(stock_path)
    index_df = load_local_index_data(index_path)
    industry_daily_df = load_local_industry_daily(industry_daily_path)
    base_candidate_df = pd.read_csv(base_candidate_path) if base_candidate_path.exists() else pd.DataFrame(columns=["symbol"])
    candidate_df = pd.read_csv(candidate_path) if candidate_path.exists() else pd.DataFrame(columns=["symbol"])
    meta = read_json_if_exists(meta_path)

    fetcher = build_fetcher(config)
    latest_closed_trade_date = fetcher.latest_closed_trade_date().date().isoformat()
    max_stale_trade_days = int(config.get("universe", {}).get("filters", {}).get("max_stale_trade_days", 20))

    latest_stock_date = latest_date_or_none(stock_df)
    latest_index_date = latest_date_or_none(index_df)
    latest_industry_date = latest_date_or_none(industry_daily_df)
    current_stale_count = count_stale_symbols(stock_df, fetcher, latest_closed_trade_date, max_stale_trade_days)
    coverage = recent_stock_coverage(
        stock_df,
        fetcher,
        latest_closed_trade_date,
        lookback_days=max(60, int(max_stale_trade_days) * 3),
    )

    stock_up_to_date = latest_stock_date == latest_closed_trade_date
    index_up_to_date = (latest_index_date == latest_closed_trade_date) if not index_df.empty else False
    industry_up_to_date = (latest_industry_date == latest_closed_trade_date) if not industry_daily_df.empty else False

    print(f"股票列表来源: {meta.get('universe_source', 'unknown')}")
    print(f"总股票数: {meta.get('universe_count', int(stock_df['symbol'].nunique()) if not stock_df.empty else 0)}")
    print(f"基础候选股票数: {int(base_candidate_df['symbol'].nunique()) if 'symbol' in base_candidate_df.columns else 0}")
    print(f"当前候选股票数: {int(candidate_df['symbol'].nunique()) if 'symbol' in candidate_df.columns else 0}")
    print(f"长期无新数据阈值: {max_stale_trade_days} 个交易日")
    print(f"当前本地 stale 股票数: {current_stale_count}")
    print(f"上次 update 排除 stale 股票数: {meta.get('stale_symbols_excluded', 0)}")
    print(f"stale registry 股票数: {meta.get('stale_symbols_registry_count', 0)}")
    print(f"最近覆盖检查交易日数: {coverage['expected_trade_days']}")
    print(f"最近覆盖缺失交易日数: {coverage['missing_trade_days']}")
    print(f"最近覆盖最少股票数: {coverage['min_daily_symbol_count']}")
    print(f"最近收盘日股票数: {coverage['latest_day_symbol_count']}")
    print(f"本地最新股票日期: {latest_stock_date or 'N/A'}")
    print(f"本地最新指数日期: {latest_index_date or 'N/A'}")
    print(f"本地最新行业日期: {latest_industry_date or 'N/A'}")
    print(f"最近已收盘交易日: {latest_closed_trade_date}")
    print(f"股票是否更新到最近收盘: {'是' if stock_up_to_date else '否'}")
    print(f"指数是否更新到最近收盘: {'是' if index_up_to_date else '否'}")
    print(f"行业是否更新到最近收盘: {'是' if industry_up_to_date else '否'}")
    print(f"本地股票数据行数: {len(stock_df)}")
    print(f"本地指数数据行数: {len(index_df)}")
    print(f"本地行业数据行数: {len(industry_daily_df)}")
    print(f"基础候选池文件: {base_candidate_path.resolve()} {'存在' if base_candidate_path.exists() else '不存在'}")
    print(f"当前候选池文件: {candidate_path.resolve()} {'存在' if candidate_path.exists() else '不存在'}")
    print(f"Stale registry 文件: {stale_symbols_path.resolve()} {'存在' if stale_symbols_path.exists() else '不存在'}")
    print(f"Universe snapshot 文件: {universe_snapshot_path.resolve()} {'存在' if universe_snapshot_path.exists() else '不存在'}")
    print(f"Meta 文件: {meta_path.resolve()} {'存在' if meta_path.exists() else '不存在'}")


if __name__ == "__main__":
    run_cli(main, label="Status")
