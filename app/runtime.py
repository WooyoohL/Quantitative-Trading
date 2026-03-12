from __future__ import annotations

import os
import random
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import yaml

from data.fetcher import (
    AkshareFetcher,
    FetchConfig,
    load_local_index_data,
    load_local_industry_daily,
    load_local_industry_map,
    load_local_stock_data,
    write_json,
)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def log_step(message: str) -> None:
    print(f"[DataLoad] {message}")


def configure_deterministic_training(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def resolve_training_frames(
    config: dict,
    as_of_date: pd.Timestamp | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_cfg = config.get("data", {})
    stock_path = Path(data_cfg.get("path", "data/eod_daily.csv"))
    index_path = Path(data_cfg.get("index_path", "data/index_daily.csv"))
    industry_map_path = Path(data_cfg.get("industry_map_path", "data/industry_map.csv"))
    industry_daily_path = Path(data_cfg.get("industry_daily_path", "data/industry_daily.csv"))

    started_at = perf_counter()
    logger = log_step if verbose else (lambda _message: None)

    logger(f"加载股票日线: {stock_path}")
    stock_df = load_local_stock_data(stock_path)
    if stock_df.empty:
        raise FileNotFoundError(f"No local stock EOD data found at {stock_path}. Run scripts/update_eod_data.py first.")
    logger(f"股票日线加载完成: rows={len(stock_df)} symbols={stock_df['symbol'].nunique()}")

    if as_of_date is not None:
        as_of_date = pd.Timestamp(as_of_date).normalize()
        stock_df = stock_df[pd.to_datetime(stock_df["date"]).dt.normalize() <= as_of_date].copy()
        if stock_df.empty:
            raise ValueError(f"No stock data available on or before as_of_date={as_of_date.date()}.")
        logger(
            f"按 T 日截断股票日线: as_of_date={as_of_date.date()} "
            f"rows={len(stock_df)} symbols={stock_df['symbol'].nunique()}"
        )

    keep_days = int(data_cfg.get("trainable_history_days", 220))
    unique_dates = sorted(pd.to_datetime(stock_df["date"]).drop_duplicates())
    if len(unique_dates) > keep_days:
        active_dates = set(unique_dates[-keep_days:])
        stock_df = stock_df[stock_df["date"].isin(active_dates)].copy()
        logger(
            f"股票日线截取最近 {keep_days} 个交易日: rows={len(stock_df)} "
            f"symbols={stock_df['symbol'].nunique()}"
        )

    if stock_df.empty:
        raise ValueError("Local stock EOD data became empty after filtering.")

    logger(f"加载指数日线: {index_path}")
    index_df = load_local_index_data(index_path)
    if as_of_date is not None and not index_df.empty:
        index_df = index_df[pd.to_datetime(index_df["date"]).dt.normalize() <= as_of_date].copy()
    logger(f"指数日线加载完成: rows={len(index_df)}")

    logger(f"加载行业映射: {industry_map_path}")
    industry_map_df = load_local_industry_map(industry_map_path)
    logger(f"行业映射加载完成: rows={len(industry_map_df)}")

    logger(f"加载行业日线: {industry_daily_path}")
    industry_daily_df = load_local_industry_daily(industry_daily_path)
    if as_of_date is not None and not industry_daily_df.empty:
        industry_daily_df = industry_daily_df[pd.to_datetime(industry_daily_df["date"]).dt.normalize() <= as_of_date].copy()
    logger(f"行业日线加载完成: rows={len(industry_daily_df)}")

    logger(f"本地数据准备完成，总耗时 {perf_counter() - started_at:.2f}s")
    return (
        stock_df.sort_values(["symbol", "date"]).reset_index(drop=True),
        index_df.sort_values(["index_key", "date"]).reset_index(drop=True),
        industry_map_df.sort_values(["symbol"]).reset_index(drop=True),
        industry_daily_df.sort_values(["industry_code", "date"]).reset_index(drop=True),
    )


def load_current_candidate_symbols(config: dict) -> tuple[set[str], Path]:
    data_cfg = config.get("data", {})
    candidate_path = Path(data_cfg.get("candidate_path", "data/current_candidates.csv"))
    if not candidate_path.exists():
        return set(), candidate_path
    candidate_df = pd.read_csv(candidate_path)
    if "symbol" not in candidate_df.columns:
        return set(), candidate_path
    return set(candidate_df["symbol"].dropna().astype(str)), candidate_path


def load_st_symbol_set(config: dict) -> set[str]:
    data_cfg = config.get("data", {})
    snapshot_path = Path(data_cfg.get("universe_snapshot_path", "data/universe_snapshot.csv"))
    if not snapshot_path.exists():
        return set()
    snapshot_df = pd.read_csv(snapshot_path)
    if "symbol" not in snapshot_df.columns or "name" not in snapshot_df.columns:
        return set()
    names = snapshot_df["name"].fillna("").astype(str).str.upper()
    symbols = snapshot_df["symbol"].fillna("").astype(str)
    return set(symbols[names.str.contains("ST", na=False)])


def load_symbol_name_map(config: dict) -> pd.Series:
    data_cfg = config.get("data", {})
    snapshot_path = Path(data_cfg.get("universe_snapshot_path", "data/universe_snapshot.csv"))
    if not snapshot_path.exists():
        return pd.Series(dtype="object")
    snapshot_df = pd.read_csv(snapshot_path)
    if "symbol" not in snapshot_df.columns or "name" not in snapshot_df.columns:
        return pd.Series(dtype="object")
    normalized = snapshot_df[["symbol", "name"]].copy()
    normalized["symbol"] = normalized["symbol"].fillna("").astype(str)
    normalized["name"] = normalized["name"].fillna("").astype(str)
    normalized = normalized.drop_duplicates(subset=["symbol"], keep="last")
    return normalized.set_index("symbol")["name"]


def build_rank_frame(scored_df: pd.DataFrame, rank_column: str) -> pd.DataFrame:
    ranked = scored_df.sort_values(["score", "symbol"], ascending=[False, True]).reset_index(drop=True).copy()
    ranked[rank_column] = ranked.index + 1
    return ranked


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


def ensure_run_dir(base_dir: Path, run_name: str | None = None) -> Path:
    run_id = run_name or pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_metadata(run_dir: Path, config: dict, summary: dict) -> None:
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    write_json(run_dir / "summary.json", summary)


def format_rank_value(value: object) -> str:
    return str(int(value)) if pd.notna(value) else "-"


def tee_stdio_to_log(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8-sig")
    tee_stdout = TeeStream(sys.stdout, log_handle)
    tee_stderr = TeeStream(sys.stderr, log_handle)
    return log_handle, redirect_stdout(tee_stdout), redirect_stderr(tee_stderr)
