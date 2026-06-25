from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from app.config_values import optional_float
from app.runtime import load_current_candidate_symbols, load_st_symbol_set
from data.dataset import AlphaDatasetBuilder
from data.splits import split_anchor_dates
from strategy.universe_selector import select_training_universe_as_of


@dataclass
class TrainingUniverse:
    training_mode: str
    universe_filters: dict
    selected_symbols: list[str]
    universe_report: pd.DataFrame
    context_stock_df: pd.DataFrame
    selection_reference_date: pd.Timestamp
    candidate_path: Path | None
    candidate_symbol_count: int | None


def resolve_training_mode(config: dict) -> str:
    training_mode = str(config.get("training", {}).get("mode", "market_rank")).strip().lower()
    if training_mode not in {"market_rank", "trade"}:
        raise ValueError(f"Unsupported training.mode={training_mode}. Expected 'market_rank' or 'trade'.")
    return training_mode


def resolve_training_universe_max_latest_price(
    *,
    config: dict,
    universe_filters: dict,
    training_mode: str,
) -> float | None:
    if training_mode != "trade":
        return None
    training_cfg = config.get("training", {})
    value = (
        training_cfg.get("universe_max_latest_price")
        if "universe_max_latest_price" in training_cfg
        else universe_filters.get("max_latest_price")
    )
    return optional_float(value)


def build_training_universe(
    *,
    config: dict,
    stock_df: pd.DataFrame,
    industry_map_df: pd.DataFrame,
    dataset_builder: AlphaDatasetBuilder,
    logger: Callable[[str], None],
) -> TrainingUniverse:
    rolling_cfg = config.get("rolling", {})
    training_cfg = config.get("training", {})
    training_mode = resolve_training_mode(config)

    required_continuous_tail_days = (
        int(rolling_cfg.get("train_days", 80))
        + int(rolling_cfg.get("valid_days", 20))
        + int(rolling_cfg.get("test_days", 20))
    )
    if training_mode == "trade":
        required_continuous_tail_days = 0

    universe_cfg = dict(config.get("universe", {}))
    universe_filters = dict(universe_cfg.get("filters", {}))
    universe_filters["min_continuous_tail_days"] = required_continuous_tail_days
    training_max_price = resolve_training_universe_max_latest_price(
        config=config,
        universe_filters=universe_filters,
        training_mode=training_mode,
    )
    if training_max_price is None:
        universe_filters.pop("training_max_latest_price", None)
    else:
        universe_filters["training_max_latest_price"] = float(training_max_price)
    universe_cfg["filters"] = universe_filters
    universe_cfg["st_symbol_set"] = sorted(load_st_symbol_set(config))

    base_frame_for_selection = dataset_builder._build_base_feature_frame(stock_df, industry_map_df)
    selection_split_dates = split_anchor_dates(
        base_frame_for_selection,
        train_days=int(rolling_cfg.get("train_days", 80)),
        valid_days=int(rolling_cfg.get("valid_days", 20)),
        test_days=int(rolling_cfg.get("test_days", 20)),
    )
    selection_reference_date = max(selection_split_dates["train"])

    logger(f"训练模式: {training_mode}")
    logger(f"开始筛选训练股票池: reference_end_date={pd.Timestamp(selection_reference_date).date()}")
    selected_symbols, universe_report = select_training_universe_as_of(
        stock_df,
        universe_cfg,
        reference_end_date=selection_reference_date,
    )

    candidate_path: Path | None = None
    candidate_symbol_count: int | None = None
    if bool(training_cfg.get("use_candidate_universe", False)):
        candidate_symbols, loaded_candidate_path = load_current_candidate_symbols(config)
        candidate_path = loaded_candidate_path
        candidate_symbol_count = int(len(candidate_symbols))
        if candidate_symbols:
            available_symbols = set(stock_df["symbol"].dropna().astype(str))
            selected_symbols = sorted(candidate_symbols & available_symbols)
            if not universe_report.empty and "symbol" in universe_report.columns:
                universe_report = universe_report[universe_report["symbol"].astype(str).isin(selected_symbols)].copy()
            logger(
                f"训练股票池按统一候选库限定: path={candidate_path} "
                f"candidate_symbols={len(candidate_symbols)} selected_symbols={len(selected_symbols)}"
            )
        else:
            logger(f"统一候选库为空或缺少 symbol 列，保留训练股票池: path={candidate_path}")

    if not selected_symbols:
        raise ValueError("Universe selection returned zero symbols.")
    logger(f"训练股票池筛选完成: selected_symbols={len(selected_symbols)}")

    selected_symbol_set = {str(symbol) for symbol in selected_symbols}
    context_stock_df = stock_df
    if training_mode == "trade":
        context_stock_df = stock_df[stock_df["symbol"].astype(str).isin(selected_symbol_set)].copy()
        logger(
            f"Trade 模式使用训练池上下文: rows={len(context_stock_df)} "
            f"symbols={context_stock_df['symbol'].nunique()}"
        )
    else:
        logger(
            f"Market-rank 模式使用全市场上下文: rows={len(context_stock_df)} "
            f"symbols={context_stock_df['symbol'].nunique()}"
        )

    return TrainingUniverse(
        training_mode=training_mode,
        universe_filters=universe_filters,
        selected_symbols=[str(symbol) for symbol in selected_symbols],
        universe_report=universe_report,
        context_stock_df=context_stock_df,
        selection_reference_date=pd.Timestamp(selection_reference_date),
        candidate_path=candidate_path,
        candidate_symbol_count=candidate_symbol_count,
    )
