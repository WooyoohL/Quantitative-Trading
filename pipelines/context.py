from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.cache import (
    build_training_context_cache_key,
    load_training_context_cache,
    save_training_context_cache,
)
from app.factories import build_dataset_builder, enabled_or_empty
from app.runtime import log_step
from data.dataset import AlphaDatasetBuilder, DatasetBundle
from pipelines.training_universe import build_training_universe, resolve_training_mode


@dataclass
class TrainingContext:
    training_mode: str
    universe_filters: dict
    selected_symbols: list[str]
    selected_symbol_set: set[str]
    universe_report: pd.DataFrame
    context_stock_df: pd.DataFrame
    dataset_builder: AlphaDatasetBuilder
    dataset_bundle: DatasetBundle


def prepare_training_context(
    *,
    config: dict,
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    industry_map_df: pd.DataFrame,
    industry_daily_df: pd.DataFrame,
    verbose: bool = True,
) -> TrainingContext:
    logger = log_step if verbose else (lambda _message: None)

    training_cfg = config.get("training", {})
    rolling_cfg = config.get("rolling", {})
    training_mode = resolve_training_mode(config)

    dataset_builder = build_dataset_builder(
        config,
        verbose=verbose,
    )

    cache_cfg = dict(config.get("cache", {}))
    use_candidate_universe = bool(training_cfg.get("use_candidate_universe", False))
    cache_enabled = bool(cache_cfg.get("enabled", False)) and not use_candidate_universe
    cache_path: Path | None = None
    if cache_enabled:
        cache_key = build_training_context_cache_key(
            config=config,
            stock_df=stock_df,
            index_df=index_df,
            industry_map_df=industry_map_df,
            industry_daily_df=industry_daily_df,
        )
        cache_dir = Path(cache_cfg.get("path", "outputs/cache/training_context"))
        cache_path = cache_dir / f"{cache_key}.pkl"
        cached_payload = load_training_context_cache(cache_path)
        if cached_payload is not None:
            logger(f"训练上下文缓存命中: {cache_path}")
            selected_symbols = [str(symbol) for symbol in cached_payload["selected_symbols"]]
            selected_symbol_set = {str(symbol) for symbol in selected_symbols}
            context_stock_df = stock_df
            if training_mode == "trade":
                context_stock_df = stock_df[stock_df["symbol"].astype(str).isin(selected_symbol_set)].copy()
            return TrainingContext(
                training_mode=str(cached_payload["training_mode"]),
                universe_filters=dict(cached_payload["universe_filters"]),
                selected_symbols=selected_symbols,
                selected_symbol_set=selected_symbol_set,
                universe_report=pd.DataFrame(cached_payload["universe_report"]),
                context_stock_df=context_stock_df,
                dataset_builder=dataset_builder,
                dataset_bundle=cached_payload["dataset_bundle"],
            )

    training_universe = build_training_universe(
        config=config,
        stock_df=stock_df,
        industry_map_df=enabled_or_empty(config, "industry", industry_map_df),
        dataset_builder=dataset_builder,
        logger=logger,
    )
    selected_symbols = training_universe.selected_symbols
    selected_symbol_set = {str(symbol) for symbol in selected_symbols}
    universe_filters = training_universe.universe_filters
    universe_report = training_universe.universe_report
    context_stock_df = training_universe.context_stock_df

    dataset_bundle = dataset_builder.build_bundle(
        raw_df=context_stock_df,
        train_days=int(rolling_cfg.get("train_days", 80)),
        valid_days=int(rolling_cfg.get("valid_days", 20)),
        test_days=int(rolling_cfg.get("test_days", 20)),
        index_df=enabled_or_empty(config, "index", index_df),
        industry_map_df=enabled_or_empty(config, "industry", industry_map_df),
        industry_daily_df=enabled_or_empty(config, "industry", industry_daily_df),
        sample_symbols=selected_symbols,
    )

    if (
        len(dataset_bundle.train_dataset) == 0
        or len(dataset_bundle.valid_dataset) == 0
        or len(dataset_bundle.test_dataset) == 0
        or len(dataset_bundle.inference_dataset) == 0
    ):
        raise ValueError(
            "At least one split is empty after sequence construction. "
            "Increase local history or reduce seq_len / rolling windows."
        )

    if cache_path is not None:
        save_training_context_cache(
            cache_path,
            {
                "training_mode": training_mode,
                "universe_filters": universe_filters,
                "selected_symbols": selected_symbols,
                "universe_report": universe_report,
                "dataset_bundle": dataset_bundle,
            },
        )
        logger(f"训练上下文缓存已写入: {cache_path}")

    return TrainingContext(
        training_mode=training_mode,
        universe_filters=universe_filters,
        selected_symbols=selected_symbols,
        selected_symbol_set=selected_symbol_set,
        universe_report=universe_report,
        context_stock_df=context_stock_df,
        dataset_builder=dataset_builder,
        dataset_bundle=dataset_bundle,
    )

