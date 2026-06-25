from __future__ import annotations

import pandas as pd

from data.dataset import AlphaDatasetBuilder
from models.trainer import TrainerConfig


def build_dataset_builder(
    config: dict,
    *,
    verbose: bool = True,
) -> AlphaDatasetBuilder:
    sequence_cfg = config.get("sequence", {})
    data_cfg = config.get("data", {})
    index_cfg = config.get("index", {})
    peer_cfg = config.get("peer", {})
    index_keys = list(index_cfg.get("symbols", {}).keys()) if index_cfg.get("enabled", True) else []
    return AlphaDatasetBuilder(
        seq_len=int(sequence_cfg.get("seq_len", 20)),
        label_horizon=int(data_cfg.get("label_horizon", 1)),
        index_keys=index_keys,
        peer_enabled=bool(peer_cfg.get("enabled", True)),
        peer_top_k=int(peer_cfg.get("top_k", 5)),
        peer_lookback_days=int(peer_cfg.get("lookback_days", 60)),
        peer_min_overlap=int(peer_cfg.get("min_overlap", 20)),
        daily_cross_sectional_norm=bool(data_cfg.get("daily_cross_sectional_norm", False)),
        verbose=bool(verbose),
    )


def build_trainer_config(config: dict) -> TrainerConfig:
    model_cfg = config.get("model", {})
    return TrainerConfig(
        epochs=int(model_cfg.get("epochs", 60)),
        lr=float(model_cfg.get("lr", 1e-3)),
        mse_loss_weight=float(model_cfg.get("mse_loss_weight", 1.0)),
        weight_decay=float(model_cfg.get("weight_decay", 1e-2)),
        pearson_loss_weight=float(model_cfg.get("pearson_loss_weight", 0.0)),
        soft_rank_loss_weight=float(model_cfg.get("soft_rank_loss_weight", 0.0)),
        pairwise_loss_weight=float(model_cfg.get("pairwise_loss_weight", 0.0)),
        ranking_tau=float(model_cfg.get("ranking_tau", 1.0)),
        pairwise_top_k_focus=int(model_cfg.get("pairwise_top_k_focus", config.get("strategy", {}).get("top_k", 3))),
        pairwise_head_boost=float(model_cfg.get("pairwise_head_boost", 3.0)),
        pairwise_top_internal_boost=float(model_cfg.get("pairwise_top_internal_boost", 1.5)),
        pairwise_tail_weight=float(model_cfg.get("pairwise_tail_weight", 0.0)),
        batch_size=int(model_cfg.get("batch_size", 256)),
        eval_batch_size=int(model_cfg.get("eval_batch_size", 512)),
        log_every=int(model_cfg.get("log_every", 1)),
        early_stopping_patience=int(model_cfg.get("early_stopping_patience", 10)),
        num_workers=int(model_cfg.get("num_workers", 0)),
        seed=int(config.get("seed", 7)),
        checkpoint_selection_mode=str(model_cfg.get("checkpoint_selection_mode", "valid_ic")),
        selection_top_k=int(model_cfg.get("selection_top_k", config.get("strategy", {}).get("top_k", 3))),
        selection_ic_tolerance=float(model_cfg.get("selection_ic_tolerance", 0.01)),
        selection_weight_ic=float(model_cfg.get("selection_weight_ic", 0.50)),
        selection_weight_top_k_return=float(model_cfg.get("selection_weight_top_k_return", 0.25)),
        selection_weight_hit_rate=float(model_cfg.get("selection_weight_hit_rate", 0.15)),
        selection_weight_excess_return=float(model_cfg.get("selection_weight_excess_return", 0.10)),
        selection_head_top_n=int(model_cfg.get("selection_head_top_n", 10)),
        selection_min_excess_return=float(model_cfg.get("selection_min_excess_return", 0.0)),
        selection_min_positive_excess_rate=float(model_cfg.get("selection_min_positive_excess_rate", 0.50)),
        selection_min_daily_ic=float(model_cfg.get("selection_min_daily_ic", 0.0)),
        selection_min_head_daily_ic=float(model_cfg.get("selection_min_head_daily_ic", -1.0)),
        selection_max_drawdown_limit=float(model_cfg.get("selection_max_drawdown_limit", -0.10)),
    )


def enabled_or_empty(config: dict, key: str, df: pd.DataFrame) -> pd.DataFrame:
    return df if config.get(key, {}).get("enabled", True) else pd.DataFrame()
