from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.factories import build_dataset_builder, build_trainer_config, enabled_or_empty
from app.config_values import optional_float
from app.runtime import (
    build_rank_frame,
    load_config,
    load_current_candidate_symbols,
    load_symbol_name_map,
    resolve_training_frames,
)
from data.dataset import AlphaDatasetBuilder, DatasetBundle
from models.trainer_factory import build_alpha_trainer
from pipelines.recommendation import _apply_right_side_filter
from pipelines.training_universe import build_training_universe
from strategy.backtest import backtest_top_k, summarize_backtest


@dataclass
class PreprocessingState:
    name: str
    training_mode: str
    selected_symbols: list[str]
    context_stock_df: pd.DataFrame
    dataset_builder: AlphaDatasetBuilder
    dataset_bundle: DatasetBundle


@dataclass
class RankedOutput:
    name: str
    signal_date: pd.Timestamp
    raw_rank: pd.DataFrame
    candidate_rank: pd.DataFrame
    review_top_k: pd.DataFrame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare daily inference rankings under current preprocessing and reconstructed source-run preprocessing."
    )
    parser.add_argument("--source-run", required=True, help="Run directory name under outputs/runs or a run directory path.")
    parser.add_argument("--checkpoint-name", default="best.ckpt", help="Checkpoint filename inside source run.")
    parser.add_argument("--as-of-date", default=None, help="Target signal date. Omit to use latest local data.")
    parser.add_argument(
        "--source-as-of-date",
        default=None,
        help="Date used to reconstruct source-run preprocessing. Defaults to source summary latest_stock_date.",
    )
    parser.add_argument("--top-n", type=int, default=None, help="Review pool size. Defaults to strategy.review_top_k.")
    parser.add_argument("--output-dir", default="outputs/preprocessing_drift", help="Base output directory.")
    return parser.parse_args(argv)


def resolve_source_run(value: str) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate.resolve()
    fallback = REPO_ROOT / "outputs" / "runs" / value
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f"Run directory not found: {value}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint_payload(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_feature_columns(payload: dict[str, Any], source_summary: dict[str, Any]) -> list[str]:
    model_config = payload.get("model_config", {})
    columns = model_config.get("feature_columns") or source_summary.get("feature_columns")
    if not columns:
        raise ValueError("Cannot find feature_columns in checkpoint or source summary.")
    return [str(column) for column in columns]


def make_dataset_builder(config: dict[str, Any], feature_columns: list[str], *, verbose: bool = False) -> AlphaDatasetBuilder:
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
        feature_columns=feature_columns,
        daily_cross_sectional_norm=bool(data_cfg.get("daily_cross_sectional_norm", False)),
        verbose=verbose,
        random_seed=int(config.get("seed", 7)),
    )


def default_feature_columns(config: dict[str, Any]) -> list[str]:
    return list(build_dataset_builder(config, verbose=False).feature_columns)


def build_preprocessing_state(
    *,
    name: str,
    config: dict[str, Any],
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    industry_map_df: pd.DataFrame,
    industry_daily_df: pd.DataFrame,
    feature_columns: list[str],
) -> PreprocessingState:
    rolling_cfg = config.get("rolling", {})
    builder = make_dataset_builder(config, feature_columns, verbose=False)
    training_universe = build_training_universe(
        config=config,
        stock_df=stock_df,
        industry_map_df=enabled_or_empty(config, "industry", industry_map_df),
        dataset_builder=builder,
        logger=lambda _message: None,
    )
    selected_symbols = training_universe.selected_symbols
    context_stock_df = training_universe.context_stock_df

    dataset_bundle = builder.build_bundle(
        raw_df=context_stock_df,
        train_days=int(rolling_cfg.get("train_days", 80)),
        valid_days=int(rolling_cfg.get("valid_days", 20)),
        test_days=int(rolling_cfg.get("test_days", 20)),
        index_df=enabled_or_empty(config, "index", index_df),
        industry_map_df=enabled_or_empty(config, "industry", industry_map_df),
        industry_daily_df=enabled_or_empty(config, "industry", industry_daily_df),
        sample_symbols=selected_symbols,
    )
    return PreprocessingState(
        name=name,
        training_mode=training_universe.training_mode,
        selected_symbols=[str(symbol) for symbol in selected_symbols],
        context_stock_df=context_stock_df,
        dataset_builder=builder,
        dataset_bundle=dataset_bundle,
    )


def score_with_state(
    *,
    state: PreprocessingState,
    trainer: Any,
    config: dict[str, Any],
    target_stock_df: pd.DataFrame,
    target_index_df: pd.DataFrame,
    target_industry_map_df: pd.DataFrame,
    target_industry_daily_df: pd.DataFrame,
    signal_date: pd.Timestamp | None,
    live_mode: bool,
    top_n: int,
) -> RankedOutput:
    if state.training_mode == "trade":
        selected_set = {str(symbol) for symbol in state.selected_symbols}
        raw_df = target_stock_df[target_stock_df["symbol"].astype(str).isin(selected_set)].copy()
    else:
        raw_df = target_stock_df

    inference_dataset, _scaled = state.dataset_builder.build_inference_dataset(
        raw_df=raw_df,
        scaler=state.dataset_bundle.scaler,
        index_df=target_index_df if config.get("index", {}).get("enabled", True) else pd.DataFrame(),
        industry_map_df=target_industry_map_df if config.get("industry", {}).get("enabled", True) else pd.DataFrame(),
        industry_daily_df=target_industry_daily_df if config.get("industry", {}).get("enabled", True) else pd.DataFrame(),
        peer_map=state.dataset_bundle.peer_map,
        signal_date=signal_date,
    )
    if len(inference_dataset) == 0:
        raise ValueError(f"{state.name}: inference dataset is empty.")

    scored = inference_dataset.meta.copy()
    scored["score"] = trainer.predict_dataset(inference_dataset)
    actual_signal_date = pd.to_datetime(scored["signal_date"]).max()
    scored = scored[pd.to_datetime(scored["signal_date"]) == actual_signal_date].copy()
    symbol_name_map = load_symbol_name_map(config)
    scored["name"] = scored["symbol"].astype(str).map(symbol_name_map).fillna("")
    ranked = build_rank_frame(scored, "candidate_rank")
    ranked["market_rank"] = pd.NA
    ranked = ranked[
        [
            "signal_date",
            "symbol",
            "name",
            "industry_name",
            "score",
            "close",
            "market_rank",
            "candidate_rank",
            "ret_1",
            "ret_5",
            "intraday_ret",
            "ma_gap_5",
            "turnover_rate_1",
            "volume_ratio_1_prev",
            "volume_ratio_3_prev",
            "volume_ratio_5_prev",
            "volume_ratio_7_prev",
            "volume_ratio_5",
            "industry_ret_1_mean",
        ]
    ].copy()
    raw_rank = ranked.copy()

    candidate_rank = ranked
    if live_mode:
        candidate_symbols, _candidate_path = load_current_candidate_symbols(config)
        if candidate_symbols:
            candidate_rank = candidate_rank[candidate_rank["symbol"].astype(str).isin(candidate_symbols)].copy()

    candidate_rank = candidate_rank.sort_values(["candidate_rank", "symbol"]).reset_index(drop=True)
    universe_filters = config.get("universe", {}).get("filters", {})
    max_price = optional_float(universe_filters.get("max_latest_price"))
    review_pool = candidate_rank.copy()
    if max_price is not None:
        review_pool = review_pool[review_pool["close"] <= max_price].copy()
    review_pool, _applied, _fallback, _before, _after = _apply_right_side_filter(review_pool, config)

    review_top_k = review_pool.nsmallest(int(top_n), "candidate_rank").copy()
    return RankedOutput(
        name=state.name,
        signal_date=pd.Timestamp(actual_signal_date),
        raw_rank=raw_rank,
        candidate_rank=candidate_rank,
        review_top_k=review_top_k,
    )


def pairwise_metrics(left: RankedOutput, right: RankedOutput, top_values: list[int]) -> dict[str, Any]:
    left_rank = left.candidate_rank[["symbol", "score", "candidate_rank"]].rename(
        columns={"score": "score_left", "candidate_rank": "rank_left"}
    )
    right_rank = right.candidate_rank[["symbol", "score", "candidate_rank"]].rename(
        columns={"score": "score_right", "candidate_rank": "rank_right"}
    )
    merged = left_rank.merge(right_rank, on="symbol", how="inner")
    metrics: dict[str, Any] = {
        "left": left.name,
        "right": right.name,
        "left_count": int(len(left.candidate_rank)),
        "right_count": int(len(right.candidate_rank)),
        "common_count": int(len(merged)),
        "review_left_count": int(len(left.review_top_k)),
        "review_right_count": int(len(right.review_top_k)),
    }
    if len(merged) >= 2:
        metrics["score_pearson_common"] = float(merged["score_left"].corr(merged["score_right"], method="pearson"))
        metrics["score_spearman_common"] = float(merged["score_left"].corr(merged["score_right"], method="spearman"))
        metrics["rank_spearman_common"] = float(merged["rank_left"].corr(merged["rank_right"], method="spearman"))
        metrics["mean_abs_rank_delta_common"] = float((merged["rank_left"] - merged["rank_right"]).abs().mean())
        metrics["median_abs_rank_delta_common"] = float((merged["rank_left"] - merged["rank_right"]).abs().median())
    else:
        metrics["score_pearson_common"] = None
        metrics["score_spearman_common"] = None
        metrics["rank_spearman_common"] = None
        metrics["mean_abs_rank_delta_common"] = None
        metrics["median_abs_rank_delta_common"] = None

    for top_n in top_values:
        left_top = set(left.candidate_rank.nsmallest(top_n, "candidate_rank")["symbol"].astype(str))
        right_top = set(right.candidate_rank.nsmallest(top_n, "candidate_rank")["symbol"].astype(str))
        overlap = len(left_top & right_top)
        denom = max(1, min(len(left_top), len(right_top)))
        metrics[f"top_{top_n}_overlap"] = int(overlap)
        metrics[f"top_{top_n}_overlap_rate"] = float(overlap / denom)

    left_review = set(left.review_top_k["symbol"].astype(str))
    right_review = set(right.review_top_k["symbol"].astype(str))
    review_overlap = len(left_review & right_review)
    review_denom = max(1, min(len(left_review), len(right_review)))
    metrics["review_top_k_overlap"] = int(review_overlap)
    metrics["review_top_k_overlap_rate"] = float(review_overlap / review_denom)
    return metrics


def scaler_delta(left: PreprocessingState, right: PreprocessingState) -> dict[str, Any]:
    left_means = left.dataset_bundle.scaler.means
    right_means = right.dataset_bundle.scaler.means
    left_stds = left.dataset_bundle.scaler.stds
    right_stds = right.dataset_bundle.scaler.stds
    columns = [column for column in left.dataset_bundle.feature_columns if column in right.dataset_bundle.feature_columns]
    mean_delta = (left_means[columns] - right_means[columns]).abs()
    std_ratio = (left_stds[columns] / right_stds[columns].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return {
        "feature_count": int(len(columns)),
        "mean_abs_delta_mean": float(mean_delta.mean()),
        "mean_abs_delta_median": float(mean_delta.median()),
        "std_ratio_median": float(std_ratio.median()),
        "std_ratio_min": float(std_ratio.min()),
        "std_ratio_max": float(std_ratio.max()),
        "largest_mean_delta_features": mean_delta.sort_values(ascending=False).head(10).to_dict(),
        "largest_std_ratio_features": (std_ratio - 1.0).abs().sort_values(ascending=False).head(10).to_dict(),
    }


def build_labeled_dataset_with_preprocessing(
    *,
    state: PreprocessingState,
    config: dict[str, Any],
    raw_df: pd.DataFrame,
    index_df: pd.DataFrame,
    industry_map_df: pd.DataFrame,
    industry_daily_df: pd.DataFrame,
    allowed_dates: list[pd.Timestamp],
    sample_symbols: list[str],
):
    if state.training_mode == "trade":
        symbol_set = {str(symbol) for symbol in sample_symbols}
        raw_df = raw_df[raw_df["symbol"].astype(str).isin(symbol_set)].copy()

    feature_frame = state.dataset_builder._build_base_feature_frame(
        raw_df,
        enabled_or_empty(config, "industry", industry_map_df),
    )
    feature_frame = state.dataset_builder._attach_market_features(feature_frame)
    feature_frame = state.dataset_builder._attach_index_features(
        feature_frame,
        enabled_or_empty(config, "index", index_df),
    )
    feature_frame = state.dataset_builder._attach_industry_features(
        feature_frame,
        enabled_or_empty(config, "industry", industry_daily_df),
    )
    feature_frame = state.dataset_builder._attach_peer_features(feature_frame, state.dataset_bundle.peer_map)
    feature_frame = state.dataset_builder._finalize_feature_frame(feature_frame)
    scaled_frame = state.dataset_bundle.scaler.transform(feature_frame, state.dataset_bundle.feature_columns)
    if state.dataset_builder.daily_cross_sectional_norm:
        scaled_frame = state.dataset_builder._apply_daily_cross_sectional_norm(scaled_frame)
    return state.dataset_builder._build_sequence_dataset(
        scaled_frame,
        allowed_dates=allowed_dates,
        require_label=True,
        sample_symbols=sample_symbols,
    )


def evaluate_labeled_dataset(
    *,
    trainer: Any,
    dataset,
    split_name: str,
    scenario: str,
    top_k: int,
) -> dict[str, Any]:
    predictions = trainer.predict_dataset(dataset)
    targets = dataset.targets_numpy
    dates = dataset.meta["date"].to_numpy() if "date" in dataset.meta.columns else None
    eval_metrics = trainer.compute_eval_metrics(predictions, targets, dates=dates)

    scored = dataset.meta.copy()
    scored["score"] = predictions
    backtest_report = backtest_top_k(scored[["date", "symbol", "label", "score"]], top_k=int(top_k))
    backtest_metrics = summarize_backtest(backtest_report)
    date_values = pd.to_datetime(dataset.meta["date"]).drop_duplicates().sort_values()

    return {
        "scenario": scenario,
        "split": split_name,
        "n_samples": int(len(dataset)),
        "start_date": date_values.iloc[0].date().isoformat() if len(date_values) else None,
        "end_date": date_values.iloc[-1].date().isoformat() if len(date_values) else None,
        "ic": float(eval_metrics["ic"]),
        "daily_ic": float(eval_metrics["daily_ic"]),
        "head_daily_ic": float(eval_metrics["head_daily_ic"]),
        "loss": float(eval_metrics["total_loss"]),
        "top_k_mean_return": backtest_metrics.get("top_k_mean_return"),
        "market_mean_return": backtest_metrics.get("market_mean_return"),
        "excess_mean_return": backtest_metrics.get("excess_mean_return"),
        "win_rate": backtest_metrics.get("win_rate"),
        "positive_excess_rate": backtest_metrics.get("positive_excess_rate"),
        "cumulative_return": backtest_metrics.get("cumulative_return"),
        "relative_return": backtest_metrics.get("relative_return"),
        "max_drawdown": backtest_metrics.get("max_drawdown"),
        "sharpe_annualized": backtest_metrics.get("sharpe_annualized"),
        "information_ratio": backtest_metrics.get("information_ratio"),
    }


def evaluate_saved_predictions(
    *,
    path: Path,
    split_name: str,
    scenario: str,
    top_k: int,
) -> dict[str, Any]:
    scored = pd.read_csv(path)
    if scored.empty:
        raise ValueError(f"Saved prediction file is empty: {path}")
    predictions = scored["score"].to_numpy(dtype=float)
    targets = scored["label"].to_numpy(dtype=float)
    dates = pd.to_datetime(scored["date"]).to_numpy()
    from models.loss_functions import daily_rank_ic_mean, head_daily_rank_ic_mean, rank_ic

    backtest_report = backtest_top_k(scored[["date", "symbol", "label", "score"]], top_k=int(top_k))
    backtest_metrics = summarize_backtest(backtest_report)
    date_values = pd.to_datetime(scored["date"]).drop_duplicates().sort_values()
    return {
        "scenario": scenario,
        "split": split_name,
        "n_samples": int(len(scored)),
        "start_date": date_values.iloc[0].date().isoformat() if len(date_values) else None,
        "end_date": date_values.iloc[-1].date().isoformat() if len(date_values) else None,
        "ic": float(rank_ic(targets, predictions)),
        "daily_ic": float(daily_rank_ic_mean(targets, predictions, dates)),
        "head_daily_ic": float(head_daily_rank_ic_mean(targets, predictions, dates, top_n=10)),
        "loss": float(np.mean(np.square(predictions - targets))) if len(scored) else float("nan"),
        "top_k_mean_return": backtest_metrics.get("top_k_mean_return"),
        "market_mean_return": backtest_metrics.get("market_mean_return"),
        "excess_mean_return": backtest_metrics.get("excess_mean_return"),
        "win_rate": backtest_metrics.get("win_rate"),
        "positive_excess_rate": backtest_metrics.get("positive_excess_rate"),
        "cumulative_return": backtest_metrics.get("cumulative_return"),
        "relative_return": backtest_metrics.get("relative_return"),
        "max_drawdown": backtest_metrics.get("max_drawdown"),
        "sharpe_annualized": backtest_metrics.get("sharpe_annualized"),
        "information_ratio": backtest_metrics.get("information_ratio"),
    }


def saved_prediction_symbols(path: Path) -> list[str]:
    if not path.exists():
        return []
    frame = pd.read_csv(path, usecols=["symbol"])
    return sorted(frame["symbol"].dropna().astype(str).unique().tolist())


def split_metric_deltas(rows: list[dict[str, Any]], left: str, right: str) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    metrics = [
        "ic",
        "daily_ic",
        "head_daily_ic",
        "top_k_mean_return",
        "excess_mean_return",
        "win_rate",
        "positive_excess_rate",
        "cumulative_return",
        "relative_return",
        "max_drawdown",
        "sharpe_annualized",
        "information_ratio",
    ]
    out: list[dict[str, Any]] = []
    for split_name in sorted(frame["split"].dropna().unique()):
        left_row = frame[(frame["scenario"] == left) & (frame["split"] == split_name)]
        right_row = frame[(frame["scenario"] == right) & (frame["split"] == split_name)]
        if left_row.empty or right_row.empty:
            continue
        entry: dict[str, Any] = {"left": left, "right": right, "split": split_name}
        for metric in metrics:
            left_value = pd.to_numeric(left_row.iloc[0].get(metric), errors="coerce")
            right_value = pd.to_numeric(right_row.iloc[0].get(metric), errors="coerce")
            entry[f"{metric}_left"] = None if pd.isna(left_value) else float(left_value)
            entry[f"{metric}_right"] = None if pd.isna(right_value) else float(right_value)
            entry[f"{metric}_delta_right_minus_left"] = (
                None if pd.isna(left_value) or pd.isna(right_value) else float(right_value - left_value)
            )
        out.append(entry)
    return out


def write_ranked_outputs(output_dir: Path, ranked_outputs: list[RankedOutput]) -> None:
    for output in ranked_outputs:
        output.raw_rank.to_csv(output_dir / f"{output.name}_raw_rank.csv", index=False, encoding="utf-8-sig")
        output.candidate_rank.to_csv(output_dir / f"{output.name}_candidate_rank.csv", index=False, encoding="utf-8-sig")
        output.review_top_k.to_csv(output_dir / f"{output.name}_review_top_k.csv", index=False, encoding="utf-8-sig")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source_run_dir = resolve_source_run(args.source_run)
    source_config_path = source_run_dir / "config.yaml"
    checkpoint_path = source_run_dir / args.checkpoint_name
    if not source_config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in source run: {source_run_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    config = load_config(source_config_path)
    source_summary = load_json(source_run_dir / "summary.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_payload = load_checkpoint_payload(checkpoint_path, device)
    feature_columns = checkpoint_feature_columns(checkpoint_payload, source_summary)
    current_default_columns = default_feature_columns(config)
    feature_compatible_with_main_infer = feature_columns == current_default_columns

    source_as_of_text = args.source_as_of_date or source_summary.get("latest_stock_date") or source_summary.get("as_of_date")
    if not source_as_of_text:
        raise ValueError("source-as-of-date is required when source summary has no latest_stock_date/as_of_date.")
    source_as_of_date = pd.Timestamp(source_as_of_text).normalize()
    target_as_of_date = pd.Timestamp(args.as_of_date).normalize() if args.as_of_date else None
    live_mode = target_as_of_date is None
    top_n = int(args.top_n or config.get("strategy", {}).get("review_top_k", 20))

    source_stock_df, source_index_df, source_industry_map_df, source_industry_daily_df = resolve_training_frames(
        config,
        as_of_date=source_as_of_date,
        verbose=False,
    )
    target_stock_df, target_index_df, target_industry_map_df, target_industry_daily_df = resolve_training_frames(
        config,
        as_of_date=target_as_of_date,
        verbose=False,
    )

    source_state = build_preprocessing_state(
        name="source_state",
        config=config,
        stock_df=source_stock_df,
        index_df=source_index_df,
        industry_map_df=source_industry_map_df,
        industry_daily_df=source_industry_daily_df,
        feature_columns=feature_columns,
    )
    current_state = build_preprocessing_state(
        name="current_state",
        config=config,
        stock_df=target_stock_df,
        index_df=target_index_df,
        industry_map_df=target_industry_map_df,
        industry_daily_df=target_industry_daily_df,
        feature_columns=feature_columns,
    )
    source_same_universe = PreprocessingState(
        name="source_preprocess_same_universe",
        training_mode=source_state.training_mode,
        selected_symbols=current_state.selected_symbols,
        context_stock_df=target_stock_df[target_stock_df["symbol"].astype(str).isin(current_state.selected_symbols)].copy(),
        dataset_builder=source_state.dataset_builder,
        dataset_bundle=source_state.dataset_bundle,
    )

    trainer, _resolved_model_config = build_alpha_trainer(
        input_dim=len(feature_columns),
        seq_len=int(config.get("sequence", {}).get("seq_len", 20)),
        feature_columns=feature_columns,
        model_cfg=config.get("model", {}),
        trainer_config=build_trainer_config(config),
        device=device,
    )
    trainer.load_checkpoint(checkpoint_path)

    signal_date_arg = target_as_of_date if target_as_of_date is not None else None
    ranked_outputs = [
        score_with_state(
            state=current_state,
            trainer=trainer,
            config=config,
            target_stock_df=target_stock_df,
            target_index_df=target_index_df,
            target_industry_map_df=target_industry_map_df,
            target_industry_daily_df=target_industry_daily_df,
            signal_date=signal_date_arg,
            live_mode=live_mode,
            top_n=top_n,
        ),
        score_with_state(
            state=source_same_universe,
            trainer=trainer,
            config=config,
            target_stock_df=target_stock_df,
            target_index_df=target_index_df,
            target_industry_map_df=target_industry_map_df,
            target_industry_daily_df=target_industry_daily_df,
            signal_date=signal_date_arg,
            live_mode=live_mode,
            top_n=top_n,
        ),
        score_with_state(
            state=source_state,
            trainer=trainer,
            config=config,
            target_stock_df=target_stock_df,
            target_index_df=target_index_df,
            target_industry_map_df=target_industry_map_df,
            target_industry_daily_df=target_industry_daily_df,
            signal_date=signal_date_arg,
            live_mode=live_mode,
            top_n=top_n,
        ),
    ]

    split_metric_rows: list[dict[str, Any]] = []
    top_k = int(config.get("strategy", {}).get("top_k", 3))
    source_prediction_paths = {
        "valid": source_run_dir / "valid_predictions.csv",
        "test": source_run_dir / "test_predictions.csv",
    }
    source_saved_symbols = sorted(
        set(saved_prediction_symbols(source_prediction_paths["valid"]))
        | set(saved_prediction_symbols(source_prediction_paths["test"]))
    )
    source_eval_symbols = source_saved_symbols or source_state.selected_symbols
    for split_name in ["valid", "test"]:
        saved_prediction_path = source_prediction_paths[split_name]
        if saved_prediction_path.exists():
            split_metric_rows.append(
                evaluate_saved_predictions(
                    path=saved_prediction_path,
                    split_name=split_name,
                    scenario="source_window_source_preprocess",
                    top_k=top_k,
                )
            )
        else:
            source_dataset = getattr(source_state.dataset_bundle, f"{split_name}_dataset")
            split_metric_rows.append(
                evaluate_labeled_dataset(
                    trainer=trainer,
                    dataset=source_dataset,
                    split_name=split_name,
                    scenario="source_window_source_preprocess",
                    top_k=top_k,
                )
            )

        source_allowed_dates = source_state.dataset_bundle.split_dates[split_name]
        source_window_current_preprocess_dataset = build_labeled_dataset_with_preprocessing(
            state=current_state,
            config=config,
            raw_df=source_stock_df,
            index_df=source_index_df,
            industry_map_df=source_industry_map_df,
            industry_daily_df=source_industry_daily_df,
            allowed_dates=source_allowed_dates,
            sample_symbols=source_eval_symbols,
        )
        split_metric_rows.append(
            evaluate_labeled_dataset(
                trainer=trainer,
                dataset=source_window_current_preprocess_dataset,
                split_name=split_name,
                scenario="source_window_current_preprocess",
                top_k=top_k,
            )
        )

    for split_name, dataset in [
        ("valid", current_state.dataset_bundle.valid_dataset),
        ("test", current_state.dataset_bundle.test_dataset),
    ]:
        split_metric_rows.append(
            evaluate_labeled_dataset(
                trainer=trainer,
                dataset=dataset,
                split_name=split_name,
                scenario="current_window_current_preprocess",
                top_k=top_k,
            )
        )

        current_allowed_dates = current_state.dataset_bundle.split_dates[split_name]
        current_window_source_preprocess_dataset = build_labeled_dataset_with_preprocessing(
            state=source_state,
            config=config,
            raw_df=target_stock_df,
            index_df=target_index_df,
            industry_map_df=target_industry_map_df,
            industry_daily_df=target_industry_daily_df,
            allowed_dates=current_allowed_dates,
            sample_symbols=current_state.selected_symbols,
        )
        split_metric_rows.append(
            evaluate_labeled_dataset(
                trainer=trainer,
                dataset=current_window_source_preprocess_dataset,
                split_name=split_name,
                scenario="current_window_source_preprocess",
                top_k=top_k,
            )
        )

    run_name = f"{source_run_dir.name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = (REPO_ROOT / args.output_dir / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_ranked_outputs(output_dir, ranked_outputs)
    pd.DataFrame(split_metric_rows).to_csv(output_dir / "split_metrics.csv", index=False, encoding="utf-8-sig")

    top_values = sorted({3, 10, 20, top_n, 50})
    pairwise_rows = [
        pairwise_metrics(ranked_outputs[0], ranked_outputs[1], top_values),
        pairwise_metrics(ranked_outputs[0], ranked_outputs[2], top_values),
        pairwise_metrics(ranked_outputs[1], ranked_outputs[2], top_values),
    ]
    pd.DataFrame(pairwise_rows).to_csv(output_dir / "pairwise_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "source_run_dir": str(source_run_dir),
        "checkpoint": str(checkpoint_path),
        "source_as_of_date": source_as_of_date.date().isoformat(),
        "target_as_of_date": None if target_as_of_date is None else target_as_of_date.date().isoformat(),
        "live_mode": bool(live_mode),
        "top_n": int(top_n),
        "feature_count": int(len(feature_columns)),
        "feature_compatible_with_main_infer": bool(feature_compatible_with_main_infer),
        "default_feature_count_from_current_code": int(len(current_default_columns)),
        "source_selected_symbols": int(len(source_state.selected_symbols)),
        "current_selected_symbols": int(len(current_state.selected_symbols)),
        "signal_dates": {output.name: output.signal_date.date().isoformat() for output in ranked_outputs},
        "candidate_counts": {output.name: int(len(output.candidate_rank)) for output in ranked_outputs},
        "review_counts": {output.name: int(len(output.review_top_k)) for output in ranked_outputs},
        "scaler_delta_source_vs_current": scaler_delta(source_state, current_state),
        "pairwise_metrics": pairwise_rows,
        "split_metrics": split_metric_rows,
        "split_metric_deltas": (
            split_metric_deltas(
                split_metric_rows,
                left="source_window_source_preprocess",
                right="source_window_current_preprocess",
            )
            + split_metric_deltas(
                split_metric_rows,
                left="current_window_current_preprocess",
                right="current_window_source_preprocess",
            )
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Output: {output_dir}")
    print(f"Feature compatible with main infer: {feature_compatible_with_main_infer}")
    for row in pairwise_rows:
        print(
            f"{row['left']} vs {row['right']}: "
            f"review_overlap={row['review_top_k_overlap']}/{min(row['review_left_count'], row['review_right_count'])} "
            f"top_{top_n}_overlap_rate={row.get(f'top_{top_n}_overlap_rate'):.3f} "
            f"score_spearman={row['score_spearman_common']}"
        )


if __name__ == "__main__":
    main()
