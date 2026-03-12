from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from app.factories import build_trainer_config
from app.reporting import assess_run_usability, print_inference_run_summary
from app.runtime import configure_deterministic_training, load_config, resolve_training_frames, save_run_metadata
from models.encoders import build_model
from models.trainer import AlphaTrainer
from pipelines.context import prepare_training_context
from pipelines.recommendation import generate_recommendation_outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily inference from an archived run checkpoint.")
    parser.add_argument(
        "--source-run",
        type=str,
        required=True,
        help="Run directory name under outputs/runs or an absolute path to a run directory.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best.ckpt",
        help="Checkpoint filename inside the source run directory. Defaults to best.ckpt.",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Use data up to this trading date as T day, for example 2026-03-10.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional output directory name under outputs/inference_runs.",
    )
    parser.add_argument(
        "--shuffle-time-blocks",
        action="store_true",
        help="Keep compatibility with the training pipeline. Usually leave disabled for live inference.",
    )
    parser.add_argument(
        "--shuffle-block-size",
        type=int,
        default=None,
        help="Block size used with --shuffle-time-blocks. Defaults to seq_len.",
    )
    return parser.parse_args(argv)


def resolve_source_run(run_value: str) -> Path:
    candidate = Path(run_value)
    if candidate.exists():
        return candidate.resolve()

    fallback = Path("outputs") / "runs" / run_value
    if fallback.exists():
        return fallback.resolve()

    raise FileNotFoundError(f"Run directory not found: {run_value}")


def ensure_output_dir(source_run_dir: Path, requested_name: str | None) -> Path:
    base_dir = Path("outputs") / "inference_runs"
    base_dir.mkdir(parents=True, exist_ok=True)
    run_name = requested_name or f"{source_run_dir.name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = base_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir.resolve()


def load_checkpoint_payload(checkpoint_path: Path, device: torch.device) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source_run_dir = resolve_source_run(args.source_run)
    source_config_path = source_run_dir / "config.yaml"
    if not source_config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in source run: {source_run_dir}")

    config = load_config(source_config_path)
    configure_deterministic_training(int(config.get("seed", 7)))

    as_of_date = pd.Timestamp(args.as_of_date).normalize() if args.as_of_date else None
    stock_df, index_df, industry_map_df, industry_daily_df = resolve_training_frames(
        config,
        as_of_date=as_of_date,
        verbose=False,
    )
    training_context = prepare_training_context(
        config=config,
        stock_df=stock_df,
        index_df=index_df,
        industry_map_df=industry_map_df,
        industry_daily_df=industry_daily_df,
        time_block_shuffle=bool(args.shuffle_time_blocks),
        time_block_size=args.shuffle_block_size,
        verbose=False,
    )

    model_cfg = config.get("model", {})
    input_dim = len(training_context.dataset_bundle.feature_columns)
    model, resolved_model_config = build_model(input_dim=input_dim, model_cfg=model_cfg)
    trainer = AlphaTrainer(
        model=model,
        config=build_trainer_config(config),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    checkpoint_path = source_run_dir / args.checkpoint_name
    checkpoint_payload = load_checkpoint_payload(checkpoint_path, trainer.device)
    checkpoint_model_config = checkpoint_payload.get("model_config", {})
    checkpoint_features = checkpoint_model_config.get("feature_columns")
    if checkpoint_features and list(checkpoint_features) != list(training_context.dataset_bundle.feature_columns):
        raise ValueError(
            "Feature columns from current data/config do not match the checkpoint. "
            "Use the matching config or restore the previous feature pipeline first."
        )
    trainer.load_checkpoint(checkpoint_path)

    output_dir = ensure_output_dir(source_run_dir, args.output_name)
    recommendations = generate_recommendation_outputs(
        config=config,
        output_dir=output_dir,
        trainer=trainer,
        dataset_builder=training_context.dataset_builder,
        dataset_bundle=training_context.dataset_bundle,
        training_mode=training_context.training_mode,
        as_of_date=as_of_date,
        context_stock_df=training_context.context_stock_df,
        stock_df=stock_df,
        index_df=index_df,
        industry_map_df=industry_map_df,
        industry_daily_df=industry_daily_df,
        universe_report=training_context.universe_report,
    )

    source_summary_path = source_run_dir / "summary.json"
    source_summary: dict = {}
    if source_summary_path.exists():
        try:
            source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            source_summary = {}

    source_decision_label, source_decision_reasons = assess_run_usability(source_summary) if source_summary else ("-", [])

    summary = {
        "run_dir": str(output_dir),
        "source_run_dir": str(source_run_dir),
        "source_checkpoint": str(checkpoint_path),
        "source_best_epoch": source_summary.get("best_epoch"),
        "source_best_valid_ic": source_summary.get("best_valid_ic"),
        "source_best_valid_daily_ic": source_summary.get("best_valid_daily_ic"),
        "source_test_ic": source_summary.get("test_ic"),
        "source_test_daily_ic": source_summary.get("test_daily_ic"),
        "source_recommendation_label": source_decision_label,
        "source_recommendation_reasons": source_decision_reasons,
        "as_of_date": str(as_of_date.date()) if as_of_date is not None else None,
        "latest_stock_date": str(pd.to_datetime(stock_df["date"]).max().date()),
        "latest_index_date": str(pd.to_datetime(index_df["date"]).max().date()) if not index_df.empty else None,
        "latest_industry_date": str(pd.to_datetime(industry_daily_df["date"]).max().date())
        if not industry_daily_df.empty
        else None,
        "training_mode": training_context.training_mode,
        "signal_date": str(recommendations.signal_date.date()),
        "next_trade_date": str(recommendations.next_trade_date.date()),
        "universe_size": int(len(training_context.selected_symbols)),
        "candidate_filter_applied": recommendations.candidate_filter_applied,
        "candidate_symbol_count": recommendations.candidate_symbol_count,
        "market_rank_count": int(len(recommendations.market_rank)),
        "candidate_rank_count": int(len(recommendations.candidate_rank)),
        "recommendation_price_filter_applied": recommendations.recommendation_price_filter_applied,
        "recommendation_pool_count": int(len(recommendations.recommendation_pool)),
        "recommendation_max_latest_price": recommendations.recommendation_max_latest_price,
        "top_k_count": int(len(recommendations.latest_top_k)),
        "n_train_samples": int(len(training_context.dataset_bundle.train_dataset)),
        "n_valid_samples": int(len(training_context.dataset_bundle.valid_dataset)),
        "n_test_samples": int(len(training_context.dataset_bundle.test_dataset)),
        "seq_len": int(config.get("sequence", {}).get("seq_len", 20)),
        "feature_dim": int(input_dim),
        "model_name": str(resolved_model_config.get("name")),
        "feature_columns": training_context.dataset_bundle.feature_columns,
        "split_dates": {
            key: [pd.Timestamp(value).date().isoformat() for value in value_list]
            for key, value_list in training_context.dataset_bundle.split_dates.items()
        },
        "peer_symbol_count": int(sum(1 for value in training_context.dataset_bundle.peer_map.values() if value)),
    }
    save_run_metadata(output_dir, config, summary)
    print_inference_run_summary(
        summary,
        recommendations.latest_top_k,
        output_dir,
        source_run_dir,
        checkpoint_path.name,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Inference] User interrupted.")
        raise SystemExit(130)
