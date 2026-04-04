from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

from app.factories import build_trainer_config
from app.interrupts import run_cli
from app.reporting import print_training_run_summary
from app.runtime import (
    configure_deterministic_training,
    ensure_run_dir,
    load_config,
    save_run_metadata,
    tee_stdio_to_log,
    resolve_training_frames,
)
from models.encoders import build_model
from models.trainer import AlphaTrainer
from pipelines.context import prepare_training_context
from pipelines.evaluation import evaluate_trained_model
from pipelines.recommendation import generate_recommendation_outputs


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train alpha model from local EOD data.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Use data up to this trading date as T day, for example 2026-02-03.",
    )
    parser.add_argument(
        "--shuffle-time-blocks",
        action="store_true",
        help="Randomly shuffle non-overlapping time blocks on each symbol timeline while keeping order inside each block.",
    )
    parser.add_argument(
        "--shuffle-block-size",
        type=int,
        default=None,
        help="Block size used with --shuffle-time-blocks. Defaults to seq_len.",
    )
    return parser.parse_args(argv)


def run_training(args: argparse.Namespace, config: dict, run_dir: Path, as_of_date: pd.Timestamp | None) -> None:
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
    sequence_cfg = config.get("sequence", {})
    input_dim = len(training_context.dataset_bundle.feature_columns)
    model, resolved_model_config = build_model(input_dim=input_dim, model_cfg=model_cfg)
    trainer = AlphaTrainer(
        model=model,
        config=build_trainer_config(config),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    history_df = trainer.fit(
        training_context.dataset_bundle.train_dataset,
        training_context.dataset_bundle.valid_dataset,
        run_dir=run_dir,
        model_config={
            **resolved_model_config,
            "feature_columns": training_context.dataset_bundle.feature_columns,
        },
    )
    trainer.load_checkpoint(run_dir / "best.ckpt")

    evaluation = evaluate_trained_model(
        trainer=trainer,
        dataset_bundle=training_context.dataset_bundle,
        run_dir=run_dir,
        top_k=int(config["strategy"]["top_k"]),
    )
    recommendations = generate_recommendation_outputs(
        config=config,
        output_dir=run_dir,
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

    summary = {
        "run_dir": str(run_dir.resolve()),
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
        "right_side_filter_applied": recommendations.right_side_filter_applied,
        "right_side_filter_before_count": recommendations.right_side_filter_before_count,
        "right_side_filter_after_count": recommendations.right_side_filter_after_count,
        "review_top_k_target": int(recommendations.review_top_k_target),
        "review_top_k_count": int(len(recommendations.review_top_k)),
        "post_filter_applied": False,
        "final_top_k_target": int(config["strategy"]["top_k"]),
        "final_top_k_count": None,
        "n_train_samples": int(len(training_context.dataset_bundle.train_dataset)),
        "n_valid_samples": int(len(training_context.dataset_bundle.valid_dataset)),
        "n_test_samples": int(len(training_context.dataset_bundle.test_dataset)),
        "seq_len": int(sequence_cfg.get("seq_len", 20)),
        "time_block_shuffle": bool(args.shuffle_time_blocks),
        "time_block_size": int(args.shuffle_block_size or sequence_cfg.get("seq_len", 20)),
        "feature_dim": int(input_dim),
        "model_name": str(resolved_model_config.get("name")),
        "feature_columns": training_context.dataset_bundle.feature_columns,
        "split_dates": {
            key: [pd.Timestamp(value).date().isoformat() for value in value_list]
            for key, value_list in training_context.dataset_bundle.split_dates.items()
        },
        "peer_symbol_count": int(sum(1 for value in training_context.dataset_bundle.peer_map.values() if value)),
        "valid_ic": float(evaluation.valid_eval["ic"]),
        "valid_daily_ic": float(evaluation.valid_eval["daily_ic"]),
        "valid_head_daily_ic": float(evaluation.valid_eval["head_daily_ic"]),
        "test_ic": float(evaluation.test_eval["ic"]),
        "test_daily_ic": float(evaluation.test_eval["daily_ic"]),
        "test_head_daily_ic": float(evaluation.test_eval["head_daily_ic"]),
        "valid_loss": float(evaluation.valid_eval["total_loss"]),
        "valid_mse_loss": float(evaluation.valid_eval["mse_loss"]),
        "valid_pearson_loss": float(evaluation.valid_eval["pearson_loss"]),
        "test_loss": float(evaluation.test_eval["total_loss"]),
        "test_mse_loss": float(evaluation.test_eval["mse_loss"]),
        "test_pearson_loss": float(evaluation.test_eval["pearson_loss"]),
        "best_epoch": int(trainer.best_epoch),
        "best_valid_ic": float(trainer.monitor_best_valid_ic),
        "best_valid_daily_ic": float(trainer.monitor_best_valid_daily_ic),
        "selected_epoch_valid_ic": float(evaluation.valid_eval["ic"]),
        "selected_epoch_valid_daily_ic": float(evaluation.valid_eval["daily_ic"]),
        "checkpoint_selection_mode": str(trainer.config.checkpoint_selection_mode),
        "best_selection_score": None if pd.isna(trainer.best_selection_score) else float(trainer.best_selection_score),
        "selection_candidate_count": int(trainer.selection_candidate_count),
        "best_selection_breakdown": trainer.best_selection_breakdown,
        "valid_backtest_metrics": evaluation.valid_backtest_metrics,
        "backtest_metrics": evaluation.backtest_metrics,
        "backtest_metric_guidance": evaluation.backtest_metric_guidance,
    }
    save_run_metadata(run_dir, config, summary)

    outputs_cfg = config.get("outputs", {})
    if bool(outputs_cfg.get("write_latest_run_metadata", True)):
        latest_run_path = Path(outputs_cfg.get("latest_run_metadata", "outputs/latest_run.json"))
        latest_run_path.parent.mkdir(parents=True, exist_ok=True)
        latest_run_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print_training_run_summary(summary, history_df, recommendations.review_top_k, run_dir)


def train_main(argv: list[str] | None = None) -> None:
    args = parse_train_args(argv)
    config = load_config(args.config)
    configure_deterministic_training(int(config.get("seed", 7)))

    run_dir = ensure_run_dir(Path(config.get("outputs", {}).get("run_dir", "outputs/runs")), run_name=args.run_name)
    log_path = run_dir / "run.log"
    as_of_date = pd.Timestamp(args.as_of_date).normalize() if args.as_of_date else None

    log_handle, stdout_cm, stderr_cm = tee_stdio_to_log(log_path)
    with log_handle, stdout_cm, stderr_cm:
        print(f"[RunLog] {log_path.resolve()}")
        run_training(args=args, config=config, run_dir=run_dir, as_of_date=as_of_date)


def print_root_help() -> None:
    print("Usage: python main.py <command> [options]")
    print("")
    print("Commands:")
    print("  train        训练模型并生成推荐")
    print("  infer        使用历史 run 权重做推理")
    print("  update-data  更新本地数据")
    print("  check-data   检查本地数据状态")
    print("  batch        跑批量实验")
    print("  analyze      分析 batch 结果")
    print("  sweep        生成或运行模型超参数 sweep")
    print("")
    print("兼容模式:")
    print("  python main.py --config config.yaml")
    print("  等价于: python main.py train --config config.yaml")


def dispatch_main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        train_main([])
        return
    if argv[0] in {"-h", "--help", "help"}:
        print_root_help()
        return

    subcommand = argv[0]
    sub_argv = argv[1:]
    if subcommand == "train":
        train_main(sub_argv)
        return
    if subcommand == "infer":
        from scripts.infer_from_run import main as infer_main

        infer_main(sub_argv)
        return
    if subcommand == "update-data":
        from scripts.update_eod_data import main as update_main

        update_main(sub_argv)
        return
    if subcommand == "check-data":
        from scripts.check_data_status import main as check_main

        check_main(sub_argv)
        return
    if subcommand == "batch":
        from scripts.batch_experiments import main as batch_main

        batch_main(sub_argv)
        return
    if subcommand == "analyze":
        from scripts.analysis import main as analysis_main

        analysis_main(sub_argv)
        return
    if subcommand == "sweep":
        from scripts.model_hparam_sweep import main as sweep_main

        sweep_main(sub_argv)
        return

    train_main(argv)


if __name__ == "__main__":
    run_cli(dispatch_main, label="Train")

