from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
import yaml

from data.dataset import AlphaDatasetBuilder
from data.fetcher import (
    AkshareFetcher,
    FetchConfig,
    load_local_index_data,
    load_local_industry_daily,
    load_local_industry_map,
    load_local_stock_data,
    write_json,
)
from execution.mock_trader import simulate_rebalance
from models.encoders import build_model
from models.loss_functions import rank_ic
from models.trainer import AlphaTrainer, TrainerConfig
from strategy.backtest import backtest_metric_guidance, backtest_top_k, summarize_backtest
from strategy.universe_selector import select_training_universe


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def log_step(message: str) -> None:
    print(f"[DataLoad] {message}")


def resolve_training_frames(
    config: dict,
    as_of_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_cfg = config.get("data", {})
    stock_path = Path(data_cfg.get("path", "data/eod_daily.csv"))
    index_path = Path(data_cfg.get("index_path", "data/index_daily.csv"))
    industry_map_path = Path(data_cfg.get("industry_map_path", "data/industry_map.csv"))
    industry_daily_path = Path(data_cfg.get("industry_daily_path", "data/industry_daily.csv"))

    started_at = perf_counter()
    log_step(f"加载股票日线: {stock_path}")
    stock_df = load_local_stock_data(stock_path)
    if stock_df.empty:
        raise FileNotFoundError(f"No local stock EOD data found at {stock_path}. Run scripts/update_eod_data.py first.")
    log_step(f"股票日线加载完成: rows={len(stock_df)} symbols={stock_df['symbol'].nunique()}")

    if as_of_date is not None:
        as_of_date = pd.Timestamp(as_of_date).normalize()
        stock_df = stock_df[pd.to_datetime(stock_df["date"]).dt.normalize() <= as_of_date].copy()
        if stock_df.empty:
            raise ValueError(f"No stock data available on or before as_of_date={as_of_date.date()}.")
        log_step(
            f"按 T 日截断股票日线: as_of_date={as_of_date.date()} "
            f"rows={len(stock_df)} symbols={stock_df['symbol'].nunique()}"
        )

    keep_days = int(data_cfg.get("trainable_history_days", 220))
    unique_dates = sorted(pd.to_datetime(stock_df["date"]).drop_duplicates())
    if len(unique_dates) > keep_days:
        active_dates = set(unique_dates[-keep_days:])
        stock_df = stock_df[stock_df["date"].isin(active_dates)].copy()
        log_step(
            f"股票日线截取最近 {keep_days} 个交易日: rows={len(stock_df)} "
            f"symbols={stock_df['symbol'].nunique()}"
        )

    if stock_df.empty:
        raise ValueError("Local stock EOD data became empty after filtering.")

    log_step(f"加载指数日线: {index_path}")
    index_df = load_local_index_data(index_path)
    if as_of_date is not None and not index_df.empty:
        index_df = index_df[pd.to_datetime(index_df["date"]).dt.normalize() <= as_of_date].copy()
    log_step(f"指数日线加载完成: rows={len(index_df)}")

    log_step(f"加载行业映射: {industry_map_path}")
    industry_map_df = load_local_industry_map(industry_map_path)
    log_step(f"行业映射加载完成: rows={len(industry_map_df)}")

    log_step(f"加载行业日线: {industry_daily_path}")
    industry_daily_df = load_local_industry_daily(industry_daily_path)
    if as_of_date is not None and not industry_daily_df.empty:
        industry_daily_df = industry_daily_df[pd.to_datetime(industry_daily_df["date"]).dt.normalize() <= as_of_date].copy()
    log_step(f"行业日线加载完成: rows={len(industry_daily_df)}")

    log_step(f"本地数据准备完成，总耗时 {perf_counter() - started_at:.2f}s")
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


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def format_rank_value(value: object) -> str:
    return str(int(value)) if pd.notna(value) else "-"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    as_of_date = pd.Timestamp(args.as_of_date).normalize() if args.as_of_date else None
    stock_df, index_df, industry_map_df, industry_daily_df = resolve_training_frames(config, as_of_date=as_of_date)

    universe_cfg = config.get("universe", {})
    training_cfg = config.get("training", {})
    rolling_cfg = config.get("rolling", {})
    training_mode = str(training_cfg.get("mode", "market_rank")).strip().lower()
    if training_mode not in {"market_rank", "trade"}:
        raise ValueError(f"Unsupported training.mode={training_mode}. Expected 'market_rank' or 'trade'.")

    required_continuous_tail_days = (
        int(rolling_cfg.get("train_days", 80))
        + int(rolling_cfg.get("valid_days", 20))
        + int(rolling_cfg.get("test_days", 20))
    )
    if training_mode == "trade":
        required_continuous_tail_days = 0

    universe_cfg = dict(universe_cfg)
    universe_filters = dict(universe_cfg.get("filters", {}))
    universe_filters["min_continuous_tail_days"] = required_continuous_tail_days
    if training_mode == "trade":
        universe_filters["training_max_latest_price"] = universe_filters.get("max_latest_price")
    else:
        universe_filters.pop("training_max_latest_price", None)
    universe_cfg["filters"] = universe_filters
    universe_cfg["st_symbol_set"] = sorted(load_st_symbol_set(config))

    log_step(f"Training mode={training_mode}")
    log_step("开始筛选训练股票池")
    selected_symbols, universe_report = select_training_universe(stock_df, universe_cfg)
    if not selected_symbols:
        raise ValueError("Universe selection returned zero symbols.")
    log_step(f"训练股票池筛选完成: selected_symbols={len(selected_symbols)}")

    selected_symbol_set = {str(symbol) for symbol in selected_symbols}
    context_stock_df = stock_df
    if training_mode == "trade":
        context_stock_df = stock_df[stock_df["symbol"].astype(str).isin(selected_symbol_set)].copy()
        log_step(
            f"Trade 模式使用训练域上下文: rows={len(context_stock_df)} "
            f"symbols={context_stock_df['symbol'].nunique()}"
        )
    else:
        log_step(
            f"Market-rank 模式使用全市场上下文: rows={len(context_stock_df)} "
            f"symbols={context_stock_df['symbol'].nunique()}"
        )

    sequence_cfg = config.get("sequence", {})
    data_cfg = config.get("data", {})
    index_cfg = config.get("index", {})
    peer_cfg = config.get("peer", {})
    index_keys = list(index_cfg.get("symbols", {}).keys()) if index_cfg.get("enabled", True) else []

    dataset_builder = AlphaDatasetBuilder(
        seq_len=int(sequence_cfg.get("seq_len", 20)),
        label_horizon=int(data_cfg.get("label_horizon", 1)),
        index_keys=index_keys,
        peer_enabled=bool(peer_cfg.get("enabled", True)),
        peer_top_k=int(peer_cfg.get("top_k", 5)),
        peer_lookback_days=int(peer_cfg.get("lookback_days", 60)),
        peer_min_overlap=int(peer_cfg.get("min_overlap", 20)),
        verbose=True,
        time_block_shuffle=bool(args.shuffle_time_blocks),
        time_block_size=args.shuffle_block_size,
        random_seed=int(config.get("seed", 7)),
    )

    dataset_bundle = dataset_builder.build_bundle(
        raw_df=context_stock_df,
        train_days=int(rolling_cfg.get("train_days", 80)),
        valid_days=int(rolling_cfg.get("valid_days", 20)),
        test_days=int(rolling_cfg.get("test_days", 20)),
        index_df=index_df if index_cfg.get("enabled", True) else pd.DataFrame(),
        industry_map_df=industry_map_df if config.get("industry", {}).get("enabled", True) else pd.DataFrame(),
        industry_daily_df=industry_daily_df if config.get("industry", {}).get("enabled", True) else pd.DataFrame(),
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

    model_cfg = config.get("model", {})
    input_dim = len(dataset_bundle.feature_columns)
    model, resolved_model_config = build_model(input_dim=input_dim, model_cfg=model_cfg)

    trainer = AlphaTrainer(
        model=model,
        config=TrainerConfig(
            epochs=int(model_cfg.get("epochs", 60)),
            lr=float(model_cfg.get("lr", 1e-3)),
            weight_decay=float(model_cfg.get("weight_decay", 1e-2)),
            batch_size=int(model_cfg.get("batch_size", 256)),
            eval_batch_size=int(model_cfg.get("eval_batch_size", 512)),
            log_every=int(model_cfg.get("log_every", 1)),
            early_stopping_patience=int(model_cfg.get("early_stopping_patience", 10)),
            num_workers=int(model_cfg.get("num_workers", 0)),
            seed=int(config.get("seed", 7)),
        ),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    run_dir = ensure_run_dir(Path(config.get("outputs", {}).get("run_dir", "outputs/runs")), run_name=args.run_name)
    history_df = trainer.fit(
        dataset_bundle.train_dataset,
        dataset_bundle.valid_dataset,
        run_dir=run_dir,
        model_config={
            **resolved_model_config,
            "feature_columns": dataset_bundle.feature_columns,
        },
    )
    trainer.load_checkpoint(run_dir / "best.ckpt")

    valid_pred = trainer.predict_dataset(dataset_bundle.valid_dataset)
    test_pred = trainer.predict_dataset(dataset_bundle.test_dataset)
    valid_target = dataset_bundle.valid_dataset.targets_numpy
    test_target = dataset_bundle.test_dataset.targets_numpy
    valid_ic = rank_ic(valid_target, valid_pred) if valid_target is not None else 0.0
    test_ic = rank_ic(test_target, test_pred) if test_target is not None else 0.0

    valid_scored = dataset_bundle.valid_dataset.meta.copy()
    valid_scored["score"] = valid_pred
    valid_scored.to_csv(run_dir / "valid_predictions.csv", index=False, encoding="utf-8-sig")

    test_scored = dataset_bundle.test_dataset.meta.copy()
    test_scored["score"] = test_pred
    test_scored.to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    bt_report = backtest_top_k(test_scored[["date", "symbol", "label", "score"]], top_k=int(config["strategy"]["top_k"]))
    bt_report.to_csv(run_dir / "backtest.csv", index=False, encoding="utf-8-sig")
    backtest_metrics = summarize_backtest(bt_report)
    backtest_guidance = backtest_metric_guidance()
    write_json(
        run_dir / "backtest_metrics.json",
        {
            "metrics": backtest_metrics,
            "guidance": backtest_guidance,
        },
    )

    candidate_symbols: set[str] = set()
    candidate_filter_applied = False
    candidate_path = Path(config.get("data", {}).get("candidate_path", "data/current_candidates.csv"))
    if as_of_date is None:
        candidate_symbols, candidate_path = load_current_candidate_symbols(config)

    inference_raw_df = context_stock_df if training_mode == "trade" else stock_df
    if training_mode == "trade":
        if as_of_date is None:
            log_step(
                f"Trade 模式使用训练域推理，不生成全市场 rank: rows={len(inference_raw_df)} "
                f"symbols={inference_raw_df['symbol'].nunique()}"
            )
        else:
            log_step("Trade 模式 + as_of_date：使用历史训练域截面推理。")
    else:
        log_step(
            f"Market-rank 模式使用全市场推理: rows={len(inference_raw_df)} "
            f"symbols={inference_raw_df['symbol'].nunique()}"
        )

    inference_dataset, _ = dataset_builder.build_inference_dataset(
        raw_df=inference_raw_df,
        scaler=dataset_bundle.scaler,
        index_df=index_df if index_cfg.get("enabled", True) else pd.DataFrame(),
        industry_map_df=industry_map_df if config.get("industry", {}).get("enabled", True) else pd.DataFrame(),
        industry_daily_df=industry_daily_df if config.get("industry", {}).get("enabled", True) else pd.DataFrame(),
        peer_map=dataset_bundle.peer_map,
    )
    if len(inference_dataset) == 0:
        raise ValueError("Inference dataset is empty.")

    inference_pred = trainer.predict_dataset(inference_dataset)
    inference_scored = inference_dataset.meta.copy()
    inference_scored["score"] = inference_pred
    inference_scored.to_csv(run_dir / "inference_predictions.csv", index=False, encoding="utf-8-sig")
    symbol_name_map = load_symbol_name_map(config)

    signal_date = pd.to_datetime(inference_scored["signal_date"]).max()
    signal_slice = inference_scored[inference_scored["signal_date"] == signal_date].copy()
    signal_slice["name"] = signal_slice["symbol"].astype(str).map(symbol_name_map).fillna("")

    market_rank = pd.DataFrame(columns=["signal_date", "symbol", "name", "industry_name", "score", "close", "market_rank"])
    if training_mode == "market_rank":
        market_rank = build_rank_frame(signal_slice.copy(), "market_rank")
        market_rank = market_rank[
            ["signal_date", "symbol", "name", "industry_name", "score", "close", "market_rank"]
        ].copy()
        market_rank.to_csv(run_dir / "market_rank.csv", index=False, encoding="utf-8-sig")

        candidate_rank = market_rank.copy()
        if as_of_date is None:
            if candidate_symbols:
                before_count = len(candidate_rank)
                candidate_rank = candidate_rank[candidate_rank["symbol"].astype(str).isin(candidate_symbols)].copy()
                candidate_filter_applied = True
                log_step(
                    f"推理阶段候选池过滤完成: candidate_symbols={len(candidate_symbols)} "
                    f"before={before_count} after={len(candidate_rank)} path={candidate_path}"
                )
            else:
                log_step("推理阶段未使用候选池过滤: current_candidates.csv 不存在或不含 symbol。")
        else:
            log_step("指定历史 T 日时跳过 current_candidates.csv，避免引入当前候选池的生存者偏差。")

        if candidate_rank.empty:
            raise ValueError("Inference cross-section became empty after candidate filtering.")

        candidate_rank = candidate_rank.sort_values(["market_rank", "symbol"]).reset_index(drop=True).copy()
        candidate_rank["candidate_rank"] = candidate_rank["market_rank"]
    else:
        if signal_slice.empty:
            raise ValueError("Trade-mode inference cross-section is empty.")
        candidate_rank = build_rank_frame(signal_slice.copy(), "candidate_rank")
        candidate_rank["market_rank"] = pd.NA
        if as_of_date is None:
            log_step(f"Trade 模式直接在训练池内推荐: count={len(candidate_rank)}")
        else:
            log_step(f"Trade 模式历史截面排序完成: count={len(candidate_rank)}")

        if candidate_rank.empty:
            raise ValueError("Trade recommendation universe is empty.")

    candidate_rank = candidate_rank[
        ["signal_date", "symbol", "name", "industry_name", "score", "close", "market_rank", "candidate_rank"]
    ].copy()
    candidate_rank.to_csv(run_dir / "candidate_rank.csv", index=False, encoding="utf-8-sig")

    recommendation_pool = candidate_rank.copy()
    max_recommend_price = universe_filters.get("max_latest_price")
    recommendation_price_filter_applied = False
    if max_recommend_price is not None:
        before_count = len(recommendation_pool)
        recommendation_pool = recommendation_pool[recommendation_pool["close"] <= float(max_recommend_price)].copy()
        recommendation_price_filter_applied = True
        log_step(
            f"推荐阶段价格上限过滤完成: max_latest_price={float(max_recommend_price):.2f} "
            f"before={before_count} after={len(recommendation_pool)}"
        )
    if recommendation_pool.empty:
        raise ValueError("Recommendation pool became empty after final price filtering.")

    fetcher = build_fetcher(config)
    next_trade_date = fetcher.next_trade_date(signal_date)

    latest_top_k = recommendation_pool.nsmallest(int(config["strategy"]["top_k"]), "candidate_rank").copy()
    latest_top_k["next_trade_date"] = next_trade_date.date().isoformat()
    latest_top_k = latest_top_k.rename(columns={"close": "buy_price"})
    latest_top_k["buy_price_basis"] = "signal_close_ref"
    latest_top_k["entry_price_ref_close"] = latest_top_k["buy_price"]
    latest_top_k = latest_top_k[
        [
            "signal_date",
            "next_trade_date",
            "symbol",
            "name",
            "industry_name",
            "score",
            "market_rank",
            "candidate_rank",
            "buy_price",
            "buy_price_basis",
            "entry_price_ref_close",
        ]
    ]
    latest_top_k.to_csv(run_dir / "top_k.csv", index=False, encoding="utf-8-sig")

    orders = simulate_rebalance(
        latest_top_k.rename(columns={"signal_date": "date"}),
        cash=float(config["execution"]["initial_cash"]),
        max_positions=int(config["execution"]["max_positions"]),
    )
    orders.to_csv(run_dir / "orders.csv", index=False, encoding="utf-8-sig")
    universe_report.to_csv(run_dir / "universe_report.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_dir": str(run_dir.resolve()),
        "as_of_date": str(as_of_date.date()) if as_of_date is not None else None,
        "latest_stock_date": str(pd.to_datetime(stock_df["date"]).max().date()),
        "latest_index_date": str(pd.to_datetime(index_df["date"]).max().date()) if not index_df.empty else None,
        "latest_industry_date": str(pd.to_datetime(industry_daily_df["date"]).max().date())
        if not industry_daily_df.empty
        else None,
        "training_mode": training_mode,
        "signal_date": str(pd.Timestamp(signal_date).date()),
        "next_trade_date": str(pd.Timestamp(next_trade_date).date()),
        "universe_size": int(len(selected_symbols)),
        "candidate_filter_applied": bool(candidate_filter_applied),
        "candidate_symbol_count": int(len(candidate_symbols)) if as_of_date is None else None,
        "market_rank_count": int(len(market_rank)),
        "candidate_rank_count": int(len(candidate_rank)),
        "recommendation_price_filter_applied": bool(recommendation_price_filter_applied),
        "recommendation_pool_count": int(len(recommendation_pool)),
        "recommendation_max_latest_price": float(max_recommend_price) if max_recommend_price is not None else None,
        "top_k_count": int(len(latest_top_k)),
        "n_train_samples": int(len(dataset_bundle.train_dataset)),
        "n_valid_samples": int(len(dataset_bundle.valid_dataset)),
        "n_test_samples": int(len(dataset_bundle.test_dataset)),
        "seq_len": int(sequence_cfg.get("seq_len", 20)),
        "time_block_shuffle": bool(args.shuffle_time_blocks),
        "time_block_size": int(args.shuffle_block_size or sequence_cfg.get("seq_len", 20)),
        "feature_dim": int(input_dim),
        "model_name": str(resolved_model_config.get("name")),
        "feature_columns": dataset_bundle.feature_columns,
        "split_dates": {
            key: [pd.Timestamp(value).date().isoformat() for value in value_list]
            for key, value_list in dataset_bundle.split_dates.items()
        },
        "peer_symbol_count": int(sum(1 for value in dataset_bundle.peer_map.values() if value)),
        "valid_ic": float(valid_ic),
        "test_ic": float(test_ic),
        "best_epoch": int(trainer.best_epoch),
        "best_valid_ic": float(trainer.best_valid_ic),
        "backtest_metrics": backtest_metrics,
        "backtest_metric_guidance": backtest_guidance,
    }
    save_run_metadata(run_dir, config, summary)

    latest_run_path = Path(config.get("outputs", {}).get("latest_run_metadata", "outputs/latest_run.json"))
    latest_run_path.parent.mkdir(parents=True, exist_ok=True)
    latest_run_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Run dir: {run_dir.resolve()}")
    print(f"As-of date: {summary['as_of_date']}")
    print(f"Latest stock date: {summary['latest_stock_date']}")
    print(f"Latest index date: {summary['latest_index_date']}")
    print(f"Latest industry date: {summary['latest_industry_date']}")
    print(f"Training mode: {summary['training_mode']}")
    print(f"Universe size: {summary['universe_size']}")
    print(f"Model: {summary['model_name']}")
    print(f"Time block shuffle: {summary['time_block_shuffle']} | block_size={summary['time_block_size']}")
    print(
        f"Samples(train/valid/test): {summary['n_train_samples']}/"
        f"{summary['n_valid_samples']}/{summary['n_test_samples']} "
        f"| seq_len={summary['seq_len']} feature_dim={summary['feature_dim']}"
    )
    print(
        f"Best epoch={summary['best_epoch']} "
        f"| best_valid_ic={summary['best_valid_ic']:.4f} "
        f"| valid_ic={summary['valid_ic']:.4f} | test_ic={summary['test_ic']:.4f}"
    )
    if backtest_metrics["n_backtest_days"]:
        print(
            "[Backtest] "
            f"days={backtest_metrics['n_backtest_days']} "
            f"mean={backtest_metrics['top_k_mean_return']:.4f} "
            f"cum={backtest_metrics['cumulative_return']:.4f} "
            f"excess={backtest_metrics['relative_return']:.4f} "
            f"win={backtest_metrics['win_rate']:.2%} "
            f"mdd={backtest_metrics['max_drawdown']:.4f} "
            f"sharpe={backtest_metrics['sharpe_annualized']:.2f}"
        )
    if not history_df.empty:
        print(
            f"Last epoch train_loss={history_df.iloc[-1]['train_loss']:.6f} "
            f"valid_loss={history_df.iloc[-1]['valid_loss']:.6f}"
        )
    print("T+1 picks:")
    for _, row in latest_top_k.iterrows():
        print(
            f"- signal_date={pd.Timestamp(row['signal_date']).date()} "
            f"next_trade_date={row['next_trade_date']} "
            f"symbol={row['symbol']} name={row['name']} industry={row['industry_name']} "
            f"score={row['score']:.4f} market_rank={format_rank_value(row['market_rank'])} "
            f"candidate_rank={format_rank_value(row['candidate_rank'])} buy_price={row['buy_price']:.3f} "
            f"price_basis={row['buy_price_basis']}"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Train] 用户中断，训练已停止。")
        raise SystemExit(130)
