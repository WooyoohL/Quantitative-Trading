from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.runtime import (
    build_fetcher,
    build_rank_frame,
    load_current_candidate_symbols,
    load_symbol_name_map,
    log_step,
)
from execution.mock_trader import simulate_rebalance
from models.trainer import AlphaTrainer


@dataclass
class RecommendationArtifacts:
    signal_date: pd.Timestamp
    next_trade_date: pd.Timestamp
    inference_scored: pd.DataFrame
    market_rank: pd.DataFrame
    candidate_rank: pd.DataFrame
    recommendation_pool: pd.DataFrame
    latest_top_k: pd.DataFrame
    orders: pd.DataFrame
    candidate_filter_applied: bool
    candidate_symbol_count: int | None
    recommendation_price_filter_applied: bool
    recommendation_max_latest_price: float | None


def generate_recommendation_outputs(
    *,
    config: dict,
    output_dir: Path,
    trainer: AlphaTrainer,
    dataset_builder,
    dataset_bundle,
    training_mode: str,
    as_of_date: pd.Timestamp | None,
    context_stock_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    industry_map_df: pd.DataFrame,
    industry_daily_df: pd.DataFrame,
    universe_report: pd.DataFrame,
) -> RecommendationArtifacts:
    candidate_symbols: set[str] = set()
    candidate_filter_applied = False
    candidate_path = Path(config.get("data", {}).get("candidate_path", "data/current_candidates.csv"))
    if as_of_date is None:
        candidate_symbols, candidate_path = load_current_candidate_symbols(config)

    inference_raw_df = context_stock_df if training_mode == "trade" else stock_df
    if training_mode == "trade":
        if as_of_date is None:
            log_step(
                f"Trade 模式使用训练池做推理，不生成全市场 rank: rows={len(inference_raw_df)} "
                f"symbols={inference_raw_df['symbol'].nunique()}"
            )
        else:
            log_step("Trade 模式 + as_of_date：使用历史训练池截面推理。")
    else:
        log_step(
            f"Market-rank 模式使用全市场做推理: rows={len(inference_raw_df)} "
            f"symbols={inference_raw_df['symbol'].nunique()}"
        )

    inference_dataset, _ = dataset_builder.build_inference_dataset(
        raw_df=inference_raw_df,
        scaler=dataset_bundle.scaler,
        index_df=index_df if config.get("index", {}).get("enabled", True) else pd.DataFrame(),
        industry_map_df=industry_map_df if config.get("industry", {}).get("enabled", True) else pd.DataFrame(),
        industry_daily_df=industry_daily_df if config.get("industry", {}).get("enabled", True) else pd.DataFrame(),
        peer_map=dataset_bundle.peer_map,
        signal_date=as_of_date,
    )
    if len(inference_dataset) == 0:
        raise ValueError("Inference dataset is empty.")

    inference_pred = trainer.predict_dataset(inference_dataset)
    inference_scored = inference_dataset.meta.copy()
    inference_scored["score"] = inference_pred
    inference_scored.to_csv(output_dir / "inference_predictions.csv", index=False, encoding="utf-8-sig")

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
        market_rank.to_csv(output_dir / "market_rank.csv", index=False, encoding="utf-8-sig")

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

    candidate_rank = candidate_rank[
        ["signal_date", "symbol", "name", "industry_name", "score", "close", "market_rank", "candidate_rank"]
    ].copy()
    candidate_rank.to_csv(output_dir / "candidate_rank.csv", index=False, encoding="utf-8-sig")

    recommendation_pool = candidate_rank.copy()
    universe_filters = config.get("universe", {}).get("filters", {})
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
    latest_top_k.to_csv(output_dir / "top_k.csv", index=False, encoding="utf-8-sig")

    orders = simulate_rebalance(
        latest_top_k.rename(columns={"signal_date": "date"}),
        cash=float(config["execution"]["initial_cash"]),
        max_positions=int(config["execution"]["max_positions"]),
    )
    orders.to_csv(output_dir / "orders.csv", index=False, encoding="utf-8-sig")
    universe_report.to_csv(output_dir / "universe_report.csv", index=False, encoding="utf-8-sig")

    return RecommendationArtifacts(
        signal_date=pd.Timestamp(signal_date),
        next_trade_date=pd.Timestamp(next_trade_date),
        inference_scored=inference_scored,
        market_rank=market_rank,
        candidate_rank=candidate_rank,
        recommendation_pool=recommendation_pool,
        latest_top_k=latest_top_k,
        orders=orders,
        candidate_filter_applied=bool(candidate_filter_applied),
        candidate_symbol_count=int(len(candidate_symbols)) if as_of_date is None else None,
        recommendation_price_filter_applied=bool(recommendation_price_filter_applied),
        recommendation_max_latest_price=float(max_recommend_price) if max_recommend_price is not None else None,
    )
