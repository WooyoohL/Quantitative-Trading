from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.backtest import backtest_top_k, summarize_backtest


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def compute_rightside_score(df: pd.DataFrame) -> pd.Series:
    return (
        0.30 * df["ret_1"].astype(float)
        + 0.25 * df["intraday_ret"].astype(float)
        + 0.20 * df["ma_gap_5"].astype(float)
        + 0.10 * df["volume_ratio_5"].astype(float)
        + 0.10 * df["industry_ret_1_mean"].astype(float)
        + 0.05 * df["ret_5"].astype(float)
    )


def apply_right_side_filter(scored: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = scored.copy()
    universe_filters = config.get("universe", {}).get("filters", {})
    max_price = universe_filters.get("max_latest_price")
    if max_price is not None and "close" in out.columns:
        out = out[pd.to_numeric(out["close"], errors="coerce") <= float(max_price)].copy()

    filter_cfg = config.get("strategy", {}).get("right_side_filter", {})
    if not bool(filter_cfg.get("enabled", False)):
        return out

    filtered = out.copy()
    for column, threshold_key in [
        ("ret_1", "min_ret_1"),
        ("ret_5", "min_ret_5"),
        ("intraday_ret", "min_intraday_ret"),
        ("ma_gap_5", "min_ma_gap_5"),
        ("volume_ratio_5", "min_volume_ratio_5"),
        ("industry_ret_1_mean", "min_industry_ret_1_mean"),
    ]:
        threshold = filter_cfg.get(threshold_key)
        if threshold is None or column not in filtered.columns:
            continue
        values = pd.to_numeric(filtered[column], errors="coerce")
        filtered = filtered[values >= float(threshold)].copy()

    if filtered.empty:
        return out
    return filtered


def backtest_raw_topk(scored: pd.DataFrame, top_k: int) -> dict[str, Any]:
    report = backtest_top_k(scored[["date", "symbol", "label", "score"]], top_k=int(top_k))
    return summarize_backtest(report)


def backtest_filtered_reranked_top3(
    scored: pd.DataFrame,
    config: dict[str, Any],
    *,
    preselect_top_k: int = 20,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for date, day_df in scored.groupby("date", sort=True):
        preselected = day_df.sort_values(["score", "symbol"], ascending=[False, True]).head(int(preselect_top_k)).copy()
        preselected["candidate_rank_proxy"] = range(1, len(preselected) + 1)
        pool = apply_right_side_filter(preselected, config)
        pool = pool.copy()
        pool["rightside_score"] = compute_rightside_score(pool)
        picks = pool.sort_values(
            ["rightside_score", "score", "candidate_rank_proxy", "symbol"],
            ascending=[False, False, True, True],
        ).head(3)
        rows.append(
            {
                "date": date,
                "strategy_return": picks["label"].astype(float).mean(),
                "market_return": day_df["label"].astype(float).mean(),
            }
        )
    report = pd.DataFrame(rows)
    if report.empty:
        return summarize_backtest(report)
    report = report.sort_values("date")
    report["strategy_return"] = report["strategy_return"].fillna(0.0)
    report["market_return"] = report["market_return"].fillna(0.0)
    report["excess_return"] = report["strategy_return"] - report["market_return"]
    report["equity_curve"] = (1.0 + report["strategy_return"]).cumprod()
    report["market_curve"] = (1.0 + report["market_return"]).cumprod()
    report["relative_curve"] = report["equity_curve"] / report["market_curve"].replace(0.0, pd.NA)
    return summarize_backtest(report)


def analyze_run(run_dir: Path) -> dict[str, Any]:
    config = load_config(run_dir / "config.yaml")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    valid_scored = pd.read_csv(run_dir / "valid_predictions.csv")
    test_scored = pd.read_csv(run_dir / "test_predictions.csv")

    raw_top20_valid = backtest_raw_topk(valid_scored, top_k=20)
    raw_top20_test = backtest_raw_topk(test_scored, top_k=20)
    filtered_top3_valid = backtest_filtered_reranked_top3(valid_scored, config, preselect_top_k=20)
    filtered_top3_test = backtest_filtered_reranked_top3(test_scored, config, preselect_top_k=20)

    return {
        "run_name": run_dir.name,
        "seed": int(config.get("seed", -1)),
        "best_epoch": summary.get("best_epoch"),
        "selection_mode": summary.get("checkpoint_selection_mode"),
        "valid_ic": summary.get("valid_ic"),
        "valid_daily_ic": summary.get("valid_daily_ic"),
        "test_ic": summary.get("test_ic"),
        "test_daily_ic": summary.get("test_daily_ic"),
        "raw_top20_valid_relative_return": raw_top20_valid.get("relative_return"),
        "raw_top20_valid_excess_mean_return": raw_top20_valid.get("excess_mean_return"),
        "raw_top20_valid_positive_excess_rate": raw_top20_valid.get("positive_excess_rate"),
        "raw_top20_test_relative_return": raw_top20_test.get("relative_return"),
        "raw_top20_test_excess_mean_return": raw_top20_test.get("excess_mean_return"),
        "raw_top20_test_positive_excess_rate": raw_top20_test.get("positive_excess_rate"),
        "filtered_top3_valid_relative_return": filtered_top3_valid.get("relative_return"),
        "filtered_top3_valid_excess_mean_return": filtered_top3_valid.get("excess_mean_return"),
        "filtered_top3_valid_positive_excess_rate": filtered_top3_valid.get("positive_excess_rate"),
        "filtered_top3_valid_max_drawdown": filtered_top3_valid.get("max_drawdown"),
        "filtered_top3_test_relative_return": filtered_top3_test.get("relative_return"),
        "filtered_top3_test_excess_mean_return": filtered_top3_test.get("excess_mean_return"),
        "filtered_top3_test_positive_excess_rate": filtered_top3_test.get("positive_excess_rate"),
        "filtered_top3_test_max_drawdown": filtered_top3_test.get("max_drawdown"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze multi-seed stability using raw top20 coverage and filtered+reraanked top3.")
    parser.add_argument("--batch-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_dir = args.batch_dir
    summary_csv = batch_dir / "batch_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing batch summary csv: {summary_csv}")
    batch_df = pd.read_csv(summary_csv)
    rows: list[dict[str, Any]] = []
    for run_dir_str in batch_df["run_dir"].dropna().tolist():
        rows.append(analyze_run(Path(run_dir_str)))
    out_df = pd.DataFrame(rows).sort_values("seed")
    out_csv = batch_dir / "seed_stability_analysis.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(out_df.to_string(index=False))
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
