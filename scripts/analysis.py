from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.decision import decide_recommendation
from strategy.backtest import backtest_top_k, summarize_backtest


DECISION_PRIORITY = {"推荐": 0, "观察": 1, "不建议": 2}


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return 0.0
    order_true = np.argsort(y_true)
    order_pred = np.argsort(y_pred)
    rank_true = np.empty_like(order_true, dtype=float)
    rank_pred = np.empty_like(order_pred, dtype=float)
    rank_true[order_true] = np.arange(len(y_true), dtype=float)
    rank_pred[order_pred] = np.arange(len(y_pred), dtype=float)
    corr = np.corrcoef(rank_true, rank_pred)[0, 1]
    return float(np.nan_to_num(corr))


def daily_rank_ic_mean(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    if len(y_true) < 2 or len(y_true) != len(y_pred) or len(y_true) != len(dates):
        return 0.0
    values: list[float] = []
    for date_value in np.unique(dates):
        mask = dates == date_value
        if int(mask.sum()) < 2:
            continue
        values.append(rank_ic(y_true[mask], y_pred[mask]))
    return float(np.nan_to_num(np.mean(values))) if values else 0.0


def head_daily_rank_ic_mean(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray, top_n: int) -> float:
    if len(y_true) < 2 or len(y_true) != len(y_pred) or len(y_true) != len(dates) or int(top_n) <= 1:
        return 0.0
    values: list[float] = []
    for date_value in np.unique(dates):
        mask = dates == date_value
        if int(mask.sum()) < 2:
            continue
        day_true = y_true[mask]
        day_pred = y_pred[mask]
        head_index = np.argsort(-day_pred)[: min(int(top_n), len(day_pred))]
        if len(head_index) < 2:
            continue
        values.append(rank_ic(day_true[head_index], day_pred[head_index]))
    return float(np.nan_to_num(np.mean(values))) if values else 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a completed batch experiment directory.")
    parser.add_argument("--batch-dir", required=True, help="Path to outputs/batch_runs/<batch_name>.")
    return parser.parse_args(argv)


def load_batch_rows(batch_dir: Path) -> pd.DataFrame:
    summary_csv = batch_dir / "batch_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Batch summary not found: {summary_csv}")
    return pd.read_csv(summary_csv)


def load_run_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def resolve_run_dir(row: pd.Series) -> Path | None:
    run_dir_value = row.get("run_dir")
    if isinstance(run_dir_value, str) and run_dir_value:
        direct = Path(run_dir_value)
        if direct.exists():
            return direct
    run_name = row.get("run_name")
    if isinstance(run_name, str) and run_name:
        fallback = REPO_ROOT / "outputs" / "runs" / run_name
        if fallback.exists():
            return fallback
    return None


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def compute_split_metrics_from_predictions(run_dir: Path, split_name: str, top_k: int) -> dict[str, float]:
    path = run_dir / f"{split_name}_predictions.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or "label" not in df.columns or "score" not in df.columns or "date" not in df.columns:
        return {}
    labels = pd.to_numeric(df["label"], errors="coerce").to_numpy()
    scores = pd.to_numeric(df["score"], errors="coerce").to_numpy()
    dates = pd.to_datetime(df["date"], errors="coerce").to_numpy()
    bt = summarize_backtest(backtest_top_k(df[["date", "label", "score"]].copy(), top_k=int(top_k)))
    return {
        "daily_ic": float(daily_rank_ic_mean(labels, scores, dates)),
        "head_daily_ic": float(head_daily_rank_ic_mean(labels, scores, dates, top_n=20)),
        "top_k_mean_return": float(bt.get("top_k_mean_return") or 0.0),
        "excess_mean_return": float(bt.get("excess_mean_return") or 0.0),
        "positive_excess_rate": float(bt.get("positive_excess_rate") or 0.0),
        "win_rate": float(bt.get("win_rate") or 0.0),
        "relative_return": float(bt.get("relative_return") or 0.0),
        "max_drawdown": float(bt.get("max_drawdown") or 0.0),
    }


def rank_score(series: pd.Series, ascending: bool = False) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    if valid.empty:
        return result.fillna(0.0)
    raw_rank = valid.rank(method="average", ascending=ascending)
    normalized = pd.Series(1.0, index=valid.index) if len(valid) == 1 else 1.0 - (raw_rank - 1.0) / float(len(valid) - 1)
    result.loc[normalized.index] = normalized.astype(float)
    return result.fillna(0.0)


def compute_raw_top20_metrics(run_dir: Path, split_name: str) -> dict[str, float]:
    path = run_dir / f"{split_name}_predictions.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or not {"date", "label", "score"}.issubset(df.columns):
        return {}
    summary = summarize_backtest(backtest_top_k(df[["date", "label", "score"]].copy(), top_k=20))
    return {
        "relative_return": float(summary.get("relative_return") or 0.0),
        "excess_mean_return": float(summary.get("excess_mean_return") or 0.0),
        "positive_excess_rate": float(summary.get("positive_excess_rate") or 0.0),
        "win_rate": float(summary.get("win_rate") or 0.0),
        "max_drawdown": float(summary.get("max_drawdown") or 0.0),
    }


def _pick_metric(summary: dict, computed: dict[str, float], key: str, fallback: object = None) -> object:
    value = summary.get(key)
    if not _is_missing(value):
        return value
    value = computed.get(key)
    if not _is_missing(value):
        return value
    return fallback


def enrich_results(batch_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in batch_df.iterrows():
        run_dir = resolve_run_dir(row)
        summary = load_run_summary(run_dir) if run_dir and run_dir.exists() else {}
        valid_bt = summary.get("valid_backtest_metrics", {}) or {}
        test_bt = summary.get("backtest_metrics", {}) or {}
        top_k = int(summary.get("final_top_k_target") or row.get("top_k") or 3)
        computed_valid = compute_split_metrics_from_predictions(run_dir, "valid", top_k=top_k) if run_dir else {}
        computed_test = compute_split_metrics_from_predictions(run_dir, "test", top_k=top_k) if run_dir else {}
        raw_top20_valid = compute_raw_top20_metrics(run_dir, "valid") if run_dir else {}
        raw_top20_test = compute_raw_top20_metrics(run_dir, "test") if run_dir else {}

        decision = decide_recommendation(
            valid_excess_mean_return=_pick_metric(valid_bt, computed_valid, "excess_mean_return"),
            valid_positive_excess_rate=_pick_metric(valid_bt, computed_valid, "positive_excess_rate"),
            valid_daily_ic=_pick_metric(summary, computed_valid, "valid_daily_ic", computed_valid.get("daily_ic")),
            valid_max_drawdown=_pick_metric(valid_bt, computed_valid, "max_drawdown"),
            test_relative_return=_pick_metric(test_bt, computed_test, "relative_return", row.get("backtest_relative_return")),
            test_positive_excess_rate=_pick_metric(
                test_bt,
                computed_test,
                "positive_excess_rate",
                row.get("backtest_positive_excess_rate"),
            ),
            test_daily_ic=_pick_metric(summary, computed_test, "test_daily_ic", computed_test.get("daily_ic")),
        )

        rows.append(
            {
                **row.to_dict(),
                "valid_daily_ic": _pick_metric(summary, computed_valid, "valid_daily_ic", computed_valid.get("daily_ic")),
                "valid_head_daily_ic": _pick_metric(summary, computed_valid, "valid_head_daily_ic", computed_valid.get("head_daily_ic")),
                "valid_top_k_mean_return": _pick_metric(valid_bt, computed_valid, "top_k_mean_return"),
                "valid_excess_mean_return": _pick_metric(valid_bt, computed_valid, "excess_mean_return"),
                "valid_positive_excess_rate": _pick_metric(valid_bt, computed_valid, "positive_excess_rate"),
                "valid_win_rate": _pick_metric(valid_bt, computed_valid, "win_rate"),
                "valid_relative_return": _pick_metric(valid_bt, computed_valid, "relative_return"),
                "valid_max_drawdown": _pick_metric(valid_bt, computed_valid, "max_drawdown"),
                "test_daily_ic": _pick_metric(summary, computed_test, "test_daily_ic", computed_test.get("daily_ic")),
                "test_head_daily_ic": _pick_metric(summary, computed_test, "test_head_daily_ic", computed_test.get("head_daily_ic")),
                "test_top_k_mean_return": _pick_metric(test_bt, computed_test, "top_k_mean_return"),
                "test_excess_mean_return": _pick_metric(test_bt, computed_test, "excess_mean_return"),
                "test_positive_excess_rate": _pick_metric(
                    test_bt,
                    computed_test,
                    "positive_excess_rate",
                    row.get("backtest_positive_excess_rate"),
                ),
                "test_win_rate": _pick_metric(test_bt, computed_test, "win_rate", row.get("backtest_win_rate")),
                "test_relative_return": _pick_metric(
                    test_bt,
                    computed_test,
                    "relative_return",
                    row.get("backtest_relative_return"),
                ),
                "test_max_drawdown": _pick_metric(
                    test_bt,
                    computed_test,
                    "max_drawdown",
                    row.get("backtest_max_drawdown"),
                ),
                "checkpoint_selection_mode": summary.get("checkpoint_selection_mode"),
                "best_epoch": summary.get("best_epoch"),
                "top_k": top_k,
                "signal_date": summary.get("signal_date"),
                "raw_top20_valid_relative_return": raw_top20_valid.get("relative_return"),
                "raw_top20_valid_excess_mean_return": raw_top20_valid.get("excess_mean_return"),
                "raw_top20_valid_positive_excess_rate": raw_top20_valid.get("positive_excess_rate"),
                "raw_top20_test_relative_return": raw_top20_test.get("relative_return"),
                "raw_top20_test_excess_mean_return": raw_top20_test.get("excess_mean_return"),
                "raw_top20_test_positive_excess_rate": raw_top20_test.get("positive_excess_rate"),
                "recommendation_label": decision.label,
                "recommendation_reasons": " | ".join(decision.reasons),
                "valid_gate_pass": decision.valid_pass,
                "test_consistent": decision.test_pass,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["score_valid_excess"] = rank_score(df["valid_excess_mean_return"], ascending=False)
    df["score_valid_pos_excess"] = rank_score(df["valid_positive_excess_rate"], ascending=False)
    df["score_valid_relative"] = rank_score(df["valid_relative_return"], ascending=False)
    df["score_valid_daily_ic"] = rank_score(df["valid_daily_ic"], ascending=False)
    df["score_valid_head_ic"] = rank_score(df["valid_head_daily_ic"], ascending=False)
    df["score_valid_mdd"] = rank_score(df["valid_max_drawdown"], ascending=False)
    df["valid_score"] = (
        0.35 * df["score_valid_excess"]
        + 0.20 * df["score_valid_pos_excess"]
        + 0.15 * df["score_valid_relative"]
        + 0.15 * df["score_valid_daily_ic"]
        + 0.10 * df["score_valid_head_ic"]
        + 0.05 * df["score_valid_mdd"]
    )

    df["score_test_relative"] = rank_score(df["test_relative_return"], ascending=False)
    df["score_test_pos_excess"] = rank_score(df["test_positive_excess_rate"], ascending=False)
    df["score_test_daily_ic"] = rank_score(df["test_daily_ic"], ascending=False)
    df["score_test_head_ic"] = rank_score(df["test_head_daily_ic"], ascending=False)
    df["score_test_mdd"] = rank_score(df["test_max_drawdown"], ascending=False)
    df["consistency_score"] = (
        0.50 * df["score_test_relative"]
        + 0.20 * df["score_test_pos_excess"]
        + 0.15 * df["score_test_daily_ic"]
        + 0.10 * df["score_test_head_ic"]
        + 0.05 * df["score_test_mdd"]
    )
    df["recommendation_score"] = 0.75 * df["valid_score"] + 0.25 * df["consistency_score"]

    df["score_raw_top20_relative"] = rank_score(df["raw_top20_test_relative_return"], ascending=False)
    df["score_raw_top20_pos_excess"] = rank_score(df["raw_top20_test_positive_excess_rate"], ascending=False)
    df["candidate_pool_score"] = (
        0.55 * df["score_raw_top20_relative"]
        + 0.25 * df["score_raw_top20_pos_excess"]
        + 0.20 * df["score_test_daily_ic"]
    )
    df["recommendation_rank"] = df["recommendation_label"].map(DECISION_PRIORITY).fillna(99).astype(int)

    df = df.sort_values(
        [
            "recommendation_rank",
            "candidate_pool_score",
            "recommendation_score",
            "valid_excess_mean_return",
            "valid_positive_excess_rate",
            "valid_relative_return",
            "valid_daily_ic",
            "valid_head_daily_ic",
        ],
        ascending=[True, False, False, False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return df


def build_reason(row: pd.Series) -> str:
    parts: list[str] = []
    if row.get("recommendation_reasons"):
        parts.append(str(row["recommendation_reasons"]))
    valid_excess = row.get("valid_excess_mean_return")
    valid_pos = row.get("valid_positive_excess_rate")
    test_relative = row.get("test_relative_return")
    test_daily_ic = row.get("test_daily_ic")
    raw_top20_rel = row.get("raw_top20_test_relative_return")
    if pd.notna(valid_excess):
        parts.append(f"valid超额 {float(valid_excess):.2%}")
    if pd.notna(valid_pos):
        parts.append(f"valid超额胜率 {float(valid_pos):.2%}")
    if pd.notna(test_relative):
        parts.append(f"test相对收益 {float(test_relative):.2%}")
    if pd.notna(test_daily_ic):
        parts.append(f"test daily_ic {float(test_daily_ic):.4f}")
    if pd.notna(raw_top20_rel):
        parts.append(f"raw_top20 {float(raw_top20_rel):.2%}")
    return " | ".join(part for part in parts if part)


def write_report(batch_dir: Path, df: pd.DataFrame) -> Path:
    report_path = batch_dir / "analysis_report.md"
    lines: list[str] = []
    lines.append("# Batch Analysis")
    lines.append("")
    lines.append("## How To Read")
    lines.append("")
    lines.append("- `recommendation_label`: 原有 valid/test 门槛下的推荐/观察/不建议。")
    lines.append("- `candidate_pool_score`: 原始模型 top20 候选池质量分。")
    lines.append("- `valid_*`: 仍然用于 checkpoint 和原生 run 质量判断。")
    lines.append("- `raw_top20_*`: 用于看模型是否适合作为候选池生成器。")
    lines.append("- 本分析不评估 post filter 之后的最终交易名单，最终 top_k 需要在事件面过滤后另行确定。")
    lines.append("")
    lines.append("## Rule")
    lines.append("")
    lines.append("- 第一步：先看 recommendation_label 和原有 valid/test 门槛。")
    lines.append("- 第二步：再看 raw_top20，判断模型候选池是否稳定。")
    lines.append("- 第三步：只把 analysis 结果用于挑选候选池生成器，不直接当最终交易建议。")
    lines.append("")
    if df.empty:
        lines.append("## Result")
        lines.append("")
        lines.append("No completed runs found.")
    else:
        top_row = df.iloc[0]
        lines.append("## Recommendation")
        lines.append("")
        lines.append(
            f"- 推荐模型: `{top_row.get('experiment_name', top_row.get('run_name'))}` | "
            f"label={top_row['recommendation_label']} | "
            f"candidate_pool_score={float(top_row['candidate_pool_score']):.4f}"
        )
        lines.append(f"- run_dir: `{top_row.get('run_dir', '')}`")
        lines.append(f"- 推荐原因: {build_reason(top_row)}")
        lines.append("")
        lines.append("## Top Results")
        lines.append("")
        top_cols = [
            "experiment_name",
            "run_name",
            "recommendation_label",
            "candidate_pool_score",
            "raw_top20_test_relative_return",
            "raw_top20_test_positive_excess_rate",
            "valid_excess_mean_return",
            "valid_positive_excess_rate",
            "valid_relative_return",
            "valid_daily_ic",
            "valid_head_daily_ic",
            "test_relative_return",
            "test_daily_ic",
        ]
        preview = df[top_cols].head(10).copy()
        lines.append("```text")
        lines.append(preview.to_string(index=False))
        lines.append("```")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    batch_dir = Path(args.batch_dir).resolve()
    batch_df = load_batch_rows(batch_dir)
    enriched = enrich_results(batch_df)

    output_csv = batch_dir / "analysis_results.csv"
    enriched.to_csv(output_csv, index=False, encoding="utf-8-sig")
    report_path = write_report(batch_dir, enriched)

    print(f"Batch dir: {batch_dir}")
    print(f"Saved analysis csv: {output_csv}")
    print(f"Saved analysis report: {report_path}")
    if not enriched.empty:
        top_row = enriched.iloc[0]
        print(
            f"Recommended: {top_row.get('experiment_name', top_row.get('run_name'))} "
            f"| label={top_row['recommendation_label']} "
            f"| candidate_pool_score={float(top_row['candidate_pool_score']):.4f}"
        )
        print(f"Reason: {build_reason(top_row)}")


if __name__ == "__main__":
    main()
