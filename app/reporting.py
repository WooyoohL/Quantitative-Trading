from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

from app.runtime import format_rank_value
from metrics.decision import decide_recommendation


CORE_ARTIFACTS = [
    "summary.json",
    "backtest_metrics.json",
    "top_k.csv",
    "orders.csv",
    "run.log",
]

DETAIL_ARTIFACTS = [
    "train_metrics.csv",
    "valid_predictions.csv",
    "test_predictions.csv",
    "valid_backtest.csv",
    "backtest.csv",
    "candidate_rank.csv",
    "inference_predictions.csv",
    "universe_report.csv",
]


def _display_width(text: str) -> int:
    width = 0
    for ch in str(text):
        width += 2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1
    return width


def _pad_cell(text: str, width: int, align: str = "left") -> str:
    text = str(text)
    pad = max(0, width - _display_width(text))
    if align == "right":
        return " " * pad + text
    return text + " " * pad


def _render_metric_value(value: float | int | None, *, percent: bool = False, integer: bool = False) -> str:
    if value is None or pd.isna(value):
        return "-"
    if integer:
        return str(int(value))
    if percent:
        return f"{float(value):.2%}"
    return f"{float(value):.4f}"


def assess_run_usability(summary: dict) -> tuple[str, list[str]]:
    valid_bt = summary.get("valid_backtest_metrics", {}) or {}
    test_bt = summary.get("backtest_metrics", {}) or {}
    decision = decide_recommendation(
        valid_excess_mean_return=valid_bt.get("excess_mean_return"),
        valid_positive_excess_rate=valid_bt.get("positive_excess_rate"),
        valid_daily_ic=summary.get("valid_daily_ic"),
        valid_max_drawdown=valid_bt.get("max_drawdown"),
        test_relative_return=test_bt.get("relative_return"),
        test_positive_excess_rate=test_bt.get("positive_excess_rate"),
        test_daily_ic=summary.get("test_daily_ic"),
    )
    return decision.label, list(decision.reasons)


def _print_metric_table(summary: dict) -> None:
    valid_bt = summary.get("valid_backtest_metrics", {}) or {}
    test_bt = summary.get("backtest_metrics", {}) or {}

    rows = [
        ("TopK", _render_metric_value(summary.get("top_k_count"), integer=True), _render_metric_value(summary.get("top_k_count"), integer=True), "与实盘一致"),
        ("回测天数", _render_metric_value(valid_bt.get("n_backtest_days"), integer=True), _render_metric_value(test_bt.get("n_backtest_days"), integer=True), ">= 20，>= 40 更稳"),
        ("组合胜率", _render_metric_value(valid_bt.get("win_rate"), percent=True), _render_metric_value(test_bt.get("win_rate"), percent=True), ">= 50%"),
        ("日均超额", _render_metric_value(valid_bt.get("excess_mean_return"), percent=True), _render_metric_value(test_bt.get("excess_mean_return"), percent=True), "> 0"),
        ("超额胜率", _render_metric_value(valid_bt.get("positive_excess_rate"), percent=True), _render_metric_value(test_bt.get("positive_excess_rate"), percent=True), ">= 50%"),
        ("相对收益", _render_metric_value(valid_bt.get("relative_return"), percent=True), _render_metric_value(test_bt.get("relative_return"), percent=True), "> 0"),
        ("最大回撤", _render_metric_value(valid_bt.get("max_drawdown"), percent=True), _render_metric_value(test_bt.get("max_drawdown"), percent=True), "-10% ~ 0"),
        ("Daily IC", _render_metric_value(summary.get("valid_daily_ic")), _render_metric_value(summary.get("test_daily_ic")), "> 0"),
        ("Head IC", _render_metric_value(summary.get("valid_head_daily_ic")), _render_metric_value(summary.get("test_head_daily_ic")), "> 0 更稳"),
        ("Loss", _render_metric_value(summary.get("valid_loss")), _render_metric_value(summary.get("test_loss")), "越低越好"),
    ]

    headers = ("指标", "Valid", "Test", "建议范围")
    widths = [
        max(_display_width(headers[0]), *(_display_width(row[0]) for row in rows)),
        max(_display_width(headers[1]), *(_display_width(row[1]) for row in rows)),
        max(_display_width(headers[2]), *(_display_width(row[2]) for row in rows)),
        max(_display_width(headers[3]), *(_display_width(row[3]) for row in rows)),
    ]

    print("[Metrics]")
    header_line = " | ".join(
        [
            _pad_cell(headers[0], widths[0]),
            _pad_cell(headers[1], widths[1], "right"),
            _pad_cell(headers[2], widths[2], "right"),
            _pad_cell(headers[3], widths[3]),
        ]
    )
    divider = "-+-".join("-" * width for width in widths)
    print(f"  {header_line}")
    print(f"  {divider}")
    for metric, valid, test, recommended in rows:
        line = " | ".join(
            [
                _pad_cell(metric, widths[0]),
                _pad_cell(valid, widths[1], "right"),
                _pad_cell(test, widths[2], "right"),
                _pad_cell(recommended, widths[3]),
            ]
        )
        print(f"  {line}")


def print_training_run_summary(summary: dict, history_df: pd.DataFrame, latest_top_k: pd.DataFrame, run_dir: Path) -> None:
    decision_label, reasons = assess_run_usability(summary)

    print(f"Run dir: {run_dir.resolve()}")
    print(
        f"[Run] mode={summary['training_mode']} model={summary['model_name']} "
        f"samples={summary['n_train_samples']}/{summary['n_valid_samples']}/{summary['n_test_samples']} "
        f"seq_len={summary['seq_len']} feature_dim={summary['feature_dim']}"
    )
    print(
        f"[Selection] mode={summary['checkpoint_selection_mode']} "
        f"best_epoch={summary['best_epoch']} "
        f"monitor_valid_ic={summary['best_valid_ic']:.4f} "
        f"monitor_daily_ic={summary['best_valid_daily_ic']:.4f} "
        f"selected_valid_ic={summary['selected_epoch_valid_ic']:.4f} "
        f"selected_daily_ic={summary['selected_epoch_valid_daily_ic']:.4f}"
    )
    _print_metric_table(summary)
    if not history_df.empty:
        print(
            f"[LastEpoch] train_loss={float(history_df.iloc[-1]['train_loss']):.4f} "
            f"valid_loss={float(history_df.iloc[-1]['valid_loss']):.4f}"
        )
    print(f"[Conclusion] {decision_label}" + (f" | {'; '.join(reasons)}" if reasons else ""))
    print(f"[Artifacts] core: {', '.join(CORE_ARTIFACTS)}")
    print(f"[Artifacts] detail: {', '.join(DETAIL_ARTIFACTS)}")
    print("[TopK]")
    for _, row in latest_top_k.iterrows():
        print(
            f"  - {row['symbol']} {row['name']} | {row['industry_name']} | "
            f"score={row['score']:.4f} | market={format_rank_value(row['market_rank'])} "
            f"| candidate={format_rank_value(row['candidate_rank'])} | buy_ref={row['buy_price']:.3f}"
        )


def print_inference_run_summary(summary: dict, latest_top_k: pd.DataFrame, output_dir: Path, source_run_dir: Path, checkpoint_name: str) -> None:
    print(f"Source run: {source_run_dir}")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Output dir: {output_dir}")
    print(
        f"[SourceModel] best_epoch={summary.get('source_best_epoch', '-')} "
        f"valid_ic={_render_metric_value(summary.get('source_best_valid_ic'))} "
        f"valid_daily_ic={_render_metric_value(summary.get('source_best_valid_daily_ic'))} "
        f"test_ic={_render_metric_value(summary.get('source_test_ic'))} "
        f"test_daily_ic={_render_metric_value(summary.get('source_test_daily_ic'))}"
    )
    source_label = summary.get("source_recommendation_label")
    source_reasons = summary.get("source_recommendation_reasons") or []
    if source_label:
        print(f"[SourceConclusion] {source_label}" + (f" | {'; '.join(source_reasons)}" if source_reasons else ""))
    print(
        f"[Inference] signal_date={summary['signal_date']} next_trade_date={summary['next_trade_date']} "
        f"mode={summary['training_mode']} universe={summary['universe_size']} candidates={summary['candidate_rank_count']}"
    )
    print("[Artifacts] core: summary.json, top_k.csv, orders.csv")
    print("[Artifacts] detail: inference_predictions.csv, candidate_rank.csv, universe_report.csv")
    print("[TopK]")
    for _, row in latest_top_k.iterrows():
        print(
            f"  - {row['symbol']} {row['name']} | {row['industry_name']} | "
            f"score={row['score']:.4f} | market={format_rank_value(row['market_rank'])} "
            f"| candidate={format_rank_value(row['candidate_rank'])} | buy_ref={row['buy_price']:.3f}"
        )
