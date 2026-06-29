from __future__ import annotations

import argparse
import copy
import io
import json
import math
import random
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.factories import build_dataset_builder, build_trainer_config
from models.trainer_factory import build_alpha_trainer


@dataclass(frozen=True)
class ScheduledUse:
    strategy: str
    group_key: str
    train_date: pd.Timestamp
    use_date: pd.Timestamp


@dataclass
class IndependentModelState:
    strategy: str
    train_date: pd.Timestamp
    label_mode: str
    label_horizon: int
    run_dir: Path
    trainer: Any
    dataset_builder: Any
    scaler: Any
    peer_map: dict[str, list[tuple[str, float]]]
    selected_symbols: list[str]
    feature_columns: list[str]
    config: dict[str, Any]
    selection_calibration: dict[str, Any]


@dataclass
class InferenceFeatureBlock:
    first_use_date: pd.Timestamp
    last_use_date: pd.Timestamp
    scaled_frame: pd.DataFrame
    cache_path: Path
    loaded_from_cache: bool


@dataclass
class TrainingFeatureBlock:
    train_date: pd.Timestamp
    config: dict[str, Any]
    builder: Any
    scaled_frame: pd.DataFrame
    scaler: Any
    peer_map: dict[str, list[tuple[str, float]]]
    selected_symbols: list[str]
    feature_columns: list[str]
    split_dates: dict[str, list[pd.Timestamp]]
    universe_report: pd.DataFrame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest rolling retrain policies with an independent historical pipeline."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--strategies",
        type=str,
        default="mon_wed,weekly",
        help="Comma separated policies: mon_wed, weekly.",
    )
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--checkpoint-name", type=str, default="best.ckpt")
    parser.add_argument("--sell-after-trading-days", type=int, default=2)
    parser.add_argument(
        "--hold-periods",
        type=str,
        default=None,
        help="Comma separated fixed holding periods. Defaults to --sell-after-trading-days.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["raw_return", "market_excess", "industry_excess"],
        default="raw_return",
        help="Training target mode used only by this independent pipeline.",
    )
    parser.add_argument(
        "--cash-filter",
        choices=["enabled", "disabled"],
        default="enabled",
        help="Skip buying when validation-calibrated score gates are not met.",
    )
    parser.add_argument(
        "--disable-price-cap",
        action="store_true",
        help="Disable universe.filters.max_latest_price only inside this independent backtest.",
    )
    parser.add_argument("--score-quantile", type=float, default=0.60)
    parser.add_argument("--topk-mean-quantile", type=float, default=0.60)
    parser.add_argument("--score-gap-quantile", type=float, default=0.50)
    parser.add_argument("--min-position-exposure", type=float, default=0.03)
    parser.add_argument("--max-position-exposure", type=float, default=0.175)
    parser.add_argument("--max-gross-exposure", type=float, default=0.70)
    parser.add_argument("--buy-slippage-rate", type=float, default=0.001)
    parser.add_argument("--sell-slippage-rate", type=float, default=0.001)
    parser.add_argument("--commission-rate", type=float, default=0.0005)
    parser.add_argument(
        "--min-use-days",
        type=int,
        default=1,
        help="Minimum evaluated signal dates required per policy.",
    )
    parser.add_argument(
        "--required-history-days",
        type=int,
        default=None,
        help="Override warmup trading days required before the first historical train date.",
    )
    parser.add_argument(
        "--auto-start-after-warmup",
        action="store_true",
        help="Move start date to the first date with enough prior data instead of failing.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Build schedules and write plan files without training or inference.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing output directory by skipping completed strategy/use-date rows.",
    )
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for _attempt in range(5):
        try:
            frame.to_csv(path, index=False, encoding="utf-8-sig")
            return
        except OSError as exc:
            last_error = exc
            sleep(0.5)
    if last_error is not None:
        raise last_error


def normalize_date(value: str | None) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    return pd.Timestamp(value).normalize()


def parse_hold_periods(value: str | None, fallback: int) -> list[int]:
    if value is None or str(value).strip() == "":
        return [int(fallback)]
    periods: list[int] = []
    for item in str(value).split(","):
        text = item.strip()
        if not text:
            continue
        period = int(text)
        if period < 1:
            raise ValueError(f"Hold period must be >= 1: {period}")
        periods.append(period)
    if not periods:
        return [int(fallback)]
    return sorted(set(periods))


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_stock_frame(config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config.get("data", {})
    stock_path = repo_path(Path(data_cfg.get("path", "data/eod_daily.csv")))
    if not stock_path.exists():
        raise FileNotFoundError(f"Missing stock data file: {stock_path}")
    frame = pd.read_csv(stock_path)
    required = {"date", "symbol", "open", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Stock data is missing columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_optional_csv(path: Path, date_columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    for column in date_columns or []:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column]).dt.normalize()
    return frame


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_support_frames(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str], set[str]]:
    data_cfg = config.get("data", {})
    index_df = load_optional_csv(repo_path(Path(data_cfg.get("index_path", "data/index_daily.csv"))), ["date"])
    industry_map_df = load_optional_csv(repo_path(Path(data_cfg.get("industry_map_path", "data/industry_map.csv"))))
    industry_daily_df = load_optional_csv(repo_path(Path(data_cfg.get("industry_daily_path", "data/industry_daily.csv"))), ["date"])
    snapshot_df = load_optional_csv(repo_path(Path(data_cfg.get("universe_snapshot_path", "data/universe_snapshot.csv"))))

    name_map: dict[str, str] = {}
    st_symbols: set[str] = set()
    if not snapshot_df.empty and {"symbol", "name"}.issubset(snapshot_df.columns):
        snapshot_df["symbol"] = snapshot_df["symbol"].fillna("").astype(str)
        snapshot_df["name"] = snapshot_df["name"].fillna("").astype(str)
        name_map = dict(snapshot_df.drop_duplicates("symbol", keep="last")[["symbol", "name"]].values)
        st_symbols = set(snapshot_df.loc[snapshot_df["name"].str.upper().str.contains("ST", na=False), "symbol"])

    return index_df, industry_map_df, industry_daily_df, name_map, st_symbols


def slice_by_date(frame: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns:
        return frame.copy()
    cutoff = pd.Timestamp(as_of_date).normalize()
    if pd.api.types.is_datetime64_any_dtype(frame["date"]):
        return frame[frame["date"] <= cutoff].copy()
    return frame[pd.to_datetime(frame["date"]).dt.normalize() <= cutoff].copy()


def date_mask(frame: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.Series:
    cutoff = pd.Timestamp(as_of_date).normalize()
    if pd.api.types.is_datetime64_any_dtype(frame["date"]):
        return frame["date"] <= cutoff
    return pd.to_datetime(frame["date"]).dt.normalize() <= cutoff


def unique_checkpoint_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10_000):
        candidate = path.with_name(f"{stem}_{index:03d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a unique checkpoint path under {path.parent}")


def trading_dates_from_stock(stock_df: pd.DataFrame) -> list[pd.Timestamp]:
    return [pd.Timestamp(value).normalize() for value in sorted(stock_df["date"].dropna().unique())]


def default_required_history_days(config: dict[str, Any]) -> int:
    rolling = config.get("rolling", {})
    sequence = config.get("sequence", {})
    data_cfg = config.get("data", {})
    return (
        int(rolling.get("train_days", 80))
        + int(rolling.get("valid_days", 30))
        + int(sequence.get("seq_len", 20))
        + int(data_cfg.get("label_horizon", 1))
        + 3
    )


def pick_eval_dates(
    *,
    trading_dates: list[pd.Timestamp],
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
    required_history_days: int,
    sell_after_trading_days: int,
    min_use_days: int,
    auto_start_after_warmup: bool,
) -> list[pd.Timestamp]:
    if len(trading_dates) <= required_history_days + sell_after_trading_days:
        raise ValueError(
            "Not enough local trading dates for rolling retrain backtest: "
            f"have={len(trading_dates)} required_history_days={required_history_days} "
            f"sell_after_trading_days={sell_after_trading_days}."
        )

    latest_eval_date = trading_dates[-(sell_after_trading_days + 1)]
    requested_end = end_date or latest_eval_date
    requested_end = min(requested_end, latest_eval_date)
    requested_start = start_date or (requested_end - pd.DateOffset(years=1))

    first_trainable_date = trading_dates[required_history_days]
    if requested_start < first_trainable_date:
        if not auto_start_after_warmup:
            raise ValueError(
                "Requested start date does not have enough prior local data for leak-free training: "
                f"requested_start={requested_start.date()} first_trainable_date={first_trainable_date.date()} "
                f"required_history_days={required_history_days}. "
                "Fetch more history or pass --auto-start-after-warmup."
            )
        requested_start = first_trainable_date

    eval_dates = [date for date in trading_dates if requested_start <= date <= requested_end]
    if len(eval_dates) < int(min_use_days):
        raise ValueError(
            "Backtest range is shorter than the requested minimum use days: "
            f"use_days={len(eval_dates)} min_use_days={min_use_days} "
            f"start={requested_start.date()} end={requested_end.date()}."
        )
    return eval_dates


def strategy_group_key(strategy: str, trade_date: pd.Timestamp) -> str:
    iso = trade_date.isocalendar()
    if strategy == "weekly":
        return f"{iso.year}-W{iso.week:02d}"
    if strategy == "mon_wed":
        weekday = int(trade_date.weekday())
        if weekday <= 1:
            half = "mon_tue"
        elif weekday <= 4:
            half = "wed_fri"
        else:
            half = "weekend"
        return f"{iso.year}-W{iso.week:02d}-{half}"
    raise ValueError(f"Unsupported strategy: {strategy}")


def build_schedule(strategy: str, eval_dates: list[pd.Timestamp]) -> list[ScheduledUse]:
    grouped: dict[str, list[pd.Timestamp]] = {}
    for date in eval_dates:
        grouped.setdefault(strategy_group_key(strategy, date), []).append(date)

    schedule: list[ScheduledUse] = []
    for group_key in sorted(grouped):
        dates = sorted(grouped[group_key])
        train_date = dates[0]
        for use_date in dates:
            schedule.append(
                ScheduledUse(
                    strategy=strategy,
                    group_key=group_key,
                    train_date=train_date,
                    use_date=use_date,
                )
            )
    return sorted(schedule, key=lambda item: (item.strategy, item.use_date, item.train_date))


def independent_config(base_config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config.setdefault("rolling", {})
    config["rolling"]["test_days"] = 0
    config.setdefault("training", {})
    config["training"]["use_candidate_universe"] = False
    return config


def apply_independent_runtime_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = copy.deepcopy(config)
    if bool(getattr(args, "disable_price_cap", False)):
        out.setdefault("universe", {}).setdefault("filters", {})["max_latest_price"] = None
        out.setdefault("training", {})["universe_max_latest_price"] = None
    return out


def policy_name(strategy: str, hold_period: int) -> str:
    return f"{strategy}_h{int(hold_period)}"


def policy_run_label(label_mode: str, hold_period: int) -> str:
    safe_mode = str(label_mode).strip().lower().replace(" ", "_")
    return f"{safe_mode}_h{int(hold_period)}"


def quantile_value(values: pd.Series | np.ndarray | list[float], quantile: float, default: float = 0.0) -> float:
    series = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return float(default)
    q = min(1.0, max(0.0, float(quantile)))
    return float(series.quantile(q))


def apply_label_mode_to_dataset(dataset: Any, label_mode: str) -> dict[str, Any]:
    mode = str(label_mode).strip().lower()
    if getattr(dataset, "targets", None) is None or dataset.meta.empty:
        return {"label_mode": mode, "n_samples": 0}

    meta = dataset.meta.copy()
    labels = pd.to_numeric(meta["label"], errors="coerce")
    market_baseline = labels.groupby(pd.to_datetime(meta["date"]).dt.normalize()).transform("mean")

    if mode == "raw_return":
        baseline = pd.Series(0.0, index=meta.index)
    elif mode == "market_excess":
        baseline = market_baseline
    elif mode == "industry_excess":
        industry_key = meta["industry_name"].fillna("__missing__").astype(str)
        baseline = labels.groupby([pd.to_datetime(meta["date"]).dt.normalize(), industry_key]).transform("mean")
        baseline = baseline.fillna(market_baseline)
    else:
        raise ValueError(f"Unsupported label mode: {label_mode}")

    adjusted = (labels - baseline).replace([np.inf, -np.inf], np.nan)
    keep_mask = adjusted.notna()
    if not bool(keep_mask.all()):
        dataset.features = dataset.features[keep_mask.to_numpy()]
        adjusted = adjusted[keep_mask].reset_index(drop=True)
        labels = labels[keep_mask].reset_index(drop=True)
        baseline = baseline[keep_mask].reset_index(drop=True)
        meta = meta[keep_mask].reset_index(drop=True)

    dataset.targets = adjusted.to_numpy(dtype=np.float32)
    meta["raw_label"] = labels.to_numpy(dtype=np.float32)
    meta["label_baseline"] = baseline.to_numpy(dtype=np.float32)
    meta["label"] = dataset.targets
    meta["label_mode"] = mode
    dataset.meta = meta.reset_index(drop=True)
    return {
        "label_mode": mode,
        "n_samples": int(len(dataset.targets)),
        "mean_raw_label": float(pd.Series(meta["raw_label"]).mean()) if len(meta) else 0.0,
        "mean_adjusted_label": float(pd.Series(dataset.targets).mean()) if len(dataset.targets) else 0.0,
    }


def build_selection_calibration(
    *,
    valid_meta: pd.DataFrame,
    valid_scores: np.ndarray,
    top_k: int,
    score_quantile: float,
    topk_mean_quantile: float,
    score_gap_quantile: float,
    label_mode: str,
    hold_period: int,
) -> dict[str, Any]:
    if valid_meta.empty or len(valid_scores) == 0:
        return {
            "label_mode": str(label_mode),
            "hold_period": int(hold_period),
            "top_k": int(top_k),
            "enabled": False,
            "reason": "empty_validation_scores",
        }

    scored = valid_meta.copy()
    scored["score"] = np.asarray(valid_scores, dtype=np.float32)
    scored["date"] = pd.to_datetime(scored["date"]).dt.normalize()
    daily_rows: list[dict[str, Any]] = []
    selected_scores: list[float] = []
    selected_returns: list[float] = []

    for signal_date, group in scored.groupby("date"):
        ranked = group.sort_values(["score", "symbol"], ascending=[False, True]).reset_index(drop=True)
        selected = ranked.head(int(top_k)).copy()
        if selected.empty:
            continue
        next_score = ranked["score"].iloc[int(top_k)] if len(ranked) > int(top_k) else selected["score"].iloc[-1]
        score_gap = float(selected["score"].iloc[0] - next_score)
        labels = pd.to_numeric(selected["label"], errors="coerce")
        daily_rows.append(
            {
                "date": pd.Timestamp(signal_date).date().isoformat(),
                "topk_mean_score": float(pd.to_numeric(selected["score"], errors="coerce").mean()),
                "topk_min_score": float(pd.to_numeric(selected["score"], errors="coerce").min()),
                "score_gap": score_gap,
                "topk_mean_label": float(labels.mean()) if labels.notna().any() else float("nan"),
                "positive_label": float((labels > 0.0).mean()) if labels.notna().any() else float("nan"),
            }
        )
        selected_scores.extend(pd.to_numeric(selected["score"], errors="coerce").dropna().astype(float).tolist())
        selected_returns.extend(labels.dropna().astype(float).tolist())

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return {
            "label_mode": str(label_mode),
            "hold_period": int(hold_period),
            "top_k": int(top_k),
            "enabled": False,
            "reason": "empty_validation_daily_stats",
        }

    calibration = {
        "label_mode": str(label_mode),
        "hold_period": int(hold_period),
        "top_k": int(top_k),
        "enabled": True,
        "score_quantile": float(score_quantile),
        "topk_mean_quantile": float(topk_mean_quantile),
        "score_gap_quantile": float(score_gap_quantile),
        "min_score": quantile_value(selected_scores, score_quantile),
        "min_topk_mean_score": quantile_value(daily["topk_mean_score"], topk_mean_quantile),
        "min_score_gap": quantile_value(daily["score_gap"], score_gap_quantile),
        "score_p50": quantile_value(selected_scores, 0.50),
        "score_p90": quantile_value(selected_scores, 0.90),
        "valid_daily_count": int(len(daily)),
        "valid_selected_count": int(len(selected_scores)),
        "valid_topk_mean_label": float(pd.to_numeric(daily["topk_mean_label"], errors="coerce").mean()),
        "valid_positive_label_rate": float(pd.to_numeric(daily["positive_label"], errors="coerce").mean()),
        "valid_selected_mean_label": float(pd.Series(selected_returns, dtype="float64").mean())
        if selected_returns
        else float("nan"),
    }
    return calibration


def score_confidence(score: Any, calibration: dict[str, Any]) -> float:
    value = pd.to_numeric(pd.Series([score]), errors="coerce").iloc[0]
    if pd.isna(value):
        return 0.0
    low = float(calibration.get("score_p50", calibration.get("min_score", 0.0)))
    high = float(calibration.get("score_p90", low + 1.0))
    denom = max(abs(high - low), 1e-9)
    return float(min(1.0, max(0.0, (float(value) - low) / denom)))


def apply_confidence_exposure(
    picks: pd.DataFrame,
    *,
    calibration: dict[str, Any],
    min_position_exposure: float,
    max_position_exposure: float,
    max_gross_exposure: float,
) -> pd.DataFrame:
    if picks.empty:
        return picks.copy()
    out = picks.copy()
    out["position_confidence"] = out["score"].apply(lambda value: score_confidence(value, calibration))
    min_exposure = max(0.0, float(min_position_exposure))
    max_exposure = max(min_exposure, float(max_position_exposure))
    out["target_weight"] = min_exposure + (max_exposure - min_exposure) * out["position_confidence"].astype(float)
    gross = float(out["target_weight"].sum())
    cap = max(0.0, float(max_gross_exposure))
    if gross > cap and gross > 0.0:
        out["target_weight"] = out["target_weight"] * (cap / gross)
    return out


def select_policy_picks(
    *,
    review: pd.DataFrame,
    top_k: int,
    expected_use_date: pd.Timestamp,
    cash_filter_enabled: bool,
    calibration: dict[str, Any],
    min_position_exposure: float,
    max_position_exposure: float,
    max_gross_exposure: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ranked = select_review_picks(review, top_k=top_k, expected_use_date=expected_use_date)
    all_ranked = review.copy()
    if "signal_date" in all_ranked.columns:
        all_ranked["signal_date"] = pd.to_datetime(all_ranked["signal_date"]).dt.normalize()
        all_ranked = all_ranked[all_ranked["signal_date"] == pd.Timestamp(expected_use_date).normalize()].copy()
    if "score" in all_ranked.columns:
        all_ranked = all_ranked.sort_values(["score", "symbol"], ascending=[False, True]).reset_index(drop=True)
    topk_mean = float(pd.to_numeric(ranked["score"], errors="coerce").mean()) if not ranked.empty else float("nan")
    min_score = float(pd.to_numeric(ranked["score"], errors="coerce").min()) if not ranked.empty else float("nan")
    if not ranked.empty and not all_ranked.empty:
        next_score = (
            pd.to_numeric(all_ranked["score"], errors="coerce").iloc[int(top_k)]
            if len(all_ranked) > int(top_k)
            else pd.to_numeric(ranked["score"], errors="coerce").iloc[-1]
        )
        score_gap = float(pd.to_numeric(ranked["score"], errors="coerce").iloc[0] - next_score)
    else:
        score_gap = float("nan")

    decision = {
        "cash_filter_enabled": bool(cash_filter_enabled),
        "cash_filter_pass": True,
        "skip_reason": "",
        "pre_filter_pick_count": int(len(ranked)),
        "topk_mean_score": topk_mean,
        "min_score": min_score,
        "score_gap": score_gap,
        "threshold_min_score": calibration.get("min_score"),
        "threshold_topk_mean_score": calibration.get("min_topk_mean_score"),
        "threshold_score_gap": calibration.get("min_score_gap"),
    }
    reasons: list[str] = []
    if ranked.empty:
        reasons.append("empty_ranked_picks")
    if cash_filter_enabled and calibration.get("enabled", False):
        if pd.isna(min_score) or min_score < float(calibration.get("min_score", 0.0)):
            reasons.append("score_below_validation_threshold")
        if pd.isna(topk_mean) or topk_mean < float(calibration.get("min_topk_mean_score", 0.0)):
            reasons.append("topk_mean_below_validation_threshold")
        if pd.isna(score_gap) or score_gap < float(calibration.get("min_score_gap", 0.0)):
            reasons.append("score_gap_below_validation_threshold")
    elif cash_filter_enabled and not calibration.get("enabled", False):
        reasons.append(str(calibration.get("reason", "calibration_unavailable")))

    if reasons:
        decision["cash_filter_pass"] = False
        decision["skip_reason"] = ";".join(reasons)
        empty = ranked.iloc[0:0].copy()
        empty["position_confidence"] = pd.Series(dtype="float64")
        empty["target_weight"] = pd.Series(dtype="float64")
        return empty, decision

    picks = apply_confidence_exposure(
        ranked,
        calibration=calibration,
        min_position_exposure=min_position_exposure,
        max_position_exposure=max_position_exposure,
        max_gross_exposure=max_gross_exposure,
    )
    decision["gross_exposure"] = float(pd.to_numeric(picks["target_weight"], errors="coerce").fillna(0.0).sum())
    decision["cash_weight"] = float(max(0.0, 1.0 - decision["gross_exposure"]))
    return picks, decision


def skip_train_dataset_epoch_prediction(trainer: Any, train_dataset: Any) -> None:
    original_predict_dataset = trainer.predict_dataset

    def predict_dataset_fast(dataset: Any) -> np.ndarray:
        if dataset is train_dataset:
            return np.zeros(len(train_dataset), dtype=np.float32)
        return original_predict_dataset(dataset)

    trainer.predict_dataset = predict_dataset_fast


def configure_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_symbols_as_of(
    stock_df: pd.DataFrame,
    config: dict[str, Any],
    st_symbols: set[str],
    as_of_date: pd.Timestamp,
) -> tuple[list[str], pd.DataFrame]:
    scoped = stock_df[date_mask(stock_df, as_of_date)].copy()
    if scoped.empty:
        raise ValueError(f"No stock rows on or before {as_of_date.date()}.")

    universe_cfg = config.get("universe", {})
    filters = universe_cfg.get("filters", {})
    lookback_days = int(universe_cfg.get("lookback_days", 60))
    min_avg_turnover = float(filters.get("min_avg_turnover", 2e8))
    min_latest_price = float(filters.get("min_latest_price", 2.0))
    max_latest_price = filters.get("max_latest_price")
    max_latest_price = None if max_latest_price is None else float(max_latest_price)
    max_avg_intraday_range = float(filters.get("max_avg_intraday_range", 0.08))
    max_avg_abs_ret1 = float(filters.get("max_avg_abs_ret1", 0.05))
    min_n_days = int(filters.get("min_n_days", max(30, lookback_days - 10)))
    exclude_symbols = {str(symbol) for symbol in universe_cfg.get("exclude_symbols", [])}

    unique_dates = sorted(scoped["date"].drop_duplicates())
    active_dates = set(unique_dates[-lookback_days:])
    recent = scoped[scoped["date"].isin(active_dates)].sort_values(["symbol", "date"]).copy()
    grouped = recent.groupby("symbol", group_keys=False)
    recent["ret_1_calc"] = grouped["close"].pct_change().fillna(0.0)
    recent["intraday_range"] = (recent["high"] - recent["low"]) / recent["close"].replace(0.0, np.nan)
    latest_close = recent.groupby("symbol")["close"].last()

    report = (
        recent.groupby("symbol", as_index=False)
        .agg(
            avg_turnover=("turnover", "mean"),
            avg_intraday_range=("intraday_range", "mean"),
            avg_abs_ret1=("ret_1_calc", lambda value: value.abs().mean()),
            n_days=("date", "nunique"),
        )
        .sort_values("symbol")
        .reset_index(drop=True)
    )
    report["latest_close"] = report["symbol"].map(latest_close)
    report["is_st_symbol"] = report["symbol"].astype(str).isin(st_symbols)
    report["passes"] = (
        ~report["symbol"].astype(str).isin(exclude_symbols)
        & ~report["is_st_symbol"]
        & (report["avg_turnover"] >= min_avg_turnover)
        & (report["avg_intraday_range"] <= max_avg_intraday_range)
        & (report["avg_abs_ret1"] <= max_avg_abs_ret1)
        & (report["n_days"] >= min_n_days)
        & (report["latest_close"] >= min_latest_price)
    )
    if max_latest_price is not None:
        report["passes"] = report["passes"] & (report["latest_close"] <= max_latest_price)

    selected = report[report["passes"]].copy()
    if selected.empty:
        selected = report[~report["is_st_symbol"]].copy()
    selected["liq_rank"] = selected["avg_turnover"].rank(method="average", ascending=False, pct=True)
    selected["stability_rank"] = selected["avg_abs_ret1"].rank(method="average", ascending=True, pct=True)
    selected["range_rank"] = selected["avg_intraday_range"].rank(method="average", ascending=True, pct=True)
    selected["price_rank"] = selected["latest_close"].rank(method="average", ascending=True, pct=True)
    selected["pool_score"] = (
        0.45 * selected["liq_rank"]
        + 0.20 * selected["stability_rank"]
        + 0.15 * selected["range_rank"]
        + 0.20 * selected["price_rank"]
    )
    selected = selected.sort_values(["pool_score", "avg_turnover"], ascending=[False, False]).reset_index(drop=True)
    return selected["symbol"].astype(str).tolist(), selected


def open_to_open_label(frame: pd.DataFrame, label_horizon: int) -> pd.Series:
    ordered = frame[["symbol", "date", "open"]].copy()
    ordered["symbol"] = ordered["symbol"].astype(str)
    ordered["date"] = pd.to_datetime(ordered["date"]).dt.normalize()
    ordered = ordered.sort_values(["symbol", "date"])
    grouped = ordered.groupby("symbol", group_keys=False)
    entry_open = grouped["open"].shift(-1)
    exit_open = grouped["open"].shift(-(int(label_horizon) + 1))
    label = exit_open / entry_open.replace(0.0, np.nan) - 1.0
    return label.reindex(frame.index)


def build_training_feature_block(
    *,
    base_config: dict[str, Any],
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    industry_map_df: pd.DataFrame,
    industry_daily_df: pd.DataFrame,
    st_symbols: set[str],
    train_date: pd.Timestamp,
) -> TrainingFeatureBlock:
    config = independent_config(base_config)
    train_date = pd.Timestamp(train_date).normalize()
    stock_slice = slice_by_date(stock_df, train_date)
    keep_days = int(config.get("data", {}).get("trainable_history_days", 260))
    unique_dates = sorted(stock_slice["date"].drop_duplicates())
    if len(unique_dates) > keep_days:
        stock_slice = stock_slice[stock_slice["date"].isin(set(unique_dates[-keep_days:]))].copy()
    selected_symbols, universe_report = select_symbols_as_of(stock_slice, config, st_symbols, train_date)
    context_stock = stock_slice[stock_slice["symbol"].astype(str).isin(selected_symbols)].copy()

    builder = build_dataset_builder(config, verbose=False)
    rolling = config.get("rolling", {})
    bundle = builder.build_bundle(
        raw_df=context_stock,
        train_days=int(rolling.get("train_days", 80)),
        valid_days=int(rolling.get("valid_days", 30)),
        test_days=0,
        index_df=slice_by_date(index_df, train_date) if config.get("index", {}).get("enabled", True) else pd.DataFrame(),
        industry_map_df=industry_map_df if config.get("industry", {}).get("enabled", True) else pd.DataFrame(),
        industry_daily_df=slice_by_date(industry_daily_df, train_date)
        if config.get("industry", {}).get("enabled", True)
        else pd.DataFrame(),
        sample_symbols=selected_symbols,
    )
    return TrainingFeatureBlock(
        train_date=train_date,
        config=config,
        builder=builder,
        scaled_frame=bundle.feature_frame,
        scaler=bundle.scaler,
        peer_map=bundle.peer_map,
        selected_symbols=selected_symbols,
        feature_columns=bundle.feature_columns,
        split_dates=bundle.split_dates,
        universe_report=universe_report,
    )


def build_training_datasets_from_feature_block(
    *,
    block: TrainingFeatureBlock,
    label_mode: str,
    hold_period: int,
) -> dict[str, Any]:
    scaled_frame = block.scaled_frame.copy()
    scaled_frame["label"] = open_to_open_label(scaled_frame, int(hold_period))
    train_dataset = block.builder._build_sequence_dataset(
        scaled_frame,
        allowed_dates=block.split_dates["train"],
        require_label=True,
        sample_symbols=block.selected_symbols,
    )
    valid_dataset = block.builder._build_sequence_dataset(
        scaled_frame,
        allowed_dates=block.split_dates["valid"],
        require_label=True,
        sample_symbols=block.selected_symbols,
    )
    test_dataset = block.builder._build_sequence_dataset(
        scaled_frame,
        allowed_dates=block.split_dates["test"],
        require_label=True,
        sample_symbols=block.selected_symbols,
    )
    latest_signal_date = pd.to_datetime(scaled_frame["date"]).max()
    inference_dataset = block.builder._build_sequence_dataset(
        scaled_frame,
        allowed_dates=[latest_signal_date],
        require_label=False,
        sample_symbols=block.selected_symbols,
    )
    label_reports = {
        "train": apply_label_mode_to_dataset(train_dataset, label_mode),
        "valid": apply_label_mode_to_dataset(valid_dataset, label_mode),
    }
    apply_label_mode_to_dataset(test_dataset, label_mode)
    return {
        "train_dataset": train_dataset,
        "valid_dataset": valid_dataset,
        "test_dataset": test_dataset,
        "inference_dataset": inference_dataset,
        "label_reports": label_reports,
    }


def assert_complete_dir(path: Path, required_files: list[str]) -> bool:
    return path.exists() and all((path / name).exists() for name in required_files)


def apply_right_side_filter(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    filter_cfg = config.get("strategy", {}).get("right_side_filter", {})
    if not bool(filter_cfg.get("enabled", False)):
        return frame.copy()
    out = frame.copy()
    rules: list[pd.Series] = []
    for column, key in [
        ("ret_1", "min_ret_1"),
        ("ret_5", "min_ret_5"),
        ("intraday_ret", "min_intraday_ret"),
        ("ma_gap_5", "min_ma_gap_5"),
        ("volume_ratio_5", "min_volume_ratio_5"),
        ("industry_ret_1_mean", "min_industry_ret_1_mean"),
    ]:
        threshold = filter_cfg.get(key)
        if threshold is not None and column in out.columns:
            rules.append(pd.to_numeric(out[column], errors="coerce") >= float(threshold))
    if not rules:
        return out
    mask = rules[0].copy()
    for rule in rules[1:]:
        mask &= rule
    filtered = out[mask.fillna(False)].copy()
    return out if filtered.empty else filtered


def write_review_outputs_from_scored(
    *,
    scored: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    use_date: pd.Timestamp,
) -> pd.DataFrame:
    use_date = pd.Timestamp(use_date).normalize()
    output_dir.mkdir(parents=True, exist_ok=True)
    scored = scored.copy()
    write_csv(output_dir / "inference_predictions.csv", scored)

    candidate_rank = scored.sort_values(["score", "symbol"], ascending=[False, True]).reset_index(drop=True).copy()
    candidate_rank["candidate_rank"] = candidate_rank.index + 1
    candidate_rank["market_rank"] = pd.NA
    keep_columns = [
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
    for column in keep_columns:
        if column not in candidate_rank.columns:
            candidate_rank[column] = pd.NA
    candidate_rank = candidate_rank[keep_columns].copy()
    write_csv(output_dir / "candidate_rank.csv", candidate_rank)

    pool = candidate_rank.copy()
    max_price = config.get("universe", {}).get("filters", {}).get("max_latest_price")
    if max_price is not None:
        pool = pool[pd.to_numeric(pool["close"], errors="coerce") <= float(max_price)].copy()
    if pool.empty:
        raise ValueError(f"Recommendation pool is empty after price filtering for {use_date.date()}.")
    pool = apply_right_side_filter(pool, config)

    review_size = int(config.get("strategy", {}).get("review_top_k", 20))
    review = pool.nsmallest(review_size, "candidate_rank").copy()
    review["next_trade_date"] = pd.NA
    review = review.rename(columns={"close": "buy_price"})
    review["buy_price_basis"] = "signal_close_ref"
    review["entry_price_ref_close"] = review["buy_price"]
    review = review[
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
    write_csv(output_dir / "review_top_k.csv", review)
    return review


def build_review_outputs(
    *,
    state: IndependentModelState,
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    industry_map_df: pd.DataFrame,
    industry_daily_df: pd.DataFrame,
    name_map: dict[str, str],
    use_date: pd.Timestamp,
    output_dir: Path,
) -> pd.DataFrame:
    use_date = pd.Timestamp(use_date).normalize()
    raw_df = stock_df[
        date_mask(stock_df, use_date)
        & stock_df["symbol"].astype(str).isin(set(state.selected_symbols))
    ].copy()
    keep_days = int(state.config.get("data", {}).get("trainable_history_days", 260))
    unique_dates = sorted(raw_df["date"].drop_duplicates())
    if len(unique_dates) > keep_days:
        raw_df = raw_df[raw_df["date"].isin(set(unique_dates[-keep_days:]))].copy()
    inference_dataset, _scaled = state.dataset_builder.build_inference_dataset(
        raw_df=raw_df,
        scaler=state.scaler,
        index_df=slice_by_date(index_df, use_date) if state.config.get("index", {}).get("enabled", True) else pd.DataFrame(),
        industry_map_df=industry_map_df if state.config.get("industry", {}).get("enabled", True) else pd.DataFrame(),
        industry_daily_df=slice_by_date(industry_daily_df, use_date)
        if state.config.get("industry", {}).get("enabled", True)
        else pd.DataFrame(),
        peer_map=state.peer_map,
        signal_date=use_date,
    )
    if len(inference_dataset) == 0:
        raise ValueError(f"Inference dataset is empty for {use_date.date()}.")

    pred = state.trainer.predict_dataset(inference_dataset)
    scored = inference_dataset.meta.copy()
    scored["score"] = pred
    scored["name"] = scored["symbol"].astype(str).map(name_map).fillna("")
    return write_review_outputs_from_scored(
        scored=scored,
        config=state.config,
        output_dir=output_dir,
        use_date=use_date,
    )


def inference_output_dir(
    *,
    state: IndependentModelState,
    out_dir: Path,
    run_id: str,
    strategy: str,
    use_date: pd.Timestamp,
) -> Path:
    use_date = pd.Timestamp(use_date).normalize()
    if use_date == state.train_date:
        return state.run_dir.resolve()
    return (
        out_dir
        / "inference_runs"
        / strategy
        / f"{run_id}_{strategy}_infer_{state.train_date.strftime('%Y%m%d')}_{use_date.strftime('%Y%m%d')}"
    ).resolve()


def feature_block_cache_path(
    *,
    out_dir: Path,
    strategy: str,
    train_date: pd.Timestamp,
    use_dates: list[pd.Timestamp],
) -> Path:
    first_use = min(pd.Timestamp(value).normalize() for value in use_dates)
    last_use = max(pd.Timestamp(value).normalize() for value in use_dates)
    return (
        out_dir
        / "feature_blocks"
        / (
            f"{strategy}_train_{pd.Timestamp(train_date).strftime('%Y%m%d')}"
            f"_use_{first_use.strftime('%Y%m%d')}_{last_use.strftime('%Y%m%d')}.pkl"
        )
    )


def build_or_load_inference_feature_block(
    *,
    state: IndependentModelState,
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    industry_map_df: pd.DataFrame,
    industry_daily_df: pd.DataFrame,
    out_dir: Path,
    strategy: str,
    use_dates: list[pd.Timestamp],
) -> InferenceFeatureBlock:
    normalized_dates = sorted({pd.Timestamp(value).normalize() for value in use_dates})
    if not normalized_dates:
        raise ValueError("No use dates supplied for inference feature block.")

    cache_path = feature_block_cache_path(
        out_dir=out_dir,
        strategy=strategy,
        train_date=state.train_date,
        use_dates=normalized_dates,
    )
    if cache_path.exists():
        scaled_frame = pd.read_pickle(cache_path)
        if "date" in scaled_frame.columns:
            scaled_frame["date"] = pd.to_datetime(scaled_frame["date"]).dt.normalize()
        return InferenceFeatureBlock(
            first_use_date=normalized_dates[0],
            last_use_date=normalized_dates[-1],
            scaled_frame=scaled_frame,
            cache_path=cache_path,
            loaded_from_cache=True,
        )

    first_use_date = normalized_dates[0]
    max_use_date = normalized_dates[-1]
    raw_df = stock_df[
        date_mask(stock_df, max_use_date)
        & stock_df["symbol"].astype(str).isin(set(state.selected_symbols))
    ].copy()
    keep_days = int(state.config.get("data", {}).get("trainable_history_days", 260))
    unique_dates = sorted(raw_df["date"].drop_duplicates())
    dates_to_first_use = [date for date in unique_dates if date <= first_use_date]
    if len(dates_to_first_use) > keep_days:
        keep_start = dates_to_first_use[-keep_days]
        raw_df = raw_df[raw_df["date"] >= keep_start].copy()
    if raw_df.empty:
        raise ValueError(f"No raw rows for inference feature block ending {max_use_date.date()}.")

    builder = state.dataset_builder
    base_frame = builder._build_base_feature_frame(
        raw_df,
        industry_map_df if state.config.get("industry", {}).get("enabled", True) else pd.DataFrame(),
    )
    feature_frame = builder._attach_market_features(base_frame)
    feature_frame = builder._attach_index_features(
        feature_frame,
        slice_by_date(index_df, max_use_date) if state.config.get("index", {}).get("enabled", True) else pd.DataFrame(),
    )
    feature_frame = builder._attach_industry_features(
        feature_frame,
        slice_by_date(industry_daily_df, max_use_date)
        if state.config.get("industry", {}).get("enabled", True)
        else pd.DataFrame(),
    )
    feature_frame = builder._attach_peer_features(feature_frame, state.peer_map)
    feature_frame = builder._finalize_feature_frame(feature_frame)
    scaled_frame = state.scaler.transform(feature_frame, state.feature_columns)
    if builder.daily_cross_sectional_norm:
        scaled_frame = builder._apply_daily_cross_sectional_norm(scaled_frame)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    scaled_frame.to_pickle(cache_path)
    return InferenceFeatureBlock(
        first_use_date=normalized_dates[0],
        last_use_date=normalized_dates[-1],
        scaled_frame=scaled_frame,
        cache_path=cache_path,
        loaded_from_cache=False,
    )


def infer_independent_model_block(
    *,
    state: IndependentModelState,
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    industry_map_df: pd.DataFrame,
    industry_daily_df: pd.DataFrame,
    name_map: dict[str, str],
    out_dir: Path,
    run_id: str,
    strategy: str,
    use_dates: list[pd.Timestamp],
) -> dict[pd.Timestamp, tuple[Path, pd.DataFrame]]:
    normalized_dates = sorted({pd.Timestamp(value).normalize() for value in use_dates})
    block = build_or_load_inference_feature_block(
        state=state,
        stock_df=stock_df,
        index_df=index_df,
        industry_map_df=industry_map_df,
        industry_daily_df=industry_daily_df,
        out_dir=out_dir,
        strategy=strategy,
        use_dates=normalized_dates,
    )
    inference_dataset = state.dataset_builder._build_sequence_dataset(
        block.scaled_frame,
        allowed_dates=normalized_dates,
        require_label=False,
        sample_symbols=state.selected_symbols,
    )
    if len(inference_dataset) == 0:
        raise ValueError(
            f"Inference dataset is empty for {strategy} train={state.train_date.date()} "
            f"use={normalized_dates[0].date()}..{normalized_dates[-1].date()}."
        )

    pred = state.trainer.predict_dataset(inference_dataset)
    scored_all = inference_dataset.meta.copy()
    scored_all["score"] = pred
    scored_all["name"] = scored_all["symbol"].astype(str).map(name_map).fillna("")
    scored_all["signal_date"] = pd.to_datetime(scored_all["signal_date"]).dt.normalize()

    results: dict[pd.Timestamp, tuple[Path, pd.DataFrame]] = {}
    for use_date in normalized_dates:
        day_scored = scored_all[scored_all["signal_date"] == use_date].copy()
        if day_scored.empty:
            raise ValueError(f"Inference dataset has no rows for {use_date.date()}.")
        output_dir = inference_output_dir(
            state=state,
            out_dir=out_dir,
            run_id=run_id,
            strategy=strategy,
            use_date=use_date,
        )
        review = write_review_outputs_from_scored(
            scored=day_scored,
            config=state.config,
            output_dir=output_dir,
            use_date=use_date,
        )
        if output_dir.resolve() != state.run_dir.resolve():
            write_json(
                output_dir / "summary.json",
                {
                    "pipeline": "independent_rolling_retrain_backtest",
                    "source_run_dir": str(state.run_dir.resolve()),
                    "train_date": state.train_date.date().isoformat(),
                    "signal_date": use_date.date().isoformat(),
                    "review_top_k_count": int(len(review)),
                    "universe_size": int(len(state.selected_symbols)),
                    "feature_block_cache_path": str(block.cache_path.resolve()),
                    "feature_block_loaded_from_cache": bool(block.loaded_from_cache),
                },
            )
        results[use_date] = (output_dir.resolve(), review)
    return results


def train_independent_model(
    *,
    base_config: dict[str, Any],
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    industry_map_df: pd.DataFrame,
    industry_daily_df: pd.DataFrame,
    name_map: dict[str, str],
    st_symbols: set[str],
    out_dir: Path,
    train_date: pd.Timestamp,
    feature_block: TrainingFeatureBlock,
    checkpoint_name: str,
    label_mode: str,
    hold_period: int,
    top_k: int,
    score_quantile: float,
    topk_mean_quantile: float,
    score_gap_quantile: float,
) -> IndependentModelState:
    config = independent_config(base_config)
    config.setdefault("data", {})
    config["data"]["label_horizon"] = int(hold_period)
    train_date = pd.Timestamp(train_date).normalize()
    run_label = policy_run_label(label_mode, hold_period)
    run_dir = out_dir / "runs" / run_label / f"train_{train_date.strftime('%Y%m%d')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "logs" / run_label / f"train_{train_date.strftime('%Y%m%d')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _train() -> IndependentModelState:
        configure_seed(int(config.get("seed", 7)))
        dataset_payload = build_training_datasets_from_feature_block(
            block=feature_block,
            label_mode=label_mode,
            hold_period=int(hold_period),
        )
        train_dataset = dataset_payload["train_dataset"]
        valid_dataset = dataset_payload["valid_dataset"]
        test_dataset = dataset_payload["test_dataset"]
        inference_dataset = dataset_payload["inference_dataset"]
        label_reports = dataset_payload["label_reports"]
        if len(train_dataset) == 0 or len(valid_dataset) == 0 or len(inference_dataset) == 0:
            raise ValueError(f"Independent split is empty for train_date={train_date.date()}.")

        input_dim = len(feature_block.feature_columns)
        seq_len = int(train_dataset.features.shape[1])
        trainer, resolved_model_config = build_alpha_trainer(
            input_dim=input_dim,
            seq_len=seq_len,
            feature_columns=feature_block.feature_columns,
            model_cfg=config.get("model", {}),
            trainer_config=build_trainer_config(config),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        skip_train_dataset_epoch_prediction(trainer, train_dataset)
        original_save_checkpoint = trainer._save_checkpoint
        original_load_checkpoint = trainer.load_checkpoint
        memory_checkpoints: dict[str, bytes] = {}
        best_checkpoint_state: dict[str, Path | str] = {
            "path": run_dir / checkpoint_name,
            "key": str((run_dir / checkpoint_name).resolve()),
        }

        def save_checkpoint_without_last(path: Path, *args: Any, **kwargs: Any) -> None:
            requested_path = Path(path)
            if requested_path.name == "last.ckpt":
                return
            checkpoint_key = str(requested_path.resolve())
            if requested_path.name == checkpoint_name:
                checkpoint_key = str(best_checkpoint_state["key"])
            buffer = io.BytesIO()
            original_save_checkpoint(buffer, *args, **kwargs)
            memory_checkpoints[checkpoint_key] = buffer.getvalue()

        def load_checkpoint_with_alias(path: Path) -> dict[str, Any]:
            requested_path = Path(path)
            checkpoint_key = str(requested_path.resolve())
            if requested_path.name == checkpoint_name:
                checkpoint_key = str(best_checkpoint_state["key"])
            payload_bytes = memory_checkpoints.get(checkpoint_key)
            if payload_bytes is not None:
                return original_load_checkpoint(io.BytesIO(payload_bytes))
            return original_load_checkpoint(requested_path)

        trainer._save_checkpoint = save_checkpoint_without_last
        trainer.load_checkpoint = load_checkpoint_with_alias
        history_df = trainer.fit(
            train_dataset,
            valid_dataset,
            run_dir=run_dir,
            model_config={**resolved_model_config, "feature_columns": feature_block.feature_columns},
        )
        selected_payload = trainer.load_checkpoint(run_dir / checkpoint_name)
        final_checkpoint_path = unique_checkpoint_path(run_dir / "selected_best.ckpt")
        torch.save(selected_payload, final_checkpoint_path)
        best_checkpoint_state["path"] = final_checkpoint_path
        valid_scores = trainer.predict_dataset(valid_dataset)
        selection_calibration = build_selection_calibration(
            valid_meta=valid_dataset.meta,
            valid_scores=valid_scores,
            top_k=int(top_k),
            score_quantile=float(score_quantile),
            topk_mean_quantile=float(topk_mean_quantile),
            score_gap_quantile=float(score_gap_quantile),
            label_mode=str(label_mode),
            hold_period=int(hold_period),
        )

        state = IndependentModelState(
            strategy="shared",
            train_date=train_date,
            label_mode=str(label_mode),
            label_horizon=int(hold_period),
            run_dir=run_dir,
            trainer=trainer,
            dataset_builder=feature_block.builder,
            scaler=feature_block.scaler,
            peer_map=feature_block.peer_map,
            selected_symbols=feature_block.selected_symbols,
            feature_columns=feature_block.feature_columns,
            config=config,
            selection_calibration=selection_calibration,
        )
        review = build_review_outputs(
            state=state,
            stock_df=stock_df,
            index_df=index_df,
            industry_map_df=industry_map_df,
            industry_daily_df=industry_daily_df,
            name_map=name_map,
            use_date=train_date,
            output_dir=run_dir,
        )
        feature_block.universe_report.to_csv(run_dir / "universe_report.csv", index=False, encoding="utf-8-sig")
        write_yaml(run_dir / "config.yaml", config)
        (run_dir / "best_checkpoint.txt").write_text(str(best_checkpoint_state["path"]), encoding="utf-8")
        write_json(run_dir / "selection_calibration.json", selection_calibration)
        write_json(
            run_dir / "summary.json",
            {
                "run_dir": str(run_dir.resolve()),
                "pipeline": "independent_rolling_retrain_backtest",
                "train_date": train_date.date().isoformat(),
                "signal_date": train_date.date().isoformat(),
                "label_mode": str(label_mode),
                "label_horizon": int(hold_period),
                "label_reports": label_reports,
                "selection_calibration": selection_calibration,
                "test_days": 0,
                "universe_size": int(len(feature_block.selected_symbols)),
                "n_train_samples": int(len(train_dataset)),
                "n_valid_samples": int(len(valid_dataset)),
                "n_test_samples": int(len(test_dataset)),
                "review_top_k_count": int(len(review)),
                "best_epoch": int(getattr(trainer, "best_epoch", 0)),
                "best_valid_ic": float(getattr(trainer, "monitor_best_valid_ic", float("nan"))),
                "best_valid_daily_ic": float(getattr(trainer, "monitor_best_valid_daily_ic", float("nan"))),
                "feature_dim": int(input_dim),
                "seq_len": int(seq_len),
                "split_dates": {
                    key: [pd.Timestamp(value).date().isoformat() for value in values]
                    for key, values in feature_block.split_dates.items()
                },
            },
        )
        history_df.to_csv(run_dir / "train_metrics.csv", index=False, encoding="utf-8-sig")
        return state

    with log_path.open("w", encoding="utf-8-sig") as log_handle:
        with redirect_stdout(log_handle), redirect_stderr(log_handle):
            return _train()


def infer_independent_model(
    *,
    state: IndependentModelState,
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    industry_map_df: pd.DataFrame,
    industry_daily_df: pd.DataFrame,
    name_map: dict[str, str],
    out_dir: Path,
    run_id: str,
    strategy: str,
    use_date: pd.Timestamp,
) -> Path:
    use_date = pd.Timestamp(use_date).normalize()
    if use_date == state.train_date:
        return state.run_dir.resolve()
    output_dir = (
        out_dir
        / "inference_runs"
        / strategy
        / f"{run_id}_{strategy}_infer_{state.train_date.strftime('%Y%m%d')}_{use_date.strftime('%Y%m%d')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    review = build_review_outputs(
        state=state,
        stock_df=stock_df,
        index_df=index_df,
        industry_map_df=industry_map_df,
        industry_daily_df=industry_daily_df,
        name_map=name_map,
        use_date=use_date,
        output_dir=output_dir,
    )
    write_json(
        output_dir / "summary.json",
        {
            "pipeline": "independent_rolling_retrain_backtest",
            "source_run_dir": str(state.run_dir.resolve()),
            "train_date": state.train_date.date().isoformat(),
            "signal_date": use_date.date().isoformat(),
            "review_top_k_count": int(len(review)),
            "universe_size": int(len(state.selected_symbols)),
        },
    )
    return output_dir.resolve()


def read_review_picks(path: Path, top_k: int, expected_use_date: pd.Timestamp) -> pd.DataFrame:
    review_path = path / "review_top_k.csv"
    if not review_path.exists():
        raise FileNotFoundError(f"Missing review_top_k.csv: {review_path}")
    frame = pd.read_csv(review_path)
    if frame.empty:
        return frame
    if "candidate_rank" in frame.columns:
        frame = frame.sort_values(["candidate_rank", "symbol"], ascending=[True, True])
    else:
        frame = frame.sort_values(["score", "symbol"], ascending=[False, True])
    frame = frame.head(int(top_k)).copy()
    if "signal_date" in frame.columns:
        actual_dates = set(pd.to_datetime(frame["signal_date"]).dt.normalize())
        if actual_dates and actual_dates != {expected_use_date}:
            raise ValueError(
                f"Signal date mismatch in {review_path}: "
                f"expected={expected_use_date.date()} actual={sorted(date.date().isoformat() for date in actual_dates)}"
            )
    return frame


def select_review_picks(review: pd.DataFrame, top_k: int, expected_use_date: pd.Timestamp) -> pd.DataFrame:
    frame = review.copy()
    if frame.empty:
        return frame
    if "candidate_rank" in frame.columns:
        frame = frame.sort_values(["candidate_rank", "symbol"], ascending=[True, True])
    else:
        frame = frame.sort_values(["score", "symbol"], ascending=[False, True])
    frame = frame.head(int(top_k)).copy()
    if "signal_date" in frame.columns:
        actual_dates = set(pd.to_datetime(frame["signal_date"]).dt.normalize())
        expected = pd.Timestamp(expected_use_date).normalize()
        if actual_dates and actual_dates != {expected}:
            raise ValueError(
                f"Signal date mismatch in review frame: "
                f"expected={expected.date()} actual={sorted(date.date().isoformat() for date in actual_dates)}"
            )
    return frame


def build_price_maps(stock_df: pd.DataFrame) -> tuple[dict[tuple[pd.Timestamp, str], float], dict[pd.Timestamp, pd.DataFrame]]:
    price_map: dict[tuple[pd.Timestamp, str], float] = {}
    by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    for date, day_df in stock_df.groupby("date"):
        day = pd.Timestamp(date).normalize()
        clean = day_df[["symbol", "open"]].copy()
        clean["open"] = pd.to_numeric(clean["open"], errors="coerce")
        clean = clean[clean["open"] > 0].copy()
        by_date[day] = clean
        for row in clean.itertuples(index=False):
            price_map[(day, str(row.symbol))] = float(row.open)
    return price_map, by_date


def shifted_trade_date(trading_dates: list[pd.Timestamp], trade_date: pd.Timestamp, offset: int) -> pd.Timestamp:
    index = trading_dates.index(pd.Timestamp(trade_date).normalize())
    target = index + int(offset)
    if target >= len(trading_dates):
        raise IndexError(f"No trading date offset={offset} after {trade_date.date()}")
    return trading_dates[target]


def market_open_to_open_return(
    by_date: dict[pd.Timestamp, pd.DataFrame],
    buy_date: pd.Timestamp,
    sell_date: pd.Timestamp,
) -> float | None:
    left = by_date.get(buy_date)
    right = by_date.get(sell_date)
    if left is None or right is None or left.empty or right.empty:
        return None
    merged = left.merge(right, on="symbol", how="inner", suffixes=("_buy", "_sell"))
    merged = merged[(merged["open_buy"] > 0) & (merged["open_sell"] > 0)].copy()
    if merged.empty:
        return None
    return float((merged["open_sell"] / merged["open_buy"] - 1.0).mean())


def evaluate_use_day(
    *,
    scheduled: ScheduledUse,
    strategy_name: str,
    inference_dir: Path,
    picks: pd.DataFrame,
    selection_decision: dict[str, Any],
    trading_dates: list[pd.Timestamp],
    price_map: dict[tuple[pd.Timestamp, str], float],
    by_date: dict[pd.Timestamp, pd.DataFrame],
    sell_after_trading_days: int,
    buy_slippage_rate: float,
    sell_slippage_rate: float,
    commission_rate: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    buy_date = shifted_trade_date(trading_dates, scheduled.use_date, 1)
    sell_date = shifted_trade_date(trading_dates, scheduled.use_date, sell_after_trading_days)
    trade_rows: list[dict[str, Any]] = []

    for rank, row in enumerate(picks.itertuples(index=False), start=1):
        row_dict = row._asdict()
        symbol = str(row_dict.get("symbol"))
        buy_open = price_map.get((buy_date, symbol))
        sell_open = price_map.get((sell_date, symbol))
        gross_return = None
        net_return = None
        target_weight = float(row_dict.get("target_weight") or 0.0)
        position_confidence = float(row_dict.get("position_confidence") or 0.0)
        gross_contribution = None
        net_contribution = None
        if buy_open is not None and sell_open is not None and buy_open > 0 and sell_open > 0:
            buy_exec = buy_open * (1.0 + float(buy_slippage_rate))
            sell_exec = sell_open * (1.0 - float(sell_slippage_rate))
            gross_return = sell_open / buy_open - 1.0
            net_return = (sell_exec * (1.0 - float(commission_rate))) / (
                buy_exec * (1.0 + float(commission_rate))
            ) - 1.0
            gross_contribution = float(target_weight * gross_return)
            net_contribution = float(target_weight * net_return)

        trade_rows.append(
            {
                "strategy": strategy_name,
                "base_strategy": scheduled.strategy,
                "group_key": scheduled.group_key,
                "train_date": scheduled.train_date.date().isoformat(),
                "signal_date": scheduled.use_date.date().isoformat(),
                "buy_date": buy_date.date().isoformat(),
                "sell_date": sell_date.date().isoformat(),
                "hold_period": int(sell_after_trading_days),
                "rank": rank,
                "symbol": symbol,
                "name": row_dict.get("name"),
                "industry_name": row_dict.get("industry_name"),
                "score": row_dict.get("score"),
                "position_confidence": position_confidence,
                "target_weight": target_weight,
                "candidate_rank": row_dict.get("candidate_rank"),
                "signal_close": row_dict.get("buy_price", row_dict.get("close")),
                "buy_open": buy_open,
                "sell_open": sell_open,
                "gross_return": gross_return,
                "net_return": net_return,
                "gross_contribution": gross_contribution,
                "net_contribution": net_contribution,
                "inference_dir": str(inference_dir),
                "ret_1": row_dict.get("ret_1"),
                "ret_5": row_dict.get("ret_5"),
                "intraday_ret": row_dict.get("intraday_ret"),
                "ma_gap_5": row_dict.get("ma_gap_5"),
                "turnover_rate_1": row_dict.get("turnover_rate_1"),
                "volume_ratio_5": row_dict.get("volume_ratio_5"),
                "industry_ret_1_mean": row_dict.get("industry_ret_1_mean"),
            }
        )

    valid = [row for row in trade_rows if row["net_return"] is not None]
    market_return = market_open_to_open_return(by_date, buy_date, sell_date)
    portfolio_net_return = float(pd.Series([row["net_contribution"] for row in valid]).sum()) if valid else 0.0
    portfolio_gross_return = float(pd.Series([row["gross_contribution"] for row in valid]).sum()) if valid else 0.0
    avg_pick_net_return = float(pd.Series([row["net_return"] for row in valid]).mean()) if valid else None
    avg_pick_gross_return = float(pd.Series([row["gross_return"] for row in valid]).mean()) if valid else None
    gross_exposure = float(pd.to_numeric(picks.get("target_weight", pd.Series(dtype="float64")), errors="coerce").fillna(0.0).sum())
    executable_exposure = float(pd.Series([row["target_weight"] for row in valid], dtype="float64").sum()) if valid else 0.0
    daily_row = {
        "strategy": strategy_name,
        "base_strategy": scheduled.strategy,
        "group_key": scheduled.group_key,
        "train_date": scheduled.train_date.date().isoformat(),
        "signal_date": scheduled.use_date.date().isoformat(),
        "buy_date": buy_date.date().isoformat(),
        "sell_date": sell_date.date().isoformat(),
        "hold_period": int(sell_after_trading_days),
        "pick_count": int(len(trade_rows)),
        "valid_pick_count": int(len(valid)),
        "gross_exposure": gross_exposure,
        "cash_weight": float(max(0.0, 1.0 - gross_exposure)),
        "executable_exposure": executable_exposure,
        "gross_return": portfolio_gross_return,
        "net_return": portfolio_net_return,
        "avg_pick_gross_return": avg_pick_gross_return,
        "avg_pick_net_return": avg_pick_net_return,
        "market_return": market_return,
        "exposure_adjusted_market_return": None if market_return is None else market_return * gross_exposure,
        "excess_net_return": None if market_return is None else portfolio_net_return - market_return,
        "exposure_adjusted_excess_net_return": None
        if market_return is None
        else portfolio_net_return - market_return * gross_exposure,
        "cash_filter_enabled": bool(selection_decision.get("cash_filter_enabled", False)),
        "cash_filter_pass": bool(selection_decision.get("cash_filter_pass", True)),
        "skip_reason": selection_decision.get("skip_reason", ""),
        "pre_filter_pick_count": selection_decision.get("pre_filter_pick_count"),
        "topk_mean_score": selection_decision.get("topk_mean_score"),
        "min_score": selection_decision.get("min_score"),
        "score_gap": selection_decision.get("score_gap"),
        "threshold_min_score": selection_decision.get("threshold_min_score"),
        "threshold_topk_mean_score": selection_decision.get("threshold_topk_mean_score"),
        "threshold_score_gap": selection_decision.get("threshold_score_gap"),
        "inference_dir": str(inference_dir),
    }
    return daily_row, trade_rows


def summarize_daily_returns(daily_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy, group in daily_df.groupby("strategy"):
        clean = group.dropna(subset=["net_return"]).sort_values("signal_date").copy()
        if clean.empty:
            rows.append({"strategy": strategy, "n_use_days": 0})
            continue
        net = pd.to_numeric(clean["net_return"], errors="coerce").fillna(0.0)
        gross = pd.to_numeric(clean["gross_return"], errors="coerce").fillna(0.0)
        market = pd.to_numeric(clean["market_return"], errors="coerce").fillna(0.0)
        excess = pd.to_numeric(clean["excess_net_return"], errors="coerce").fillna(0.0)
        exposure_adjusted_excess = pd.to_numeric(
            clean.get("exposure_adjusted_excess_net_return", pd.Series(dtype="float64")),
            errors="coerce",
        ).fillna(0.0)
        gross_exposure = pd.to_numeric(clean.get("gross_exposure", pd.Series(dtype="float64")), errors="coerce").fillna(0.0)
        equity = (1.0 + net).cumprod()
        peak = equity.cummax()
        drawdown = equity / peak.replace(0.0, pd.NA) - 1.0
        std = float(net.std(ddof=1)) if len(net) > 1 else 0.0
        excess_std = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
        rows.append(
            {
                "strategy": strategy,
                "n_use_days": int(len(clean)),
                "n_trades": int(clean["valid_pick_count"].sum()),
                "invested_days": int((gross_exposure > 0.0).sum()),
                "cash_days": int((gross_exposure <= 0.0).sum()),
                "mean_gross_exposure": float(gross_exposure.mean()),
                "mean_daily_gross_return": float(gross.mean()),
                "mean_daily_net_return": float(net.mean()),
                "market_mean_return": float(market.mean()),
                "excess_mean_return": float(excess.mean()),
                "exposure_adjusted_excess_mean_return": float(exposure_adjusted_excess.mean()),
                "win_rate": float((net > 0.0).mean()),
                "positive_excess_rate": float((excess > 0.0).mean()),
                "positive_exposure_adjusted_excess_rate": float((exposure_adjusted_excess > 0.0).mean()),
                "cumulative_net_return": float(equity.iloc[-1] - 1.0),
                "market_cumulative_return": float((1.0 + market.fillna(0.0)).cumprod().iloc[-1] - 1.0),
                "max_drawdown": float(drawdown.min()),
                "sharpe_annualized": float(net.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0,
                "information_ratio": float(excess.mean() / excess_std * math.sqrt(252.0)) if excess_std > 0 else 0.0,
                "first_signal_date": str(clean["signal_date"].iloc[0]),
                "last_signal_date": str(clean["signal_date"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def attach_equity_curve(daily_df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, group in daily_df.groupby("strategy"):
        out = group.sort_values("signal_date").copy()
        net = pd.to_numeric(out["net_return"], errors="coerce").fillna(0.0)
        market = pd.to_numeric(out["market_return"], errors="coerce").fillna(0.0)
        out["equity_curve"] = (1.0 + net).cumprod()
        out["market_curve"] = (1.0 + market).cumprod()
        out["relative_curve"] = out["equity_curve"] / out["market_curve"].replace(0.0, pd.NA)
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else daily_df


def summarize_bias(trades_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    numeric_cols = [
        "signal_close",
        "score",
        "position_confidence",
        "target_weight",
        "candidate_rank",
        "ret_1",
        "ret_5",
        "intraday_ret",
        "ma_gap_5",
        "turnover_rate_1",
        "volume_ratio_5",
        "industry_ret_1_mean",
        "gross_return",
        "net_return",
    ]
    frame = trades_df.copy()
    for column in numeric_cols:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    bias_rows: list[dict[str, Any]] = []
    industry_rows: list[dict[str, Any]] = []
    for strategy, group in frame.groupby("strategy"):
        industry_counts = group["industry_name"].fillna("").astype(str).value_counts()
        top_industry = industry_counts.index[0] if not industry_counts.empty else ""
        top_share = float(industry_counts.iloc[0] / max(1, len(group))) if not industry_counts.empty else 0.0
        bias_rows.append(
            {
                "strategy": strategy,
                "pick_count": int(len(group)),
                "avg_signal_close": float(group["signal_close"].mean()),
                "median_signal_close": float(group["signal_close"].median()),
                "avg_score": float(group["score"].mean()),
                "avg_position_confidence": float(group["position_confidence"].mean())
                if "position_confidence" in group.columns
                else float("nan"),
                "avg_target_weight": float(group["target_weight"].mean())
                if "target_weight" in group.columns
                else float("nan"),
                "avg_candidate_rank": float(group["candidate_rank"].mean()),
                "avg_ret_1": float(group["ret_1"].mean()),
                "avg_ret_5": float(group["ret_5"].mean()),
                "avg_intraday_ret": float(group["intraday_ret"].mean()),
                "avg_ma_gap_5": float(group["ma_gap_5"].mean()),
                "avg_turnover_rate_1": float(group["turnover_rate_1"].mean()),
                "avg_volume_ratio_5": float(group["volume_ratio_5"].mean()),
                "avg_industry_ret_1_mean": float(group["industry_ret_1_mean"].mean()),
                "positive_ret_1_share": float((group["ret_1"] > 0.0).mean()),
                "positive_ret_5_share": float((group["ret_5"] > 0.0).mean()),
                "top_industry": top_industry,
                "top_industry_share": top_share,
            }
        )
        industry_group = (
            group.groupby("industry_name", dropna=False)
            .agg(
                pick_count=("symbol", "count"),
                avg_net_return=("net_return", "mean"),
                avg_score=("score", "mean"),
            )
            .reset_index()
            .sort_values(["pick_count", "avg_net_return"], ascending=[False, False])
        )
        industry_group.insert(0, "strategy", strategy)
        industry_rows.extend(industry_group.to_dict(orient="records"))

    return pd.DataFrame(bias_rows), pd.DataFrame(industry_rows)


def progress_key(strategy: str, signal_date: Any) -> str:
    return f"{strategy}|{pd.Timestamp(signal_date).date().isoformat()}"


def trim_model_cache(cache: dict[str, IndependentModelState], max_items: int = 6) -> None:
    while len(cache) > int(max_items):
        oldest_key = next(iter(cache))
        state = cache.pop(oldest_key)
        trainer = getattr(state, "trainer", None)
        model = getattr(trainer, "model", None)
        if model is not None:
            model.to(torch.device("cpu"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def trim_training_feature_cache(cache: dict[str, TrainingFeatureBlock], max_items: int = 2) -> None:
    while len(cache) > int(max_items):
        oldest_key = next(iter(cache))
        cache.pop(oldest_key, None)


def load_resume_progress(out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    daily_path = out_dir / "daily_returns.csv"
    trades_path = out_dir / "trades.csv"
    schedule_path = out_dir / "model_schedule.csv"
    if not daily_path.exists():
        return [], [], [], set()

    daily_df = read_csv_or_empty(daily_path)
    trades_df = read_csv_or_empty(trades_path)
    schedule_df = read_csv_or_empty(schedule_path)
    if daily_df.empty:
        return [], [], [], set()

    completed_keys = {
        progress_key(row.strategy, row.signal_date)
        for row in daily_df[["strategy", "signal_date"]].drop_duplicates().itertuples(index=False)
    }
    if not trades_df.empty:
        trades_df = trades_df[
            trades_df.apply(lambda row: progress_key(row["strategy"], row["signal_date"]) in completed_keys, axis=1)
        ].copy()
    if not schedule_df.empty:
        schedule_df = schedule_df[
            schedule_df.apply(lambda row: progress_key(row["strategy"], row["use_date"]) in completed_keys, axis=1)
        ].copy()

    return (
        daily_df.to_dict(orient="records"),
        trades_df.to_dict(orient="records"),
        schedule_df.to_dict(orient="records") if not schedule_df.empty else [],
        completed_keys,
    )


def resolve_checkpoint_path(run_dir: Path, checkpoint_name: str) -> Path:
    pointer = run_dir / "best_checkpoint.txt"
    if pointer.exists():
        value = pointer.read_text(encoding="utf-8").strip()
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = run_dir / path
            if path.exists():
                return path
    for candidate in [run_dir / "selected_best.ckpt", run_dir / checkpoint_name, run_dir / "best.ckpt"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing checkpoint for existing run: {run_dir}")


def load_existing_model_state(
    *,
    base_config: dict[str, Any],
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    industry_map_df: pd.DataFrame,
    industry_daily_df: pd.DataFrame,
    st_symbols: set[str],
    out_dir: Path,
    train_date: pd.Timestamp,
    checkpoint_name: str,
    label_mode: str,
    hold_period: int,
    top_k: int,
    score_quantile: float,
    topk_mean_quantile: float,
    score_gap_quantile: float,
) -> IndependentModelState:
    train_date = pd.Timestamp(train_date).normalize()
    run_label = policy_run_label(label_mode, hold_period)
    run_dir = out_dir / "runs" / run_label / f"train_{train_date.strftime('%Y%m%d')}"
    config = load_yaml(run_dir / "config.yaml") if (run_dir / "config.yaml").exists() else independent_config(base_config)
    config.setdefault("data", {})
    config["data"]["label_horizon"] = int(hold_period)

    stock_slice = slice_by_date(stock_df, train_date)
    keep_days = int(config.get("data", {}).get("trainable_history_days", 260))
    unique_dates = sorted(stock_slice["date"].drop_duplicates())
    if len(unique_dates) > keep_days:
        stock_slice = stock_slice[stock_slice["date"].isin(set(unique_dates[-keep_days:]))].copy()

    universe_path = run_dir / "universe_report.csv"
    if universe_path.exists():
        universe_report = pd.read_csv(universe_path)
        selected_symbols = universe_report["symbol"].astype(str).tolist()
    else:
        selected_symbols, _universe_report = select_symbols_as_of(stock_slice, config, st_symbols, train_date)
    if not selected_symbols:
        raise ValueError(f"No selected symbols available for existing run: {run_dir}")

    context_stock = stock_slice[stock_slice["symbol"].astype(str).isin(selected_symbols)].copy()
    builder = build_dataset_builder(config, verbose=False)
    rolling = config.get("rolling", {})
    bundle = builder.build_bundle(
        raw_df=context_stock,
        train_days=int(rolling.get("train_days", 80)),
        valid_days=int(rolling.get("valid_days", 30)),
        test_days=0,
        index_df=slice_by_date(index_df, train_date) if config.get("index", {}).get("enabled", True) else pd.DataFrame(),
        industry_map_df=industry_map_df if config.get("industry", {}).get("enabled", True) else pd.DataFrame(),
        industry_daily_df=slice_by_date(industry_daily_df, train_date)
        if config.get("industry", {}).get("enabled", True)
        else pd.DataFrame(),
        sample_symbols=selected_symbols,
    )
    apply_label_mode_to_dataset(bundle.train_dataset, label_mode)
    apply_label_mode_to_dataset(bundle.valid_dataset, label_mode)
    input_dim = len(bundle.feature_columns)
    seq_len = int(bundle.train_dataset.features.shape[1])
    trainer, _resolved_model_config = build_alpha_trainer(
        input_dim=input_dim,
        seq_len=seq_len,
        feature_columns=bundle.feature_columns,
        model_cfg=config.get("model", {}),
        trainer_config=build_trainer_config(config),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    trainer.load_checkpoint(resolve_checkpoint_path(run_dir, checkpoint_name))
    calibration_path = run_dir / "selection_calibration.json"
    if calibration_path.exists():
        selection_calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    else:
        valid_scores = trainer.predict_dataset(bundle.valid_dataset)
        selection_calibration = build_selection_calibration(
            valid_meta=bundle.valid_dataset.meta,
            valid_scores=valid_scores,
            top_k=int(top_k),
            score_quantile=float(score_quantile),
            topk_mean_quantile=float(topk_mean_quantile),
            score_gap_quantile=float(score_gap_quantile),
            label_mode=str(label_mode),
            hold_period=int(hold_period),
        )
    return IndependentModelState(
        strategy="shared",
        train_date=train_date,
        label_mode=str(label_mode),
        label_horizon=int(hold_period),
        run_dir=run_dir,
        trainer=trainer,
        dataset_builder=builder,
        scaler=bundle.scaler,
        peer_map=bundle.peer_map,
        selected_symbols=selected_symbols,
        feature_columns=bundle.feature_columns,
        config=config,
        selection_calibration=selection_calibration,
    )


def run_backtest(args: argparse.Namespace) -> Path:
    base_config_path = repo_path(args.config)
    base_config = apply_independent_runtime_overrides(load_yaml(base_config_path), args)
    stock_df = load_stock_frame(base_config)
    index_df, industry_map_df, industry_daily_df, name_map, st_symbols = load_support_frames(base_config)
    trading_dates = trading_dates_from_stock(stock_df)
    hold_periods = parse_hold_periods(args.hold_periods, int(args.sell_after_trading_days))
    max_hold_period = max(hold_periods)
    if args.required_history_days is None:
        history_config = copy.deepcopy(base_config)
        history_config.setdefault("data", {})
        history_config["data"]["label_horizon"] = int(max_hold_period)
        required_history_days = default_required_history_days(history_config)
    else:
        required_history_days = int(args.required_history_days)
    eval_dates = pick_eval_dates(
        trading_dates=trading_dates,
        start_date=normalize_date(args.start_date),
        end_date=normalize_date(args.end_date),
        required_history_days=required_history_days,
        sell_after_trading_days=int(max_hold_period),
        min_use_days=int(args.min_use_days),
        auto_start_after_warmup=bool(args.auto_start_after_warmup),
    )

    run_id = args.run_id or pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = repo_path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "retrain_backtest" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    top_k = int(args.top_k or base_config.get("strategy", {}).get("top_k", 3))
    strategies = [item.strip() for item in str(args.strategies).split(",") if item.strip()]
    label_mode = str(args.label_mode)
    cash_filter_enabled = str(args.cash_filter).strip().lower() == "enabled"
    price_map, by_date = build_price_maps(stock_df)

    schedules: list[ScheduledUse] = []
    for strategy in strategies:
        schedules.extend(build_schedule(strategy, eval_dates))

    schedule_df = pd.DataFrame(
        [
            {
                "strategy": policy_name(item.strategy, hold_period),
                "base_strategy": item.strategy,
                "hold_period": int(hold_period),
                "label_mode": label_mode,
                "group_key": item.group_key,
                "train_date": item.train_date.date().isoformat(),
                "use_date": item.use_date.date().isoformat(),
            }
            for item in schedules
            for hold_period in hold_periods
        ]
    )
    write_csv(out_dir / "model_schedule_plan.csv", schedule_df)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "config": str(base_config_path.resolve()),
        "output_dir": str(out_dir.resolve()),
        "strategies": strategies,
        "hold_periods": hold_periods,
        "label_mode": label_mode,
        "top_k": top_k,
        "start_date": eval_dates[0].date().isoformat(),
        "end_date": eval_dates[-1].date().isoformat(),
        "use_days": len(eval_dates),
        "required_history_days": required_history_days,
        "sell_after_trading_days": int(args.sell_after_trading_days),
        "max_hold_period": int(max_hold_period),
        "cash_filter": str(args.cash_filter),
        "disable_price_cap": bool(args.disable_price_cap),
        "effective_max_latest_price": base_config.get("universe", {}).get("filters", {}).get("max_latest_price"),
        "score_quantile": float(args.score_quantile),
        "topk_mean_quantile": float(args.topk_mean_quantile),
        "score_gap_quantile": float(args.score_gap_quantile),
        "min_position_exposure": float(args.min_position_exposure),
        "max_position_exposure": float(args.max_position_exposure),
        "max_gross_exposure": float(args.max_gross_exposure),
        "buy_slippage_rate": float(args.buy_slippage_rate),
        "sell_slippage_rate": float(args.sell_slippage_rate),
        "commission_rate": float(args.commission_rate),
        "plan_only": bool(args.plan_only),
        "pipeline": "independent_rolling_retrain_backtest",
        "shared_pipeline_invocation": False,
        "test_days": 0,
        "cleanup_target": str(out_dir.resolve()),
    }
    write_json(out_dir / "manifest.json", manifest)
    (out_dir / "cleanup_targets.txt").write_text(
        str(out_dir.resolve()) + "\n",
        encoding="utf-8",
    )

    if args.plan_only:
        print(f"Plan written to: {out_dir.resolve()}")
        return out_dir

    write_yaml(out_dir / "config_independent.yaml", independent_config(base_config))
    train_cache: dict[str, IndependentModelState] = {}
    training_feature_cache: dict[str, TrainingFeatureBlock] = {}
    if args.resume:
        daily_rows, trade_rows, schedule_rows, completed_keys = load_resume_progress(out_dir)
        print(f"Resume loaded: completed_days={len(completed_keys)} daily_rows={len(daily_rows)} trades={len(trade_rows)}")
    else:
        schedule_rows = []
        daily_rows = []
        trade_rows = []
        completed_keys = set()

    pending_groups: list[tuple[tuple[str, str, int], list[ScheduledUse]]] = []
    pending_by_group: dict[tuple[str, str, int], list[ScheduledUse]] = {}
    for scheduled in schedules:
        for hold_period in hold_periods:
            strategy_variant = policy_name(scheduled.strategy, hold_period)
            scheduled_key = progress_key(strategy_variant, scheduled.use_date)
            if scheduled_key in completed_keys:
                continue
            group_key = (scheduled.strategy, scheduled.train_date.date().isoformat(), int(hold_period))
            if group_key not in pending_by_group:
                pending_by_group[group_key] = []
                pending_groups.append((group_key, pending_by_group[group_key]))
            pending_by_group[group_key].append(scheduled)

    for block_index, (group_key, block_schedules) in enumerate(pending_groups, start=1):
        base_strategy, _train_date_text, hold_period = group_key
        strategy_variant = policy_name(base_strategy, hold_period)
        first_scheduled = block_schedules[0]
        last_scheduled = block_schedules[-1]
        print(
            f"[block {block_index}/{len(pending_groups)}] strategy={strategy_variant} "
            f"train={first_scheduled.train_date.date()} hold={hold_period} uses={len(block_schedules)} "
            f"first={first_scheduled.use_date.date()} last={last_scheduled.use_date.date()}"
        )
        cache_key = (
            f"{first_scheduled.train_date.date().isoformat()}|"
            f"{policy_run_label(label_mode, hold_period)}"
        )
        state = train_cache.get(cache_key)
        if state is None:
            existing_run_dir = (
                out_dir
                / "runs"
                / policy_run_label(label_mode, hold_period)
                / f"train_{first_scheduled.train_date.strftime('%Y%m%d')}"
            )
            if args.resume and (existing_run_dir / "best_checkpoint.txt").exists():
                state = load_existing_model_state(
                    base_config=base_config,
                    stock_df=stock_df,
                    index_df=index_df,
                    industry_map_df=industry_map_df,
                    industry_daily_df=industry_daily_df,
                    st_symbols=st_symbols,
                    out_dir=out_dir,
                    train_date=first_scheduled.train_date,
                    checkpoint_name=str(args.checkpoint_name),
                    label_mode=label_mode,
                    hold_period=int(hold_period),
                    top_k=top_k,
                    score_quantile=float(args.score_quantile),
                    topk_mean_quantile=float(args.topk_mean_quantile),
                    score_gap_quantile=float(args.score_gap_quantile),
                )
            else:
                feature_key = first_scheduled.train_date.date().isoformat()
                feature_block = training_feature_cache.get(feature_key)
                if feature_block is None:
                    feature_block = build_training_feature_block(
                        base_config=base_config,
                        stock_df=stock_df,
                        index_df=index_df,
                        industry_map_df=industry_map_df,
                        industry_daily_df=industry_daily_df,
                        st_symbols=st_symbols,
                        train_date=first_scheduled.train_date,
                    )
                    training_feature_cache[feature_key] = feature_block
                    trim_training_feature_cache(training_feature_cache)
                state = train_independent_model(
                    base_config=base_config,
                    stock_df=stock_df,
                    index_df=index_df,
                    industry_map_df=industry_map_df,
                    industry_daily_df=industry_daily_df,
                    name_map=name_map,
                    st_symbols=st_symbols,
                    out_dir=out_dir,
                    train_date=first_scheduled.train_date,
                    feature_block=feature_block,
                    checkpoint_name=str(args.checkpoint_name),
                    label_mode=label_mode,
                    hold_period=int(hold_period),
                    top_k=top_k,
                    score_quantile=float(args.score_quantile),
                    topk_mean_quantile=float(args.topk_mean_quantile),
                    score_gap_quantile=float(args.score_gap_quantile),
                )
            train_cache[cache_key] = state
            trim_model_cache(train_cache)

        block_results = infer_independent_model_block(
            state=state,
            stock_df=stock_df,
            index_df=index_df,
            industry_map_df=industry_map_df,
            industry_daily_df=industry_daily_df,
            name_map=name_map,
            out_dir=out_dir,
            run_id=run_id,
            strategy=strategy_variant,
            use_dates=[scheduled.use_date for scheduled in block_schedules],
        )
        for scheduled in block_schedules:
            scheduled_key = progress_key(strategy_variant, scheduled.use_date)
            if scheduled_key in completed_keys:
                continue
            inference_dir, review = block_results[pd.Timestamp(scheduled.use_date).normalize()]
            picks, selection_decision = select_policy_picks(
                review=review,
                top_k=top_k,
                expected_use_date=scheduled.use_date,
                cash_filter_enabled=cash_filter_enabled,
                calibration=state.selection_calibration,
                min_position_exposure=float(args.min_position_exposure),
                max_position_exposure=float(args.max_position_exposure),
                max_gross_exposure=float(args.max_gross_exposure),
            )
            daily_row, day_trades = evaluate_use_day(
                scheduled=scheduled,
                strategy_name=strategy_variant,
                inference_dir=inference_dir,
                picks=picks,
                selection_decision=selection_decision,
                trading_dates=trading_dates,
                price_map=price_map,
                by_date=by_date,
                sell_after_trading_days=int(hold_period),
                buy_slippage_rate=float(args.buy_slippage_rate),
                sell_slippage_rate=float(args.sell_slippage_rate),
                commission_rate=float(args.commission_rate),
            )
            daily_rows.append(daily_row)
            trade_rows.extend(day_trades)
            schedule_rows.append(
                {
                    "strategy": strategy_variant,
                    "base_strategy": scheduled.strategy,
                    "hold_period": int(hold_period),
                    "label_mode": label_mode,
                    "group_key": scheduled.group_key,
                    "train_date": scheduled.train_date.date().isoformat(),
                    "use_date": scheduled.use_date.date().isoformat(),
                    "source_run": str(state.run_dir),
                    "inference_dir": str(inference_dir),
                    "cash_filter_pass": bool(selection_decision.get("cash_filter_pass", True)),
                    "pick_count": int(len(picks)),
                }
            )
            completed_keys.add(scheduled_key)

        write_csv(out_dir / "model_schedule.csv", pd.DataFrame(schedule_rows))
        write_csv(out_dir / "daily_returns.csv", attach_equity_curve(pd.DataFrame(daily_rows)))
        write_csv(out_dir / "trades.csv", pd.DataFrame(trade_rows))

    daily_df = attach_equity_curve(pd.DataFrame(daily_rows))
    trades_df = pd.DataFrame(trade_rows)
    summary_df = summarize_daily_returns(daily_df)
    bias_df, industry_df = summarize_bias(trades_df)

    write_csv(out_dir / "daily_returns.csv", daily_df)
    write_csv(out_dir / "trades.csv", trades_df)
    write_csv(out_dir / "strategy_summary.csv", summary_df)
    write_csv(out_dir / "bias_summary.csv", bias_df)
    write_csv(out_dir / "industry_exposure.csv", industry_df)
    write_csv(out_dir / "model_schedule.csv", pd.DataFrame(schedule_rows))

    manifest["finished_at"] = pd.Timestamp.now().isoformat()
    manifest["train_runs"] = {key: str(value.run_dir) for key, value in train_cache.items()}
    write_json(out_dir / "manifest.json", manifest)
    print(f"Backtest written to: {out_dir.resolve()}")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_backtest(args)


if __name__ == "__main__":
    main()
