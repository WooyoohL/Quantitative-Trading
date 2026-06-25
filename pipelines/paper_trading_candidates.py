from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config_values import optional_float
from app.paper_trading_config import paper_cfg
from app.runtime import load_symbol_name_map
from strategy.market_heat import build_market_heat_candidates


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(out):
        return None
    return out


def first_notna(series: pd.Series) -> Any:
    values = series.dropna()
    if values.empty:
        return pd.NA
    return values.iloc[0]


def first_from_columns(group: pd.DataFrame, columns: list[str]) -> Any:
    for column in columns:
        if column in group.columns:
            value = first_notna(group[column])
            if pd.notna(value):
                return value
    return pd.NA


def ordered_unique_text(group: pd.DataFrame, value_column: str, order_column: str | None = None) -> list[str]:
    if value_column not in group.columns:
        return []
    frame = group[[value_column]].copy()
    if order_column and order_column in group.columns:
        frame[order_column] = pd.to_numeric(group[order_column], errors="coerce").fillna(9999).astype(int)
        frame = frame.sort_values(order_column)
    values: list[str] = []
    for value in frame[value_column].dropna().astype(str):
        if value and value not in values:
            values.append(value)
    return values


def current_candidate_path(project_root: Path, base_config: dict[str, Any]) -> Path:
    return project_root / str(base_config.get("data", {}).get("candidate_path", "data/current_candidates.csv"))


def base_candidate_path(project_root: Path, base_config: dict[str, Any]) -> Path:
    data_cfg = base_config.get("data", {})
    return project_root / str(data_cfg.get("base_candidate_path", data_cfg.get("candidate_path", "data/current_candidates.csv")))


def read_candidate_symbols(project_root: Path, base_config: dict[str, Any]) -> set[str]:
    candidate_path = current_candidate_path(project_root, base_config)
    if not candidate_path.exists():
        return set()
    df = pd.read_csv(candidate_path)
    if "symbol" not in df.columns:
        return set()
    return set(df["symbol"].dropna().astype(str))


def latest_symbol_rows(stock_df: pd.DataFrame, trade_date: pd.Timestamp) -> pd.DataFrame:
    if stock_df.empty:
        return pd.DataFrame(columns=["symbol", "last_price", "latest_date"])
    frame = stock_df.copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame[frame["date"] <= pd.Timestamp(trade_date).normalize()].copy()
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "last_price", "latest_date"])
    latest = frame.sort_values(["symbol", "date"]).groupby("symbol", as_index=False).tail(1).copy()
    return latest[["symbol", "close", "date"]].rename(columns={"close": "last_price", "date": "latest_date"})


def normalize_base_candidate_frame(
    base_candidates: pd.DataFrame,
    stock_df: pd.DataFrame,
    symbol_name_map: pd.Series,
    trade_date: pd.Timestamp,
) -> pd.DataFrame:
    columns = ["symbol", "name", "last_price", "latest_date", "candidate_source"]
    if base_candidates.empty or "symbol" not in base_candidates.columns:
        return pd.DataFrame(columns=columns)

    out = base_candidates.copy()
    out["symbol"] = out["symbol"].astype(str)
    if "name" not in out.columns:
        out["name"] = out["symbol"].map(symbol_name_map).fillna("")
    else:
        out["name"] = out["name"].fillna(out["symbol"].map(symbol_name_map)).fillna("")
    latest = latest_symbol_rows(stock_df, trade_date)
    if "last_price" not in out.columns:
        out["last_price"] = pd.NA
    if "latest_date" not in out.columns:
        out["latest_date"] = pd.NaT
    out = out.merge(latest, on="symbol", how="left", suffixes=("", "_latest"))
    out["last_price"] = pd.to_numeric(out["last_price"], errors="coerce").where(
        pd.to_numeric(out["last_price"], errors="coerce").notna(),
        pd.to_numeric(out["last_price_latest"], errors="coerce"),
    )
    out["latest_date"] = pd.to_datetime(out["latest_date"], errors="coerce").where(
        pd.to_datetime(out["latest_date"], errors="coerce").notna(),
        pd.to_datetime(out["latest_date_latest"], errors="coerce"),
    )
    out["candidate_source"] = "base"
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out[columns].drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)


def normalize_heat_candidate_frame(heat_candidates: pd.DataFrame, symbol_name_map: pd.Series) -> pd.DataFrame:
    columns = [
        "symbol",
        "name",
        "last_price",
        "latest_date",
        "candidate_source",
        "heat_rank",
        "heat_score",
        "heat_reasons",
    ]
    if heat_candidates.empty or "symbol" not in heat_candidates.columns:
        return pd.DataFrame(columns=columns)
    out = heat_candidates.copy()
    out["symbol"] = out["symbol"].astype(str)
    out["name"] = out["symbol"].map(symbol_name_map).fillna("")
    out["last_price"] = pd.to_numeric(out.get("close"), errors="coerce")
    out["latest_date"] = pd.to_datetime(out.get("signal_date"), errors="coerce")
    out["candidate_source"] = "heat"
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out[columns].drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)


def apply_candidate_universe_filters(
    candidates: pd.DataFrame,
    base_config: dict[str, Any],
    trade_date: pd.Timestamp,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    out = candidates.copy()
    filters = base_config.get("universe", {}).get("filters", {})
    excluded = {str(symbol) for symbol in base_config.get("universe", {}).get("exclude_symbols", [])}
    if excluded:
        out = out[~out["symbol"].astype(str).isin(excluded)].copy()
    if bool(filters.get("exclude_st", True)) and "name" in out.columns:
        out = out[~out["name"].astype(str).str.upper().str.contains("ST", na=False)].copy()

    out["last_price"] = pd.to_numeric(out["last_price"], errors="coerce")
    latest_date = pd.to_datetime(out["latest_date"], errors="coerce").dt.normalize()
    out = out[latest_date == pd.Timestamp(trade_date).normalize()].copy()

    min_price = filters.get("min_latest_price")
    max_price = optional_float(filters.get("max_latest_price"))
    if min_price is not None:
        out = out[out["last_price"] >= float(min_price)].copy()
    if max_price is not None:
        out = out[out["last_price"] <= max_price].copy()
    return out.sort_values(["symbol"]).reset_index(drop=True)


def combine_candidate_universe(base_candidates: pd.DataFrame, heat_candidates: pd.DataFrame) -> pd.DataFrame:
    if base_candidates.empty and heat_candidates.empty:
        return pd.DataFrame()
    combined = pd.concat([base_candidates, heat_candidates], ignore_index=True, sort=False)
    combined["symbol"] = combined["symbol"].astype(str)
    rows: list[dict[str, Any]] = []
    for symbol, group in combined.groupby("symbol", sort=False):
        sources = ordered_unique_text(group, "candidate_source")
        row = {
            "symbol": symbol,
            "name": first_notna(group.get("name", pd.Series(dtype=object))),
            "last_price": first_notna(group.get("last_price", pd.Series(dtype=object))),
            "latest_date": first_notna(group.get("latest_date", pd.Series(dtype=object))),
            "candidate_source": "+".join(sources),
            "heat_rank": pd.to_numeric(group.get("heat_rank", pd.Series(dtype=float)), errors="coerce").min(),
            "heat_score": pd.to_numeric(group.get("heat_score", pd.Series(dtype=float)), errors="coerce").max(),
            "heat_reasons": ",".join(sorted(set(group.get("heat_reasons", pd.Series(dtype=str)).dropna().astype(str)))),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def write_candidate_universe_files(
    *,
    project_root: Path,
    base_config: dict[str, Any],
    candidates: pd.DataFrame,
    trade_date: pd.Timestamp,
) -> tuple[Path, Path]:
    candidate_path = current_candidate_path(project_root, base_config)
    daily_dir = project_root / "outputs" / "paper_trading" / trade_date.strftime("%Y%m%d")
    daily_dir.mkdir(parents=True, exist_ok=True)
    daily_path = daily_dir / "candidate_universe.csv"
    latest_path = project_root / "outputs" / "paper_trading" / "latest_candidate_universe.csv"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    out = candidates.copy()
    if "latest_date" in out.columns:
        out["latest_date"] = pd.to_datetime(out["latest_date"], errors="coerce").dt.date.astype(str)
    out.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    out.to_csv(daily_path, index=False, encoding="utf-8-sig")
    out.to_csv(latest_path, index=False, encoding="utf-8-sig")
    return candidate_path, daily_path


def build_unified_candidate_universe(
    *,
    config: dict[str, Any],
    base_config: dict[str, Any],
    stock_df: pd.DataFrame,
    industry_map_df: pd.DataFrame,
    industry_daily_df: pd.DataFrame,
    trade_date: pd.Timestamp,
    project_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    symbol_name_map = load_symbol_name_map(base_config)
    source_base_path = base_candidate_path(project_root, base_config)
    if source_base_path.exists():
        base_raw = pd.read_csv(source_base_path)
    else:
        base_raw = pd.DataFrame(columns=["symbol"])
    base_candidates = normalize_base_candidate_frame(base_raw, stock_df, symbol_name_map, trade_date)

    candidates_cfg = paper_cfg(config).get("candidates", {})
    heat_add_limit = int(candidates_cfg.get("heat_add_limit", candidates_cfg.get("heat_limit", 30)))
    heat_raw = build_market_heat_candidates(
        stock_df=stock_df,
        industry_map_df=industry_map_df,
        industry_daily_df=industry_daily_df,
        signal_date=trade_date,
        config=paper_cfg(config).get("heat", {}),
        candidate_symbols=None,
        limit=heat_add_limit,
    )
    heat_candidates = normalize_heat_candidate_frame(heat_raw, symbol_name_map)
    unified_before_limits = combine_candidate_universe(base_candidates, heat_candidates)
    unified = apply_candidate_universe_filters(unified_before_limits, base_config, trade_date)
    current_path, daily_path = write_candidate_universe_files(
        project_root=project_root,
        base_config=base_config,
        candidates=unified,
        trade_date=trade_date,
    )
    heat_after_limits = (
        int(unified["candidate_source"].astype(str).str.contains("heat", na=False).sum())
        if "candidate_source" in unified.columns
        else 0
    )
    return unified, {
        "base_candidate_count": int(len(base_candidates)),
        "heat_candidate_count": int(len(heat_candidates)),
        "candidate_universe_before_limits_count": int(len(unified_before_limits)),
        "candidate_universe_count": int(len(unified)),
        "candidate_universe_removed_by_limits_count": int(len(unified_before_limits) - len(unified)),
        "heat_candidate_after_limits_count": heat_after_limits,
        "base_candidate_path": str(source_base_path.relative_to(project_root)),
        "candidate_universe_path": str(current_path.relative_to(project_root)),
        "daily_candidate_universe_path": str(daily_path.relative_to(project_root)),
    }


def standardize_model_candidates(
    review_path: Path,
    limit: int,
    *,
    model_source: str = "model",
    source_run: str = "",
    source_order: int = 0,
    model_quality: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if not review_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(review_path).head(int(limit)).copy()
    if df.empty:
        return pd.DataFrame()
    if "candidate_rank" not in df.columns:
        df["candidate_rank"] = range(1, len(df) + 1)
    if "buy_price" in df.columns and "close" not in df.columns:
        df["close"] = df["buy_price"]
    df["model_rank"] = range(1, len(df) + 1)
    df["model_group_rank"] = df["model_rank"]
    df["model_source"] = model_source
    df["model_source_order"] = int(source_order)
    df["source_run"] = source_run
    quality = model_quality or {}
    quality_label = str(quality.get("label") or "未知")
    quality_reasons = quality.get("reasons") or []
    if isinstance(quality_reasons, list):
        quality_reason_text = "; ".join(str(item) for item in quality_reasons if str(item))
    else:
        quality_reason_text = str(quality_reasons)
    if quality_label == "不建议":
        quality_risk = "高"
    elif quality_label == "观察":
        quality_risk = "中"
    elif quality_label == "推荐":
        quality_risk = "低"
    else:
        quality_risk = "未知"
    df["model_layer_conclusion"] = f"{model_source}:{quality_label}"
    df["model_layer_risk"] = quality_risk
    df["model_layer_reason"] = quality_reason_text
    df["source_valid_daily_ic"] = quality.get("valid_daily_ic", pd.NA)
    df["source_test_daily_ic"] = quality.get("test_daily_ic", pd.NA)
    df["from_model"] = True
    df["from_heat"] = False
    df["source"] = model_source
    return df


def read_model_quality(inference_dir: Path) -> dict[str, Any]:
    summary_path = inference_dir / "summary.json"
    if not summary_path.exists():
        return {"label": "未知", "reasons": ["缺少推理摘要"]}
    summary = read_json(summary_path)
    return {
        "label": summary.get("source_recommendation_label") or "未知",
        "reasons": summary.get("source_recommendation_reasons") or [],
        "valid_daily_ic": summary.get("source_best_valid_daily_ic"),
        "test_daily_ic": summary.get("source_test_daily_ic"),
    }


def standardize_heat_candidates(
    heat_df: pd.DataFrame,
    symbol_name_map: pd.Series,
    next_trade_date: pd.Timestamp,
) -> pd.DataFrame:
    if heat_df.empty:
        return pd.DataFrame()
    out = heat_df.copy()
    out["name"] = out["symbol"].astype(str).map(symbol_name_map).fillna("")
    out["next_trade_date"] = next_trade_date.date().isoformat()
    out["candidate_rank"] = pd.NA
    out["score"] = pd.NA
    out["model_rank"] = pd.NA
    out["from_model"] = False
    out["from_heat"] = True
    out["source"] = "heat"
    out["buy_price"] = out["close"]
    return out


def combine_candidates(model_df: pd.DataFrame, heat_df: pd.DataFrame, final_limit: int) -> pd.DataFrame:
    if model_df.empty and heat_df.empty:
        return pd.DataFrame()
    combined = pd.concat([model_df, heat_df], ignore_index=True, sort=False)
    combined["symbol"] = combined["symbol"].astype(str)
    rows: list[dict[str, Any]] = []
    for symbol, group in combined.groupby("symbol", sort=False):
        from_model = bool(group["from_model"].fillna(False).any())
        from_heat = bool(group["from_heat"].fillna(False).any())
        model_sources = ordered_unique_text(group[group["from_model"].fillna(False)], "model_source", "model_source_order")
        source_parts = model_sources if from_model and model_sources else (["model"] if from_model else [])
        if from_heat:
            source_parts.append("heat")
        source = "+".join(source_parts) if source_parts else "unknown"
        heat_reasons = ",".join(sorted(set(group.get("heat_reasons", pd.Series(dtype=str)).dropna().astype(str))))
        source_runs = ",".join(ordered_unique_text(group[group["from_model"].fillna(False)], "source_run", "model_source_order"))
        model_layer_conclusions = ordered_unique_text(
            group[group["from_model"].fillna(False)], "model_layer_conclusion", "model_source_order"
        )
        model_layer_risks = ordered_unique_text(
            group[group["from_model"].fillna(False)], "model_layer_risk", "model_source_order"
        )
        model_layer_reasons = ordered_unique_text(
            group[group["from_model"].fillna(False)], "model_layer_reason", "model_source_order"
        )
        row = {
            "signal_date": first_notna(group.get("signal_date", pd.Series(dtype=object))),
            "next_trade_date": first_notna(group.get("next_trade_date", pd.Series(dtype=object))),
            "symbol": symbol,
            "name": first_notna(group.get("name", pd.Series(dtype=object))),
            "industry_name": first_notna(group.get("industry_name", pd.Series(dtype=object))),
            "score": first_notna(group.get("score", pd.Series(dtype=object))),
            "candidate_rank": pd.to_numeric(group.get("candidate_rank", pd.Series(dtype=float)), errors="coerce").min(),
            "model_rank": pd.to_numeric(group.get("model_rank", pd.Series(dtype=float)), errors="coerce").min(),
            "model_group_rank": pd.to_numeric(group.get("model_group_rank", pd.Series(dtype=float)), errors="coerce").min(),
            "model_source": "+".join(model_sources),
            "model_source_order": pd.to_numeric(group.get("model_source_order", pd.Series(dtype=float)), errors="coerce").min(),
            "model_layer_conclusion": "; ".join(model_layer_conclusions),
            "model_layer_risk": "+".join(model_layer_risks),
            "model_layer_reason": "; ".join(model_layer_reasons),
            "source_valid_daily_ic": first_from_columns(group, ["source_valid_daily_ic"]),
            "source_test_daily_ic": first_from_columns(group, ["source_test_daily_ic"]),
            "source_run": source_runs,
            "heat_rank": pd.to_numeric(group.get("heat_rank", pd.Series(dtype=float)), errors="coerce").min(),
            "heat_score": pd.to_numeric(group.get("heat_score", pd.Series(dtype=float)), errors="coerce").max(),
            "close": first_notna(group.get("close", pd.Series(dtype=object))),
            "buy_price": first_notna(group.get("buy_price", pd.Series(dtype=object))),
            "ret_1": first_from_columns(group, ["ret_1", "ret_1_effective"]),
            "ret_5": first_from_columns(group, ["ret_5"]),
            "intraday_ret": first_from_columns(group, ["intraday_ret"]),
            "ma_gap_5": first_from_columns(group, ["ma_gap_5"]),
            "turnover_rate_1": first_from_columns(group, ["turnover_rate_1", "turnover_rate"]),
            "volume_ratio_1_prev": first_from_columns(group, ["volume_ratio_1_prev"]),
            "volume_ratio_3_prev": first_from_columns(group, ["volume_ratio_3_prev"]),
            "volume_ratio_5_prev": first_from_columns(group, ["volume_ratio_5_prev"]),
            "volume_ratio_7_prev": first_from_columns(group, ["volume_ratio_7_prev"]),
            "volume_ratio_5": first_from_columns(group, ["volume_ratio_5"]),
            "industry_ret_1_mean": first_from_columns(group, ["industry_ret_1_mean", "industry_ret_1"]),
            "source": source,
            "heat_reasons": heat_reasons,
            "_source_priority": 0 if from_model and from_heat else (1 if from_model else 2),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out["source_priority"] = pd.to_numeric(out["_source_priority"], errors="coerce").fillna(9).astype(int)
    out["model_source_order_sort"] = pd.to_numeric(out["model_source_order"], errors="coerce").fillna(9999.0)
    out["model_rank_sort"] = pd.to_numeric(out["model_group_rank"], errors="coerce").fillna(9999.0)
    out["heat_rank_sort"] = pd.to_numeric(out["heat_rank"], errors="coerce").fillna(9999.0)
    out["heat_score_sort"] = pd.to_numeric(out["heat_score"], errors="coerce").fillna(-9999.0)
    out = out.sort_values(
        ["source_priority", "model_source_order_sort", "model_rank_sort", "heat_rank_sort", "heat_score_sort", "symbol"],
        ascending=[True, True, True, True, False, True],
    ).head(int(final_limit)).reset_index(drop=True)
    out["review_rank"] = range(1, len(out) + 1)
    return out.drop(
        columns=["_source_priority", "source_priority", "model_source_order_sort", "model_rank_sort", "heat_rank_sort", "heat_score_sort"]
    )


def classify_price_volume_layer(row: pd.Series) -> tuple[str, str, str]:
    turnover = maybe_float(row.get("turnover_rate_1"))
    vr1 = maybe_float(row.get("volume_ratio_1_prev"))
    vr3 = maybe_float(row.get("volume_ratio_3_prev"))
    vr5 = maybe_float(row.get("volume_ratio_5_prev"))
    vr7 = maybe_float(row.get("volume_ratio_7_prev"))

    if turnover is None or vr5 is None:
        return "中性", "未知", "缺少换手率或五日量比，量价层不作正向确认。"
    if turnover > 0.10 and vr5 > 5:
        return "不支持", "高", "换手率超过10%且五日量比超过5，量价层判断为过热。"
    if turnover > 0.05 and vr5 < 2:
        return "不支持", "高", "换手率超过5%但五日量比低于2，量价层判断为高换手但量能确认不足。"
    if vr1 is not None and vr3 is not None and vr1 > 2.5 and vr3 < 1.5 and vr5 < 1.5:
        return "不支持", "中", "一日量比突增但三日、五日量比不足，量价层判断为单日放量。"
    if vr3 is not None and vr7 is not None and vr3 >= 2 and vr5 >= 2 and vr7 >= 1.5:
        return "支持", "低", "三日、五日、七日量比持续放大，量价层判断为持续量能确认。"
    if turnover > 0.05 and 2 <= vr5 < 3:
        return "支持", "中", "换手率超过5%且五日量比在2到3之间，量价层判断为启动信号。"
    if 0.05 <= turnover <= 0.10 and 3 <= vr5 <= 5:
        return "支持", "中", "换手率在5%到10%且五日量比在3到5之间，量价层判断为加速信号。"
    if vr1 is not None and vr3 is not None and vr1 < 1 and vr3 < vr5:
        return "中性", "中", "一日量比低于1且三日量比低于五日量比，量价层判断为量能走弱。"
    if turnover <= 0.05 and vr5 < 2:
        return "中性", "中", "换手率不高且五日量比低于2，量价层不给正向确认。"
    return "中性", "中", "量价条件未触发明确支持或不支持。"


def attach_raw_liquidity_metrics(candidates: pd.DataFrame, stock_df: pd.DataFrame, trade_date: pd.Timestamp) -> pd.DataFrame:
    if candidates.empty or stock_df.empty:
        return candidates
    frame = stock_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["volume"] = pd.to_numeric(frame.get("volume"), errors="coerce")
    frame["turnover_rate_1"] = pd.to_numeric(frame.get("turnover_rate"), errors="coerce")
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    grouped = frame.groupby("symbol", sort=False)
    frame["volume_ratio_1_prev"] = frame["volume"] / grouped["volume"].shift(1).replace(0.0, np.nan)
    for window in (3, 5, 7):
        previous_mean = grouped["volume"].transform(
            lambda value, size=window: value.shift(1).rolling(size, min_periods=1).mean()
        )
        frame[f"volume_ratio_{window}_prev"] = frame["volume"] / previous_mean.replace(0.0, np.nan)
    current_mean = grouped["volume"].transform(lambda value: value.rolling(5, min_periods=2).mean())
    frame["volume_ratio_5"] = frame["volume"] / current_mean.replace(0.0, np.nan)

    metrics = frame[frame["date"] == pd.Timestamp(trade_date).normalize()][
        [
            "symbol",
            "turnover_rate_1",
            "volume_ratio_1_prev",
            "volume_ratio_3_prev",
            "volume_ratio_5_prev",
            "volume_ratio_7_prev",
            "volume_ratio_5",
        ]
    ].drop_duplicates(subset=["symbol"], keep="last")
    if metrics.empty:
        return candidates

    out = candidates.copy()
    out["symbol"] = out["symbol"].astype(str)
    out = out.drop(
        columns=[
            "turnover_rate_1",
            "volume_ratio_1_prev",
            "volume_ratio_3_prev",
            "volume_ratio_5_prev",
            "volume_ratio_7_prev",
            "volume_ratio_5",
        ],
        errors="ignore",
    )
    out = out.merge(metrics, on="symbol", how="left")
    price_volume = out.apply(classify_price_volume_layer, axis=1, result_type="expand")
    out["price_volume_layer_conclusion"] = price_volume[0]
    out["price_volume_layer_risk"] = price_volume[1]
    out["price_volume_layer_reason"] = price_volume[2]
    return out


def apply_basic_candidate_filters(candidates: pd.DataFrame, base_config: dict[str, Any]) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    out = candidates.copy()
    filters = base_config.get("universe", {}).get("filters", {})
    min_price = filters.get("min_latest_price")
    max_price = optional_float(filters.get("max_latest_price"))
    close = pd.to_numeric(out["close"], errors="coerce")
    if min_price is not None:
        out = out[close >= float(min_price)].copy()
        close = pd.to_numeric(out["close"], errors="coerce")
    if max_price is not None:
        out = out[close <= max_price].copy()
    return out.reset_index(drop=True)


def validate_candidate_signal_date(candidates: pd.DataFrame, trade_date: pd.Timestamp) -> None:
    if candidates.empty or "signal_date" not in candidates.columns:
        raise ValueError("Model inference produced no dated candidates.")
    signal_dates = pd.to_datetime(candidates["signal_date"], errors="coerce").dropna().dt.normalize()
    if signal_dates.empty:
        raise ValueError("Model inference candidates have no valid signal_date.")
    latest_signal_date = signal_dates.max()
    expected = pd.Timestamp(trade_date).normalize()
    if latest_signal_date != expected:
        raise ValueError(
            f"Model inference signal_date mismatch: latest_signal_date={latest_signal_date.date()} "
            f"expected_trade_date={expected.date()}."
        )
