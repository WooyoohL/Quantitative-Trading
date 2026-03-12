from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from data.dataset import DatasetBundle, FeatureScaler, SequenceDataset


CACHE_VERSION = "training_context_v3"


def _frame_signature(df: pd.DataFrame, *, entity_col: str | None = None) -> dict[str, Any]:
    if df is None or df.empty:
        return {"rows": 0}
    signature: dict[str, Any] = {"rows": int(len(df))}
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        signature["min_date"] = str(dates.min().date()) if not dates.empty else None
        signature["max_date"] = str(dates.max().date()) if not dates.empty else None
        signature["n_dates"] = int(dates.nunique())
    if entity_col and entity_col in df.columns:
        signature[f"n_{entity_col}"] = int(df[entity_col].astype(str).nunique())
    return signature


def build_training_context_cache_key(
    *,
    config: dict,
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    industry_map_df: pd.DataFrame,
    industry_daily_df: pd.DataFrame,
    time_block_shuffle: bool,
    time_block_size: int | None,
) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "seed": int(config.get("seed", 7)),
        "training": config.get("training", {}),
        "rolling": config.get("rolling", {}),
        "sequence": config.get("sequence", {}),
        "data": {
            "label_horizon": config.get("data", {}).get("label_horizon", 1),
            "trainable_history_days": config.get("data", {}).get("trainable_history_days", 260),
            "daily_cross_sectional_norm": bool(config.get("data", {}).get("daily_cross_sectional_norm", False)),
        },
        "universe": config.get("universe", {}),
        "index": config.get("index", {}),
        "industry": config.get("industry", {}),
        "peer": config.get("peer", {}),
        "shuffle": {
            "enabled": bool(time_block_shuffle),
            "block_size": int(time_block_size or 0),
        },
        "stock_sig": _frame_signature(stock_df, entity_col="symbol"),
        "index_sig": _frame_signature(index_df, entity_col="index_key"),
        "industry_map_sig": _frame_signature(industry_map_df, entity_col="symbol"),
        "industry_daily_sig": _frame_signature(industry_daily_df, entity_col="industry_code"),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _serialize_dataset_bundle(bundle: DatasetBundle) -> dict[str, Any]:
    return {
        "train_dataset": {
            "features": bundle.train_dataset.features,
            "targets": bundle.train_dataset.targets,
            "meta": bundle.train_dataset.meta,
        },
        "valid_dataset": {
            "features": bundle.valid_dataset.features,
            "targets": bundle.valid_dataset.targets,
            "meta": bundle.valid_dataset.meta,
        },
        "test_dataset": {
            "features": bundle.test_dataset.features,
            "targets": bundle.test_dataset.targets,
            "meta": bundle.test_dataset.meta,
        },
        "inference_dataset": {
            "features": bundle.inference_dataset.features,
            "targets": bundle.inference_dataset.targets,
            "meta": bundle.inference_dataset.meta,
        },
        "feature_columns": list(bundle.feature_columns),
        "scaler": {
            "means": bundle.scaler.means,
            "stds": bundle.scaler.stds,
        },
        "split_dates": bundle.split_dates,
        "peer_map": bundle.peer_map,
    }


def _deserialize_dataset_bundle(payload: dict[str, Any]) -> DatasetBundle:
    scaler_payload = payload["scaler"]
    scaler = FeatureScaler(means=scaler_payload["means"], stds=scaler_payload["stds"])
    return DatasetBundle(
        train_dataset=SequenceDataset(
            payload["train_dataset"]["features"],
            payload["train_dataset"]["targets"],
            payload["train_dataset"]["meta"],
        ),
        valid_dataset=SequenceDataset(
            payload["valid_dataset"]["features"],
            payload["valid_dataset"]["targets"],
            payload["valid_dataset"]["meta"],
        ),
        test_dataset=SequenceDataset(
            payload["test_dataset"]["features"],
            payload["test_dataset"]["targets"],
            payload["test_dataset"]["meta"],
        ),
        inference_dataset=SequenceDataset(
            payload["inference_dataset"]["features"],
            payload["inference_dataset"]["targets"],
            payload["inference_dataset"]["meta"],
        ),
        feature_frame=pd.DataFrame(),
        feature_columns=list(payload["feature_columns"]),
        scaler=scaler,
        split_dates=payload["split_dates"],
        peer_map=payload["peer_map"],
    )


def load_training_context_cache(cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        payload["dataset_bundle"] = _deserialize_dataset_bundle(payload["dataset_bundle"])
        payload["universe_report"] = pd.DataFrame(payload["universe_report"])
        payload["selected_symbols"] = [str(symbol) for symbol in payload["selected_symbols"]]
        return payload
    except Exception:
        return None


def save_training_context_cache(cache_path: Path, payload: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(payload)
    serializable["dataset_bundle"] = _serialize_dataset_bundle(payload["dataset_bundle"])
    serializable["universe_report"] = payload["universe_report"].to_dict(orient="records")
    temp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
    with temp_path.open("wb") as handle:
        pickle.dump(serializable, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_path, cache_path)
