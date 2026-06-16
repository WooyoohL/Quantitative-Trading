from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from data.dataset import SequenceDataset
from models.loss_functions import daily_rank_ic_mean, head_daily_rank_ic_mean, rank_ic
from models.trainer import TrainerConfig
from strategy.backtest import backtest_top_k, summarize_backtest


TABULAR_MODEL_ALIASES = {
    "ridge": "ridge",
    "ridge_regressor": "ridge",
    "ridge_regression": "ridge",
    "elasticnet": "elasticnet",
    "elastic_net": "elasticnet",
    "linearsvr": "linear_svr",
    "linear_svr": "linear_svr",
    "histgradientboostingregressor": "hist_gradient_boosting",
    "hist_gradient_boosting": "hist_gradient_boosting",
    "hist_gbr": "hist_gradient_boosting",
    "xgb": "xgboost",
    "xgboost": "xgboost",
    "xgbregressor": "xgboost",
    "lightgbm": "lightgbm",
    "lgbm": "lightgbm",
    "lgbmregressor": "lightgbm",
    "catboost": "catboost",
    "catboostregressor": "catboost",
}


def normalize_tabular_model_name(name: Any) -> str | None:
    key = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    key = key.replace("__", "_")
    return TABULAR_MODEL_ALIASES.get(key)


def is_tabular_model_name(name: Any) -> bool:
    return normalize_tabular_model_name(name) is not None


def _load_torch_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _flatten_dataset_features(dataset: SequenceDataset) -> np.ndarray:
    features = np.asarray(dataset.features, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError(f"Expected tabular source features with shape [samples, seq_len, feature_dim], got {features.shape}.")
    flat = features.reshape(features.shape[0], features.shape[1] * features.shape[2])
    return np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _pearson_loss(targets: np.ndarray, predictions: np.ndarray) -> float:
    if len(targets) < 2 or float(np.std(targets)) == 0.0 or float(np.std(predictions)) == 0.0:
        return float("nan")
    corr = float(np.corrcoef(targets, predictions)[0, 1])
    return float(1.0 - corr)


class TabularTrainer:
    def __init__(
        self,
        *,
        model_cfg: dict[str, Any],
        config: TrainerConfig,
        seq_len: int,
        feature_columns: list[str],
    ) -> None:
        self.model_name = normalize_tabular_model_name(model_cfg.get("name"))
        if self.model_name is None:
            raise ValueError(f"Unsupported tabular model.name={model_cfg.get('name')}")
        self.device = torch.device("cpu")
        self.model_cfg = dict(model_cfg)
        self.config = config
        self.seq_len = int(seq_len)
        self.feature_columns = list(feature_columns)
        self.model: Any | None = None
        self.history: list[dict[str, float | int]] = []
        self.best_epoch = 0
        self.best_valid_ic = float("-inf")
        self.best_valid_daily_ic = float("-inf")
        self.monitor_best_valid_ic = float("-inf")
        self.monitor_best_valid_daily_ic = float("-inf")
        self.best_selection_score = float("nan")
        self.selection_candidate_count = 0
        self.best_selection_breakdown: dict[str, float | int] = {}
        self.resolved_model_config = self._resolved_model_config()

    def _model_params(self) -> dict[str, Any]:
        params = dict(self.model_cfg.get("params", {}) or {})
        params.update(dict(self.model_cfg.get("sklearn_params", {}) or {}))
        return params

    def _resolved_model_config(self) -> dict[str, Any]:
        return {
            "name": self.model_name,
            "raw_name": str(self.model_cfg.get("name", self.model_name)),
            "input_dim": len(self.feature_columns),
            "seq_len": int(self.seq_len),
            "flattened_dim": int(self.seq_len * len(self.feature_columns)),
            "params": self._model_params(),
        }

    def _build_estimator(self) -> Any:
        params = self._model_params()
        seed = int(self.config.seed)

        if self.model_name == "ridge":
            from sklearn.linear_model import Ridge

            defaults = {"alpha": 1.0}
            defaults.update(params)
            return Ridge(**defaults)

        if self.model_name == "elasticnet":
            from sklearn.linear_model import ElasticNet

            defaults = {"alpha": 0.001, "l1_ratio": 0.2, "max_iter": 5000, "random_state": seed}
            defaults.update(params)
            return ElasticNet(**defaults)

        if self.model_name == "linear_svr":
            from sklearn.svm import LinearSVR

            defaults = {"C": 0.3, "epsilon": 0.0, "max_iter": 5000, "random_state": seed}
            defaults.update(params)
            return LinearSVR(**defaults)

        if self.model_name == "hist_gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingRegressor

            defaults = {
                "max_iter": 250,
                "learning_rate": 0.04,
                "l2_regularization": 0.1,
                "max_leaf_nodes": 31,
                "random_state": seed,
            }
            defaults.update(params)
            return HistGradientBoostingRegressor(**defaults)

        if self.model_name == "xgboost":
            from xgboost import XGBRegressor

            defaults = {
                "n_estimators": 350,
                "max_depth": 3,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "n_jobs": 4,
                "random_state": seed,
                "verbosity": 0,
            }
            defaults.update(params)
            return XGBRegressor(**defaults)

        if self.model_name == "lightgbm":
            from lightgbm import LGBMRegressor

            defaults = {
                "n_estimators": 350,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "n_jobs": 4,
                "random_state": seed,
                "verbosity": -1,
            }
            defaults.update(params)
            return LGBMRegressor(**defaults)

        if self.model_name == "catboost":
            from catboost import CatBoostRegressor

            defaults = {
                "iterations": 350,
                "depth": 4,
                "learning_rate": 0.03,
                "loss_function": "RMSE",
                "random_seed": seed,
                "verbose": False,
                "allow_writing_files": False,
                "thread_count": 4,
            }
            defaults.update(params)
            return CatBoostRegressor(**defaults)

        raise ValueError(f"Unsupported tabular model.name={self.model_name}")

    def fit(
        self,
        train_dataset: SequenceDataset,
        valid_dataset: SequenceDataset,
        run_dir: Path,
        model_config: dict[str, Any],
    ) -> pd.DataFrame:
        if train_dataset.targets_numpy is None:
            raise ValueError("Training targets are required for tabular models.")

        run_dir.mkdir(parents=True, exist_ok=True)
        x_train = _flatten_dataset_features(train_dataset)
        y_train = np.asarray(train_dataset.targets_numpy, dtype=np.float32)
        y_valid = np.asarray(valid_dataset.targets_numpy, dtype=np.float32)

        self.model = self._build_estimator()
        self.model.fit(x_train, y_train)

        train_pred = self.predict_dataset(train_dataset)
        valid_pred = self.predict_dataset(valid_dataset)
        train_dates = train_dataset.meta["date"].to_numpy() if "date" in train_dataset.meta.columns else np.asarray([])
        valid_dates = valid_dataset.meta["date"].to_numpy() if "date" in valid_dataset.meta.columns else np.asarray([])

        train_eval = self.compute_eval_metrics(train_pred, y_train, dates=train_dates)
        valid_eval = self.compute_eval_metrics(valid_pred, y_valid, dates=valid_dates)
        selection_metrics = self._compute_selection_metrics(valid_dataset, valid_pred)

        self.best_epoch = 1
        self.best_valid_ic = float(valid_eval["ic"])
        self.best_valid_daily_ic = float(valid_eval["daily_ic"])
        self.monitor_best_valid_ic = float(valid_eval["ic"])
        self.monitor_best_valid_daily_ic = float(valid_eval["daily_ic"])

        record = {
            "epoch": 1,
            "train_loss": float(train_eval["total_loss"]),
            "train_ic": float(train_eval["ic"]),
            "train_daily_ic": float(train_eval["daily_ic"]),
            "valid_loss": float(valid_eval["total_loss"]),
            "valid_mse_loss": float(valid_eval["mse_loss"]),
            "valid_pearson_loss": float(valid_eval["pearson_loss"]),
            "valid_soft_rank_loss": float("nan"),
            "valid_pairwise_loss": float("nan"),
            "valid_ic": float(valid_eval["ic"]),
            "valid_daily_ic": float(valid_eval["daily_ic"]),
            "valid_top_k_return": float(selection_metrics["valid_top_k_return"]),
            "valid_hit_rate": float(selection_metrics["valid_hit_rate"]),
            "valid_excess_return": float(selection_metrics["valid_excess_return"]),
            "valid_positive_excess_rate": float(selection_metrics["valid_positive_excess_rate"]),
            "valid_relative_return": float(selection_metrics["valid_relative_return"]),
            "valid_max_drawdown": float(selection_metrics["valid_max_drawdown"]),
            "valid_head_daily_ic": float(valid_eval["head_daily_ic"]),
            "lr": float(self.model_cfg.get("lr", 0.0) or 0.0),
        }
        self.history = [record]
        self._save_checkpoint(run_dir / "best.ckpt", epoch=1, metrics=record, model_config=model_config)
        self._save_checkpoint(run_dir / "last.ckpt", epoch=1, metrics=record, model_config=model_config)

        history_df = pd.DataFrame(self.history)
        history_df.to_csv(run_dir / "train_metrics.csv", index=False, encoding="utf-8-sig")
        self._write_trainer_state(run_dir, model_config=model_config)
        print(
            f"[Train] model={self.model_name} train_loss={record['train_loss']:.4f} "
            f"valid_loss={record['valid_loss']:.4f} valid_daily_ic={record['valid_daily_ic']:.4f} "
            f"valid_excess={record['valid_excess_return']:.4f}"
        )
        return history_df

    def predict_dataset(self, dataset: SequenceDataset) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Tabular model is not fitted or loaded.")
        features = _flatten_dataset_features(dataset)
        predictions = self.model.predict(features)
        return np.asarray(predictions, dtype=np.float32).reshape(-1)

    def _compute_selection_metrics(self, dataset: SequenceDataset, predictions: np.ndarray) -> dict[str, float]:
        if dataset.meta.empty or "date" not in dataset.meta.columns or "label" not in dataset.meta.columns:
            return {
                "valid_top_k_return": float("nan"),
                "valid_hit_rate": float("nan"),
                "valid_excess_return": float("nan"),
                "valid_positive_excess_rate": float("nan"),
                "valid_relative_return": float("nan"),
                "valid_max_drawdown": float("nan"),
            }

        scored = dataset.meta[["date", "label"]].copy()
        scored["score"] = predictions
        report = backtest_top_k(scored, top_k=int(self.config.selection_top_k))
        summary = summarize_backtest(report)
        return {
            "valid_top_k_return": float(summary.get("top_k_mean_return") or 0.0),
            "valid_hit_rate": float(summary.get("win_rate") or 0.0),
            "valid_excess_return": float(summary.get("excess_mean_return") or 0.0),
            "valid_positive_excess_rate": float(summary.get("positive_excess_rate") or 0.0),
            "valid_relative_return": float(summary.get("relative_return") or 0.0),
            "valid_max_drawdown": float(summary.get("max_drawdown") or 0.0),
        }

    def compute_eval_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray | None,
        dates: np.ndarray | None = None,
    ) -> dict[str, float]:
        if targets is None or len(targets) == 0:
            return {
                "mse_loss": float("nan"),
                "pearson_loss": float("nan"),
                "soft_rank_loss": float("nan"),
                "pairwise_loss": float("nan"),
                "total_loss": float("nan"),
                "ic": float("nan"),
                "daily_ic": float("nan"),
                "head_daily_ic": float("nan"),
            }

        target_values = np.asarray(targets, dtype=np.float32)
        pred_values = np.asarray(predictions, dtype=np.float32)
        mse_loss = float(np.mean(np.square(pred_values - target_values)))
        pearson_loss = _pearson_loss(target_values, pred_values)
        daily_ic = float("nan")
        head_daily_ic = float("nan")
        if dates is not None and len(dates) == len(predictions):
            daily_ic = float(daily_rank_ic_mean(target_values, pred_values, dates))
            head_daily_ic = float(
                head_daily_rank_ic_mean(
                    target_values,
                    pred_values,
                    dates,
                    top_n=int(self.config.selection_head_top_n),
                )
            )
        return {
            "mse_loss": float(mse_loss),
            "pearson_loss": float(pearson_loss),
            "soft_rank_loss": float("nan"),
            "pairwise_loss": float("nan"),
            "total_loss": float(mse_loss),
            "ic": float(rank_ic(target_values, pred_values)),
            "daily_ic": float(daily_ic),
            "head_daily_ic": float(head_daily_ic),
        }

    def _save_checkpoint(self, path: Path, epoch: int, metrics: dict[str, Any], model_config: dict[str, Any]) -> None:
        payload = {
            "checkpoint_format": "tabular_estimator_v1",
            "epoch": int(epoch),
            "metrics": metrics,
            "model_config": model_config,
            "trainer_config": asdict(self.config),
            "trainer_state": {
                "monitor_best_valid_ic": float(self.monitor_best_valid_ic),
                "monitor_best_valid_daily_ic": float(self.monitor_best_valid_daily_ic),
            },
            "estimator": self.model,
            "history": self.history,
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        payload = _load_torch_payload(path)
        self.model = payload["estimator"]
        self.history = list(payload.get("history", []))
        self.best_epoch = int(payload.get("epoch", self.best_epoch))
        metrics = payload.get("metrics", {})
        self.best_valid_ic = float(metrics.get("valid_ic", self.best_valid_ic))
        self.best_valid_daily_ic = float(metrics.get("valid_daily_ic", self.best_valid_daily_ic))
        trainer_state = payload.get("trainer_state", {})
        self.monitor_best_valid_ic = float(trainer_state.get("monitor_best_valid_ic", self.monitor_best_valid_ic))
        self.monitor_best_valid_daily_ic = float(
            trainer_state.get("monitor_best_valid_daily_ic", self.monitor_best_valid_daily_ic)
        )
        self.best_selection_score = float(metrics.get("selection_score", self.best_selection_score))
        self.selection_candidate_count = int(metrics.get("selection_candidate_count", self.selection_candidate_count))
        self.best_selection_breakdown = dict(metrics.get("selection_breakdown", self.best_selection_breakdown))
        return payload

    def _write_trainer_state(self, run_dir: Path, model_config: dict[str, Any]) -> None:
        payload = {
            "best_epoch": int(self.best_epoch),
            "best_valid_ic": float(self.monitor_best_valid_ic),
            "best_valid_daily_ic": float(self.monitor_best_valid_daily_ic),
            "selected_epoch_valid_ic": float(self.best_valid_ic),
            "selected_epoch_valid_daily_ic": float(self.best_valid_daily_ic),
            "checkpoint_selection_mode": str(self.config.checkpoint_selection_mode),
            "best_selection_score": None if not np.isfinite(self.best_selection_score) else float(self.best_selection_score),
            "selection_candidate_count": int(self.selection_candidate_count),
            "best_selection_breakdown": self.best_selection_breakdown,
            "trainer_config": asdict(self.config),
            "model_config": model_config,
        }
        (run_dir / "trainer_state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
