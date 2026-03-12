from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.dataset import DatasetBundle
from data.fetcher import write_json
from models.trainer import AlphaTrainer
from strategy.backtest import backtest_metric_guidance, backtest_top_k, summarize_backtest


@dataclass
class EvaluationArtifacts:
    valid_pred: pd.Series
    test_pred: pd.Series
    valid_eval: dict[str, float]
    test_eval: dict[str, float]
    valid_scored: pd.DataFrame
    test_scored: pd.DataFrame
    valid_backtest_report: pd.DataFrame
    test_backtest_report: pd.DataFrame
    valid_backtest_metrics: dict[str, float]
    backtest_metrics: dict[str, float]
    backtest_metric_guidance: dict[str, str]


def evaluate_trained_model(
    *,
    trainer: AlphaTrainer,
    dataset_bundle: DatasetBundle,
    run_dir: Path,
    top_k: int,
) -> EvaluationArtifacts:
    valid_pred = trainer.predict_dataset(dataset_bundle.valid_dataset)
    test_pred = trainer.predict_dataset(dataset_bundle.test_dataset)

    valid_target = dataset_bundle.valid_dataset.targets_numpy
    test_target = dataset_bundle.test_dataset.targets_numpy
    valid_dates = (
        dataset_bundle.valid_dataset.meta["date"].to_numpy()
        if "date" in dataset_bundle.valid_dataset.meta.columns
        else None
    )
    test_dates = (
        dataset_bundle.test_dataset.meta["date"].to_numpy()
        if "date" in dataset_bundle.test_dataset.meta.columns
        else None
    )

    valid_eval = trainer.compute_eval_metrics(valid_pred, valid_target, dates=valid_dates)
    test_eval = trainer.compute_eval_metrics(test_pred, test_target, dates=test_dates)

    valid_scored = dataset_bundle.valid_dataset.meta.copy()
    valid_scored["score"] = valid_pred
    valid_scored.to_csv(run_dir / "valid_predictions.csv", index=False, encoding="utf-8-sig")

    test_scored = dataset_bundle.test_dataset.meta.copy()
    test_scored["score"] = test_pred
    test_scored.to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    valid_backtest_report = backtest_top_k(valid_scored[["date", "symbol", "label", "score"]], top_k=int(top_k))
    valid_backtest_report.to_csv(run_dir / "valid_backtest.csv", index=False, encoding="utf-8-sig")
    valid_backtest_metrics = summarize_backtest(valid_backtest_report)

    test_backtest_report = backtest_top_k(test_scored[["date", "symbol", "label", "score"]], top_k=int(top_k))
    test_backtest_report.to_csv(run_dir / "backtest.csv", index=False, encoding="utf-8-sig")
    backtest_metrics = summarize_backtest(test_backtest_report)
    metric_guidance = backtest_metric_guidance()
    write_json(
        run_dir / "backtest_metrics.json",
        {
            "metrics": backtest_metrics,
            "guidance": metric_guidance,
        },
    )

    return EvaluationArtifacts(
        valid_pred=pd.Series(valid_pred),
        test_pred=pd.Series(test_pred),
        valid_eval=valid_eval,
        test_eval=test_eval,
        valid_scored=valid_scored,
        test_scored=test_scored,
        valid_backtest_report=valid_backtest_report,
        test_backtest_report=test_backtest_report,
        valid_backtest_metrics=valid_backtest_metrics,
        backtest_metrics=backtest_metrics,
        backtest_metric_guidance=metric_guidance,
    )
