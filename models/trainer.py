from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import SequenceDataset
from metrics.selection import (
    annotate_ic_gate_selection,
    annotate_topk_valid_selection,
    build_topk_valid_monitor_tuple,
)
from models.loss_functions import (
    HeadWeightedPairwiseLoss,
    PearsonLoss,
    SoftRankICLoss,
    daily_rank_ic_mean,
    head_daily_rank_ic_mean,
    rank_ic,
)
from strategy.backtest import backtest_top_k, summarize_backtest


@dataclass
class TrainerConfig:
    epochs: int = 60
    lr: float = 1e-3
    mse_loss_weight: float = 1.0
    weight_decay: float = 1e-2
    pearson_loss_weight: float = 0.0
    soft_rank_loss_weight: float = 0.0
    pairwise_loss_weight: float = 0.0
    ranking_tau: float = 1.0
    pairwise_top_k_focus: int = 3
    pairwise_head_boost: float = 3.0
    pairwise_top_internal_boost: float = 1.5
    pairwise_tail_weight: float = 0.0
    batch_size: int = 256
    eval_batch_size: int = 512
    log_every: int = 1
    early_stopping_patience: int = 10
    num_workers: int = 0
    seed: int = 7
    checkpoint_selection_mode: str = "valid_ic"
    selection_top_k: int = 3
    selection_ic_tolerance: float = 0.01
    selection_weight_ic: float = 0.50
    selection_weight_top_k_return: float = 0.25
    selection_weight_hit_rate: float = 0.15
    selection_weight_excess_return: float = 0.10
    selection_head_top_n: int = 10
    selection_min_excess_return: float = 0.0
    selection_min_positive_excess_rate: float = 0.50
    selection_min_daily_ic: float = 0.0
    selection_min_head_daily_ic: float = -1.0
    selection_max_drawdown_limit: float = -0.10


class AlphaTrainer:
    def __init__(self, model: nn.Module, config: TrainerConfig, device: torch.device | None = None) -> None:
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        torch.manual_seed(int(config.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config.seed))

        self.criterion = nn.MSELoss()
        self.criterion_1 = PearsonLoss()
        self.criterion_2 = SoftRankICLoss(tau=float(config.ranking_tau))
        self.criterion_3 = HeadWeightedPairwiseLoss(
            tau=float(config.ranking_tau),
            top_k_focus=int(config.pairwise_top_k_focus),
            head_boost=float(config.pairwise_head_boost),
            top_internal_boost=float(config.pairwise_top_internal_boost),
            tail_weight=float(config.pairwise_tail_weight),
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
        )
        self.history: list[dict[str, float | int]] = []
        self.best_epoch = 0
        self.best_valid_ic = float("-inf")
        self.best_valid_daily_ic = float("-inf")
        self.monitor_best_valid_ic = float("-inf")
        self.monitor_best_valid_daily_ic = float("-inf")
        self.best_selection_score = float("nan")
        self.selection_candidate_count = 0
        self.best_selection_breakdown: dict[str, float | int] = {}

    def fit(
        self,
        train_dataset: SequenceDataset,
        valid_dataset: SequenceDataset,
        run_dir: Path,
        model_config: dict,
    ) -> pd.DataFrame:
        run_dir.mkdir(parents=True, exist_ok=True)
        train_generator = torch.Generator()
        train_generator.manual_seed(int(self.config.seed))
        eval_generator = torch.Generator()
        eval_generator.manual_seed(int(self.config.seed))
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(self.config.batch_size),
            shuffle=True,
            num_workers=int(self.config.num_workers),
            drop_last=False,
            generator=train_generator,
            worker_init_fn=self._seed_worker,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=int(self.config.eval_batch_size),
            shuffle=False,
            num_workers=int(self.config.num_workers),
            drop_last=False,
            generator=eval_generator,
            worker_init_fn=self._seed_worker,
        )

        checkpoint_mode = str(self.config.checkpoint_selection_mode).strip().lower()
        use_composite_selection = checkpoint_mode == "ic_gate_composite"
        use_topk_valid_selection = checkpoint_mode == "topk_valid"
        monitor_daily_ic = use_topk_valid_selection
        best_topk_monitor_tuple: tuple[float | int, ...] | None = None
        checkpoint_dir = run_dir / "epoch_checkpoints"
        if use_composite_selection or use_topk_valid_selection:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        epochs_without_improvement = 0
        for epoch in range(1, int(self.config.epochs) + 1):
            train_loss = self._train_epoch(train_loader, train_dataset=train_dataset, epoch=epoch)
            train_pred = self.predict_dataset(train_dataset)
            valid_pred = self.predict_loader(valid_loader)

            train_target = train_dataset.targets_numpy
            valid_target = valid_dataset.targets_numpy
            train_dates = train_dataset.meta["date"].to_numpy() if "date" in train_dataset.meta.columns else np.asarray([])
            valid_dates = valid_dataset.meta["date"].to_numpy() if "date" in valid_dataset.meta.columns else np.asarray([])

            train_ic = 0.0 if train_target is None else rank_ic(train_target, train_pred)
            train_daily_ic = 0.0 if train_target is None or len(train_dates) != len(train_pred) else daily_rank_ic_mean(
                train_target, train_pred, train_dates
            )
            valid_loss = float("nan")
            valid_mse_loss = float("nan")
            valid_pearson_loss = float("nan")
            valid_soft_rank_loss = float("nan")
            valid_pairwise_loss = float("nan")
            valid_ic = float("nan")
            valid_daily_ic = float("nan")
            valid_top_k_return = float("nan")
            valid_hit_rate = float("nan")
            valid_excess_return = float("nan")
            valid_positive_excess_rate = float("nan")
            valid_relative_return = float("nan")
            valid_max_drawdown = float("nan")
            valid_head_daily_ic = float("nan")
            if valid_target is not None and len(valid_target) > 0:
                (
                    valid_mse_loss,
                    valid_pearson_loss,
                    valid_soft_rank_loss,
                    valid_pairwise_loss,
                    valid_loss,
                ) = self._compute_loss_metrics(valid_pred, valid_target, valid_dates)
                valid_ic = rank_ic(valid_target, valid_pred)
                if len(valid_dates) == len(valid_pred):
                    valid_daily_ic = daily_rank_ic_mean(valid_target, valid_pred, valid_dates)
                    valid_head_daily_ic = head_daily_rank_ic_mean(
                        valid_target,
                        valid_pred,
                        valid_dates,
                        top_n=int(self.config.selection_head_top_n),
                    )
                selection_metrics = self._compute_selection_metrics(valid_dataset, valid_pred)
                valid_top_k_return = float(selection_metrics["valid_top_k_return"])
                valid_hit_rate = float(selection_metrics["valid_hit_rate"])
                valid_excess_return = float(selection_metrics["valid_excess_return"])
                valid_positive_excess_rate = float(selection_metrics["valid_positive_excess_rate"])
                valid_relative_return = float(selection_metrics["valid_relative_return"])
                valid_max_drawdown = float(selection_metrics["valid_max_drawdown"])

            record = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_ic": float(train_ic),
                "train_daily_ic": float(train_daily_ic),
                "valid_loss": float(valid_loss),
                "valid_mse_loss": float(valid_mse_loss),
                "valid_pearson_loss": float(valid_pearson_loss),
                "valid_soft_rank_loss": float(valid_soft_rank_loss),
                "valid_pairwise_loss": float(valid_pairwise_loss),
                "valid_ic": float(valid_ic),
                "valid_daily_ic": float(valid_daily_ic),
                "valid_top_k_return": valid_top_k_return,
                "valid_hit_rate": valid_hit_rate,
                "valid_excess_return": valid_excess_return,
                "valid_positive_excess_rate": valid_positive_excess_rate,
                "valid_relative_return": valid_relative_return,
                "valid_max_drawdown": valid_max_drawdown,
                "valid_head_daily_ic": valid_head_daily_ic,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }
            self.history.append(record)

            if use_topk_valid_selection:
                current_topk_tuple = build_topk_valid_monitor_tuple(
                    record,
                    selection_min_excess_return=float(self.config.selection_min_excess_return),
                    selection_min_positive_excess_rate=float(self.config.selection_min_positive_excess_rate),
                    selection_min_daily_ic=float(self.config.selection_min_daily_ic),
                    selection_min_head_daily_ic=float(self.config.selection_min_head_daily_ic),
                    selection_max_drawdown_limit=float(self.config.selection_max_drawdown_limit),
                )
                improved = best_topk_monitor_tuple is None or current_topk_tuple > best_topk_monitor_tuple
            else:
                monitor_value = valid_daily_ic if monitor_daily_ic else valid_ic
                best_monitor_value = self.monitor_best_valid_daily_ic if monitor_daily_ic else self.monitor_best_valid_ic
                improved = monitor_value > best_monitor_value

            if improved:
                self.monitor_best_valid_ic = float(valid_ic)
                self.monitor_best_valid_daily_ic = float(valid_daily_ic)
                self.best_valid_ic = float(valid_ic)
                self.best_valid_daily_ic = float(valid_daily_ic)
                self.best_epoch = int(epoch)
                if use_topk_valid_selection:
                    best_topk_monitor_tuple = current_topk_tuple
                epochs_without_improvement = 0
                self._save_checkpoint(
                    run_dir / "best.ckpt",
                    epoch=epoch,
                    metrics=record,
                    model_config=model_config,
                )
            else:
                epochs_without_improvement += 1

            if use_composite_selection or use_topk_valid_selection:
                self._save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch:03d}.ckpt",
                    epoch=epoch,
                    metrics=record,
                    model_config=model_config,
                )

            self._save_checkpoint(
                run_dir / "last.ckpt",
                epoch=epoch,
                metrics=record,
                model_config=model_config,
            )

            if epoch == 1 or epoch % int(self.config.log_every) == 0 or epoch == int(self.config.epochs):
                print(
                    f"[Train] epoch={epoch:03d}/{self.config.epochs} "
                    f"train_loss={train_loss:.4f} "
                    f"valid_loss={valid_loss:.4f} "
                    f"valid_daily_ic={valid_daily_ic:.4f} "
                    f"valid_excess={valid_excess_return:.4f} "
                    f"valid_pos_excess={valid_positive_excess_rate:.2%}"
                )

            if epochs_without_improvement >= int(self.config.early_stopping_patience):
                if use_topk_valid_selection:
                    print(f"[Train] early stopping at epoch={epoch}, best_topk_valid_epoch={self.best_epoch}")
                else:
                    best_monitor_label = "best_valid_daily_ic" if monitor_daily_ic else "best_valid_ic"
                    best_monitor_print = self.monitor_best_valid_daily_ic if monitor_daily_ic else self.monitor_best_valid_ic
                    print(f"[Train] early stopping at epoch={epoch}, {best_monitor_label}={best_monitor_print:.4f}")
                break

        history_df = pd.DataFrame(self.history)
        if use_composite_selection and not history_df.empty:
            final_monitor_best_valid_ic = float(self.monitor_best_valid_ic)
            final_monitor_best_valid_daily_ic = float(self.monitor_best_valid_daily_ic)
            history_df, selection_result = annotate_ic_gate_selection(
                history_df,
                selection_ic_tolerance=float(self.config.selection_ic_tolerance),
                selection_weight_ic=float(self.config.selection_weight_ic),
                selection_weight_top_k_return=float(self.config.selection_weight_top_k_return),
                selection_weight_hit_rate=float(self.config.selection_weight_hit_rate),
                selection_weight_excess_return=float(self.config.selection_weight_excess_return),
            )
            if selection_result is not None:
                self.best_epoch = int(selection_result.epoch)
                self.best_selection_score = float(selection_result.selection_score)
                self.selection_candidate_count = int(selection_result.candidate_count)
                self.best_selection_breakdown = dict(selection_result.breakdown)
                self.load_checkpoint(checkpoint_dir / f"epoch_{self.best_epoch:03d}.ckpt")
                self.monitor_best_valid_ic = final_monitor_best_valid_ic
                self.monitor_best_valid_daily_ic = final_monitor_best_valid_daily_ic

                selected_metrics = dict(self.history[self.best_epoch - 1])
                selected_metrics["selection_score"] = float(self.best_selection_score)
                selected_metrics["selection_candidate_count"] = int(self.selection_candidate_count)
                selected_metrics["selection_breakdown"] = self.best_selection_breakdown
                self._save_checkpoint(
                    run_dir / "best.ckpt",
                    epoch=self.best_epoch,
                    metrics=selected_metrics,
                    model_config=model_config,
                )
        elif use_topk_valid_selection and not history_df.empty:
            final_monitor_best_valid_ic = float(self.monitor_best_valid_ic)
            final_monitor_best_valid_daily_ic = float(self.monitor_best_valid_daily_ic)
            history_df, selection_result = annotate_topk_valid_selection(
                history_df,
                selection_min_excess_return=float(self.config.selection_min_excess_return),
                selection_min_positive_excess_rate=float(self.config.selection_min_positive_excess_rate),
                selection_min_daily_ic=float(self.config.selection_min_daily_ic),
                selection_min_head_daily_ic=float(self.config.selection_min_head_daily_ic),
                selection_max_drawdown_limit=float(self.config.selection_max_drawdown_limit),
                selection_head_top_n=int(self.config.selection_head_top_n),
            )
            if selection_result is not None:
                self.best_epoch = int(selection_result.epoch)
                self.best_selection_score = float(selection_result.selection_score)
                self.selection_candidate_count = int(selection_result.candidate_count)
                self.best_selection_breakdown = dict(selection_result.breakdown)
                self.load_checkpoint(checkpoint_dir / f"epoch_{self.best_epoch:03d}.ckpt")
                self.monitor_best_valid_ic = final_monitor_best_valid_ic
                self.monitor_best_valid_daily_ic = final_monitor_best_valid_daily_ic

                selected_metrics = dict(self.history[self.best_epoch - 1])
                selected_metrics["selection_score"] = float(self.best_selection_score)
                selected_metrics["selection_candidate_count"] = int(self.selection_candidate_count)
                selected_metrics["selection_breakdown"] = self.best_selection_breakdown
                self._save_checkpoint(
                    run_dir / "best.ckpt",
                    epoch=self.best_epoch,
                    metrics=selected_metrics,
                    model_config=model_config,
                )

        history_df.to_csv(run_dir / "train_metrics.csv", index=False, encoding="utf-8-sig")
        self._write_trainer_state(run_dir, model_config=model_config)
        self.load_checkpoint(run_dir / "best.ckpt")
        return history_df

    def _train_epoch(
        self,
        train_loader: DataLoader,
        *,
        train_dataset: SequenceDataset | None = None,
        epoch: int = 1,
    ) -> float:
        if train_dataset is not None and (
            float(self.config.soft_rank_loss_weight) > 0.0 or float(self.config.pairwise_loss_weight) > 0.0
        ):
            return self._train_epoch_grouped_by_date(train_dataset, epoch=epoch)

        self.model.train()
        total_loss = 0.0
        total_items = 0

        for features, targets in train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.model(features)
            mse_loss = self.criterion(predictions, targets)
            pearson_loss = self.criterion_1(predictions, targets)
            loss = (
                float(self.config.mse_loss_weight) * mse_loss
                + float(self.config.pearson_loss_weight) * pearson_loss
            )
            loss.backward()
            self.optimizer.step()

            batch_size = int(features.size(0))
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size

        return total_loss / max(1, total_items)

    def _train_epoch_grouped_by_date(self, dataset: SequenceDataset, *, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_items = 0
        dates = pd.to_datetime(dataset.meta["date"]).dt.normalize().to_numpy()
        unique_dates = np.asarray(pd.unique(dates))
        rng = np.random.default_rng(int(self.config.seed) + int(epoch))
        ordered_dates = unique_dates[rng.permutation(len(unique_dates))]

        for date_value in ordered_dates:
            idx = np.flatnonzero(dates == date_value)
            if idx.size < 2:
                continue

            features = torch.as_tensor(dataset.features[idx], dtype=torch.float32, device=self.device)
            targets = torch.as_tensor(dataset.targets[idx], dtype=torch.float32, device=self.device)

            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.model(features)

            mse_loss = self.criterion(predictions, targets)
            pearson_loss = self.criterion_1(predictions, targets)
            soft_rank_loss = self.criterion_2(predictions, targets)
            pairwise_loss = self.criterion_3(predictions, targets)
            loss = (
                float(self.config.mse_loss_weight) * mse_loss
                + float(self.config.pearson_loss_weight) * pearson_loss
                + float(self.config.soft_rank_loss_weight) * soft_rank_loss
                + float(self.config.pairwise_loss_weight) * pairwise_loss
            )
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item()) * int(idx.size)
            total_items += int(idx.size)

        return total_loss / max(1, total_items)

    def predict_dataset(self, dataset: SequenceDataset) -> np.ndarray:
        eval_generator = torch.Generator()
        eval_generator.manual_seed(int(self.config.seed))
        loader = DataLoader(
            dataset,
            batch_size=int(self.config.eval_batch_size),
            shuffle=False,
            num_workers=int(self.config.num_workers),
            generator=eval_generator,
            worker_init_fn=self._seed_worker,
        )
        return self.predict_loader(loader)

    def predict_loader(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    features = batch[0]
                else:
                    features = batch
                features = features.to(self.device)
                pred = self.model(features).detach().cpu().numpy()
                predictions.append(pred)

        if not predictions:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(predictions, axis=0).astype(np.float32)

    @staticmethod
    def _seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

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

    def _compute_loss_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        dates: np.ndarray | None = None,
    ) -> tuple[float, float, float, float, float]:
        pred_tensor = torch.as_tensor(predictions, dtype=torch.float32, device=self.device)
        target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        mse_loss = float(self.criterion(pred_tensor, target_tensor).detach().cpu().item())
        pearson_loss = float(self.criterion_1(pred_tensor, target_tensor).detach().cpu().item())
        soft_rank_loss = float("nan")
        pairwise_loss = float("nan")
        if dates is not None and len(dates) == len(predictions):
            date_values = np.asarray(dates)
            soft_rank_losses: list[float] = []
            pairwise_losses: list[float] = []
            for date_value in np.unique(date_values):
                mask = date_values == date_value
                if int(mask.sum()) < 2:
                    continue
                day_pred = pred_tensor[mask]
                day_target = target_tensor[mask]
                soft_rank_losses.append(float(self.criterion_2(day_pred, day_target).detach().cpu().item()))
                pairwise_losses.append(float(self.criterion_3(day_pred, day_target).detach().cpu().item()))
            if soft_rank_losses:
                soft_rank_loss = float(np.mean(soft_rank_losses))
            if pairwise_losses:
                pairwise_loss = float(np.mean(pairwise_losses))
        total_loss = float(
            float(self.config.mse_loss_weight) * mse_loss
            + float(self.config.pearson_loss_weight) * pearson_loss
            + float(self.config.soft_rank_loss_weight) * (0.0 if np.isnan(soft_rank_loss) else soft_rank_loss)
            + float(self.config.pairwise_loss_weight) * (0.0 if np.isnan(pairwise_loss) else pairwise_loss)
        )
        return mse_loss, pearson_loss, soft_rank_loss, pairwise_loss, total_loss

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
        mse_loss, pearson_loss, soft_rank_loss, pairwise_loss, total_loss = self._compute_loss_metrics(
            predictions,
            targets,
            dates,
        )
        daily_ic = float("nan")
        head_daily_ic = float("nan")
        if dates is not None and len(dates) == len(predictions):
            daily_ic = float(daily_rank_ic_mean(targets, predictions, dates))
            head_daily_ic = float(
                head_daily_rank_ic_mean(
                    targets,
                    predictions,
                    dates,
                    top_n=int(self.config.selection_head_top_n),
                )
            )
        return {
            "mse_loss": float(mse_loss),
            "pearson_loss": float(pearson_loss),
            "soft_rank_loss": float(soft_rank_loss),
            "pairwise_loss": float(pairwise_loss),
            "total_loss": float(total_loss),
            "ic": float(rank_ic(targets, predictions)),
            "daily_ic": float(daily_ic),
            "head_daily_ic": float(head_daily_ic),
        }

    def _save_checkpoint(self, path: Path, epoch: int, metrics: dict, model_config: dict) -> None:
        payload = {
            "epoch": int(epoch),
            "metrics": metrics,
            "model_config": model_config,
            "trainer_config": asdict(self.config),
            "trainer_state": {
                "monitor_best_valid_ic": float(self.monitor_best_valid_ic),
                "monitor_best_valid_daily_ic": float(self.monitor_best_valid_daily_ic),
            },
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "device": str(self.device),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: Path) -> dict:
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
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
        self.model.to(self.device)
        self.model.eval()
        return payload

    def _write_trainer_state(self, run_dir: Path, model_config: dict) -> None:
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
