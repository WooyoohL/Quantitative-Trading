from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import SequenceDataset
from models.loss_functions import rank_ic, PearsonLoss


@dataclass
class TrainerConfig:
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 256
    eval_batch_size: int = 512
    log_every: int = 1
    early_stopping_patience: int = 10
    num_workers: int = 0
    seed: int = 7


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
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
        )
        self.history: list[dict[str, float | int]] = []
        self.best_epoch = 0
        self.best_valid_ic = float("-inf")

    def fit(
        self,
        train_dataset: SequenceDataset,
        valid_dataset: SequenceDataset,
        run_dir: Path,
        model_config: dict,
    ) -> pd.DataFrame:
        run_dir.mkdir(parents=True, exist_ok=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(self.config.batch_size),
            shuffle=True,
            num_workers=int(self.config.num_workers),
            drop_last=False
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=int(self.config.eval_batch_size),
            shuffle=False,
            num_workers=int(self.config.num_workers),
            drop_last=False
        )

        # 用验证集 IC 选择 best checkpoint，而不是只看训练损失。
        epochs_without_improvement = 0
        for epoch in range(1, int(self.config.epochs) + 1):
            train_loss = self._train_epoch(train_loader)
            train_pred = self.predict_dataset(train_dataset)
            valid_pred = self.predict_loader(valid_loader)

            train_target = train_dataset.targets_numpy
            valid_target = valid_dataset.targets_numpy

            train_ic = 0.0 if train_target is None else rank_ic(train_target, train_pred)
            valid_loss = float("nan")
            valid_ic = float("nan")
            if valid_target is not None and len(valid_target) > 0:
                valid_loss = float(np.mean((valid_pred - valid_target) ** 2))
                valid_ic = rank_ic(valid_target, valid_pred)

            record = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_ic": float(train_ic),
                "valid_loss": float(valid_loss),
                "valid_ic": float(valid_ic),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }
            self.history.append(record)

            if valid_ic > self.best_valid_ic:
                self.best_valid_ic = float(valid_ic)
                self.best_epoch = int(epoch)
                epochs_without_improvement = 0
                self._save_checkpoint(
                    run_dir / "best.ckpt",
                    epoch=epoch,
                    metrics=record,
                    model_config=model_config,
                )
            else:
                epochs_without_improvement += 1

            self._save_checkpoint(
                run_dir / "last.ckpt",
                epoch=epoch,
                metrics=record,
                model_config=model_config,
            )

            if epoch == 1 or epoch % int(self.config.log_every) == 0 or epoch == int(self.config.epochs):
                print(
                    f"[Train] epoch={epoch:03d}/{self.config.epochs} "
                    f"train_loss={train_loss:.6f} train_ic={train_ic:.4f} "
                    f"valid_loss={valid_loss:.6f} valid_ic={valid_ic:.4f}"
                )

            if epochs_without_improvement >= int(self.config.early_stopping_patience):
                print(f"[Train] early stopping at epoch={epoch}, best_valid_ic={self.best_valid_ic:.4f}")
                break

        history_df = pd.DataFrame(self.history)
        # 训练过程完整落盘，后续可以直接画 loss 和 IC 曲线。
        history_df.to_csv(run_dir / "train_metrics.csv", index=False, encoding="utf-8-sig")
        self._write_trainer_state(run_dir, model_config=model_config)
        self.load_checkpoint(run_dir / "best.ckpt")
        return history_df

    def _train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_items = 0

        for features, targets in train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)

            # 标准 PyTorch 训练流程：forward -> loss -> backward -> step。
            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.model(features)
            loss = self.criterion(predictions, targets)
            loss.backward()
            self.optimizer.step()

            batch_size = int(features.size(0))
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size

        return total_loss / max(1, total_items)

    def predict_dataset(self, dataset: SequenceDataset) -> np.ndarray:
        loader = DataLoader(
            dataset,
            batch_size=int(self.config.eval_batch_size),
            shuffle=False,
            num_workers=int(self.config.num_workers),
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

    def _save_checkpoint(self, path: Path, epoch: int, metrics: dict, model_config: dict) -> None:
        payload = {
            "epoch": int(epoch),
            "metrics": metrics,
            "model_config": model_config,
            "trainer_config": asdict(self.config),
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
        self.model.to(self.device)
        self.model.eval()
        return payload

    def _write_trainer_state(self, run_dir: Path, model_config: dict) -> None:
        payload = {
            "best_epoch": int(self.best_epoch),
            "best_valid_ic": float(self.best_valid_ic),
            "trainer_config": asdict(self.config),
            "model_config": model_config,
        }
        (run_dir / "trainer_state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
