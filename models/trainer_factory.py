from __future__ import annotations

from typing import Any

import torch

from models.encoders import build_model
from models.tabular_trainer import TabularTrainer, is_tabular_model_name
from models.trainer import AlphaTrainer, TrainerConfig


def build_alpha_trainer(
    *,
    input_dim: int,
    seq_len: int,
    feature_columns: list[str],
    model_cfg: dict[str, Any],
    trainer_config: TrainerConfig,
    device: torch.device | None = None,
) -> tuple[Any, dict[str, Any]]:
    if is_tabular_model_name(model_cfg.get("name")):
        trainer = TabularTrainer(
            model_cfg=model_cfg,
            config=trainer_config,
            seq_len=seq_len,
            feature_columns=feature_columns,
        )
        return trainer, dict(trainer.resolved_model_config)

    model, resolved_model_config = build_model(input_dim=input_dim, model_cfg=model_cfg)
    trainer = AlphaTrainer(model=model, config=trainer_config, device=device)
    return trainer, resolved_model_config
