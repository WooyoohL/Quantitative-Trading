from __future__ import annotations

from typing import Any

__all__ = ["AlphaSequenceEncoder", "AlphaTrainer", "TabularTrainer", "TrainerConfig"]


def __getattr__(name: str) -> Any:
    if name == "AlphaSequenceEncoder":
        from models.encoders import AlphaSequenceEncoder

        return AlphaSequenceEncoder
    if name in {"AlphaTrainer", "TrainerConfig"}:
        from models.trainer import AlphaTrainer, TrainerConfig

        return {"AlphaTrainer": AlphaTrainer, "TrainerConfig": TrainerConfig}[name]
    if name == "TabularTrainer":
        from models.tabular_trainer import TabularTrainer

        return TabularTrainer
    raise AttributeError(f"module 'models' has no attribute {name!r}")
