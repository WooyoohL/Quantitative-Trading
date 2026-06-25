from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from metrics.selection import SelectionResult


@dataclass
class CheckpointMonitor:
    mode: str
    best_topk_monitor_tuple: tuple[float | int, ...] | None = None

    @classmethod
    def from_config(cls, config: Any) -> "CheckpointMonitor":
        return cls(mode=str(config.checkpoint_selection_mode).strip().lower())

    @property
    def uses_composite_selection(self) -> bool:
        return self.mode == "ic_gate_composite"

    @property
    def uses_topk_valid_selection(self) -> bool:
        return self.mode == "topk_valid"

    @property
    def needs_epoch_checkpoints(self) -> bool:
        return self.uses_composite_selection or self.uses_topk_valid_selection

    @property
    def monitors_daily_ic(self) -> bool:
        return self.uses_topk_valid_selection

    @property
    def best_monitor_label(self) -> str:
        return "best_valid_daily_ic" if self.monitors_daily_ic else "best_valid_ic"

    def is_improved(
        self,
        record: dict[str, float | int],
        config: Any,
        *,
        best_valid_ic: float,
        best_valid_daily_ic: float,
    ) -> bool:
        if self.uses_topk_valid_selection:
            from metrics.selection import build_topk_valid_monitor_tuple

            current_topk_tuple = build_topk_valid_monitor_tuple(
                record,
                selection_min_excess_return=float(config.selection_min_excess_return),
                selection_min_positive_excess_rate=float(config.selection_min_positive_excess_rate),
                selection_min_daily_ic=float(config.selection_min_daily_ic),
                selection_min_head_daily_ic=float(config.selection_min_head_daily_ic),
                selection_max_drawdown_limit=float(config.selection_max_drawdown_limit),
            )
            improved = self.best_topk_monitor_tuple is None or current_topk_tuple > self.best_topk_monitor_tuple
            if improved:
                self.best_topk_monitor_tuple = current_topk_tuple
            return bool(improved)

        monitor_value = float(record["valid_daily_ic"] if self.monitors_daily_ic else record["valid_ic"])
        best_monitor_value = float(best_valid_daily_ic if self.monitors_daily_ic else best_valid_ic)
        return bool(monitor_value > best_monitor_value)

    def annotate_history(
        self,
        history_df: "pd.DataFrame",
        config: Any,
    ) -> tuple["pd.DataFrame", "SelectionResult | None"]:
        if self.uses_composite_selection:
            from metrics.selection import annotate_ic_gate_selection

            return annotate_ic_gate_selection(
                history_df,
                selection_ic_tolerance=float(config.selection_ic_tolerance),
                selection_weight_ic=float(config.selection_weight_ic),
                selection_weight_top_k_return=float(config.selection_weight_top_k_return),
                selection_weight_hit_rate=float(config.selection_weight_hit_rate),
                selection_weight_excess_return=float(config.selection_weight_excess_return),
            )
        if self.uses_topk_valid_selection:
            from metrics.selection import annotate_topk_valid_selection

            return annotate_topk_valid_selection(
                history_df,
                selection_min_excess_return=float(config.selection_min_excess_return),
                selection_min_positive_excess_rate=float(config.selection_min_positive_excess_rate),
                selection_min_daily_ic=float(config.selection_min_daily_ic),
                selection_min_head_daily_ic=float(config.selection_min_head_daily_ic),
                selection_max_drawdown_limit=float(config.selection_max_drawdown_limit),
                selection_head_top_n=int(config.selection_head_top_n),
            )
        return history_df, None
