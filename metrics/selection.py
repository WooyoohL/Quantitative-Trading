from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SelectionResult:
    epoch: int
    selection_score: float
    candidate_count: int
    breakdown: dict[str, float | int]


def topk_valid_rank_columns() -> list[str]:
    return [
        "valid_excess_return",
        "valid_top_k_return",
        "valid_positive_excess_rate",
        "valid_head_daily_ic",
        "valid_relative_return",
        "valid_daily_ic",
        "valid_max_drawdown",
    ]


def build_topk_valid_monitor_tuple(
    metrics: dict[str, float | int],
    *,
    selection_min_excess_return: float,
    selection_min_positive_excess_rate: float,
    selection_min_daily_ic: float,
    selection_min_head_daily_ic: float,
    selection_max_drawdown_limit: float,
) -> tuple[float | int, ...]:
    valid_excess = float(metrics.get("valid_excess_return", float("-inf")))
    valid_top_k = float(metrics.get("valid_top_k_return", float("-inf")))
    valid_pos_excess = float(metrics.get("valid_positive_excess_rate", float("-inf")))
    valid_head_ic = float(metrics.get("valid_head_daily_ic", float("-inf")))
    valid_relative = float(metrics.get("valid_relative_return", float("-inf")))
    valid_daily_ic = float(metrics.get("valid_daily_ic", float("-inf")))
    valid_mdd = float(metrics.get("valid_max_drawdown", float("-inf")))

    return (
        int(valid_excess > float(selection_min_excess_return)),
        int(valid_pos_excess >= float(selection_min_positive_excess_rate)),
        int(valid_daily_ic > float(selection_min_daily_ic)),
        int(valid_head_ic >= float(selection_min_head_daily_ic)),
        int(valid_mdd >= float(selection_max_drawdown_limit)),
        valid_excess,
        valid_top_k,
        valid_pos_excess,
        valid_head_ic,
        valid_relative,
        valid_daily_ic,
        valid_mdd,
    )


def annotate_ic_gate_selection(
    history_df: pd.DataFrame,
    *,
    selection_ic_tolerance: float,
    selection_weight_ic: float,
    selection_weight_top_k_return: float,
    selection_weight_hit_rate: float,
    selection_weight_excess_return: float,
) -> tuple[pd.DataFrame, SelectionResult | None]:
    scored = history_df.copy()
    for column in [
        "selection_candidate",
        "selection_z_ic",
        "selection_z_top_k_return",
        "selection_z_hit_rate",
        "selection_z_excess_return",
        "selection_score",
    ]:
        scored[column] = np.nan

    valid_rows = scored.dropna(subset=["valid_ic"]).copy()
    if valid_rows.empty:
        return scored, None

    best_ic_value = float(valid_rows["valid_ic"].max())
    candidate_mask = scored["valid_ic"] >= best_ic_value - float(selection_ic_tolerance)
    candidates = scored.loc[candidate_mask].copy()
    if candidates.empty:
        return scored, None

    scored.loc[candidate_mask, "selection_candidate"] = 1.0

    z_column_map = {
        "valid_ic": "selection_z_ic",
        "valid_top_k_return": "selection_z_top_k_return",
        "valid_hit_rate": "selection_z_hit_rate",
        "valid_excess_return": "selection_z_excess_return",
    }
    for raw_column, z_column in z_column_map.items():
        series = candidates[raw_column].astype(float)
        std_value = float(series.std(ddof=0)) if len(series) > 1 else 0.0
        if not np.isfinite(std_value) or std_value <= 1e-12:
            z_values = pd.Series(0.0, index=series.index)
        else:
            z_values = (series - float(series.mean())) / std_value
        scored.loc[z_values.index, z_column] = z_values.astype(float)

    scored.loc[candidate_mask, "selection_score"] = (
        float(selection_weight_ic) * scored.loc[candidate_mask, "selection_z_ic"].fillna(0.0)
        + float(selection_weight_top_k_return) * scored.loc[candidate_mask, "selection_z_top_k_return"].fillna(0.0)
        + float(selection_weight_hit_rate) * scored.loc[candidate_mask, "selection_z_hit_rate"].fillna(0.0)
        + float(selection_weight_excess_return) * scored.loc[candidate_mask, "selection_z_excess_return"].fillna(0.0)
    )

    selected = (
        scored.loc[candidate_mask]
        .sort_values(
            ["selection_score", "valid_ic", "valid_top_k_return", "valid_hit_rate", "valid_excess_return", "epoch"],
            ascending=[False, False, False, False, False, True],
        )
        .iloc[0]
    )
    breakdown = {
        "epoch": int(selected["epoch"]),
        "valid_ic": float(selected["valid_ic"]),
        "valid_top_k_return": float(selected["valid_top_k_return"]),
        "valid_hit_rate": float(selected["valid_hit_rate"]),
        "valid_excess_return": float(selected["valid_excess_return"]),
        "z_ic": float(selected["selection_z_ic"]),
        "z_top_k_return": float(selected["selection_z_top_k_return"]),
        "z_hit_rate": float(selected["selection_z_hit_rate"]),
        "z_excess_return": float(selected["selection_z_excess_return"]),
        "contrib_ic": float(selection_weight_ic) * float(selected["selection_z_ic"]),
        "contrib_top_k_return": float(selection_weight_top_k_return) * float(selected["selection_z_top_k_return"]),
        "contrib_hit_rate": float(selection_weight_hit_rate) * float(selected["selection_z_hit_rate"]),
        "contrib_excess_return": float(selection_weight_excess_return) * float(selected["selection_z_excess_return"]),
    }
    return scored, SelectionResult(
        epoch=int(selected["epoch"]),
        selection_score=float(selected["selection_score"]),
        candidate_count=int(candidate_mask.sum()),
        breakdown=breakdown,
    )


def annotate_topk_valid_selection(
    history_df: pd.DataFrame,
    *,
    selection_min_excess_return: float,
    selection_min_positive_excess_rate: float,
    selection_min_daily_ic: float,
    selection_min_head_daily_ic: float,
    selection_max_drawdown_limit: float,
    selection_head_top_n: int,
) -> tuple[pd.DataFrame, SelectionResult | None]:
    scored = history_df.copy()
    scored["selection_candidate"] = 0.0
    scored["selection_score"] = np.nan

    valid_rows = scored.dropna(
        subset=[
            "valid_excess_return",
            "valid_positive_excess_rate",
            "valid_daily_ic",
            "valid_head_daily_ic",
            "valid_max_drawdown",
        ]
    ).copy()
    if valid_rows.empty:
        return scored, None

    candidate_mask = (
        (scored["valid_excess_return"] > float(selection_min_excess_return))
        & (scored["valid_positive_excess_rate"] >= float(selection_min_positive_excess_rate))
        & (scored["valid_daily_ic"] > float(selection_min_daily_ic))
        & (scored["valid_head_daily_ic"] >= float(selection_min_head_daily_ic))
        & (scored["valid_max_drawdown"] >= float(selection_max_drawdown_limit))
    )
    if not candidate_mask.any():
        candidate_mask = scored["valid_daily_ic"] > float(selection_min_daily_ic)
    if not candidate_mask.any():
        candidate_mask = scored["valid_excess_return"] > float(selection_min_excess_return)
    if not candidate_mask.any():
        candidate_mask = scored["valid_daily_ic"].notna()

    candidates = scored.loc[candidate_mask].copy()
    if candidates.empty:
        return scored, None

    scored.loc[candidate_mask, "selection_candidate"] = 1.0
    rank_columns = topk_valid_rank_columns()
    for index, (_, row) in enumerate(
        candidates.sort_values(
            rank_columns + ["epoch"],
            ascending=[False, False, False, False, False, False, False, True],
        ).iterrows(),
        start=1,
    ):
        scored.loc[row.name, "selection_score"] = float(len(candidates) - index + 1)

    selected = (
        scored.loc[candidate_mask]
        .sort_values(
            rank_columns + ["epoch"],
            ascending=[False, False, False, False, False, False, False, True],
        )
        .iloc[0]
    )
    breakdown = {
        "epoch": int(selected["epoch"]),
        "valid_excess_return": float(selected["valid_excess_return"]),
        "valid_positive_excess_rate": float(selected["valid_positive_excess_rate"]),
        "valid_relative_return": float(selected["valid_relative_return"]),
        "valid_daily_ic": float(selected["valid_daily_ic"]),
        "valid_head_daily_ic": float(selected["valid_head_daily_ic"]),
        "valid_top_k_return": float(selected["valid_top_k_return"]),
        "valid_hit_rate": float(selected["valid_hit_rate"]),
        "valid_max_drawdown": float(selected["valid_max_drawdown"]),
        "head_top_n": int(selection_head_top_n),
    }
    return scored, SelectionResult(
        epoch=int(selected["epoch"]),
        selection_score=float(selected["selection_score"]),
        candidate_count=int(candidate_mask.sum()),
        breakdown=breakdown,
    )
