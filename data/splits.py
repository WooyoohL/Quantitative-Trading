from __future__ import annotations

import pandas as pd


def split_anchor_dates(
    frame: pd.DataFrame,
    *,
    train_days: int,
    valid_days: int,
    test_days: int,
) -> dict[str, list[pd.Timestamp]]:
    usable = frame[frame["label"].notna()].copy()
    anchor_dates = sorted(pd.to_datetime(usable["date"]).drop_duplicates())
    needed = int(train_days) + int(valid_days) + int(test_days)
    if len(anchor_dates) < needed:
        raise ValueError(
            f"Not enough anchor dates for split: have {len(anchor_dates)}, need at least {needed}. "
            "Increase local history or reduce rolling windows."
        )

    train_end = len(anchor_dates) - valid_days - test_days
    valid_end = len(anchor_dates) - test_days
    train_dates = list(anchor_dates[train_end - train_days : train_end])
    valid_dates = list(anchor_dates[train_end:valid_end])
    test_dates = list(anchor_dates[valid_end:])
    return {"train": train_dates, "valid": valid_dates, "test": test_dates}
