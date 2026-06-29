from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rolling_retrain_backtest import main as run_rolling_backtest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the server rolling-retrain backtest preset."
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    if output_dir is None:
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / "retrain_backtest" / f"server_monwed_h12_quality_nopricecap_{stamp}"

    command = [
        "--config",
        str(args.config),
        "--output-dir",
        str(output_dir),
        "--strategies",
        "mon_wed",
        "--hold-periods",
        "1,2",
        "--label-mode",
        "market_excess",
        "--cash-filter",
        "enabled",
        "--cash-filter-policy",
        "quality",
        "--disable-price-cap",
        "--score-quantile",
        "0.50",
        "--topk-mean-quantile",
        "0.45",
        "--score-gap-quantile",
        "0.20",
        "--min-position-exposure",
        "0.05",
        "--mid-position-exposure",
        "0.10",
        "--max-position-exposure",
        "0.175",
        "--max-gross-exposure",
        "0.60",
        "--auto-start-after-warmup",
    ]
    if args.start_date:
        command.extend(["--start-date", args.start_date])
    if args.end_date:
        command.extend(["--end-date", args.end_date])
    if args.resume:
        command.append("--resume")

    run_rolling_backtest(command)


if __name__ == "__main__":
    main()
