from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from app.interrupts import run_cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export daily top-k picks from a test_predictions.csv file.")
    parser.add_argument("csv_path", type=Path, help="Path to test_predictions.csv")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k symbols to keep for each test date. Defaults to strategy.top_k in config.yaml.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Config file used to resolve default strategy.top_k.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="test_top_k_daily.csv",
        help="Output file name saved to the same folder as the input csv.",
    )
    return parser.parse_args()


def load_default_top_k(config_path: Path) -> int:
    if not config_path.exists():
        return 3
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return int(config.get("strategy", {}).get("top_k", 3))


def export_daily_top_k(csv_path: Path, top_k: int, output_name: str) -> Path:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"date", "symbol", "score"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["date"] = pd.to_datetime(df["date"])
    daily_top_k = (
        df.sort_values(["date", "score"], ascending=[True, False])
        .groupby("date", group_keys=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    # 给每天的入选股票加上名次，方便直接查看当日推荐顺序。
    daily_top_k["rank"] = daily_top_k.groupby("date")["score"].rank(method="first", ascending=False).astype(int)
    ordered_columns = ["date", "rank", "symbol", "score"] + [
        column for column in daily_top_k.columns if column not in {"date", "rank", "symbol", "score"}
    ]
    daily_top_k = daily_top_k[ordered_columns]
    daily_top_k["date"] = pd.to_datetime(daily_top_k["date"]).dt.date.astype(str)

    output_path = csv_path.parent / output_name
    daily_top_k.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main() -> None:
    args = parse_args()
    top_k = int(args.top_k or load_default_top_k(args.config))
    output_path = export_daily_top_k(args.csv_path, top_k=top_k, output_name=args.output_name)
    print(f"Input: {args.csv_path.resolve()}")
    print(f"Top-k: {top_k}")
    print(f"Output: {output_path.resolve()}")


if __name__ == "__main__":
    run_cli(main, label="ExportTopK")
