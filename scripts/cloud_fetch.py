from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from data.fetcher import AkshareFetcher, FetchConfig


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch A-share daily data in cloud runner.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument("--output-dir", default="cloud_outputs", help="Output directory.")
    parser.add_argument("--start-date", default=None, help="Override start date, format YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Override end date, format YYYY-MM-DD.")
    parser.add_argument("--history-days", type=int, default=None, help="Override history_days in config.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries for transient network failure.")
    parser.add_argument("--retry-delay", type=float, default=3.0, help="Seconds between retries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    history_days = args.history_days if args.history_days is not None else int(cfg.get("history_days", 520))
    start_date, end_date = AkshareFetcher.resolve_date_range(
        start_date=args.start_date if args.start_date else cfg.get("start_date"),
        end_date=args.end_date if args.end_date else cfg.get("end_date"),
        history_days=history_days,
    )
    symbols = cfg.get("candidate_symbols", cfg.get("symbols", []))
    if not symbols:
        raise ValueError("No symbols found in config (candidate_symbols/symbols).")

    fetcher = AkshareFetcher(
        FetchConfig(
            seed=int(cfg.get("seed", 7)),
            use_real_data=True,
            fallback_to_synthetic=False,
        )
    )

    last_err: Exception | None = None
    raw_df = None
    for attempt in range(1, args.max_retries + 1):
        try:
            raw_df = fetcher.fetch_daily_data(symbols=symbols, start_date=start_date, end_date=end_date)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt == args.max_retries:
                raise
            time.sleep(args.retry_delay)

    if raw_df is None:
        raise RuntimeError(f"Fetch failed unexpectedly: {last_err}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "raw_daily.csv"
    meta_path = out_dir / "fetch_meta.json"
    raw_df.to_csv(raw_path, index=False)
    meta = {
        "source": fetcher.last_source,
        "start_date": start_date,
        "end_date": end_date,
        "n_rows": int(len(raw_df)),
        "n_symbols": int(raw_df["symbol"].nunique()),
        "symbols": symbols,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"source={fetcher.last_source}")
    print(f"date_range={start_date}->{end_date}")
    print(f"rows={len(raw_df)}")
    print(f"symbols={raw_df['symbol'].nunique()}")
    print(f"saved={raw_path.resolve()}")
    print(f"saved={meta_path.resolve()}")


if __name__ == "__main__":
    main()
