from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

try:
    import akshare as ak
except ImportError:  # pragma: no cover
    ak = None


@dataclass
class FetchConfig:
    seed: int = 7
    use_real_data: bool = True
    fallback_to_synthetic: bool = True


class AkshareFetcher:
    """Fetch A-share daily data from AkShare, with synthetic fallback."""

    def __init__(self, config: FetchConfig | None = None) -> None:
        self.config = config or FetchConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.last_source = "unknown"

    @staticmethod
    def resolve_date_range(
        start_date: str | None,
        end_date: str | None,
        history_days: int = 520,
    ) -> tuple[str, str]:
        end = pd.to_datetime(end_date).date() if end_date else date.today()
        start = pd.to_datetime(start_date).date() if start_date else (end - timedelta(days=history_days))
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    def _fetch_daily_data_akshare(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        if ak is None:
            raise RuntimeError("akshare is not installed.")

        frames: list[pd.DataFrame] = []
        start_compact = start_date.replace("-", "")
        end_compact = end_date.replace("-", "")
        for symbol in symbols:
            # AkShare `stock_zh_a_hist` 使用 6 位证券代码（如 600000 / 000001）。
            code = symbol.split(".")[0]
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_compact,
                end_date=end_compact,
                adjust="qfq",
            )
            if df is None or df.empty:
                continue

            # AkShare chinese columns map
            mapped = pd.DataFrame(
                {
                    "date": pd.to_datetime(df["日期"]),
                    "symbol": symbol,
                    "open": pd.to_numeric(df["开盘"], errors="coerce"),
                    "high": pd.to_numeric(df["最高"], errors="coerce"),
                    "low": pd.to_numeric(df["最低"], errors="coerce"),
                    "close": pd.to_numeric(df["收盘"], errors="coerce"),
                    "volume": pd.to_numeric(df["成交量"], errors="coerce"),
                    "turnover": pd.to_numeric(df["成交额"], errors="coerce"),
                }
            )
            frames.append(mapped.dropna())

        if not frames:
            raise RuntimeError("No data returned from AkShare for requested symbols.")
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
        return out

    def _fetch_daily_data_synthetic(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        dates = pd.bdate_range(start=start_date, end=end_date)
        frames: list[pd.DataFrame] = []

        for idx, symbol in enumerate(symbols):
            base_price = 8.0 + 3.5 * idx
            drift = self.rng.normal(0.0006, 0.0008, size=len(dates))
            noise = self.rng.normal(0.0, 0.02, size=len(dates))
            returns = drift + noise

            close = base_price * np.cumprod(1.0 + returns)
            open_ = close * (1.0 + self.rng.normal(0.0, 0.004, size=len(dates)))
            high = np.maximum(open_, close) * (1.0 + np.abs(self.rng.normal(0.003, 0.002, size=len(dates))))
            low = np.minimum(open_, close) * (1.0 - np.abs(self.rng.normal(0.003, 0.002, size=len(dates))))
            volume = self.rng.integers(900_000, 7_500_000, size=len(dates))
            turnover = volume * (open_ + close) / 2.0

            frame = pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "turnover": turnover,
                }
            )
            frames.append(frame)

        out = pd.concat(frames, ignore_index=True)
        price_cols = ["open", "high", "low", "close", "turnover"]
        out[price_cols] = out[price_cols].round(4)
        return out

    def fetch_daily_data(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        if self.config.use_real_data:
            try:
                out = self._fetch_daily_data_akshare(symbols, start_date, end_date)
                self.last_source = "akshare"
                return out
            except Exception:
                if not self.config.fallback_to_synthetic:
                    raise

        out = self._fetch_daily_data_synthetic(symbols, start_date, end_date)
        self.last_source = "synthetic"
        return out
