from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

try:
    import akshare as ak
except ImportError:  # pragma: no cover
    ak = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


STOCK_EOD_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "amplitude",
    "pct_chg",
    "chg",
    "turnover_rate",
    "outstanding_share",
    "source",
]

STOCK_TURNOVER_COLUMNS = [
    "date",
    "symbol",
    "turnover_rate",
    "outstanding_share",
]

INDEX_EOD_COLUMNS = [
    "date",
    "index_key",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "amplitude",
    "pct_chg",
    "chg",
    "source",
]

INDUSTRY_MAP_COLUMNS = ["symbol", "industry_name", "industry_code", "updated_at", "source"]

INDUSTRY_DAILY_COLUMNS = [
    "date",
    "industry_name",
    "industry_code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "amplitude",
    "pct_chg",
    "chg",
    "turnover_rate",
    "source",
]

NUMERIC_STOCK_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "amplitude",
    "pct_chg",
    "chg",
    "turnover_rate",
    "outstanding_share",
]

NUMERIC_STOCK_TURNOVER_COLUMNS = [
    "turnover_rate",
    "outstanding_share",
]

NUMERIC_INDEX_COLUMNS = ["open", "high", "low", "close", "volume", "turnover", "amplitude", "pct_chg", "chg"]

NUMERIC_INDUSTRY_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "amplitude",
    "pct_chg",
    "chg",
    "turnover_rate",
]


@dataclass
class FetchConfig:
    seed: int = 7
    use_real_data: bool = True
    fallback_to_synthetic: bool = False
    max_workers: int = 4
    request_timeout: float | None = 15.0
    show_progress: bool = True


def _ensure_columns(df: pd.DataFrame, columns: list[str], numeric_columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    special = {"date", "symbol", "index_key", "industry_name", "industry_code", "source", "updated_at"}
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan if column not in special else pd.NA
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out[columns]


def load_local_stock_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=STOCK_EOD_COLUMNS)
    return _ensure_columns(pd.read_csv(path, parse_dates=["date"]), STOCK_EOD_COLUMNS, NUMERIC_STOCK_COLUMNS)


def load_local_index_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=INDEX_EOD_COLUMNS)
    return _ensure_columns(pd.read_csv(path, parse_dates=["date"]), INDEX_EOD_COLUMNS, NUMERIC_INDEX_COLUMNS)


def load_local_industry_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=INDUSTRY_MAP_COLUMNS)
    df = pd.read_csv(path)
    for column in INDUSTRY_MAP_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[INDUSTRY_MAP_COLUMNS]


def load_local_industry_daily(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=INDUSTRY_DAILY_COLUMNS)
    return _ensure_columns(pd.read_csv(path, parse_dates=["date"]), INDUSTRY_DAILY_COLUMNS, NUMERIC_INDUSTRY_COLUMNS)


def save_frame(df: pd.DataFrame, path: Path, columns: list[str], date_columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for column in date_columns or []:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column])
    special = {"date", "symbol", "index_key", "industry_name", "industry_code", "source", "updated_at"}
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan if column not in special else pd.NA
    out[columns].to_csv(path, index=False, encoding="utf-8")


def merge_time_series_frames(existing_df: pd.DataFrame, new_df: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    if existing_df.empty:
        return new_df.copy()
    if new_df.empty:
        return existing_df.copy()
    merged = pd.concat([existing_df, new_df], ignore_index=True)
    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"])
    return merged.drop_duplicates(subset=key_columns, keep="last").sort_values(key_columns).reset_index(drop=True)


def merge_stock_turnover_columns(stock_df: pd.DataFrame, turnover_df: pd.DataFrame) -> pd.DataFrame:
    if stock_df.empty or turnover_df.empty:
        return stock_df.copy()

    out = stock_df.copy()
    supplement = turnover_df.copy()
    out["date"] = pd.to_datetime(out["date"])
    supplement["date"] = pd.to_datetime(supplement["date"])
    supplement = supplement.drop_duplicates(subset=["symbol", "date"], keep="last")

    merged = out.merge(
        supplement,
        on=["symbol", "date"],
        how="left",
        suffixes=("", "_sina"),
    )

    for column in ["turnover_rate", "outstanding_share"]:
        supplement_column = f"{column}_sina"
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
        merged[supplement_column] = pd.to_numeric(merged[supplement_column], errors="coerce")
        merged[column] = merged[column].where(merged[column].notna(), merged[supplement_column])
        merged = merged.drop(columns=[supplement_column])

    return merged.sort_values(["symbol", "date"]).reset_index(drop=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class AkshareFetcher:
    def __init__(self, config: FetchConfig | None = None) -> None:
        self.config = config or FetchConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.last_source = "unknown"

    @staticmethod
    def resolve_date_range(start_date: str | None, end_date: str | None, history_days: int = 520) -> tuple[str, str]:
        end = pd.to_datetime(end_date).date() if end_date else date.today()
        start = pd.to_datetime(start_date).date() if start_date else (end - timedelta(days=history_days))
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    @staticmethod
    def normalize_symbol(code: str) -> str:
        token = str(code).strip().upper()
        if "." in token:
            return token
        if token.startswith(("600", "601", "603", "605")):
            return f"{token}.SH"
        if token.startswith(("000", "001", "002")):
            return f"{token}.SZ"
        return token

    @staticmethod
    def is_main_board_symbol(symbol: str) -> bool:
        return symbol.split(".")[0].startswith(("600", "601", "603", "605", "000", "001", "002"))

    @staticmethod
    def _first_column(df: pd.DataFrame, names: list[str]) -> str:
        for name in names:
            if name in df.columns:
                return name
        raise KeyError(f"None of columns {names} found in {list(df.columns)}")

    @staticmethod
    def _identity_tqdm(iterable, *args, **kwargs):
        return iterable

    @contextmanager
    def _progress_bar(self, total: int, desc: str):
        bar = tqdm(total=total, desc=desc, unit="item", dynamic_ncols=True, leave=True) if self.config.show_progress and tqdm is not None else None
        try:
            yield bar
        finally:
            if bar is not None:
                bar.close()

    @contextmanager
    def _suppress_tx_inner_progress(self):
        if ak is None:
            yield
            return
        try:
            import akshare.index.index_stock_zh as index_module
            import akshare.stock_feature.stock_hist_tx as stock_tx_module
        except Exception:
            yield
            return
        original_index_tqdm = getattr(index_module, "get_tqdm", None)
        original_stock_tqdm = getattr(stock_tx_module, "get_tqdm", None)
        if original_index_tqdm is not None:
            index_module.get_tqdm = lambda: self._identity_tqdm
        if original_stock_tqdm is not None:
            stock_tx_module.get_tqdm = lambda: self._identity_tqdm
        try:
            yield
        finally:
            if original_index_tqdm is not None:
                index_module.get_tqdm = original_index_tqdm
            if original_stock_tqdm is not None:
                stock_tx_module.get_tqdm = original_stock_tqdm

    def _collect_items(self, items: list, fetch_one, desc: str, unit: str, max_workers: int | None = None) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        errors: list[str] = []
        empty_items: list[str] = []
        total = len(items)
        if total == 0:
            return pd.DataFrame()
        worker_limit = int(self.config.max_workers) if max_workers is None else int(max_workers)
        max_workers = max(1, min(worker_limit, total))
        with self._progress_bar(total=total, desc=desc) as bar:
            if max_workers == 1:
                try:
                    for item in items:
                        try:
                            frame = fetch_one(item)
                            if frame is not None and not frame.empty:
                                frames.append(frame)
                            else:
                                empty_items.append(str(item))
                        except Exception as exc:
                            errors.append(f"{item}: {exc}")
                        if bar is not None:
                            bar.update(1)
                            bar.set_postfix_str(f"ok={len(frames)} empty={len(empty_items)} fail={len(errors)}")
                except KeyboardInterrupt:
                    if bar is not None:
                        bar.write(f"[DataFetch] {desc} interrupted by user.")
                    raise
            else:
                executor = ThreadPoolExecutor(max_workers=max_workers)
                future_map = {}
                try:
                    future_map = {executor.submit(fetch_one, item): item for item in items}
                    for future in as_completed(future_map):
                        item = future_map[future]
                        try:
                            frame = future.result()
                            if frame is not None and not frame.empty:
                                frames.append(frame)
                            else:
                                empty_items.append(str(item))
                        except Exception as exc:
                            errors.append(f"{item}: {exc}")
                        if bar is not None:
                            bar.update(1)
                            bar.set_postfix_str(f"ok={len(frames)} empty={len(empty_items)} fail={len(errors)}")
                except KeyboardInterrupt:
                    for future in future_map:
                        future.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    if bar is not None:
                        bar.write(f"[DataFetch] {desc} interrupted by user. Pending tasks cancelled.")
                    raise
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)
        if not frames:
            detail_parts: list[str] = []
            if errors:
                detail_parts.append(" | ".join(errors[:5]))
            if empty_items:
                detail_parts.append(f"empty samples: {', '.join(empty_items[:5])}")
            detail = " | ".join(detail_parts) if detail_parts else "no rows returned"
            if empty_items and not errors:
                raise RuntimeError(f"{desc} failed for all {unit}s. no rows returned | {detail}")
            raise RuntimeError(f"{desc} failed for all {unit}s. {detail}")
        if empty_items:
            print(f"[DataFetch] {desc} empty results: {len(empty_items)}/{total}")
            for preview in empty_items[:5]:
                print(f"[DataFetch] empty {unit}: {preview}")
        if errors:
            print(f"[DataFetch] {desc} partial failures: {len(errors)}/{total}")
            for preview in errors[:5]:
                print(f"[DataFetch] {preview}")
        return pd.concat(frames, ignore_index=True)

    def latest_closed_trade_date(self, as_of: datetime | date | str | None = None) -> pd.Timestamp:
        ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
        ts = ts.tz_localize(None) if ts.tzinfo is not None else ts
        try:
            calendar = self.get_trade_calendar()
            if ts.normalize() in set(calendar) and ts >= ts.normalize() + pd.Timedelta(hours=15):
                return ts.normalize()
            closed_days = calendar[calendar < ts.normalize()]
            if len(closed_days) > 0:
                return closed_days[-1]
        except Exception:
            pass
        return ts.normalize() if ts.hour >= 15 and ts.weekday() < 5 else (ts.normalize() - BDay(1)).normalize()

    def next_trade_date(self, trade_date: pd.Timestamp | str | date) -> pd.Timestamp:
        ts = pd.Timestamp(trade_date).normalize()
        try:
            calendar = self.get_trade_calendar()
            future_days = calendar[calendar > ts]
            if len(future_days) > 0:
                return future_days[0]
        except Exception:
            pass
        return (ts + BDay(1)).normalize()

    def get_trade_calendar(self) -> pd.DatetimeIndex:
        if ak is None:
            raise RuntimeError("akshare is not installed.")
        trade_dates = ak.tool_trade_date_hist_sina()
        return pd.DatetimeIndex(pd.to_datetime(trade_dates[self._first_column(trade_dates, ["trade_date", "日期"])]).sort_values().unique())

    def fetch_stock_universe(self) -> pd.DataFrame:
        if ak is None:
            raise RuntimeError("akshare is not installed.")
        raw = ak.stock_info_a_code_name()
        if raw is None or raw.empty:
            raise RuntimeError("stock_info_a_code_name returned empty result.")
        code_column = self._first_column(raw, ["code", "证券代码"])
        name_column = self._first_column(raw, ["name", "证券简称"])
        universe = pd.DataFrame({"symbol": raw[code_column].astype(str).map(self.normalize_symbol), "name": raw[name_column].astype(str)})
        return universe.dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)

    def fetch_stock_daily_data(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame(columns=STOCK_EOD_COLUMNS)
        if self.config.use_real_data:
            endpoints = [("akshare_tx", self._fetch_stock_daily_tx), ("akshare_daily", self._fetch_stock_daily_sina)]
            errors: list[str] = []
            for source_name, endpoint in endpoints:
                try:
                    out = endpoint(symbols, start_date, end_date)
                    self.last_source = source_name
                    return _ensure_columns(out, STOCK_EOD_COLUMNS, NUMERIC_STOCK_COLUMNS)
                except Exception as exc:
                    errors.append(f"{source_name}: {exc}")
            if not self.config.fallback_to_synthetic:
                raise RuntimeError(" | ".join(errors))
        out = self._fetch_stock_daily_synthetic(symbols, start_date, end_date)
        self.last_source = "synthetic"
        return _ensure_columns(out, STOCK_EOD_COLUMNS, NUMERIC_STOCK_COLUMNS)

    def fetch_daily_data(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        return self.fetch_stock_daily_data(symbols, start_date, end_date)

    def fetch_stock_turnover_data(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame(columns=STOCK_TURNOVER_COLUMNS)
        if not self.config.use_real_data:
            return pd.DataFrame(columns=STOCK_TURNOVER_COLUMNS)
        out = self._fetch_stock_turnover_sina(symbols, start_date, end_date)
        return _ensure_columns(out, STOCK_TURNOVER_COLUMNS, NUMERIC_STOCK_TURNOVER_COLUMNS)

    def _fetch_stock_daily_tx(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        if ak is None:
            raise RuntimeError("akshare is not installed.")

        def fetch_one(symbol: str) -> pd.DataFrame:
            code, exch = symbol.split(".")
            tx_symbol = f"{exch.lower()}{code}"
            df = ak.stock_zh_a_hist_tx(symbol=tx_symbol, start_date=start_date, end_date=end_date, adjust="qfq", timeout=self.config.request_timeout)
            if df is None or df.empty:
                return pd.DataFrame()
            close = pd.to_numeric(df["close"], errors="coerce")
            high = pd.to_numeric(df["high"], errors="coerce")
            low = pd.to_numeric(df["low"], errors="coerce")
            volume = pd.to_numeric(df["amount"], errors="coerce") * 100.0
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(df["date"]),
                    "symbol": symbol,
                    "open": pd.to_numeric(df["open"], errors="coerce"),
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "turnover": volume * close,
                    "amplitude": (high - low) / close.replace(0.0, np.nan),
                    "pct_chg": close.pct_change(),
                    "chg": close.diff(),
                    "turnover_rate": np.nan,
                    "outstanding_share": np.nan,
                    "source": "akshare_tx",
                }
            ).dropna(subset=["date", "open", "high", "low", "close", "volume", "turnover"])

        with self._suppress_tx_inner_progress():
            out = self._collect_items(symbols, fetch_one, desc="Stock TX", unit="symbol")
        return out.sort_values(["symbol", "date"]).reset_index(drop=True)

    def _fetch_stock_daily_sina(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        if ak is None:
            raise RuntimeError("akshare is not installed.")
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        def fetch_one(symbol: str) -> pd.DataFrame:
            code, exch = symbol.split(".")
            df = ak.stock_zh_a_daily(symbol=f"{exch.lower()}{code}", adjust="qfq")
            if df is None or df.empty:
                return pd.DataFrame()
            tmp = df.copy()
            tmp["date"] = pd.to_datetime(tmp["date"])
            tmp = tmp[(tmp["date"] >= start_ts) & (tmp["date"] <= end_ts)]
            if tmp.empty:
                return pd.DataFrame()
            close = pd.to_numeric(tmp["close"], errors="coerce")
            high = pd.to_numeric(tmp["high"], errors="coerce")
            low = pd.to_numeric(tmp["low"], errors="coerce")
            return pd.DataFrame(
                {
                    "date": tmp["date"],
                    "symbol": symbol,
                    "open": pd.to_numeric(tmp["open"], errors="coerce"),
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": pd.to_numeric(tmp["volume"], errors="coerce"),
                    "turnover": pd.to_numeric(tmp["amount"], errors="coerce"),
                    "amplitude": (high - low) / close.replace(0.0, np.nan),
                    "pct_chg": close.pct_change(),
                    "chg": close.diff(),
                    "turnover_rate": pd.to_numeric(tmp["turnover"], errors="coerce"),
                    "outstanding_share": pd.to_numeric(tmp["outstanding_share"], errors="coerce"),
                    "source": "akshare_daily",
                }
            ).dropna(subset=["date", "open", "high", "low", "close", "volume", "turnover"])

        out = self._collect_items(symbols, fetch_one, desc="Stock Daily", unit="symbol")
        return out.sort_values(["symbol", "date"]).reset_index(drop=True)

    def _fetch_stock_turnover_sina(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        if ak is None:
            raise RuntimeError("akshare is not installed.")
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        def fetch_one(symbol: str) -> pd.DataFrame:
            code, exch = symbol.split(".")
            # Sina computes turnover from raw volume/outstanding_share; no price adjustment is needed here.
            df = ak.stock_zh_a_daily(symbol=f"{exch.lower()}{code}", adjust="")
            if df is None or df.empty:
                return pd.DataFrame()
            tmp = df.copy()
            tmp["date"] = pd.to_datetime(tmp["date"])
            tmp = tmp[(tmp["date"] >= start_ts) & (tmp["date"] <= end_ts)]
            if tmp.empty:
                return pd.DataFrame()
            return pd.DataFrame(
                {
                    "date": tmp["date"],
                    "symbol": symbol,
                    "turnover_rate": pd.to_numeric(tmp["turnover"], errors="coerce"),
                    "outstanding_share": pd.to_numeric(tmp["outstanding_share"], errors="coerce"),
                }
            ).dropna(subset=["date"])

        # Sina daily uses JS decoding via py_mini_racer; high parallelism can crash the process with JS OOM.
        out = self._collect_items(symbols, fetch_one, desc="Stock Turnover Sina", unit="symbol", max_workers=1)
        return out.sort_values(["symbol", "date"]).reset_index(drop=True)

    def _fetch_stock_daily_synthetic(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        dates = pd.bdate_range(start=start_date, end=end_date)
        frames: list[pd.DataFrame] = []
        for idx, symbol in enumerate(symbols):
            base_price = 8.0 + 1.5 * idx
            returns = self.rng.normal(0.0005, 0.02, size=len(dates))
            close = base_price * np.cumprod(1.0 + returns)
            open_ = close * (1.0 + self.rng.normal(0.0, 0.004, size=len(dates)))
            high = np.maximum(open_, close) * (1.0 + np.abs(self.rng.normal(0.003, 0.002, size=len(dates))))
            low = np.minimum(open_, close) * (1.0 - np.abs(self.rng.normal(0.003, 0.002, size=len(dates))))
            volume = self.rng.integers(500_000, 8_000_000, size=len(dates)).astype(float)
            turnover = volume * (open_ + close) / 2.0
            frames.append(pd.DataFrame({"date": dates, "symbol": symbol, "open": open_, "high": high, "low": low, "close": close, "volume": volume, "turnover": turnover, "amplitude": (high - low) / np.where(close == 0.0, np.nan, close), "pct_chg": returns, "chg": np.concatenate([[np.nan], np.diff(close)]), "turnover_rate": np.nan, "outstanding_share": np.nan, "source": "synthetic"}))
        out = pd.concat(frames, ignore_index=True)
        out[NUMERIC_STOCK_COLUMNS] = out[NUMERIC_STOCK_COLUMNS].round(6)
        return out

    def fetch_index_daily_data(self, index_symbols: dict[str, str], start_date: str, end_date: str) -> pd.DataFrame:
        if not index_symbols:
            return pd.DataFrame(columns=INDEX_EOD_COLUMNS)
        if self.config.use_real_data:
            endpoints = [("akshare_index_tx", self._fetch_index_daily_tx), ("akshare_index_sina", self._fetch_index_daily_sina)]
            errors: list[str] = []
            for source_name, endpoint in endpoints:
                try:
                    out = endpoint(index_symbols, start_date, end_date)
                    self.last_source = source_name
                    return _ensure_columns(out, INDEX_EOD_COLUMNS, NUMERIC_INDEX_COLUMNS)
                except Exception as exc:
                    errors.append(f"{source_name}: {exc}")
            if not self.config.fallback_to_synthetic:
                raise RuntimeError(" | ".join(errors))
        out = self._fetch_index_daily_synthetic(index_symbols, start_date, end_date)
        self.last_source = "synthetic_index"
        return _ensure_columns(out, INDEX_EOD_COLUMNS, NUMERIC_INDEX_COLUMNS)

    def _fetch_index_daily_tx(self, index_symbols: dict[str, str], start_date: str, end_date: str) -> pd.DataFrame:
        if ak is None:
            raise RuntimeError("akshare is not installed.")
        items = list(index_symbols.items())
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        def fetch_one(item: tuple[str, str]) -> pd.DataFrame:
            index_key, symbol = item
            df = ak.stock_zh_index_daily_tx(symbol=symbol.lower())
            if df is None or df.empty:
                return pd.DataFrame()
            tmp = df.copy()
            tmp["date"] = pd.to_datetime(tmp["date"])
            tmp = tmp[(tmp["date"] >= start_ts) & (tmp["date"] <= end_ts)]
            if tmp.empty:
                return pd.DataFrame()
            close = pd.to_numeric(tmp["close"], errors="coerce")
            high = pd.to_numeric(tmp["high"], errors="coerce")
            low = pd.to_numeric(tmp["low"], errors="coerce")
            return pd.DataFrame({"date": tmp["date"], "index_key": index_key, "symbol": symbol.upper(), "open": pd.to_numeric(tmp["open"], errors="coerce"), "high": high, "low": low, "close": close, "volume": np.nan, "turnover": pd.to_numeric(tmp["amount"], errors="coerce"), "amplitude": (high - low) / close.replace(0.0, np.nan), "pct_chg": close.pct_change(), "chg": close.diff(), "source": "akshare_index_tx"}).dropna(subset=["date", "open", "high", "low", "close"])

        with self._suppress_tx_inner_progress():
            out = self._collect_items(items, fetch_one, desc="Index TX", unit="index")
        return out.sort_values(["index_key", "date"]).reset_index(drop=True)

    def _fetch_index_daily_sina(self, index_symbols: dict[str, str], start_date: str, end_date: str) -> pd.DataFrame:
        if ak is None:
            raise RuntimeError("akshare is not installed.")
        items = list(index_symbols.items())
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        def fetch_one(item: tuple[str, str]) -> pd.DataFrame:
            index_key, symbol = item
            df = ak.stock_zh_index_daily(symbol=symbol.lower())
            if df is None or df.empty:
                return pd.DataFrame()
            tmp = df.copy()
            tmp["date"] = pd.to_datetime(tmp["date"])
            tmp = tmp[(tmp["date"] >= start_ts) & (tmp["date"] <= end_ts)]
            if tmp.empty:
                return pd.DataFrame()
            close = pd.to_numeric(tmp["close"], errors="coerce")
            high = pd.to_numeric(tmp["high"], errors="coerce")
            low = pd.to_numeric(tmp["low"], errors="coerce")
            return pd.DataFrame({"date": tmp["date"], "index_key": index_key, "symbol": symbol.upper(), "open": pd.to_numeric(tmp["open"], errors="coerce"), "high": high, "low": low, "close": close, "volume": pd.to_numeric(tmp["volume"], errors="coerce"), "turnover": np.nan, "amplitude": (high - low) / close.replace(0.0, np.nan), "pct_chg": close.pct_change(), "chg": close.diff(), "source": "akshare_index_sina"}).dropna(subset=["date", "open", "high", "low", "close"])

        out = self._collect_items(items, fetch_one, desc="Index Sina", unit="index")
        return out.sort_values(["index_key", "date"]).reset_index(drop=True)

    def _fetch_index_daily_synthetic(self, index_symbols: dict[str, str], start_date: str, end_date: str) -> pd.DataFrame:
        dates = pd.bdate_range(start=start_date, end=end_date)
        frames: list[pd.DataFrame] = []
        for idx, (index_key, symbol) in enumerate(index_symbols.items()):
            base_price = 2500.0 + idx * 500.0
            returns = self.rng.normal(0.0003, 0.01, size=len(dates))
            close = base_price * np.cumprod(1.0 + returns)
            open_ = close * (1.0 + self.rng.normal(0.0, 0.002, size=len(dates)))
            high = np.maximum(open_, close) * (1.0 + np.abs(self.rng.normal(0.002, 0.001, size=len(dates))))
            low = np.minimum(open_, close) * (1.0 - np.abs(self.rng.normal(0.002, 0.001, size=len(dates))))
            frames.append(pd.DataFrame({"date": dates, "index_key": index_key, "symbol": symbol.upper(), "open": open_, "high": high, "low": low, "close": close, "volume": np.nan, "turnover": np.nan, "amplitude": (high - low) / np.where(close == 0.0, np.nan, close), "pct_chg": returns, "chg": np.concatenate([[np.nan], np.diff(close)]), "source": "synthetic_index"}))
        out = pd.concat(frames, ignore_index=True)
        out[NUMERIC_INDEX_COLUMNS] = out[NUMERIC_INDEX_COLUMNS].round(6)
        return out

    def fetch_industry_map(
        self,
        symbols_filter: set[str] | None = None,
        start_date: str = "20000101",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        if ak is None:
            raise RuntimeError("akshare is not installed.")

        symbols = sorted(symbols_filter or [])
        if not symbols:
            return pd.DataFrame(columns=INDUSTRY_MAP_COLUMNS)

        end_compact = (end_date or pd.Timestamp.today().strftime("%Y%m%d")).replace("-", "")

        def fetch_one(symbol: str) -> pd.DataFrame:
            code = symbol.split(".")[0]
            history = ak.stock_industry_change_cninfo(symbol=code, start_date=start_date, end_date=end_compact)
            if history is None or history.empty:
                return pd.DataFrame()

            latest = history.sort_values("变更日期").iloc[-1]
            industry_name = latest.get("行业中类")
            if pd.isna(industry_name):
                industry_name = latest.get("行业大类")
            if pd.isna(industry_name):
                industry_name = latest.get("行业门类")

            return pd.DataFrame(
                {
                    "symbol": [symbol],
                    "industry_name": [industry_name],
                    "industry_code": [latest.get("行业编码")],
                    "updated_at": [pd.Timestamp.now().isoformat()],
                    "source": ["cninfo_industry_change"],
                }
            ).dropna(subset=["industry_name", "industry_code"])

        out = self._collect_items(symbols, fetch_one, desc="Industry Map", unit="symbol")
        return out.drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)
