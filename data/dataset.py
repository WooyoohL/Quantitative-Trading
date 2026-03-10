from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_INDEX_KEYS = ["sse", "szse", "hs300", "zz500", "cyb"]

BASE_FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "ret_10",
    "pct_chg_1",
    "gap_open",
    "intraday_ret",
    "amplitude_1",
    "amplitude_5_mean",
    "range_pct",
    "turnover_rate_1",
    "turnover_rate_missing",
    "volume_ratio_5",
    "turnover_ratio_5",
    "volatility_5",
    "volatility_10",
    "ma_gap_5",
    "ma_gap_10",
    "rsi_14",
    "macd_pct",
]

MARKET_FEATURE_COLUMNS = [
    "market_ret_1_mean",
    "market_ret_5_mean",
    "market_volatility_5_mean",
    "market_breadth_up",
]

INDUSTRY_FEATURE_COLUMNS = [
    "industry_ret_1_mean",
    "industry_ret_5_mean",
    "industry_breadth_up",
    "industry_strength_vs_market",
    "industry_board_pct_chg_1",
    "industry_board_ret_5",
    "industry_board_turnover_rate",
    "industry_board_missing",
    "industry_missing",
]

PEER_FEATURE_COLUMNS = [
    "peer_ret_1_mean",
    "peer_ret_5_mean",
    "peer_amplitude_mean",
    "peer_turnover_rate_mean",
    "peer_top1_ret_1",
    "peer_corr_mean",
    "peer_count",
]


def build_feature_columns(index_keys: list[str] | None = None) -> list[str]:
    keys = index_keys or DEFAULT_INDEX_KEYS
    index_features: list[str] = []
    for index_key in keys:
        index_features.extend([f"idx_{index_key}_ret_1", f"idx_{index_key}_ret_5"])
    return BASE_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS + index_features + INDUSTRY_FEATURE_COLUMNS + PEER_FEATURE_COLUMNS


FEATURE_COLUMNS = build_feature_columns(DEFAULT_INDEX_KEYS)


@dataclass
class FeatureScaler:
    means: pd.Series
    stds: pd.Series

    @classmethod
    def fit(cls, df: pd.DataFrame, feature_columns: list[str]) -> "FeatureScaler":
        means = df[feature_columns].mean(axis=0).fillna(0.0)
        stds = df[feature_columns].std(axis=0).replace(0.0, np.nan).fillna(1.0)
        return cls(means=means, stds=stds)

    def transform(self, df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        out = df.copy()
        out = out.astype({column: np.float32 for column in feature_columns}, copy=False)
        transformed = (out[feature_columns] - self.means[feature_columns]) / self.stds[feature_columns]
        out.loc[:, feature_columns] = transformed.astype(np.float32)
        out.loc[:, feature_columns] = out[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out


class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray | None, meta: pd.DataFrame) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.targets = None if targets is None else np.asarray(targets, dtype=np.float32)
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return int(len(self.features))

    def __getitem__(self, index: int):
        feature_tensor = torch.tensor(self.features[index], dtype=torch.float32)
        if self.targets is None:
            return feature_tensor
        target_tensor = torch.tensor(self.targets[index], dtype=torch.float32)
        return feature_tensor, target_tensor

    @property
    def targets_numpy(self) -> np.ndarray | None:
        return self.targets


@dataclass
class DatasetBundle:
    train_dataset: SequenceDataset
    valid_dataset: SequenceDataset
    test_dataset: SequenceDataset
    inference_dataset: SequenceDataset
    feature_frame: pd.DataFrame
    feature_columns: list[str]
    scaler: FeatureScaler
    split_dates: dict[str, list[pd.Timestamp]]
    peer_map: dict[str, list[tuple[str, float]]]


class AlphaDatasetBuilder:
    def __init__(
        self,
        seq_len: int,
        label_horizon: int = 1,
        index_keys: list[str] | None = None,
        peer_enabled: bool = True,
        peer_top_k: int = 5,
        peer_lookback_days: int = 60,
        peer_min_overlap: int = 20,
        feature_columns: list[str] | None = None,
        verbose: bool = False,
        time_block_shuffle: bool = False,
        time_block_size: int | None = None,
        random_seed: int = 7,
    ) -> None:
        self.seq_len = int(seq_len)
        self.label_horizon = int(label_horizon)
        self.index_keys = list(index_keys or [])
        self.peer_enabled = bool(peer_enabled)
        self.peer_top_k = int(peer_top_k)
        self.peer_lookback_days = int(peer_lookback_days)
        self.peer_min_overlap = int(peer_min_overlap)
        self.feature_columns = list(feature_columns or build_feature_columns(self.index_keys))
        self.verbose = bool(verbose)
        self.time_block_shuffle = bool(time_block_shuffle)
        self.time_block_size = int(time_block_size or self.seq_len)
        self.random_seed = int(random_seed)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[Dataset] {message}")

    def build_bundle(
        self,
        raw_df: pd.DataFrame,
        train_days: int,
        valid_days: int,
        test_days: int,
        index_df: pd.DataFrame | None = None,
        industry_map_df: pd.DataFrame | None = None,
        industry_daily_df: pd.DataFrame | None = None,
        sample_symbols: set[str] | list[str] | None = None,
    ) -> DatasetBundle:
        started_at = perf_counter()
        self._log(
            f"开始构建数据集: raw_rows={len(raw_df)} symbols={raw_df['symbol'].nunique() if not raw_df.empty else 0} "
            f"seq_len={self.seq_len}"
        )

        step_started = perf_counter()
        base_frame = self._build_base_feature_frame(raw_df, industry_map_df)
        self._log(f"基础因子完成: rows={len(base_frame)} 耗时={perf_counter() - step_started:.2f}s")

        step_started = perf_counter()
        feature_frame = self._attach_market_features(base_frame)
        self._log(f"市场特征完成: rows={len(feature_frame)} 耗时={perf_counter() - step_started:.2f}s")

        step_started = perf_counter()
        feature_frame = self._attach_index_features(feature_frame, index_df)
        self._log(f"指数特征完成: rows={len(feature_frame)} 耗时={perf_counter() - step_started:.2f}s")

        step_started = perf_counter()
        feature_frame = self._attach_industry_features(feature_frame, industry_daily_df)
        self._log(f"行业特征完成: rows={len(feature_frame)} 耗时={perf_counter() - step_started:.2f}s")

        step_started = perf_counter()
        feature_frame = self._maybe_shuffle_time_blocks(feature_frame)
        if self.time_block_shuffle:
            self._log(
                f"时间块随机打乱完成: rows={len(feature_frame)} "
                f"block_size={self.time_block_size} 耗时={perf_counter() - step_started:.2f}s"
            )

        step_started = perf_counter()
        split_dates = self._split_anchor_dates(feature_frame, train_days=train_days, valid_days=valid_days, test_days=test_days)
        self._log(
            "锚点日期切分完成: "
            f"train={len(split_dates['train'])} valid={len(split_dates['valid'])} test={len(split_dates['test'])} "
            f"耗时={perf_counter() - step_started:.2f}s"
        )

        step_started = perf_counter()
        peer_map = self._build_peer_map(feature_frame, split_dates["train"])
        self._log(f"peer_map 构建完成: symbols={len(peer_map)} 耗时={perf_counter() - step_started:.2f}s")

        step_started = perf_counter()
        feature_frame = self._attach_peer_features(feature_frame, peer_map)
        self._log(f"peer 特征拼接完成: rows={len(feature_frame)} 耗时={perf_counter() - step_started:.2f}s")

        step_started = perf_counter()
        feature_frame = self._finalize_feature_frame(feature_frame)
        self._log(f"缺失值收口完成: rows={len(feature_frame)} 耗时={perf_counter() - step_started:.2f}s")

        train_cutoff = max(split_dates["train"]) if split_dates["train"] else pd.Timestamp.min
        scaler_fit_frame = feature_frame[feature_frame["date"] <= train_cutoff].copy()

        step_started = perf_counter()
        scaler = FeatureScaler.fit(scaler_fit_frame, self.feature_columns)
        scaled_frame = scaler.transform(feature_frame, self.feature_columns)
        self._log(
            f"标准化完成: rows={len(scaled_frame)} feature_dim={len(self.feature_columns)} "
            f"耗时={perf_counter() - step_started:.2f}s"
        )

        step_started = perf_counter()
        train_dataset = self._build_sequence_dataset(
            scaled_frame,
            allowed_dates=split_dates["train"],
            require_label=True,
            sample_symbols=sample_symbols,
        )
        self._log(f"训练集序列完成: samples={len(train_dataset)} 耗时={perf_counter() - step_started:.2f}s")

        step_started = perf_counter()
        valid_dataset = self._build_sequence_dataset(
            scaled_frame,
            allowed_dates=split_dates["valid"],
            require_label=True,
            sample_symbols=sample_symbols,
        )
        self._log(f"验证集序列完成: samples={len(valid_dataset)} 耗时={perf_counter() - step_started:.2f}s")

        step_started = perf_counter()
        test_dataset = self._build_sequence_dataset(
            scaled_frame,
            allowed_dates=split_dates["test"],
            require_label=True,
            sample_symbols=sample_symbols,
        )
        self._log(f"测试集序列完成: samples={len(test_dataset)} 耗时={perf_counter() - step_started:.2f}s")

        latest_signal_date = pd.to_datetime(scaled_frame["date"]).max()
        step_started = perf_counter()
        inference_dataset = self._build_sequence_dataset(
            scaled_frame,
            allowed_dates=[latest_signal_date],
            require_label=False,
            sample_symbols=sample_symbols,
        )
        self._log(f"推理集序列完成: samples={len(inference_dataset)} 耗时={perf_counter() - step_started:.2f}s")
        self._log(f"数据集构建完成，总耗时 {perf_counter() - started_at:.2f}s")

        return DatasetBundle(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            inference_dataset=inference_dataset,
            feature_frame=scaled_frame,
            feature_columns=self.feature_columns,
            scaler=scaler,
            split_dates=split_dates,
            peer_map=peer_map,
        )

    def build_inference_dataset(
        self,
        raw_df: pd.DataFrame,
        scaler: FeatureScaler,
        index_df: pd.DataFrame | None = None,
        industry_map_df: pd.DataFrame | None = None,
        industry_daily_df: pd.DataFrame | None = None,
        peer_map: dict[str, list[tuple[str, float]]] | None = None,
        signal_date: pd.Timestamp | None = None,
    ) -> tuple[SequenceDataset, pd.DataFrame]:
        if raw_df.empty:
            raise ValueError("Raw stock data is empty.")

        base_frame = self._build_base_feature_frame(raw_df, industry_map_df)
        feature_frame = self._attach_market_features(base_frame)
        feature_frame = self._attach_index_features(feature_frame, index_df)
        feature_frame = self._attach_industry_features(feature_frame, industry_daily_df)
        feature_frame = self._attach_peer_features(feature_frame, peer_map or {})
        feature_frame = self._finalize_feature_frame(feature_frame)
        scaled_frame = scaler.transform(feature_frame, self.feature_columns)

        target_signal_date = pd.Timestamp(signal_date).normalize() if signal_date is not None else pd.to_datetime(scaled_frame["date"]).max()
        inference_dataset = self._build_sequence_dataset(
            scaled_frame,
            allowed_dates=[target_signal_date],
            require_label=False,
        )
        return inference_dataset, scaled_frame

    def _build_base_feature_frame(self, raw_df: pd.DataFrame, industry_map_df: pd.DataFrame | None) -> pd.DataFrame:
        if raw_df.empty:
            raise ValueError("Raw stock data is empty.")

        frame = raw_df.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["symbol"] = frame["symbol"].astype(str)
        frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)

        grouped = frame.groupby("symbol", group_keys=False)
        close = grouped["close"]
        open_ = frame["open"]
        prev_close = close.shift(1)
        ema12 = close.transform(lambda x: x.ewm(span=12, adjust=False).mean())
        ema26 = close.transform(lambda x: x.ewm(span=26, adjust=False).mean())
        ma5 = close.transform(lambda x: x.rolling(5, min_periods=2).mean())
        ma10 = close.transform(lambda x: x.rolling(10, min_periods=3).mean())

        frame["ret_1"] = close.pct_change()
        frame["ret_5"] = close.pct_change(5)
        frame["ret_10"] = close.pct_change(10)
        frame["pct_chg_1"] = frame["pct_chg"].where(frame["pct_chg"].notna(), frame["ret_1"])
        frame["gap_open"] = open_ / prev_close.replace(0.0, np.nan) - 1.0
        frame["intraday_ret"] = frame["close"] / open_.replace(0.0, np.nan) - 1.0
        frame["amplitude_1"] = frame["amplitude"].where(
            frame["amplitude"].notna(),
            (frame["high"] - frame["low"]) / frame["close"].replace(0.0, np.nan),
        )
        frame["amplitude_5_mean"] = frame.groupby("symbol")["amplitude_1"].transform(
            lambda x: x.rolling(5, min_periods=2).mean()
        )
        frame["range_pct"] = (frame["high"] - frame["low"]) / frame["close"].replace(0.0, np.nan)
        frame["turnover_rate_1"] = frame["turnover_rate"]
        frame["turnover_rate_missing"] = frame["turnover_rate_1"].isna().astype(float)
        frame["volume_ratio_5"] = frame["volume"] / frame.groupby("symbol")["volume"].transform(
            lambda x: x.rolling(5, min_periods=2).mean()
        )
        frame["turnover_ratio_5"] = frame["turnover"] / frame.groupby("symbol")["turnover"].transform(
            lambda x: x.rolling(5, min_periods=2).mean()
        )
        frame["volatility_5"] = frame.groupby("symbol")["ret_1"].transform(lambda x: x.rolling(5, min_periods=3).std())
        frame["volatility_10"] = frame.groupby("symbol")["ret_1"].transform(
            lambda x: x.rolling(10, min_periods=5).std()
        )
        frame["ma_gap_5"] = frame["close"] / ma5.replace(0.0, np.nan) - 1.0
        frame["ma_gap_10"] = frame["close"] / ma10.replace(0.0, np.nan) - 1.0
        frame["rsi_14"] = grouped["close"].transform(self._compute_rsi)
        frame["macd_pct"] = (ema12 - ema26) / frame["close"].replace(0.0, np.nan)

        entry_open = grouped["open"].shift(-1)
        exit_open = grouped["open"].shift(-(self.label_horizon + 1))
        # 标签固定定义为：T 收盘出信号，T+1 开盘买入，T+1+h 开盘卖出。
        frame["label"] = exit_open / entry_open.replace(0.0, np.nan) - 1.0

        if industry_map_df is not None and not industry_map_df.empty:
            industry_map = (
                industry_map_df[["symbol", "industry_name", "industry_code"]]
                .dropna(subset=["symbol"])
                .drop_duplicates(subset=["symbol"], keep="last")
            )
            frame = frame.merge(industry_map, on="symbol", how="left")
        else:
            frame["industry_name"] = pd.NA
            frame["industry_code"] = pd.NA

        return frame

    def _attach_market_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        market = (
            frame.groupby("date", as_index=False)
            .agg(
                market_ret_1_mean=("ret_1", "mean"),
                market_ret_5_mean=("ret_5", "mean"),
                market_volatility_5_mean=("volatility_5", "mean"),
                market_breadth_up=("ret_1", lambda x: np.mean((x.fillna(0.0) > 0.0).astype(float))),
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
        return frame.merge(market, on="date", how="left")

    def _attach_index_features(self, frame: pd.DataFrame, index_df: pd.DataFrame | None) -> pd.DataFrame:
        out = frame.copy()
        for index_key in self.index_keys:
            out[f"idx_{index_key}_ret_1"] = 0.0
            out[f"idx_{index_key}_ret_5"] = 0.0

        if index_df is None or index_df.empty or not self.index_keys:
            return out

        idx = index_df.copy()
        idx["date"] = pd.to_datetime(idx["date"])
        idx = idx.sort_values(["index_key", "date"]).reset_index(drop=True)
        idx["idx_ret_1"] = idx.groupby("index_key")["close"].pct_change()
        idx["idx_ret_5"] = idx.groupby("index_key")["close"].pct_change(5)

        ret1_pivot = idx.pivot_table(index="date", columns="index_key", values="idx_ret_1")
        ret5_pivot = idx.pivot_table(index="date", columns="index_key", values="idx_ret_5")
        if not ret1_pivot.empty:
            ret1_pivot = ret1_pivot.rename(columns={col: f"idx_{col}_ret_1" for col in ret1_pivot.columns}).reset_index()
            out = out.merge(ret1_pivot, on="date", how="left")
        if not ret5_pivot.empty:
            ret5_pivot = ret5_pivot.rename(columns={col: f"idx_{col}_ret_5" for col in ret5_pivot.columns}).reset_index()
            out = out.merge(ret5_pivot, on="date", how="left")

        for index_key in self.index_keys:
            col_ret_1 = f"idx_{index_key}_ret_1"
            col_ret_5 = f"idx_{index_key}_ret_5"
            if col_ret_1 not in out.columns:
                out[col_ret_1] = 0.0
            if col_ret_5 not in out.columns:
                out[col_ret_5] = 0.0
            out[col_ret_1] = pd.to_numeric(out[col_ret_1], errors="coerce").fillna(0.0)
            out[col_ret_5] = pd.to_numeric(out[col_ret_5], errors="coerce").fillna(0.0)

        return out

    def _attach_industry_features(self, frame: pd.DataFrame, industry_daily_df: pd.DataFrame | None) -> pd.DataFrame:
        out = frame.copy()
        industry_agg = (
            out.groupby(["date", "industry_name"], as_index=False)
            .agg(
                industry_ret_1_mean=("ret_1", "mean"),
                industry_ret_5_mean=("ret_5", "mean"),
                industry_breadth_up=("ret_1", lambda x: np.mean((x.fillna(0.0) > 0.0).astype(float))),
            )
        )
        out = out.merge(industry_agg, on=["date", "industry_name"], how="left")
        out["industry_strength_vs_market"] = out["industry_ret_1_mean"] - out["market_ret_1_mean"]
        out["industry_missing"] = out["industry_name"].isna().astype(float)

        if industry_daily_df is not None and not industry_daily_df.empty:
            industry_daily = industry_daily_df.copy()
            industry_daily["date"] = pd.to_datetime(industry_daily["date"])
            industry_daily = industry_daily.sort_values(["industry_code", "date"]).reset_index(drop=True)
            industry_daily["industry_board_pct_chg_1"] = industry_daily["pct_chg"].where(
                industry_daily["pct_chg"].notna(),
                industry_daily.groupby("industry_code")["close"].pct_change(),
            )
            industry_daily["industry_board_ret_5"] = industry_daily.groupby("industry_code")["close"].pct_change(5)
            industry_daily["industry_board_turnover_rate"] = industry_daily["turnover_rate"]
            industry_daily["industry_board_missing"] = industry_daily["close"].isna().astype(float)
            board_features = industry_daily[
                [
                    "date",
                    "industry_code",
                    "industry_board_pct_chg_1",
                    "industry_board_ret_5",
                    "industry_board_turnover_rate",
                    "industry_board_missing",
                ]
            ].copy()
            out = out.merge(board_features, on=["date", "industry_code"], how="left")
        else:
            out["industry_board_pct_chg_1"] = np.nan
            out["industry_board_ret_5"] = np.nan
            out["industry_board_turnover_rate"] = np.nan
            out["industry_board_missing"] = 1.0

        out["industry_ret_1_mean"] = out["industry_ret_1_mean"].fillna(out["market_ret_1_mean"])
        out["industry_ret_5_mean"] = out["industry_ret_5_mean"].fillna(out["market_ret_5_mean"])
        out["industry_breadth_up"] = out["industry_breadth_up"].fillna(out["market_breadth_up"])
        out["industry_strength_vs_market"] = out["industry_strength_vs_market"].fillna(0.0)
        out["industry_board_pct_chg_1"] = out["industry_board_pct_chg_1"].fillna(out["industry_ret_1_mean"])
        out["industry_board_ret_5"] = out["industry_board_ret_5"].fillna(out["industry_ret_5_mean"])
        out["industry_board_turnover_rate"] = pd.to_numeric(out["industry_board_turnover_rate"], errors="coerce").fillna(0.0)
        out["industry_board_missing"] = pd.to_numeric(out["industry_board_missing"], errors="coerce").fillna(1.0)
        return out

    def _maybe_shuffle_time_blocks(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.time_block_shuffle or frame.empty:
            return frame

        rng = np.random.default_rng(self.random_seed)
        block_size = max(1, int(self.time_block_size))
        shuffled_groups: list[pd.DataFrame] = []

        for symbol, symbol_df in frame.groupby("symbol", sort=False):
            group = symbol_df.sort_values("date").reset_index(drop=True)
            ordered_dates = pd.to_datetime(group["date"]).to_numpy()
            blocks = [group.iloc[start : start + block_size].copy() for start in range(0, len(group), block_size)]

            if len(blocks) <= 1:
                shuffled_groups.append(group)
                continue

            shuffled = pd.concat([blocks[idx] for idx in rng.permutation(len(blocks))], ignore_index=True)
            # 只打乱 20 天大块在时间轴上的位置，块内相对顺序保持不变。
            shuffled["date"] = ordered_dates[: len(shuffled)]
            shuffled_groups.append(shuffled)

        out = pd.concat(shuffled_groups, ignore_index=True)
        return out.sort_values(["symbol", "date"]).reset_index(drop=True)

    def _build_peer_map(self, frame: pd.DataFrame, train_dates: list[pd.Timestamp]) -> dict[str, list[tuple[str, float]]]:
        if not self.peer_enabled or not train_dates:
            return {}

        train_date_set = set(pd.to_datetime(train_dates))
        peer_frame = frame[frame["date"].isin(train_date_set)].copy()
        if peer_frame.empty:
            return {}

        keep_dates = sorted(peer_frame["date"].drop_duplicates())[-self.peer_lookback_days :]
        peer_frame = peer_frame[peer_frame["date"].isin(set(keep_dates))].copy()
        if peer_frame.empty:
            return {}

        returns_pivot = peer_frame.pivot_table(index="date", columns="symbol", values="ret_1")
        corr_matrix = returns_pivot.corr(min_periods=self.peer_min_overlap)
        if corr_matrix.empty:
            return {}

        symbol_to_industry = (
            peer_frame[["symbol", "industry_name"]]
            .drop_duplicates(subset=["symbol"], keep="last")
            .set_index("symbol")["industry_name"]
            .to_dict()
        )

        peer_map: dict[str, list[tuple[str, float]]] = {}
        for symbol in corr_matrix.columns:
            series = corr_matrix[symbol].drop(labels=[symbol], errors="ignore").dropna()
            if series.empty:
                peer_map[symbol] = []
                continue

            industry_name = symbol_to_industry.get(symbol)
            if pd.notna(industry_name):
                same_industry = [peer for peer in series.index if symbol_to_industry.get(peer) == industry_name]
                if same_industry:
                    series = series.loc[same_industry]

            series = series.sort_values(ascending=False).head(self.peer_top_k)
            peer_map[symbol] = [(peer_symbol, float(corr_value)) for peer_symbol, corr_value in series.items()]

        return peer_map

    def _attach_peer_features(self, frame: pd.DataFrame, peer_map: dict[str, list[tuple[str, float]]]) -> pd.DataFrame:
        out = frame.copy()
        for column in PEER_FEATURE_COLUMNS:
            out[column] = np.nan

        if not peer_map:
            out["peer_count"] = 0.0
            return out

        relation_rows: list[dict[str, object]] = []
        total_symbols = len(peer_map)
        for idx, (symbol, peers) in enumerate(peer_map.items(), start=1):
            if not peers:
                continue
            relation_rows.extend(
                {
                    "symbol": str(symbol),
                    "peer_symbol": str(peer_symbol),
                    "peer_rank": int(rank),
                    "peer_corr": float(corr_value),
                }
                for rank, (peer_symbol, corr_value) in enumerate(peers)
            )
            if self.verbose and (idx == 1 or idx % 50 == 0 or idx == total_symbols):
                self._log(f"peer 特征处理中。 {idx}/{total_symbols} symbols")

        if not relation_rows:
            out["peer_count"] = 0.0
            return out

        relation_df = pd.DataFrame(relation_rows)
        peer_source = out[["date", "symbol", "ret_1", "ret_5", "amplitude_1", "turnover_rate_1"]].copy()
        peer_source = peer_source.rename(columns={"symbol": "peer_symbol"})
        peer_rows_df = relation_df.merge(peer_source, on="peer_symbol", how="left")
        peer_rows_df = peer_rows_df.dropna(subset=["date"])

        if peer_rows_df.empty:
            out["peer_count"] = 0.0
            return out

        aggregated = (
            peer_rows_df.groupby(["symbol", "date"], as_index=False)
            .agg(
                peer_ret_1_mean=("ret_1", "mean"),
                peer_ret_5_mean=("ret_5", "mean"),
                peer_amplitude_mean=("amplitude_1", "mean"),
                peer_turnover_rate_mean=("turnover_rate_1", "mean"),
                peer_corr_mean=("peer_corr", "mean"),
                peer_count=("peer_symbol", "count"),
            )
        )
        top1 = (
            peer_rows_df.sort_values(["symbol", "date", "peer_rank"])
            .groupby(["symbol", "date"], as_index=False)
            .first()[["symbol", "date", "ret_1"]]
            .rename(columns={"ret_1": "peer_top1_ret_1"})
        )
        peer_feature_frame = aggregated.merge(top1, on=["symbol", "date"], how="left")
        out = out.drop(columns=PEER_FEATURE_COLUMNS, errors="ignore").merge(
            peer_feature_frame,
            on=["date", "symbol"],
            how="left",
        )
        return out

        relation_rows: list[dict[str, object]] = []
        total_symbols = len(peer_map)

        for idx, symbol in enumerate(all_symbols, start=1):
            peers = peer_map.get(symbol, [])
            if not peers:
                continue

            peer_rank = {peer_symbol: rank for rank, (peer_symbol, _) in enumerate(peers)}
            peer_corr = {peer_symbol: corr for peer_symbol, corr in peers}
            peer_rows_df = peer_source[peer_source["symbol"].isin(peer_rank)].copy()
            if peer_rows_df.empty:
                continue

            peer_rows_df["peer_rank"] = peer_rows_df["symbol"].map(peer_rank)
            peer_rows_df["peer_corr"] = peer_rows_df["symbol"].map(peer_corr)

            aggregated = (
                peer_rows_df.groupby("date", as_index=False)
                .agg(
                    peer_ret_1_mean=("ret_1", "mean"),
                    peer_ret_5_mean=("ret_5", "mean"),
                    peer_amplitude_mean=("amplitude_1", "mean"),
                    peer_turnover_rate_mean=("turnover_rate_1", "mean"),
                    peer_corr_mean=("peer_corr", "mean"),
                    peer_count=("symbol", "count"),
                )
            )
            top1 = (
                peer_rows_df.sort_values(["date", "peer_rank"])
                .groupby("date", as_index=False)
                .first()[["date", "ret_1"]]
                .rename(columns={"ret_1": "peer_top1_ret_1"})
            )
            aggregated = aggregated.merge(top1, on="date", how="left")
            aggregated["symbol"] = symbol
            values.extend(aggregated.to_dict("records"))

            if self.verbose and (idx == 1 or idx % 50 == 0 or idx == len(all_symbols)):
                self._log(f"peer 特征处理中: {idx}/{len(all_symbols)} symbols")

        peer_feature_frame = pd.DataFrame(values)
        out = out.drop(columns=PEER_FEATURE_COLUMNS, errors="ignore").merge(
            peer_feature_frame,
            on=["date", "symbol"],
            how="left",
        )
        return out

    def _finalize_feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out[self.feature_columns] = out[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        # 缺失值统一在这里收口，避免数据源字段缺失直接打断样本构造。
        fill_map = {
            "turnover_rate_1": 0.0,
            "turnover_rate_missing": 1.0,
            "peer_count": 0.0,
            "peer_corr_mean": 0.0,
            "industry_missing": 1.0,
            "industry_board_missing": 1.0,
        }
        for column in self.feature_columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(fill_map.get(column, 0.0))
        return out.sort_values(["symbol", "date"]).reset_index(drop=True)

    def _split_anchor_dates(
        self,
        frame: pd.DataFrame,
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

    def _build_sequence_dataset(
        self,
        frame: pd.DataFrame,
        allowed_dates: list[pd.Timestamp],
        require_label: bool,
        sample_symbols: set[str] | list[str] | None = None,
    ) -> SequenceDataset:
        allowed_date_set = {pd.Timestamp(value).normalize() for value in allowed_dates}
        allowed_symbols = None if sample_symbols is None else {str(symbol) for symbol in sample_symbols}
        features: list[np.ndarray] = []
        targets: list[float] = []
        meta_rows: list[dict[str, object]] = []

        symbol_groups = list(frame.groupby("symbol"))
        if allowed_symbols is not None:
            symbol_groups = [(symbol, symbol_df) for symbol, symbol_df in symbol_groups if str(symbol) in allowed_symbols]
        for group_idx, (symbol, symbol_df) in enumerate(symbol_groups, start=1):
            group = symbol_df.sort_values("date").reset_index(drop=True)
            feature_values = group[self.feature_columns].to_numpy(dtype=np.float32)
            dates = pd.to_datetime(group["date"]).dt.normalize().to_list()

            for idx in range(self.seq_len - 1, len(group)):
                anchor_date = dates[idx]
                if anchor_date not in allowed_date_set:
                    continue

                label_value = group.at[idx, "label"]
                if require_label and pd.isna(label_value):
                    continue

                window = feature_values[idx - self.seq_len + 1 : idx + 1]
                if len(window) != self.seq_len:
                    continue

                features.append(window)
                if not pd.isna(label_value):
                    targets.append(float(label_value))

                meta_rows.append(
                    {
                        "date": anchor_date,
                        "signal_date": anchor_date,
                        "symbol": symbol,
                        "label": float(label_value) if not pd.isna(label_value) else np.nan,
                        "close": float(group.at[idx, "close"]),
                        "industry_name": group.at[idx, "industry_name"] if "industry_name" in group.columns else pd.NA,
                    }
                )

            if self.verbose and (group_idx == 1 or group_idx % 50 == 0 or group_idx == len(symbol_groups)):
                split_name = "带标签数据集" if require_label else "推理数据集"
                self._log(f"{split_name} 构建中: {group_idx}/{len(symbol_groups)} symbols")

        targets_array = np.asarray(targets, dtype=np.float32) if require_label else None
        meta_df = pd.DataFrame(meta_rows)
        if not features:
            empty = np.empty((0, self.seq_len, len(self.feature_columns)), dtype=np.float32)
            return SequenceDataset(empty, targets_array, meta_df)
        feature_array = np.stack(features).astype(np.float32)
        return SequenceDataset(feature_array, targets_array, meta_df)

    @staticmethod
    def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return (rsi / 100.0).clip(lower=0.0, upper=1.0)
