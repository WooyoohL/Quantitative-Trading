from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = ["ret_1", "ret_5", "vol_5", "rsi_14", "macd"]


class RollingFeatureProcessor:
    def __init__(self, label_horizon: int = 1) -> None:
        self.label_horizon = label_horizon

    @staticmethod
    def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        # 先按股票、日期排序，确保滚动指标和 shift 标签方向正确。
        raw_df = raw_df.sort_values(["symbol", "date"]).copy()
        groups: list[pd.DataFrame] = []

        for _, group in raw_df.groupby("symbol", sort=False):
            g = group.copy()
            g["ret_1"] = g["close"].pct_change()
            g["ret_5"] = g["close"].pct_change(5)
            g["vol_5"] = g["ret_1"].rolling(5).std()
            ema_12 = g["close"].ewm(span=12, adjust=False).mean()
            ema_26 = g["close"].ewm(span=26, adjust=False).mean()
            g["macd"] = ema_12 - ema_26
            g["rsi_14"] = self._rsi(g["close"], window=14)

            # 标签定义：以 T+1 开盘买入、T+2 开盘卖出收益，贴合 A 股 T+1 约束。
            g["label"] = g["open"].shift(-(self.label_horizon + 1)) / g["open"].shift(-1) - 1.0
            groups.append(g)

        feat_df = pd.concat(groups, ignore_index=True)
        # 去除指标初始化窗口与末尾无标签样本。
        feat_df = feat_df.dropna(subset=FEATURE_COLUMNS + ["label"])
        return feat_df
