from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from data.fetcher import AkshareFetcher, FetchConfig
from data.processor import FEATURE_COLUMNS, RollingFeatureProcessor
from execution.mock_trader import simulate_rebalance
from models.encoders import SimpleAlphaModel
from models.loss_functions import rank_ic
from strategy.alpha_generator import generate_daily_alpha
from strategy.backtest import backtest_top_k
from strategy.universe_selector import select_training_universe


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_windows(
    df: pd.DataFrame, train_days: int, valid_days: int, test_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 严格按“最近 train/valid/test 天”切分，避免把过旧数据混入训练集。
    dates = sorted(df["date"].unique())
    needed = train_days + valid_days + test_days
    if len(dates) < needed:
        raise ValueError(f"Not enough samples: need {needed} trading days, found {len(dates)}.")

    window_dates = dates[-needed:]
    train_cut = train_days
    valid_cut = train_days + valid_days
    train_dates = set(window_dates[:train_cut])
    valid_dates = set(window_dates[train_cut:valid_cut])
    test_dates = set(window_dates[valid_cut:])

    train_df = df[df["date"].isin(train_dates)].copy()
    valid_df = df[df["date"].isin(valid_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()
    return train_df, valid_df, test_df


def ensure_outputs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    config = load_config(Path("config.yaml"))

    # 日期未显式指定时，默认回看 history_days 天。
    start_date, end_date = AkshareFetcher.resolve_date_range(
        start_date=config.get("start_date"),
        end_date=config.get("end_date"),
        history_days=int(config.get("history_days", 180)),
    )
    fetcher = AkshareFetcher(
        FetchConfig(
            seed=config["seed"],
            use_real_data=bool(config.get("use_real_data", True)),
            fallback_to_synthetic=bool(config.get("fallback_to_synthetic", True)),
        )
    )
    candidate_symbols = config.get("candidate_symbols", config["symbols"])
    raw_df_all = fetcher.fetch_daily_data(
        symbols=candidate_symbols,
        start_date=start_date,
        end_date=end_date,
    )
    universe_cfg = config.get("universe", {})
    if universe_cfg.get("enabled", False):
        # 先从候选池自动筛出训练股票池，再做特征与训练。
        selected_symbols, universe_report = select_training_universe(raw_df_all, universe_cfg)
    else:
        selected_symbols = config["symbols"]
        universe_report = pd.DataFrame({"symbol": selected_symbols})

    if not selected_symbols:
        raise ValueError("Universe selection returned zero symbols. Check universe filters.")

    raw_df = raw_df_all[raw_df_all["symbol"].isin(selected_symbols)].copy()

    processor = RollingFeatureProcessor(label_horizon=1)
    feature_df = processor.transform(raw_df)

    rolling_cfg = config["rolling"]
    train_df, valid_df, test_df = split_windows(
        feature_df,
        train_days=rolling_cfg["train_days"],
        valid_days=rolling_cfg["valid_days"],
        test_days=rolling_cfg["test_days"],
    )

    model_cfg = config["model"]
    model = SimpleAlphaModel(
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=model_cfg["hidden_dim"],
        epochs=model_cfg["epochs"],
        lr=model_cfg["lr"],
        dropout=model_cfg["dropout"],
        l2=model_cfg["l2"],
        use_torch=model_cfg["use_torch"],
    )
    model.fit(train_df[FEATURE_COLUMNS].to_numpy(), train_df["label"].to_numpy())

    # 验证集用于监控泛化质量，测试集用于最终回测表现评估。
    valid_pred = model.predict(valid_df[FEATURE_COLUMNS].to_numpy())
    test_pred = model.predict(test_df[FEATURE_COLUMNS].to_numpy())
    valid_ic = rank_ic(valid_df["label"].to_numpy(), valid_pred)
    test_ic = rank_ic(test_df["label"].to_numpy(), test_pred)

    scored_test = test_df.copy()
    scored_test["score"] = test_pred
    bt_report = backtest_top_k(scored_test, top_k=config["strategy"]["top_k"])

    _, latest_top_k = generate_daily_alpha(
        feature_df=test_df,
        model=model,
        feature_columns=FEATURE_COLUMNS,
        top_k=config["strategy"]["top_k"],
    )
    orders = simulate_rebalance(
        latest_top_k,
        cash=float(config["execution"]["initial_cash"]),
        max_positions=int(config["execution"]["max_positions"]),
    )

    out_dir = Path("outputs")
    ensure_outputs(out_dir)
    universe_report.to_csv(out_dir / "universe_report.csv", index=False)
    latest_top_k.to_csv(out_dir / "top_k.csv", index=False)
    bt_report.to_csv(out_dir / "backtest.csv", index=False)
    orders.to_csv(out_dir / "orders.csv", index=False)

    print(f"Data source: {fetcher.last_source}")
    print(f"Date range: {start_date} -> {end_date}")
    print(f"Model backend: {model.backend}")
    print(f"Universe size: {len(selected_symbols)}")
    print(f"Rows(raw/features): {len(raw_df)}/{len(feature_df)}")
    print(f"Validation IC: {valid_ic:.4f}")
    print(f"Test IC: {test_ic:.4f}")
    if not bt_report.empty:
        print(f"Backtest final equity: {bt_report['equity_curve'].iloc[-1]:.4f}")
    print(f"Top-k and reports saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
