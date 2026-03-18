from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.interrupts import run_cli
from data.fetcher import load_local_stock_data


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model replay returns and compare with actual executed trades.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory that contains top_k.csv.")
    parser.add_argument("--top-k-path", type=Path, default=None, help="Explicit top_k.csv path.")
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols actually bought. Default is to evaluate all rows in top_k.csv.",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=None,
        help="Total cash deployed for this batch. Default uses execution.initial_cash from config.",
    )
    parser.add_argument(
        "--label-horizon",
        type=int,
        default=None,
        help="Holding horizon in model label definition. Default uses data.label_horizon from config.",
    )
    parser.add_argument(
        "--actual-trades-path",
        type=Path,
        default=None,
        help="CSV of actual executed trades for comparison.",
    )
    return parser.parse_args()


def resolve_run_dir(args: argparse.Namespace, config: dict) -> Path:
    if args.run_dir is not None:
        return args.run_dir.resolve()

    latest_run_path = Path(config.get("outputs", {}).get("latest_run_metadata", "outputs/latest_run.json"))
    if not latest_run_path.exists():
        raise FileNotFoundError("latest_run.json not found. Please provide --run-dir or run training first.")

    latest_payload = json.loads(latest_run_path.read_text(encoding="utf-8"))
    run_dir = latest_payload.get("run_dir")
    if not run_dir:
        raise ValueError("latest_run.json does not contain run_dir.")
    return Path(run_dir)


def resolve_top_k_path(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.top_k_path is not None:
        return args.top_k_path.resolve()
    return run_dir / "top_k.csv"


def parse_symbol_filter(symbols_arg: str | None) -> set[str] | None:
    if not symbols_arg:
        return None
    symbols = {token.strip().upper() for token in symbols_arg.split(",") if token.strip()}
    return symbols or None


def normalize_actual_trade_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "symbol",
        "actual_entry_date",
        "actual_entry_price",
        "actual_exit_date",
        "actual_exit_price",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Actual trades CSV is missing required columns: "
            + ", ".join(sorted(missing))
            + ". Required columns are: symbol, actual_entry_date, actual_entry_price, actual_exit_date, actual_exit_price."
        )

    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["actual_entry_date"] = pd.to_datetime(out["actual_entry_date"]).dt.normalize()
    out["actual_exit_date"] = pd.to_datetime(out["actual_exit_date"]).dt.normalize()
    out["actual_entry_price"] = pd.to_numeric(out["actual_entry_price"], errors="coerce")
    out["actual_exit_price"] = pd.to_numeric(out["actual_exit_price"], errors="coerce")
    if "signal_date" in out.columns:
        out["signal_date"] = pd.to_datetime(out["signal_date"]).dt.normalize()
    if "shares" in out.columns:
        out["shares"] = pd.to_numeric(out["shares"], errors="coerce")
    if "actual_entry_amount" in out.columns:
        out["actual_entry_amount"] = pd.to_numeric(out["actual_entry_amount"], errors="coerce")
    return out


def get_trade_rows(symbol_df: pd.DataFrame, next_trade_date: pd.Timestamp, label_horizon: int) -> dict:
    group = symbol_df.sort_values("date").reset_index(drop=True).copy()
    group["date"] = pd.to_datetime(group["date"]).dt.normalize()

    entry_rows = group[group["date"] == next_trade_date].copy()
    if entry_rows.empty:
        return {
            "status": "no_entry_bar",
            "entry_date": None,
            "entry_open": None,
            "exit_date": None,
            "exit_open": None,
            "model_return": None,
        }

    entry_idx = int(entry_rows.index[0])
    exit_idx = entry_idx + int(label_horizon)
    if exit_idx >= len(group):
        return {
            "status": "no_exit_bar",
            "entry_date": str(group.at[entry_idx, "date"].date()),
            "entry_open": float(group.at[entry_idx, "open"]),
            "exit_date": None,
            "exit_open": None,
            "model_return": None,
        }

    entry_open = float(group.at[entry_idx, "open"])
    exit_open = float(group.at[exit_idx, "open"])
    model_return = exit_open / entry_open - 1.0

    return {
        "status": "filled",
        "entry_date": str(group.at[entry_idx, "date"].date()),
        "entry_open": entry_open,
        "exit_date": str(group.at[exit_idx, "date"].date()),
        "exit_open": exit_open,
        "model_return": float(model_return),
    }


def summarize_model_replay(df: pd.DataFrame, cash: float) -> dict:
    filled = df[df["status"] == "filled"].copy()
    if filled.empty:
        return {
            "n_requested": int(len(df)),
            "n_filled": 0,
            "fill_rate": 0.0,
            "win_rate": None,
            "avg_return": None,
            "median_return": None,
            "best_return": None,
            "worst_return": None,
            "portfolio_return_equal_weight": None,
            "portfolio_pnl_equal_weight": None,
            "cash_assumption": float(cash),
        }

    per_position_cash = float(cash) / max(1, len(filled))
    pnl = (filled["model_return"].astype(float) * per_position_cash).sum()
    portfolio_return = pnl / float(cash) if cash > 0 else None

    return {
        "n_requested": int(len(df)),
        "n_filled": int(len(filled)),
        "fill_rate": float(len(filled) / max(1, len(df))),
        "win_rate": float((filled["model_return"].astype(float) > 0.0).mean()),
        "avg_return": float(filled["model_return"].astype(float).mean()),
        "median_return": float(filled["model_return"].astype(float).median()),
        "best_return": float(filled["model_return"].astype(float).max()),
        "worst_return": float(filled["model_return"].astype(float).min()),
        "portfolio_return_equal_weight": float(portfolio_return) if portfolio_return is not None else None,
        "portfolio_pnl_equal_weight": float(pnl),
        "cash_assumption": float(cash),
    }


def build_actual_comparison(model_df: pd.DataFrame, actual_df: pd.DataFrame) -> pd.DataFrame:
    actual = actual_df.copy()
    actual["actual_return"] = actual["actual_exit_price"] / actual["actual_entry_price"] - 1.0

    if "signal_date" in actual.columns and "signal_date" in model_df.columns:
        merged = model_df.merge(actual, on=["symbol", "signal_date"], how="left", suffixes=("", "_actual"))
    else:
        merged = model_df.merge(actual, on=["symbol"], how="left", suffixes=("", "_actual"))

    merged["entry_slippage_pct"] = merged["actual_entry_price"] / merged["entry_open"] - 1.0
    merged["exit_slippage_pct"] = merged["actual_exit_price"] / merged["exit_open"] - 1.0
    merged["return_gap"] = merged["actual_return"] - merged["model_return"]

    if "shares" in merged.columns:
        merged["actual_pnl"] = (merged["actual_exit_price"] - merged["actual_entry_price"]) * merged["shares"]
    elif "actual_entry_amount" in merged.columns:
        merged["actual_pnl"] = merged["actual_entry_amount"] * merged["actual_return"]
    else:
        merged["actual_pnl"] = pd.NA

    return merged


def summarize_actual_comparison(df: pd.DataFrame, cash: float) -> dict:
    matched = df[df["actual_return"].notna()].copy()
    if matched.empty:
        return {
            "n_model_rows": int(len(df)),
            "n_actual_matched": 0,
            "match_rate": 0.0,
            "actual_win_rate": None,
            "actual_avg_return": None,
            "actual_median_return": None,
            "actual_best_return": None,
            "actual_worst_return": None,
            "model_avg_return_matched": None,
            "avg_return_gap": None,
            "avg_entry_slippage_pct": None,
            "avg_exit_slippage_pct": None,
            "actual_portfolio_return_equal_weight": None,
            "actual_portfolio_pnl_equal_weight": None,
            "cash_assumption": float(cash),
        }

    per_position_cash = float(cash) / max(1, len(matched))
    pnl_equal_weight = (matched["actual_return"].astype(float) * per_position_cash).sum()
    portfolio_return = pnl_equal_weight / float(cash) if cash > 0 else None

    summary = {
        "n_model_rows": int(len(df)),
        "n_actual_matched": int(len(matched)),
        "match_rate": float(len(matched) / max(1, len(df))),
        "actual_win_rate": float((matched["actual_return"].astype(float) > 0.0).mean()),
        "actual_avg_return": float(matched["actual_return"].astype(float).mean()),
        "actual_median_return": float(matched["actual_return"].astype(float).median()),
        "actual_best_return": float(matched["actual_return"].astype(float).max()),
        "actual_worst_return": float(matched["actual_return"].astype(float).min()),
        "model_avg_return_matched": float(matched["model_return"].astype(float).mean()),
        "avg_return_gap": float(matched["return_gap"].astype(float).mean()),
        "avg_entry_slippage_pct": float(matched["entry_slippage_pct"].astype(float).mean()),
        "avg_exit_slippage_pct": float(matched["exit_slippage_pct"].astype(float).mean()),
        "actual_portfolio_return_equal_weight": float(portfolio_return) if portfolio_return is not None else None,
        "actual_portfolio_pnl_equal_weight": float(pnl_equal_weight),
        "cash_assumption": float(cash),
    }

    if matched["actual_pnl"].notna().any():
        summary["actual_portfolio_pnl_reported"] = float(pd.to_numeric(matched["actual_pnl"], errors="coerce").fillna(0.0).sum())
    else:
        summary["actual_portfolio_pnl_reported"] = None

    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    run_dir = resolve_run_dir(args, config)
    top_k_path = resolve_top_k_path(args, run_dir)
    if not top_k_path.exists():
        raise FileNotFoundError(f"top_k.csv not found at {top_k_path}")

    stock_path = Path(config.get("data", {}).get("path", "data/eod_daily.csv"))
    stock_df = load_local_stock_data(stock_path)
    if stock_df.empty:
        raise FileNotFoundError(f"No local stock EOD data found at {stock_path}")
    stock_df["date"] = pd.to_datetime(stock_df["date"]).dt.normalize()

    top_k_df = pd.read_csv(top_k_path)
    if top_k_df.empty:
        raise ValueError(f"{top_k_path} is empty.")

    symbol_filter = parse_symbol_filter(args.symbols)
    if symbol_filter is not None:
        top_k_df = top_k_df[top_k_df["symbol"].astype(str).str.upper().isin(symbol_filter)].copy()
        if top_k_df.empty:
            raise ValueError("No rows remain in top_k after applying --symbols filter.")

    top_k_df["symbol"] = top_k_df["symbol"].astype(str).str.upper()
    top_k_df["signal_date"] = pd.to_datetime(top_k_df["signal_date"]).dt.normalize()
    top_k_df["next_trade_date"] = pd.to_datetime(top_k_df["next_trade_date"]).dt.normalize()

    label_horizon = int(args.label_horizon) if args.label_horizon is not None else int(config.get("data", {}).get("label_horizon", 1))
    cash = float(args.cash) if args.cash is not None else float(config.get("execution", {}).get("initial_cash", 10000))

    rows: list[dict] = []
    for _, row in top_k_df.iterrows():
        symbol = str(row["symbol"]).upper()
        symbol_df = stock_df[stock_df["symbol"].astype(str).str.upper() == symbol].copy()
        replay = get_trade_rows(
            symbol_df=symbol_df,
            next_trade_date=pd.Timestamp(row["next_trade_date"]),
            label_horizon=label_horizon,
        )
        rows.append(
            {
                "signal_date": pd.Timestamp(row["signal_date"]).normalize(),
                "planned_entry_date": pd.Timestamp(row["next_trade_date"]).normalize(),
                "symbol": symbol,
                "industry_name": row.get("industry_name"),
                "model_score": float(row["score"]) if "score" in row and pd.notna(row["score"]) else None,
                "buy_price": float(row["buy_price"]) if "buy_price" in row and pd.notna(row["buy_price"]) else None,
                "buy_price_basis": row.get("buy_price_basis"),
                "entry_price_ref_close": float(row["entry_price_ref_close"])
                if "entry_price_ref_close" in row and pd.notna(row["entry_price_ref_close"])
                else (float(row["buy_price"]) if "buy_price" in row and pd.notna(row["buy_price"]) else None),
                **replay,
            }
        )

    model_eval_df = pd.DataFrame(rows)
    model_summary = summarize_model_replay(model_eval_df, cash=cash)
    model_summary.update(
        {
            "run_dir": str(run_dir.resolve()),
            "top_k_path": str(top_k_path.resolve()),
            "label_horizon": int(label_horizon),
        }
    )

    output_csv = run_dir / "realized_trade_eval.csv"
    output_json = run_dir / "realized_trade_summary.json"
    export_model_df = model_eval_df.copy()
    for column in ["signal_date", "planned_entry_date"]:
        export_model_df[column] = pd.to_datetime(export_model_df[column]).dt.date.astype(str)
    export_model_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    output_json.write_text(json.dumps(model_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Run dir: {run_dir.resolve()}")
    print(f"Top-k path: {top_k_path.resolve()}")
    print(f"Model replay requested picks: {model_summary['n_requested']}")
    print(f"Model replay filled picks: {model_summary['n_filled']}")
    print(f"Model replay fill rate: {model_summary['fill_rate']:.2%}")
    if model_summary["n_filled"] > 0:
        print(f"Model replay win rate: {model_summary['win_rate']:.2%}")
        print(f"Model replay avg return: {model_summary['avg_return']:.4f}")
        print(f"Model replay portfolio return(equal weight): {model_summary['portfolio_return_equal_weight']:.4f}")
    print(f"Saved model replay detail to: {output_csv.resolve()}")
    print(f"Saved model replay summary to: {output_json.resolve()}")

    if args.actual_trades_path is None:
        return

    actual_path = args.actual_trades_path.resolve()
    if not actual_path.exists():
        raise FileNotFoundError(f"Actual trades CSV not found at {actual_path}")

    actual_df = normalize_actual_trade_columns(pd.read_csv(actual_path))
    comparison_df = build_actual_comparison(model_eval_df, actual_df)
    comparison_summary = summarize_actual_comparison(comparison_df, cash=cash)
    comparison_summary.update(
        {
            "run_dir": str(run_dir.resolve()),
            "actual_trades_path": str(actual_path),
            "label_horizon": int(label_horizon),
        }
    )

    compare_csv = run_dir / "actual_trade_compare.csv"
    compare_json = run_dir / "actual_trade_compare_summary.json"
    export_compare_df = comparison_df.copy()
    for column in [
        "signal_date",
        "planned_entry_date",
        "actual_entry_date",
        "actual_exit_date",
    ]:
        if column in export_compare_df.columns:
            export_compare_df[column] = pd.to_datetime(export_compare_df[column], errors="coerce").dt.date.astype(str)
    export_compare_df.to_csv(compare_csv, index=False, encoding="utf-8-sig")
    compare_json.write_text(json.dumps(comparison_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Actual matched trades: {comparison_summary['n_actual_matched']}")
    print(f"Actual match rate: {comparison_summary['match_rate']:.2%}")
    if comparison_summary["n_actual_matched"] > 0:
        print(f"Actual win rate: {comparison_summary['actual_win_rate']:.2%}")
        print(f"Actual avg return: {comparison_summary['actual_avg_return']:.4f}")
        print(f"Average return gap(actual-model): {comparison_summary['avg_return_gap']:.4f}")
        print(f"Average entry slippage: {comparison_summary['avg_entry_slippage_pct']:.4f}")
        print(f"Average exit slippage: {comparison_summary['avg_exit_slippage_pct']:.4f}")
    print(f"Saved actual-vs-model detail to: {compare_csv.resolve()}")
    print(f"Saved actual-vs-model summary to: {compare_json.resolve()}")


if __name__ == "__main__":
    run_cli(main, label="Eval")
