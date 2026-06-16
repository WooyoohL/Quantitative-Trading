from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.runtime import build_fetcher, load_config, load_symbol_name_map
from data.fetcher import (
    load_local_industry_daily,
    load_local_industry_map,
    load_local_stock_data,
)
from strategy.market_heat import build_market_heat_candidates


WEEKDAY_NAME_TO_INT = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily paper-trading workflow.")
    parser.add_argument("--config", type=Path, default=Path("paper_trading.yaml"))
    parser.add_argument("--stage", choices=["prepare", "finalize"], default="prepare")
    parser.add_argument("--event-decisions", type=Path, default=None)
    parser.add_argument("--skip-update", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--allow-stale-data", action="store_true")
    return parser.parse_args(argv)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command: list[str]) -> None:
    print("[PaperTrade] " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def paper_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("paper_trading", {}))


def state_path(config: dict[str, Any]) -> Path:
    return PROJECT_ROOT / str(paper_cfg(config).get("state_path", "outputs/paper_trading/state.json"))


def workbook_path(config: dict[str, Any]) -> Path:
    return PROJECT_ROOT / str(paper_cfg(config).get("workbook_path", "outputs/paper_trading/paper_account.xlsx"))


def current_source_path(config: dict[str, Any]) -> Path:
    return PROJECT_ROOT / str(
        paper_cfg(config).get("current_source_run_path", "outputs/paper_trading/current_source_run.txt")
    )


def default_state(config: dict[str, Any]) -> dict[str, Any]:
    cfg = paper_cfg(config)
    return {
        "initial_cash": float(cfg.get("initial_cash", 100000.0)),
        "cash": float(cfg.get("initial_cash", 100000.0)),
        "current_source_run": None,
        "last_training_date": None,
        "last_prepare_date": None,
        "last_finalize_date": None,
        "positions": [],
        "pending_buys": [],
        "trades": [],
        "equity_curve": [],
        "runs": [],
        "pending_filter_input_path": None,
        "pending_event_decisions_template_path": None,
    }


def load_state(config: dict[str, Any]) -> dict[str, Any]:
    path = state_path(config)
    if not path.exists():
        return default_state(config)
    state = read_json(path)
    base = default_state(config)
    base.update(state)
    return base


def save_state(config: dict[str, Any], state: dict[str, Any]) -> None:
    write_json(state_path(config), state)


def load_base_config(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    base_config_path = PROJECT_ROOT / str(config.get("base_config", "config.yaml"))
    return base_config_path, load_config(base_config_path)


def load_local_frames(base_config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_cfg = base_config.get("data", {})
    stock_df = load_local_stock_data(PROJECT_ROOT / str(data_cfg.get("path", "data/eod_daily.csv")))
    industry_map_df = load_local_industry_map(PROJECT_ROOT / str(data_cfg.get("industry_map_path", "data/industry_map.csv")))
    industry_daily_df = load_local_industry_daily(
        PROJECT_ROOT / str(data_cfg.get("industry_daily_path", "data/industry_daily.csv"))
    )
    return stock_df, industry_map_df, industry_daily_df


def latest_trade_date(stock_df: pd.DataFrame, base_config: dict[str, Any], allow_stale_data: bool = False) -> pd.Timestamp:
    if stock_df.empty:
        raise ValueError("Local stock data is empty after update-data.")
    fetcher = build_fetcher(base_config)
    expected = fetcher.latest_closed_trade_date().normalize()
    actual = pd.to_datetime(stock_df["date"]).max().normalize()
    if actual < expected and not allow_stale_data:
        raise ValueError(f"Local stock data is stale: latest={actual.date()} expected={expected.date()}.")
    if actual < expected:
        print(f"[PaperTrade] Stale local data allowed for this run: latest={actual.date()} expected={expected.date()}.")
    return actual


def price_row(stock_df: pd.DataFrame, symbol: str, trade_date: pd.Timestamp) -> pd.Series | None:
    rows = stock_df[
        (stock_df["symbol"].astype(str) == str(symbol))
        & (pd.to_datetime(stock_df["date"]).dt.normalize() == pd.Timestamp(trade_date).normalize())
    ]
    if rows.empty:
        return None
    return rows.iloc[-1]


def append_trade(state: dict[str, Any], trade: dict[str, Any]) -> None:
    state.setdefault("trades", []).append(trade)


def execute_due_sells(state: dict[str, Any], stock_df: pd.DataFrame, trade_date: pd.Timestamp) -> None:
    remaining_positions: list[dict[str, Any]] = []
    for position in state.get("positions", []):
        planned_sell_date = pd.Timestamp(position["planned_sell_date"]).normalize()
        if planned_sell_date > trade_date:
            remaining_positions.append(position)
            continue

        row = price_row(stock_df, position["symbol"], trade_date)
        if row is None or pd.isna(row.get("open")):
            remaining_positions.append(position)
            append_trade(
                state,
                {
                    "date": trade_date.date().isoformat(),
                    "action": "SELL_SKIPPED_NO_OPEN",
                    "symbol": position["symbol"],
                    "name": position.get("name", ""),
                    "shares": int(position["shares"]),
                    "price": None,
                    "notional": 0.0,
                    "cash_after": float(state["cash"]),
                    "realized_pnl": 0.0,
                },
            )
            continue

        price = float(row["open"])
        shares = int(position["shares"])
        notional = float(price * shares)
        cost = float(position.get("cost", 0.0))
        pnl = float(notional - cost)
        state["cash"] = float(state["cash"]) + notional
        append_trade(
            state,
            {
                "date": trade_date.date().isoformat(),
                "action": "SELL",
                "symbol": position["symbol"],
                "name": position.get("name", ""),
                "shares": shares,
                "price": price,
                "notional": notional,
                "cash_after": float(state["cash"]),
                "realized_pnl": pnl,
                "source": position.get("source", ""),
                "signal_date": position.get("signal_date"),
                "buy_date": position.get("buy_date"),
            },
        )
    state["positions"] = remaining_positions


def execute_due_buys(
    state: dict[str, Any],
    stock_df: pd.DataFrame,
    trade_date: pd.Timestamp,
    config: dict[str, Any],
) -> None:
    cfg = paper_cfg(config)
    lot_size = int(cfg.get("lot_size", 100))
    target_gross_exposure = float(cfg.get("target_gross_exposure", 0.70))
    due = [
        item
        for item in state.get("pending_buys", [])
        if pd.Timestamp(item["buy_date"]).normalize() <= trade_date
    ]
    future = [
        item
        for item in state.get("pending_buys", [])
        if pd.Timestamp(item["buy_date"]).normalize() > trade_date
    ]
    if not due:
        state["pending_buys"] = future
        return

    cash_before_buys = float(state["cash"])
    allocation = cash_before_buys * target_gross_exposure / max(1, len(due))
    for plan in due:
        row = price_row(stock_df, plan["symbol"], trade_date)
        if row is None or pd.isna(row.get("open")):
            append_trade(
                state,
                {
                    "date": trade_date.date().isoformat(),
                    "action": "BUY_SKIPPED_NO_OPEN",
                    "symbol": plan["symbol"],
                    "name": plan.get("name", ""),
                    "shares": 0,
                    "price": None,
                    "notional": 0.0,
                    "cash_after": float(state["cash"]),
                    "source": plan.get("source", ""),
                    "signal_date": plan.get("signal_date"),
                },
            )
            continue

        price = float(row["open"])
        if price <= 0:
            continue
        shares = int(math.floor(allocation / price / lot_size) * lot_size)
        affordable_shares = int(math.floor(float(state["cash"]) / price / lot_size) * lot_size)
        shares = min(shares, affordable_shares)
        if shares <= 0:
            append_trade(
                state,
                {
                    "date": trade_date.date().isoformat(),
                    "action": "BUY_SKIPPED_CASH_OR_LOT",
                    "symbol": plan["symbol"],
                    "name": plan.get("name", ""),
                    "shares": 0,
                    "price": price,
                    "notional": 0.0,
                    "cash_after": float(state["cash"]),
                    "source": plan.get("source", ""),
                    "signal_date": plan.get("signal_date"),
                },
            )
            continue

        notional = float(shares * price)
        state["cash"] = float(state["cash"]) - notional
        position = {
            "symbol": plan["symbol"],
            "name": plan.get("name", ""),
            "shares": shares,
            "entry_price": price,
            "cost": notional,
            "signal_date": plan.get("signal_date"),
            "buy_date": trade_date.date().isoformat(),
            "planned_sell_date": plan["sell_date"],
            "source": plan.get("source", ""),
            "rank": plan.get("rank"),
        }
        state.setdefault("positions", []).append(position)
        append_trade(
            state,
            {
                "date": trade_date.date().isoformat(),
                "action": "BUY",
                "symbol": plan["symbol"],
                "name": plan.get("name", ""),
                "shares": shares,
                "price": price,
                "notional": notional,
                "cash_after": float(state["cash"]),
                "source": plan.get("source", ""),
                "signal_date": plan.get("signal_date"),
                "planned_sell_date": plan["sell_date"],
            },
        )

    state["pending_buys"] = future


def mark_to_market(state: dict[str, Any], stock_df: pd.DataFrame, trade_date: pd.Timestamp) -> None:
    market_value = 0.0
    unrealized_pnl = 0.0
    for position in state.get("positions", []):
        row = price_row(stock_df, position["symbol"], trade_date)
        close_price = float(row["close"]) if row is not None and pd.notna(row.get("close")) else float(position["entry_price"])
        value = close_price * int(position["shares"])
        market_value += value
        unrealized_pnl += value - float(position.get("cost", 0.0))
    total_equity = float(state["cash"]) + float(market_value)
    curve = [
        row
        for row in state.get("equity_curve", [])
        if str(row.get("date")) != trade_date.date().isoformat()
    ]
    curve.append(
        {
            "date": trade_date.date().isoformat(),
            "cash": float(state["cash"]),
            "market_value": float(market_value),
            "total_equity": float(total_equity),
            "unrealized_pnl": float(unrealized_pnl),
            "position_count": int(len(state.get("positions", []))),
            "pending_buy_count": int(len(state.get("pending_buys", []))),
        }
    )
    state["equity_curve"] = sorted(curve, key=lambda row: row["date"])


def update_ledger_for_trade_date(
    state: dict[str, Any],
    stock_df: pd.DataFrame,
    trade_date: pd.Timestamp,
    config: dict[str, Any],
) -> None:
    execute_due_sells(state, stock_df, trade_date)
    execute_due_buys(state, stock_df, trade_date, config)
    mark_to_market(state, stock_df, trade_date)


def retrain_weekday_set(config: dict[str, Any]) -> set[int]:
    names = paper_cfg(config).get("retrain_weekdays", ["Monday", "Wednesday"])
    out: set[int] = set()
    for name in names:
        if isinstance(name, int):
            out.add(int(name))
        else:
            normalized = str(name).strip().lower()
            if normalized in WEEKDAY_NAME_TO_INT:
                out.add(WEEKDAY_NAME_TO_INT[normalized])
    return out


def source_run_exists(source_run: str | None) -> bool:
    if not source_run:
        return False
    path = Path(source_run)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return (path / "config.yaml").exists() and (path / "best.ckpt").exists()


def resolve_source_run(config: dict[str, Any], state: dict[str, Any]) -> str | None:
    if source_run_exists(state.get("current_source_run")):
        return str(state.get("current_source_run"))
    pointer = current_source_path(config)
    if pointer.exists():
        value = pointer.read_text(encoding="utf-8").strip()
        if source_run_exists(value):
            state["current_source_run"] = value
            return value
    configured = paper_cfg(config).get("model", {}).get("source_run")
    if source_run_exists(configured):
        state["current_source_run"] = str(configured)
        return str(configured)
    return None


def maybe_train_model(
    *,
    config: dict[str, Any],
    base_config_path: Path,
    state: dict[str, Any],
    trade_date: pd.Timestamp,
    force_if_missing: bool,
    skip_train: bool,
) -> str | None:
    current_source = resolve_source_run(config, state)
    if skip_train:
        return current_source

    should_train = trade_date.weekday() in retrain_weekday_set(config)
    should_train = should_train and state.get("last_training_date") != trade_date.date().isoformat()
    should_train = should_train or (force_if_missing and not source_run_exists(current_source))
    if not should_train:
        return current_source

    prefix = str(paper_cfg(config).get("model", {}).get("train_run_prefix", "paper"))
    run_name = f"{prefix}_{trade_date.strftime('%Y%m%d')}_{pd.Timestamp.now().strftime('%H%M%S')}"
    run_dir = Path("outputs") / "runs" / run_name
    run_command([sys.executable, "main.py", "train", "--config", str(base_config_path), "--run-name", run_name])
    state["current_source_run"] = str(run_dir)
    state["last_training_date"] = trade_date.date().isoformat()
    state.setdefault("runs", []).append(
        {
            "date": trade_date.date().isoformat(),
            "run_dir": str(run_dir),
            "reason": "scheduled_or_missing_source",
        }
    )
    pointer = current_source_path(config)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(run_dir), encoding="utf-8")
    return str(run_dir)


def run_inference(
    *,
    config: dict[str, Any],
    source_run: str,
    trade_date: pd.Timestamp,
) -> Path:
    infer_cfg = paper_cfg(config).get("inference", {})
    prefix = str(infer_cfg.get("output_prefix", "paper_signal"))
    output_name = f"{prefix}_{trade_date.strftime('%Y%m%d')}_{pd.Timestamp.now().strftime('%H%M%S')}"
    checkpoint_name = str(infer_cfg.get("checkpoint_name", "best.ckpt"))
    run_command(
        [
            sys.executable,
            "main.py",
            "infer",
            "--source-run",
            source_run,
            "--checkpoint-name",
            checkpoint_name,
            "--output-name",
            output_name,
        ]
    )
    return PROJECT_ROOT / "outputs" / "inference_runs" / output_name


def read_candidate_symbols(base_config: dict[str, Any]) -> set[str]:
    candidate_path = PROJECT_ROOT / str(base_config.get("data", {}).get("candidate_path", "data/current_candidates.csv"))
    if not candidate_path.exists():
        return set()
    df = pd.read_csv(candidate_path)
    if "symbol" not in df.columns:
        return set()
    return set(df["symbol"].dropna().astype(str))


def standardize_model_candidates(review_path: Path, limit: int) -> pd.DataFrame:
    if not review_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(review_path).head(int(limit)).copy()
    if df.empty:
        return pd.DataFrame()
    if "candidate_rank" not in df.columns:
        df["candidate_rank"] = range(1, len(df) + 1)
    if "buy_price" in df.columns and "close" not in df.columns:
        df["close"] = df["buy_price"]
    rank = pd.to_numeric(df["candidate_rank"], errors="coerce")
    df["model_rank"] = rank.where(rank.notna(), pd.Series(range(1, len(df) + 1), index=df.index))
    df["from_model"] = True
    df["from_heat"] = False
    df["source"] = "model"
    return df


def standardize_heat_candidates(
    heat_df: pd.DataFrame,
    symbol_name_map: pd.Series,
    next_trade_date: pd.Timestamp,
) -> pd.DataFrame:
    if heat_df.empty:
        return pd.DataFrame()
    out = heat_df.copy()
    out["name"] = out["symbol"].astype(str).map(symbol_name_map).fillna("")
    out["next_trade_date"] = next_trade_date.date().isoformat()
    out["candidate_rank"] = pd.NA
    out["score"] = pd.NA
    out["model_rank"] = pd.NA
    out["from_model"] = False
    out["from_heat"] = True
    out["source"] = "heat"
    out["buy_price"] = out["close"]
    return out


def first_notna(series: pd.Series) -> Any:
    values = series.dropna()
    if values.empty:
        return pd.NA
    return values.iloc[0]


def first_from_columns(group: pd.DataFrame, columns: list[str]) -> Any:
    for column in columns:
        if column in group.columns:
            value = first_notna(group[column])
            if pd.notna(value):
                return value
    return pd.NA


def combine_candidates(model_df: pd.DataFrame, heat_df: pd.DataFrame, final_limit: int) -> pd.DataFrame:
    if model_df.empty and heat_df.empty:
        return pd.DataFrame()
    combined = pd.concat([model_df, heat_df], ignore_index=True, sort=False)
    combined["symbol"] = combined["symbol"].astype(str)
    rows: list[dict[str, Any]] = []
    for symbol, group in combined.groupby("symbol", sort=False):
        from_model = bool(group["from_model"].fillna(False).any())
        from_heat = bool(group["from_heat"].fillna(False).any())
        source = "model+heat" if from_model and from_heat else ("model" if from_model else "heat")
        heat_reasons = ",".join(sorted(set(group.get("heat_reasons", pd.Series(dtype=str)).dropna().astype(str))))
        row = {
            "signal_date": first_notna(group.get("signal_date", pd.Series(dtype=object))),
            "next_trade_date": first_notna(group.get("next_trade_date", pd.Series(dtype=object))),
            "symbol": symbol,
            "name": first_notna(group.get("name", pd.Series(dtype=object))),
            "industry_name": first_notna(group.get("industry_name", pd.Series(dtype=object))),
            "score": first_notna(group.get("score", pd.Series(dtype=object))),
            "candidate_rank": pd.to_numeric(group.get("candidate_rank", pd.Series(dtype=float)), errors="coerce").min(),
            "model_rank": pd.to_numeric(group.get("model_rank", pd.Series(dtype=float)), errors="coerce").min(),
            "heat_rank": pd.to_numeric(group.get("heat_rank", pd.Series(dtype=float)), errors="coerce").min(),
            "heat_score": pd.to_numeric(group.get("heat_score", pd.Series(dtype=float)), errors="coerce").max(),
            "close": first_notna(group.get("close", pd.Series(dtype=object))),
            "buy_price": first_notna(group.get("buy_price", pd.Series(dtype=object))),
            "ret_1": first_from_columns(group, ["ret_1", "ret_1_effective"]),
            "ret_5": first_from_columns(group, ["ret_5"]),
            "intraday_ret": first_from_columns(group, ["intraday_ret"]),
            "ma_gap_5": first_from_columns(group, ["ma_gap_5"]),
            "volume_ratio_5": first_from_columns(group, ["volume_ratio_5"]),
            "industry_ret_1_mean": first_from_columns(group, ["industry_ret_1_mean", "industry_ret_1"]),
            "source": source,
            "heat_reasons": heat_reasons,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out["source_priority"] = out["source"].map({"model+heat": 0, "model": 1, "heat": 2}).fillna(9).astype(int)
    out["model_rank_sort"] = pd.to_numeric(out["model_rank"], errors="coerce").fillna(9999.0)
    out["heat_rank_sort"] = pd.to_numeric(out["heat_rank"], errors="coerce").fillna(9999.0)
    out["heat_score_sort"] = pd.to_numeric(out["heat_score"], errors="coerce").fillna(-9999.0)
    out = out.sort_values(
        ["source_priority", "model_rank_sort", "heat_rank_sort", "heat_score_sort", "symbol"],
        ascending=[True, True, True, False, True],
    ).head(int(final_limit)).reset_index(drop=True)
    out["review_rank"] = range(1, len(out) + 1)
    return out.drop(columns=["source_priority", "model_rank_sort", "heat_rank_sort", "heat_score_sort"])


def apply_basic_candidate_filters(candidates: pd.DataFrame, base_config: dict[str, Any]) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    out = candidates.copy()
    filters = base_config.get("universe", {}).get("filters", {})
    min_price = filters.get("min_latest_price")
    max_price = filters.get("max_latest_price")
    close = pd.to_numeric(out["close"], errors="coerce")
    if min_price is not None:
        out = out[close >= float(min_price)].copy()
        close = pd.to_numeric(out["close"], errors="coerce")
    if max_price is not None:
        out = out[close <= float(max_price)].copy()
    return out.reset_index(drop=True)


def write_event_filter_files(
    *,
    config: dict[str, Any],
    candidates: pd.DataFrame,
    trade_date: pd.Timestamp,
) -> tuple[Path, Path]:
    base_dir = PROJECT_ROOT / "outputs" / "paper_trading" / trade_date.strftime("%Y%m%d")
    base_dir.mkdir(parents=True, exist_ok=True)
    filter_input = base_dir / "event_filter_input.csv"
    decision_template = base_dir / "event_filter_decisions.csv"
    latest_input = PROJECT_ROOT / "outputs" / "paper_trading" / "latest_filter_input.csv"
    latest_template = PROJECT_ROOT / "outputs" / "paper_trading" / "latest_event_filter_decisions.csv"

    candidates.to_csv(filter_input, index=False, encoding="utf-8-sig")
    candidates.to_csv(latest_input, index=False, encoding="utf-8-sig")

    decision_columns = [
        "symbol",
        "name",
        "recommended_action",
        "risk_level",
        "positive_catalyst_level",
        "key_negative_events",
        "key_positive_events",
        "summary",
        "sources",
    ]
    template = candidates[["symbol", "name"]].copy()
    for column in decision_columns:
        if column not in template.columns:
            template[column] = ""
    template = template[decision_columns]
    template.to_csv(decision_template, index=False, encoding="utf-8-sig")
    template.to_csv(latest_template, index=False, encoding="utf-8-sig")

    prompt_path = PROJECT_ROOT / str(paper_cfg(config).get("event_filter_prompt_path", "strategy/post_filter.md"))
    review_note = base_dir / "event_filter_instruction.md"
    review_note.write_text(
        "\n".join(
            [
                "# Event Filter Task",
                "",
                f"Prompt file: {prompt_path}",
                f"Input CSV: {filter_input}",
                f"Decision CSV to fill: {decision_template}",
                "",
                "Only rows with recommended_action=Keep are eligible for paper-trading buys.",
            ]
        ),
        encoding="utf-8",
    )
    return filter_input, decision_template


def write_workbook(
    *,
    config: dict[str, Any],
    state: dict[str, Any],
    latest_candidates: pd.DataFrame | None = None,
    event_decisions: pd.DataFrame | None = None,
) -> None:
    path = workbook_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        [
            {
                "initial_cash": state.get("initial_cash"),
                "cash": state.get("cash"),
                "current_source_run": state.get("current_source_run"),
                "last_training_date": state.get("last_training_date"),
                "last_prepare_date": state.get("last_prepare_date"),
                "last_finalize_date": state.get("last_finalize_date"),
                "position_count": len(state.get("positions", [])),
                "pending_buy_count": len(state.get("pending_buys", [])),
            }
        ]
    )
    frames = {
        "account_summary": summary,
        "positions": pd.DataFrame(state.get("positions", [])),
        "pending_buys": pd.DataFrame(state.get("pending_buys", [])),
        "trades": pd.DataFrame(state.get("trades", [])),
        "equity_curve": pd.DataFrame(state.get("equity_curve", [])),
        "latest_candidates": latest_candidates if latest_candidates is not None else pd.DataFrame(),
        "event_decisions": event_decisions if event_decisions is not None else pd.DataFrame(),
    }
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def prepare_stage(args: argparse.Namespace, config: dict[str, Any]) -> None:
    base_config_path, base_config = load_base_config(config)
    state = load_state(config)

    if not args.skip_update:
        run_command([sys.executable, "main.py", "update-data", "--config", str(base_config_path)])

    stock_df, industry_map_df, industry_daily_df = load_local_frames(base_config)
    trade_date = latest_trade_date(stock_df, base_config, allow_stale_data=bool(args.allow_stale_data))
    update_ledger_for_trade_date(state, stock_df, trade_date, config)

    source_run = maybe_train_model(
        config=config,
        base_config_path=base_config_path,
        state=state,
        trade_date=trade_date,
        force_if_missing=True,
        skip_train=bool(args.skip_train),
    )
    if not source_run or not source_run_exists(source_run):
        raise FileNotFoundError("No usable source run is available for paper-trading inference.")

    inference_dir = run_inference(config=config, source_run=source_run, trade_date=trade_date)
    fetcher = build_fetcher(base_config)
    next_trade_date = fetcher.next_trade_date(trade_date)
    model_limit = int(paper_cfg(config).get("candidates", {}).get("model_review_limit", 20))
    heat_limit = int(paper_cfg(config).get("candidates", {}).get("heat_limit", 30))
    event_review_limit = int(paper_cfg(config).get("candidates", {}).get("event_review_limit", 50))

    model_candidates = standardize_model_candidates(inference_dir / "review_top_k.csv", limit=model_limit)
    heat_candidates = build_market_heat_candidates(
        stock_df=stock_df,
        industry_map_df=industry_map_df,
        industry_daily_df=industry_daily_df,
        signal_date=trade_date,
        config=paper_cfg(config).get("heat", {}),
        candidate_symbols=read_candidate_symbols(base_config),
        limit=heat_limit,
    )
    heat_candidates = standardize_heat_candidates(
        heat_candidates,
        symbol_name_map=load_symbol_name_map(base_config),
        next_trade_date=next_trade_date,
    )

    combined = combine_candidates(model_candidates, heat_candidates, final_limit=event_review_limit)
    combined = apply_basic_candidate_filters(combined, base_config)
    filter_input, decision_template = write_event_filter_files(config=config, candidates=combined, trade_date=trade_date)

    state["current_source_run"] = source_run
    state["last_prepare_date"] = trade_date.date().isoformat()
    state["last_inference_dir"] = str(inference_dir.relative_to(PROJECT_ROOT))
    state["pending_filter_input_path"] = str(filter_input.relative_to(PROJECT_ROOT))
    state["pending_event_decisions_template_path"] = str(decision_template.relative_to(PROJECT_ROOT))
    save_state(config, state)
    write_workbook(config=config, state=state, latest_candidates=combined)

    print(f"Trade date: {trade_date.date()}")
    print(f"Source run: {source_run}")
    print(f"Inference dir: {inference_dir}")
    print(f"Event filter input: {filter_input}")
    print(f"Event decision CSV: {decision_template}")
    print(f"Candidates for event review: {len(combined)}")


def normalize_action(value: Any) -> str:
    return str(value or "").strip().lower()


def finalize_stage(args: argparse.Namespace, config: dict[str, Any]) -> None:
    state = load_state(config)
    base_config_path, base_config = load_base_config(config)
    stock_df, _, _ = load_local_frames(base_config)
    trade_date = latest_trade_date(stock_df, base_config, allow_stale_data=bool(args.allow_stale_data))

    filter_input_path = PROJECT_ROOT / str(state.get("pending_filter_input_path") or "")
    if not filter_input_path.exists():
        raise FileNotFoundError(f"Missing pending filter input: {filter_input_path}")

    event_decisions_path = args.event_decisions
    if event_decisions_path is None:
        event_decisions_path = PROJECT_ROOT / str(state.get("pending_event_decisions_template_path") or "")
    if not event_decisions_path.is_absolute():
        event_decisions_path = PROJECT_ROOT / event_decisions_path
    if not event_decisions_path.exists():
        raise FileNotFoundError(f"Missing event decisions CSV: {event_decisions_path}")

    candidates = pd.read_csv(filter_input_path)
    decisions = pd.read_csv(event_decisions_path)
    if "recommended_action" not in decisions.columns:
        raise ValueError("event decisions CSV must contain recommended_action.")
    decisions["symbol"] = decisions["symbol"].astype(str)
    candidates["symbol"] = candidates["symbol"].astype(str)
    merged = candidates.merge(decisions, on=["symbol"], how="left", suffixes=("", "_event"))
    keep = merged[merged["recommended_action"].map(normalize_action) == "keep"].copy()
    final_count = int(paper_cfg(config).get("candidates", {}).get("final_buy_count", 3))
    keep = keep.sort_values(["review_rank", "symbol"], ascending=[True, True]).head(final_count).copy()

    signal_date = pd.Timestamp(state.get("last_prepare_date") or trade_date.date().isoformat()).normalize()
    fetcher = build_fetcher(base_config)
    if keep.empty:
        buy_date = fetcher.next_trade_date(signal_date)
    else:
        buy_date = pd.Timestamp(keep["next_trade_date"].dropna().iloc[0]).normalize()
    sell_date = fetcher.next_trade_date(buy_date)

    state["pending_buys"] = [
        plan
        for plan in state.get("pending_buys", [])
        if str(plan.get("signal_date")) != signal_date.date().isoformat()
    ]
    for index, row in keep.reset_index(drop=True).iterrows():
        state.setdefault("pending_buys", []).append(
            {
                "signal_date": signal_date.date().isoformat(),
                "buy_date": buy_date.date().isoformat(),
                "sell_date": sell_date.date().isoformat(),
                "symbol": str(row["symbol"]),
                "name": str(row.get("name", row.get("name_event", "")) or ""),
                "source": str(row.get("source", "")),
                "rank": int(index + 1),
                "review_rank": int(row.get("review_rank", index + 1)),
                "recommended_action": str(row.get("recommended_action", "")),
                "risk_level": str(row.get("risk_level", "")),
                "positive_catalyst_level": str(row.get("positive_catalyst_level", "")),
            }
        )

    final_plan_dir = PROJECT_ROOT / "outputs" / "paper_trading" / signal_date.strftime("%Y%m%d")
    final_plan_dir.mkdir(parents=True, exist_ok=True)
    final_plan_path = final_plan_dir / "final_buy_plan.csv"
    pd.DataFrame(state.get("pending_buys", [])).to_csv(final_plan_path, index=False, encoding="utf-8-sig")

    state["last_finalize_date"] = trade_date.date().isoformat()
    state["last_final_buy_plan_path"] = str(final_plan_path.relative_to(PROJECT_ROOT))
    save_state(config, state)
    write_workbook(config=config, state=state, latest_candidates=candidates, event_decisions=decisions)

    print(f"Trade date: {trade_date.date()}")
    print(f"Signal date: {signal_date.date()}")
    print(f"Buy date: {buy_date.date()}")
    print(f"Sell date: {sell_date.date()}")
    print(f"Final buy count: {len(keep)}")
    print(f"Final buy plan: {final_plan_path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = read_yaml(PROJECT_ROOT / args.config)
    if args.stage == "prepare":
        prepare_stage(args, config)
    else:
        finalize_stage(args, config)


if __name__ == "__main__":
    main()
