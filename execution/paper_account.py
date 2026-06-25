from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.paper_trading_config import paper_cfg


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


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def trading_cost_rates(config: dict[str, Any]) -> dict[str, float]:
    cfg = paper_cfg(config)
    default_slippage = float(cfg.get("slippage_rate", 0.001))
    return {
        "buy_slippage_rate": float(cfg.get("buy_slippage_rate", default_slippage)),
        "sell_slippage_rate": float(cfg.get("sell_slippage_rate", default_slippage)),
        "fee_rate": float(cfg.get("commission_rate", cfg.get("fee_rate", 0.0005))),
    }


def expected_open_gap(
    actual_open_price: float | None,
    expected_buy_price: float | None,
) -> tuple[float | None, float | None]:
    if actual_open_price is None or expected_buy_price is None or expected_buy_price <= 0:
        return None, None
    gap = float(actual_open_price - expected_buy_price)
    return gap, float(gap / expected_buy_price)


def buy_plan_key(plan: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(plan.get("signal_date", "")),
        str(plan.get("buy_date", plan.get("actual_buy_date", ""))),
        str(plan.get("symbol", "")),
        str(plan.get("rank", "")),
    )


def upsert_buy_plan_history(state: dict[str, Any], plan: dict[str, Any]) -> None:
    key = buy_plan_key(plan)
    history: list[dict[str, Any]] = []
    merged = dict(plan)
    for item in state.get("buy_plan_history", []):
        if buy_plan_key(item) == key:
            merged = {**item, **plan}
        else:
            history.append(item)
    history.append(merged)
    state["buy_plan_history"] = sorted(
        history,
        key=lambda item: (
            str(item.get("signal_date", "")),
            str(item.get("buy_date", item.get("actual_buy_date", ""))),
            int(item.get("rank") or 9999),
            str(item.get("symbol", "")),
        ),
    )


BUY_PLAN_COLUMNS = [
    "signal_date",
    "buy_date",
    "sell_date",
    "symbol",
    "name",
    "source",
    "buy_intent",
    "rank",
    "review_rank",
    "recommended_action",
    "risk_level",
    "positive_catalyst_level",
    "expected_buy_price",
    "buy_slippage_rate",
    "expected_buy_execution_price",
    "fee_rate",
    "target_exposure",
    "execution_status",
]

BUY_EXECUTION_TRACKING_COLUMNS = BUY_PLAN_COLUMNS + [
    "actual_buy_date",
    "actual_open_price",
    "open_vs_expected_price",
    "open_vs_expected_pct",
    "actual_execution_price",
    "actual_shares",
    "actual_notional",
    "buy_fee",
    "total_cash_out",
]


def buy_plan_frame(records: list[dict[str, Any]], *, include_execution_columns: bool) -> pd.DataFrame:
    columns = BUY_EXECUTION_TRACKING_COLUMNS if include_execution_columns else BUY_PLAN_COLUMNS
    frame = pd.DataFrame(records)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    extras = [column for column in frame.columns if column not in columns]
    return frame[columns + extras]


def execute_due_sells(
    state: dict[str, Any],
    stock_df: pd.DataFrame,
    trade_date: pd.Timestamp,
    config: dict[str, Any],
) -> None:
    rates = trading_cost_rates(config)
    sell_slippage_rate = rates["sell_slippage_rate"]
    fee_rate = rates["fee_rate"]
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
                    "gross_notional": 0.0,
                    "fee": 0.0,
                    "net_cash_flow": 0.0,
                    "cash_after": float(state["cash"]),
                    "realized_pnl": 0.0,
                },
            )
            continue

        open_price = float(row["open"])
        price = float(open_price * (1.0 - sell_slippage_rate))
        shares = int(position["shares"])
        gross_notional = float(price * shares)
        fee = float(gross_notional * fee_rate)
        net_cash_flow = float(gross_notional - fee)
        cost = float(position.get("cost", 0.0))
        pnl = float(net_cash_flow - cost)
        state["cash"] = float(state["cash"]) + net_cash_flow
        append_trade(
            state,
            {
                "date": trade_date.date().isoformat(),
                "action": "SELL",
                "symbol": position["symbol"],
                "name": position.get("name", ""),
                "shares": shares,
                "price": price,
                "open_price": open_price,
                "notional": gross_notional,
                "gross_notional": gross_notional,
                "fee_rate": fee_rate,
                "sell_slippage_rate": sell_slippage_rate,
                "fee": fee,
                "net_cash_flow": net_cash_flow,
                "cash_after": float(state["cash"]),
                "realized_pnl": pnl,
                "source": position.get("source", ""),
                "signal_date": position.get("signal_date"),
                "buy_date": position.get("buy_date"),
            },
        )
    state["positions"] = remaining_positions


def current_position_market_value(state: dict[str, Any], stock_df: pd.DataFrame, trade_date: pd.Timestamp) -> float:
    market_value = 0.0
    for position in state.get("positions", []):
        row = price_row(stock_df, str(position.get("symbol")), trade_date)
        if row is not None and pd.notna(row.get("open")):
            price = float(row["open"])
        elif row is not None and pd.notna(row.get("close")):
            price = float(row["close"])
        else:
            price = float(position.get("entry_price", 0.0) or 0.0)
        market_value += price * int(position.get("shares", 0) or 0)
    return float(market_value)


def execute_due_buys(
    state: dict[str, Any],
    stock_df: pd.DataFrame,
    trade_date: pd.Timestamp,
    config: dict[str, Any],
) -> None:
    cfg = paper_cfg(config)
    lot_size = int(cfg.get("lot_size", 100))
    max_positions = int(cfg.get("max_positions", 3))
    target_gross_exposure = float(cfg.get("target_gross_exposure", 0.70))
    per_position_target_exposure = float(
        cfg.get("per_position_target_exposure", target_gross_exposure / max(1, max_positions))
    )
    rates = trading_cost_rates(config)
    buy_slippage_rate = rates["buy_slippage_rate"]
    fee_rate = rates["fee_rate"]
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
    existing_market_value = current_position_market_value(state, stock_df, trade_date)
    total_equity_before_buys = cash_before_buys + existing_market_value
    max_gross_budget = total_equity_before_buys * target_gross_exposure
    available_gross_budget = max(0.0, max_gross_budget - existing_market_value)
    remaining_slots = max(0, max_positions - len(state.get("positions", [])))
    if remaining_slots <= 0 or available_gross_budget <= 0:
        for plan in due:
            upsert_buy_plan_history(
                state,
                {
                    **plan,
                    "actual_buy_date": trade_date.date().isoformat(),
                    "execution_status": "SKIPPED_EXPOSURE_CAP",
                    "actual_shares": 0,
                    "actual_notional": 0.0,
                    "buy_fee": 0.0,
                    "total_cash_out": 0.0,
                },
            )
        state["pending_buys"] = future
        return

    process_due = due[:remaining_slots]
    skipped_due = due[remaining_slots:]
    for plan in skipped_due:
        upsert_buy_plan_history(
            state,
            {
                **plan,
                "actual_buy_date": trade_date.date().isoformat(),
                "execution_status": "SKIPPED_MAX_POSITIONS",
                "actual_shares": 0,
                "actual_notional": 0.0,
                "buy_fee": 0.0,
                "total_cash_out": 0.0,
            },
        )

    remaining_gross_budget = available_gross_budget
    for plan in process_due:
        plan_target_exposure = maybe_float(plan.get("target_exposure"))
        if plan_target_exposure is None:
            plan_target_exposure = per_position_target_exposure
        allocation = min(total_equity_before_buys * plan_target_exposure, remaining_gross_budget)
        if allocation <= 0:
            upsert_buy_plan_history(
                state,
                {
                    **plan,
                    "actual_buy_date": trade_date.date().isoformat(),
                    "execution_status": "SKIPPED_EXPOSURE_CAP",
                    "actual_shares": 0,
                    "actual_notional": 0.0,
                    "buy_fee": 0.0,
                    "total_cash_out": 0.0,
                },
            )
            continue
        row = price_row(stock_df, plan["symbol"], trade_date)
        expected_buy_price = maybe_float(plan.get("expected_buy_price"))
        if row is None or pd.isna(row.get("open")):
            upsert_buy_plan_history(
                state,
                {
                    **plan,
                    "actual_buy_date": trade_date.date().isoformat(),
                    "execution_status": "SKIPPED_NO_OPEN",
                    "actual_open_price": None,
                    "open_vs_expected_price": None,
                    "open_vs_expected_pct": None,
                    "actual_execution_price": None,
                    "actual_shares": 0,
                    "actual_notional": 0.0,
                    "buy_fee": 0.0,
                    "total_cash_out": 0.0,
                },
            )
            append_trade(
                state,
                {
                    "date": trade_date.date().isoformat(),
                    "action": "BUY_SKIPPED_NO_OPEN",
                    "symbol": plan["symbol"],
                    "name": plan.get("name", ""),
                    "shares": 0,
                    "price": None,
                    "expected_buy_price": expected_buy_price,
                    "notional": 0.0,
                    "gross_notional": 0.0,
                    "fee": 0.0,
                    "net_cash_flow": 0.0,
                    "cash_after": float(state["cash"]),
                    "source": plan.get("source", ""),
                    "signal_date": plan.get("signal_date"),
                },
            )
            continue

        open_price = float(row["open"])
        price = float(open_price * (1.0 + buy_slippage_rate))
        open_gap, open_gap_pct = expected_open_gap(open_price, expected_buy_price)
        if price <= 0:
            continue
        cash_per_share = price * (1.0 + fee_rate)
        shares = int(math.floor(allocation / cash_per_share / lot_size) * lot_size)
        affordable_shares = int(math.floor(float(state["cash"]) / cash_per_share / lot_size) * lot_size)
        shares = min(shares, affordable_shares)
        if shares <= 0:
            upsert_buy_plan_history(
                state,
                {
                    **plan,
                    "actual_buy_date": trade_date.date().isoformat(),
                    "execution_status": "SKIPPED_CASH_OR_LOT",
                    "actual_open_price": open_price,
                    "open_vs_expected_price": open_gap,
                    "open_vs_expected_pct": open_gap_pct,
                    "actual_execution_price": price,
                    "actual_shares": 0,
                    "actual_notional": 0.0,
                    "buy_fee": 0.0,
                    "total_cash_out": 0.0,
                },
            )
            append_trade(
                state,
                {
                    "date": trade_date.date().isoformat(),
                    "action": "BUY_SKIPPED_CASH_OR_LOT",
                    "symbol": plan["symbol"],
                    "name": plan.get("name", ""),
                    "shares": 0,
                    "price": price,
                    "open_price": open_price,
                    "expected_buy_price": expected_buy_price,
                    "open_vs_expected_price": open_gap,
                    "open_vs_expected_pct": open_gap_pct,
                    "notional": 0.0,
                    "gross_notional": 0.0,
                    "fee": 0.0,
                    "net_cash_flow": 0.0,
                    "cash_after": float(state["cash"]),
                    "source": plan.get("source", ""),
                    "signal_date": plan.get("signal_date"),
                },
            )
            continue

        gross_notional = float(shares * price)
        fee = float(gross_notional * fee_rate)
        total_cash_out = float(gross_notional + fee)
        state["cash"] = float(state["cash"]) - total_cash_out
        remaining_gross_budget = max(0.0, remaining_gross_budget - gross_notional)
        position = {
            "symbol": plan["symbol"],
            "name": plan.get("name", ""),
            "buy_intent": plan.get("buy_intent", ""),
            "shares": shares,
            "entry_price": price,
            "entry_open_price": open_price,
            "expected_buy_price": expected_buy_price,
            "open_vs_expected_price": open_gap,
            "open_vs_expected_pct": open_gap_pct,
            "buy_slippage_rate": buy_slippage_rate,
            "fee_rate": fee_rate,
            "buy_fee": fee,
            "gross_cost": gross_notional,
            "cost": total_cash_out,
            "signal_date": plan.get("signal_date"),
            "buy_date": trade_date.date().isoformat(),
            "planned_sell_date": plan["sell_date"],
            "source": plan.get("source", ""),
            "rank": plan.get("rank"),
        }
        state.setdefault("positions", []).append(position)
        upsert_buy_plan_history(
            state,
            {
                **plan,
                "actual_buy_date": trade_date.date().isoformat(),
                "execution_status": "BOUGHT",
                "actual_open_price": open_price,
                "open_vs_expected_price": open_gap,
                "open_vs_expected_pct": open_gap_pct,
                "actual_execution_price": price,
                "actual_shares": shares,
                "actual_notional": gross_notional,
                "buy_fee": fee,
                "total_cash_out": total_cash_out,
            },
        )
        append_trade(
            state,
            {
                "date": trade_date.date().isoformat(),
                "action": "BUY",
                "symbol": plan["symbol"],
                "name": plan.get("name", ""),
                "shares": shares,
                "price": price,
                "open_price": open_price,
                "expected_buy_price": expected_buy_price,
                "open_vs_expected_price": open_gap,
                "open_vs_expected_pct": open_gap_pct,
                "notional": gross_notional,
                "gross_notional": gross_notional,
                "buy_slippage_rate": buy_slippage_rate,
                "fee_rate": fee_rate,
                "fee": fee,
                "net_cash_flow": -total_cash_out,
                "cash_after": float(state["cash"]),
                "source": plan.get("source", ""),
                "buy_intent": plan.get("buy_intent", ""),
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
    execute_due_sells(state, stock_df, trade_date, config)
    execute_due_buys(state, stock_df, trade_date, config)
    mark_to_market(state, stock_df, trade_date)


ACCOUNT_LEDGER_COLUMNS = [
    "date",
    "trade_count",
    "buy_count",
    "buy_shares",
    "buy_notional",
    "buy_fee",
    "buy_cash_out",
    "sell_count",
    "sell_shares",
    "sell_notional",
    "sell_fee",
    "sell_cash_in",
    "net_cash_flow",
    "realized_pnl",
    "cash",
    "market_value",
    "total_equity",
    "unrealized_pnl",
    "position_count",
    "pending_buy_count",
]


def account_ledger_frame(state: dict[str, Any]) -> pd.DataFrame:
    trades = pd.DataFrame(state.get("trades", []))
    equity = pd.DataFrame(state.get("equity_curve", []))
    dates: set[str] = set()
    if not trades.empty and "date" in trades.columns:
        dates.update(str(value) for value in trades["date"].dropna().unique())
    if not equity.empty and "date" in equity.columns:
        dates.update(str(value) for value in equity["date"].dropna().unique())
    if not dates:
        return pd.DataFrame(columns=ACCOUNT_LEDGER_COLUMNS)

    if not trades.empty:
        for column in ["shares", "gross_notional", "notional", "fee", "net_cash_flow", "realized_pnl", "cash_after"]:
            if column in trades.columns:
                trades[column] = pd.to_numeric(trades[column], errors="coerce")
        if "gross_notional" not in trades.columns and "notional" in trades.columns:
            trades["gross_notional"] = trades["notional"]
    if not equity.empty:
        for column in ["cash", "market_value", "total_equity", "unrealized_pnl", "position_count", "pending_buy_count"]:
            if column in equity.columns:
                equity[column] = pd.to_numeric(equity[column], errors="coerce")

    rows: list[dict[str, Any]] = []
    for date in sorted(dates):
        day_trades = trades[trades["date"].astype(str) == date] if not trades.empty and "date" in trades.columns else pd.DataFrame()
        buys = day_trades[day_trades.get("action", pd.Series(index=day_trades.index, dtype=object)) == "BUY"]
        sells = day_trades[day_trades.get("action", pd.Series(index=day_trades.index, dtype=object)) == "SELL"]

        equity_row: pd.Series | None = None
        if not equity.empty and "date" in equity.columns:
            day_equity = equity[equity["date"].astype(str) == date]
            if not day_equity.empty:
                equity_row = day_equity.iloc[-1]
        last_cash_after = None
        if not day_trades.empty and "cash_after" in day_trades.columns:
            cash_values = day_trades["cash_after"].dropna()
            if not cash_values.empty:
                last_cash_after = float(cash_values.iloc[-1])

        rows.append(
            {
                "date": date,
                "trade_count": int(len(buys) + len(sells)),
                "buy_count": int(len(buys)),
                "buy_shares": int(buys["shares"].sum()) if "shares" in buys.columns else 0,
                "buy_notional": float(buys["gross_notional"].sum()) if "gross_notional" in buys.columns else 0.0,
                "buy_fee": float(buys["fee"].sum()) if "fee" in buys.columns else 0.0,
                "buy_cash_out": float(-buys["net_cash_flow"].sum()) if "net_cash_flow" in buys.columns else 0.0,
                "sell_count": int(len(sells)),
                "sell_shares": int(sells["shares"].sum()) if "shares" in sells.columns else 0,
                "sell_notional": float(sells["gross_notional"].sum()) if "gross_notional" in sells.columns else 0.0,
                "sell_fee": float(sells["fee"].sum()) if "fee" in sells.columns else 0.0,
                "sell_cash_in": float(sells["net_cash_flow"].sum()) if "net_cash_flow" in sells.columns else 0.0,
                "net_cash_flow": float(day_trades["net_cash_flow"].sum()) if "net_cash_flow" in day_trades.columns else 0.0,
                "realized_pnl": float(sells["realized_pnl"].sum()) if "realized_pnl" in sells.columns else 0.0,
                "cash": float(equity_row.get("cash")) if equity_row is not None and pd.notna(equity_row.get("cash")) else last_cash_after,
                "market_value": float(equity_row.get("market_value"))
                if equity_row is not None and pd.notna(equity_row.get("market_value"))
                else None,
                "total_equity": float(equity_row.get("total_equity"))
                if equity_row is not None and pd.notna(equity_row.get("total_equity"))
                else None,
                "unrealized_pnl": float(equity_row.get("unrealized_pnl"))
                if equity_row is not None and pd.notna(equity_row.get("unrealized_pnl"))
                else None,
                "position_count": int(equity_row.get("position_count"))
                if equity_row is not None and pd.notna(equity_row.get("position_count"))
                else None,
                "pending_buy_count": int(equity_row.get("pending_buy_count"))
                if equity_row is not None and pd.notna(equity_row.get("pending_buy_count"))
                else None,
            }
        )
    return pd.DataFrame(rows, columns=ACCOUNT_LEDGER_COLUMNS)


def trade_ledger_frame(state: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(state.get("trades", []))
