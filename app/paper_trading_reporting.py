from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.paper_trading_config import paper_cfg
from execution.paper_account import account_ledger_frame, buy_plan_frame, trade_ledger_frame


def workbook_path(config: dict[str, Any], project_root: Path) -> Path:
    return project_root / str(paper_cfg(config).get("workbook_path", "outputs/paper_trading/paper_account.xlsx"))


DISPLAY_COLUMN_NAMES = {
    "initial_cash": "初始现金",
    "cash": "剩余现金",
    "current_source_run": "当前模型源",
    "last_training_date": "最近训练日期",
    "last_prepare_date": "最近准备日期",
    "last_finalize_date": "最近定稿日期",
    "position_count": "持仓数量",
    "pending_buy_count": "待买数量",
    "date": "日期",
    "action": "操作",
    "trade_count": "成交笔数",
    "buy_count": "买入笔数",
    "buy_shares": "买入股数",
    "buy_notional": "买入成交额",
    "buy_cash_out": "买入现金支出",
    "sell_count": "卖出笔数",
    "sell_shares": "卖出股数",
    "sell_notional": "卖出成交额",
    "sell_cash_in": "卖出现金回收",
    "symbol": "代码",
    "name": "名称",
    "shares": "股数",
    "price": "价格",
    "open_price": "开盘价",
    "notional": "成交金额",
    "gross_notional": "成交总额",
    "net_cash_flow": "净现金流",
    "cash_after": "交易后现金",
    "realized_pnl": "已实现盈亏",
    "entry_price": "入场价",
    "entry_open_price": "入场开盘价",
    "cost": "成本",
    "gross_cost": "买入成交金额",
    "signal_date": "信号日期",
    "buy_date": "买入日期",
    "actual_buy_date": "实际买入日期",
    "sell_date": "卖出日期",
    "planned_sell_date": "计划卖出日期",
    "source": "来源",
    "buy_intent": "买入类型",
    "rank": "计划序号",
    "review_rank": "审查排序",
    "recommended_action": "建议动作",
    "risk_level": "风险级别",
    "positive_catalyst_level": "正向催化级别",
    "market_value": "持仓市值",
    "total_equity": "总权益",
    "unrealized_pnl": "未实现盈亏",
    "next_trade_date": "下个交易日",
    "industry_name": "行业",
    "score": "模型分数",
    "candidate_rank": "候选排序",
    "model_rank": "模型排序",
    "model_group_rank": "模型内排序",
    "model_source": "模型来源",
    "model_source_order": "模型来源序号",
    "model_layer_conclusion": "模型层结论",
    "model_layer_risk": "模型层风险",
    "model_layer_reason": "模型层原因",
    "source_valid_daily_ic": "来源验证DailyIC",
    "source_test_daily_ic": "来源测试DailyIC",
    "source_run": "模型运行目录",
    "heat_rank": "热度排序",
    "heat_score": "热度分数",
    "close": "收盘价",
    "buy_price": "参考买入价",
    "expected_buy_price": "预期买入价",
    "expected_buy_execution_price": "预估滑点成交价",
    "actual_open_price": "次日开盘价",
    "open_vs_expected_price": "开盘价偏差",
    "open_vs_expected_pct": "开盘价偏差率",
    "actual_execution_price": "实际成交价",
    "actual_shares": "实际股数",
    "actual_notional": "实际成交金额",
    "execution_status": "执行状态",
    "target_exposure": "目标仓位",
    "buy_slippage_rate": "买入滑点率",
    "sell_slippage_rate": "卖出滑点率",
    "fee_rate": "手续费率",
    "buy_fee": "买入手续费",
    "sell_fee": "卖出手续费",
    "fee": "手续费",
    "total_cash_out": "买入总支出",
    "ret_1": "一日收益",
    "ret_5": "五日收益",
    "intraday_ret": "日内收益",
    "ma_gap_5": "五日均线偏离",
    "turnover_rate_1": "换手率",
    "volume_ratio_1_prev": "一日量比",
    "volume_ratio_3_prev": "三日量比",
    "volume_ratio_5_prev": "五日量比_前期均量",
    "volume_ratio_7_prev": "七日量比",
    "volume_ratio_5": "五日量比",
    "price_volume_layer_conclusion": "量价层结论",
    "price_volume_layer_risk": "量价层风险",
    "price_volume_layer_reason": "量价层原因",
    "industry_ret_1_mean": "行业一日均值",
    "heat_reasons": "热度原因",
    "event_layer_conclusion": "事件层结论",
    "event_layer_risk": "事件层风险",
    "event_layer_reason": "事件层原因",
    "key_negative_events": "关键负面事件",
    "key_positive_events": "关键正面事件",
    "summary": "结论",
    "sources": "来源链接",
}


DISPLAY_VALUE_MAP = {
    "action": {
        "BUY": "买入",
        "SELL": "卖出",
        "BUY_SKIPPED_NO_OPEN": "买入跳过：无开盘价",
        "BUY_SKIPPED_CASH_OR_LOT": "买入跳过：现金或手数不足",
        "SELL_SKIPPED_NO_OPEN": "卖出跳过：无开盘价",
    },
    "recommended_action": {
        "Keep": "保留",
        "Watch buy": "观察买入",
        "Exclude": "排除",
        "Manual review": "人工复审",
        "keep": "保留",
        "watch buy": "观察买入",
        "exclude": "排除",
        "manual review": "人工复审",
    },
    "buy_intent": {
        "Strong buy": "强买入",
        "Watch buy": "观察买入",
        "strong buy": "强买入",
        "watch buy": "观察买入",
    },
    "risk_level": {
        "High": "高",
        "Medium": "中",
        "Low": "低",
        "high": "高",
        "medium": "中",
        "low": "低",
    },
    "positive_catalyst_level": {
        "High": "高",
        "Medium": "中",
        "Low": "低",
        "high": "高",
        "medium": "中",
        "low": "低",
    },
    "source": {
        "model": "模型",
        "heat": "热度",
        "model+heat": "模型+热度",
        "attention": "Attention模型",
        "ridge": "岭回归模型",
        "attention+ridge": "Attention模型+岭回归模型",
        "attention+heat": "Attention模型+热度",
        "ridge+heat": "岭回归模型+热度",
        "attention+ridge+heat": "Attention模型+岭回归模型+热度",
    },
    "model_source": {
        "attention": "Attention模型",
        "ridge": "岭回归模型",
        "attention+ridge": "Attention模型+岭回归模型",
    },
    "execution_status": {
        "PLANNED": "计划买入",
        "BOUGHT": "已买入",
        "SKIPPED_NO_OPEN": "未买入：无开盘价",
        "SKIPPED_CASH_OR_LOT": "未买入：现金或手数不足",
        "SKIPPED_EXPOSURE_CAP": "未买入：敞口上限",
        "SKIPPED_MAX_POSITIONS": "未买入：持仓数量上限",
    },
}


SHEET_NAME_MAP = {
    "account_summary": "账户概览",
    "account_ledger": "资金总账",
    "positions": "当前持仓",
    "pending_buys": "待买计划",
    "buy_plan_history": "买入执行跟踪",
    "trades": "交易流水",
    "equity_curve": "权益曲线",
    "latest_candidates": "候选池",
    "event_decisions": "事件筛选",
}


def to_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.rename(columns=DISPLAY_COLUMN_NAMES)
    out = frame.copy()
    for column, mapping in DISPLAY_VALUE_MAP.items():
        if column in out.columns:
            out[column] = out[column].map(lambda value: mapping.get(str(value), value) if pd.notna(value) else value)
    return out.rename(columns=DISPLAY_COLUMN_NAMES)


def dated_paper_trading_path(project_root: Path, state: dict[str, Any], filename: str) -> Path | None:
    date_value = state.get("last_finalize_date") or state.get("last_prepare_date")
    if not date_value:
        return None
    return project_root / "outputs" / "paper_trading" / pd.Timestamp(date_value).strftime("%Y%m%d") / filename


def write_buy_plan_tracking_files(
    config: dict[str, Any],
    state: dict[str, Any],
    *,
    project_root: Path,
    extra_signal_dates: list[str] | None = None,
) -> None:
    history = buy_plan_frame(state.get("buy_plan_history", []), include_execution_columns=True)
    latest_path = project_root / "outputs" / "paper_trading" / "latest_buy_execution_tracking.csv"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    to_display_frame(history).to_csv(latest_path, index=False, encoding="utf-8-sig")
    written_days: set[str] = set()
    for signal_date, group in history.groupby("signal_date", dropna=True):
        if pd.isna(signal_date):
            continue
        day = pd.Timestamp(signal_date).strftime("%Y%m%d")
        written_days.add(day)
        dated_path = project_root / "outputs" / "paper_trading" / day / "buy_execution_tracking.csv"
        dated_path.parent.mkdir(parents=True, exist_ok=True)
        to_display_frame(group.reset_index(drop=True)).to_csv(dated_path, index=False, encoding="utf-8-sig")
    for signal_date in extra_signal_dates or []:
        day = pd.Timestamp(signal_date).strftime("%Y%m%d")
        if day in written_days:
            continue
        dated_path = project_root / "outputs" / "paper_trading" / day / "buy_execution_tracking.csv"
        dated_path.parent.mkdir(parents=True, exist_ok=True)
        empty = buy_plan_frame([], include_execution_columns=True)
        to_display_frame(empty).to_csv(dated_path, index=False, encoding="utf-8-sig")


def write_account_ledger_files(config: dict[str, Any], state: dict[str, Any], *, project_root: Path) -> None:
    output_dir = workbook_path(config, project_root).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    to_display_frame(account_ledger_frame(state)).to_csv(
        output_dir / "account_ledger.csv",
        index=False,
        encoding="utf-8-sig",
    )
    to_display_frame(trade_ledger_frame(state)).to_csv(
        output_dir / "trade_ledger.csv",
        index=False,
        encoding="utf-8-sig",
    )


def write_workbook(
    *,
    config: dict[str, Any],
    state: dict[str, Any],
    project_root: Path,
    latest_candidates: pd.DataFrame | None = None,
    event_decisions: pd.DataFrame | None = None,
) -> None:
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
        "account_ledger": account_ledger_frame(state),
        "positions": pd.DataFrame(state.get("positions", [])),
        "pending_buys": pd.DataFrame(state.get("pending_buys", [])),
        "buy_plan_history": pd.DataFrame(state.get("buy_plan_history", [])),
        "trades": pd.DataFrame(state.get("trades", [])),
        "equity_curve": pd.DataFrame(state.get("equity_curve", [])),
        "latest_candidates": latest_candidates if latest_candidates is not None else pd.DataFrame(),
        "event_decisions": event_decisions if event_decisions is not None else pd.DataFrame(),
    }

    paths = [workbook_path(config, project_root)]
    dated_path = dated_paper_trading_path(project_root, state, "paper_account.xlsx")
    if dated_path is not None and str(dated_path) not in {str(path) for path in paths}:
        paths.append(dated_path)

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, frame in frames.items():
                display_frame = to_display_frame(frame)
                display_sheet_name = SHEET_NAME_MAP.get(sheet_name, sheet_name)[:31]
                display_frame.to_excel(writer, sheet_name=display_sheet_name, index=False)
