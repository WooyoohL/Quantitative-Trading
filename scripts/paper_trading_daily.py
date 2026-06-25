from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.runtime import build_fetcher, load_config
from app.paper_trading_config import paper_cfg
from app.paper_trading_reporting import (
    to_display_frame,
    write_account_ledger_files,
    write_buy_plan_tracking_files,
    write_workbook,
)
from data.fetcher import (
    load_local_industry_daily,
    load_local_industry_map,
    load_local_stock_data,
)
from execution.paper_account import (
    buy_plan_frame,
    maybe_float,
    mark_to_market,
    trading_cost_rates,
    update_ledger_for_trade_date,
    upsert_buy_plan_history,
)
from pipelines.paper_trading_candidates import (
    apply_basic_candidate_filters,
    attach_raw_liquidity_metrics,
    build_unified_candidate_universe,
    combine_candidates,
    read_model_quality,
    standardize_model_candidates,
    validate_candidate_signal_date,
)


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


def state_path(config: dict[str, Any]) -> Path:
    return PROJECT_ROOT / str(paper_cfg(config).get("state_path", "outputs/paper_trading/state.json"))


def current_source_path(config: dict[str, Any]) -> Path:
    return PROJECT_ROOT / str(
        paper_cfg(config).get("current_source_run_path", "outputs/paper_trading/current_source_run.txt")
    )


def as_project_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


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
        "buy_plan_history": [],
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


def source_run_exists(source_run: str | None, checkpoint_name: str = "best.ckpt") -> bool:
    if not source_run:
        return False
    path = as_project_path(source_run)
    return (path / "config.yaml").exists() and (path / checkpoint_name).exists()


def resolve_source_run(config: dict[str, Any], state: dict[str, Any]) -> str | None:
    configured = paper_cfg(config).get("model", {}).get("source_run")
    pointer = current_source_path(config)
    if pointer.exists():
        value = pointer.read_text(encoding="utf-8").strip()
        if source_run_exists(value):
            state["current_source_run"] = value
            return value
    if source_run_exists(state.get("current_source_run")):
        return str(state.get("current_source_run"))
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


def safe_name_part(value: str) -> str:
    text = str(value).strip().lower()
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text) or "model"


def source_run_model_name(source_run: str | Path | None) -> str | None:
    if not source_run:
        return None
    path = as_project_path(source_run)
    summary_path = path / "summary.json"
    if summary_path.exists():
        value = read_json(summary_path).get("model_name")
        if value:
            return safe_name_part(str(value))
    config_path = path / "config.yaml"
    if config_path.exists():
        model_cfg = read_yaml(config_path).get("model", {})
        value = model_cfg.get("name")
        if value:
            return safe_name_part(str(value))
    return None


def candidate_model_entries(
    *,
    config: dict[str, Any],
    primary_source_run: str,
) -> list[dict[str, Any]]:
    cfg = paper_cfg(config)
    infer_cfg = cfg.get("inference", {})
    candidates_cfg = cfg.get("candidates", {})
    default_checkpoint = str(infer_cfg.get("checkpoint_name", "best.ckpt"))
    default_limit = int(candidates_cfg.get("model_review_limit", 20))
    configured_models = cfg.get("candidate_models")

    if not configured_models:
        return [
            {
                "label": "model",
                "source_run": primary_source_run,
                "checkpoint_name": default_checkpoint,
                "output_prefix": str(infer_cfg.get("output_prefix", "paper_signal")),
                "review_limit": default_limit,
            }
        ]

    entries: list[dict[str, Any]] = []
    primary_model_name = source_run_model_name(primary_source_run)
    for index, item in enumerate(configured_models, start=1):
        if not isinstance(item, dict):
            raise ValueError("paper_trading.candidate_models entries must be mappings.")
        label = safe_name_part(str(item.get("label") or item.get("name") or f"model_{index}"))
        if primary_model_name and label == primary_model_name:
            source_run = str(primary_source_run)
        else:
            source_run = str(item.get("source_run") or primary_source_run)
        checkpoint_name = str(item.get("checkpoint_name") or default_checkpoint)
        if not source_run_exists(source_run, checkpoint_name=checkpoint_name):
            raise FileNotFoundError(f"Configured source run is not usable for {label}: {source_run}")
        output_prefix = str(item.get("output_prefix") or f"{infer_cfg.get('output_prefix', 'paper_signal')}_{label}")
        entries.append(
            {
                "label": label,
                "source_run": source_run,
                "checkpoint_name": checkpoint_name,
                "output_prefix": output_prefix,
                "review_limit": int(item.get("review_limit", default_limit)),
            }
        )
    return entries


def run_inference(
    *,
    config: dict[str, Any],
    source_run: str,
    trade_date: pd.Timestamp,
    checkpoint_name: str | None = None,
    output_prefix: str | None = None,
) -> Path:
    infer_cfg = paper_cfg(config).get("inference", {})
    prefix = str(output_prefix or infer_cfg.get("output_prefix", "paper_signal"))
    output_name = f"{prefix}_{trade_date.strftime('%Y%m%d')}_{pd.Timestamp.now().strftime('%H%M%S')}"
    checkpoint_name = str(checkpoint_name or infer_cfg.get("checkpoint_name", "best.ckpt"))
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
        "event_layer_conclusion",
        "event_layer_risk",
        "event_layer_reason",
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
                "# 事件筛选任务",
                "",
                f"规则文件: {prompt_path}",
                f"候选输入: {filter_input}",
                f"待填写决策表: {decision_template}",
                "",
                "候选输入已包含模型层和量价层结论。事件筛选只补充事件层结论，最终动作再综合三层判断。",
                "recommended_action=Keep 的行进入强买入计划；recommended_action=Watch buy 的行在通过结构性校验后进入观察买入计划。",
            ]
        ),
        encoding="utf-8",
    )
    return filter_input, decision_template


def prepare_stage(args: argparse.Namespace, config: dict[str, Any]) -> None:
    base_config_path, base_config = load_base_config(config)
    state = load_state(config)

    if not args.skip_update:
        run_command([sys.executable, "main.py", "update-data", "--config", str(base_config_path)])

    stock_df, industry_map_df, industry_daily_df = load_local_frames(base_config)
    trade_date = latest_trade_date(stock_df, base_config, allow_stale_data=bool(args.allow_stale_data))
    update_ledger_for_trade_date(state, stock_df, trade_date, config)
    save_state(config, state)
    write_buy_plan_tracking_files(config, state, project_root=PROJECT_ROOT)
    write_account_ledger_files(config, state, project_root=PROJECT_ROOT)
    write_workbook(config=config, state=state, project_root=PROJECT_ROOT)
    candidate_universe, candidate_universe_meta = build_unified_candidate_universe(
        config=config,
        base_config=base_config,
        stock_df=stock_df,
        industry_map_df=industry_map_df,
        industry_daily_df=industry_daily_df,
        trade_date=trade_date,
        project_root=PROJECT_ROOT,
    )
    if candidate_universe.empty:
        raise ValueError("Unified candidate universe is empty after base filters and heat additions.")

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

    model_entries = candidate_model_entries(config=config, primary_source_run=source_run)
    inference_dirs: list[Path] = []
    model_frames: list[pd.DataFrame] = []
    for source_order, entry in enumerate(model_entries):
        inference_dir = run_inference(
            config=config,
            source_run=str(entry["source_run"]),
            trade_date=trade_date,
            checkpoint_name=str(entry["checkpoint_name"]),
            output_prefix=str(entry["output_prefix"]),
        )
        inference_dirs.append(inference_dir)
        model_frames.append(
            standardize_model_candidates(
                inference_dir / "review_top_k.csv",
                limit=int(entry["review_limit"]),
                model_source=str(entry["label"]),
                source_run=str(entry["source_run"]),
                source_order=source_order,
                model_quality=read_model_quality(inference_dir),
            )
        )

    event_review_limit = int(paper_cfg(config).get("candidates", {}).get("event_review_limit", 50))

    model_candidates = pd.concat(model_frames, ignore_index=True, sort=False) if model_frames else pd.DataFrame()
    validate_candidate_signal_date(model_candidates, trade_date)

    combined = combine_candidates(model_candidates, pd.DataFrame(), final_limit=event_review_limit)
    combined = apply_basic_candidate_filters(combined, base_config)
    combined = attach_raw_liquidity_metrics(combined, stock_df, trade_date)
    filter_input, decision_template = write_event_filter_files(config=config, candidates=combined, trade_date=trade_date)

    state["current_source_run"] = str(model_entries[0]["source_run"]) if model_entries else source_run
    state["model_source_runs"] = [
        {"label": str(entry["label"]), "source_run": str(entry["source_run"])} for entry in model_entries
    ]
    state["last_prepare_date"] = trade_date.date().isoformat()
    state["last_inference_dir"] = str(inference_dirs[0].relative_to(PROJECT_ROOT)) if inference_dirs else None
    state["last_inference_dirs"] = [str(path.relative_to(PROJECT_ROOT)) for path in inference_dirs]
    state["candidate_universe"] = candidate_universe_meta
    state["pending_filter_input_path"] = str(filter_input.relative_to(PROJECT_ROOT))
    state["pending_event_decisions_template_path"] = str(decision_template.relative_to(PROJECT_ROOT))
    save_state(config, state)
    write_buy_plan_tracking_files(config, state, project_root=PROJECT_ROOT)
    write_account_ledger_files(config, state, project_root=PROJECT_ROOT)
    write_workbook(config=config, state=state, project_root=PROJECT_ROOT, latest_candidates=combined)

    print(f"交易日期: {trade_date.date()}")
    print("模型源: " + "; ".join(f"{entry['label']}={entry['source_run']}" for entry in model_entries))
    print("推理目录: " + "; ".join(str(path) for path in inference_dirs))
    print(
        "统一候选库: "
        f"base={candidate_universe_meta['base_candidate_count']} "
        f"heat={candidate_universe_meta['heat_candidate_count']} "
        f"before_limits={candidate_universe_meta['candidate_universe_before_limits_count']} "
        f"final={candidate_universe_meta['candidate_universe_count']} "
        f"removed={candidate_universe_meta['candidate_universe_removed_by_limits_count']} "
        f"heat_after_limits={candidate_universe_meta['heat_candidate_after_limits_count']}"
    )
    print(f"基础候选池: {candidate_universe_meta['base_candidate_path']}")
    print(f"事件筛选输入: {filter_input}")
    print(f"事件决策表: {decision_template}")
    print(f"待事件审查候选数: {len(combined)}")


def normalize_action(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_layer_value(value: Any) -> str:
    return str(value or "").strip().lower()


def apply_clear_keep_policy(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty:
        return merged
    if "event_layer_conclusion" not in merged.columns or "price_volume_layer_conclusion" not in merged.columns:
        return merged
    out = merged.copy()
    event_conclusion = out.get("event_layer_conclusion", pd.Series("", index=out.index)).map(normalize_layer_value)
    price_volume_conclusion = out.get("price_volume_layer_conclusion", pd.Series("", index=out.index)).map(normalize_layer_value)

    hard_event_negative = event_conclusion.isin({"负向", "negative"})
    price_volume_reject = price_volume_conclusion.isin({"不支持", "unsupported"})
    event_neutral = event_conclusion.isin({"中性", "neutral", ""})
    price_volume_neutral_or_weak = price_volume_conclusion.isin({"中性", "neutral", "不支持", "unsupported", ""})

    clear_keep = ~(hard_event_negative | price_volume_reject | (event_neutral & price_volume_neutral_or_weak))
    return out[clear_keep].copy()


def apply_watch_buy_policy(merged: pd.DataFrame, *, review_rank_limit: int | None = None) -> pd.DataFrame:
    if merged.empty:
        return merged
    required_columns = {"event_layer_conclusion", "event_layer_risk", "price_volume_layer_conclusion"}
    if not required_columns.issubset(set(merged.columns)):
        return merged.iloc[0:0].copy()
    out = merged.copy()
    event_conclusion = out.get("event_layer_conclusion", pd.Series("", index=out.index)).map(normalize_layer_value)
    event_risk = out.get("event_layer_risk", pd.Series("", index=out.index)).map(normalize_layer_value)
    price_volume_conclusion = out.get("price_volume_layer_conclusion", pd.Series("", index=out.index)).map(
        normalize_layer_value
    )
    price_volume_reason = out.get("price_volume_layer_reason", pd.Series("", index=out.index)).astype(str)
    model_conclusion = out.get("model_layer_conclusion", pd.Series("", index=out.index)).astype(str).str.lower()

    event_ok = event_conclusion.isin({"中性", "neutral"})
    event_risk_ok = ~event_risk.isin({"高", "high"})
    price_volume_ok = price_volume_conclusion.isin({"中性", "neutral"})
    no_obvious_volume_weakening = ~price_volume_reason.str.contains("走弱", na=False)
    model_ok = model_conclusion.str.contains("观察|watch", na=False)
    rank_ok = pd.Series(True, index=out.index)
    if review_rank_limit is not None and "review_rank" in out.columns:
        rank_ok = pd.to_numeric(out["review_rank"], errors="coerce").le(int(review_rank_limit))

    watch_buy = event_ok & event_risk_ok & price_volume_ok & no_obvious_volume_weakening & model_ok & rank_ok
    return out[watch_buy].copy()


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
    candidates_cfg = paper_cfg(config).get("candidates", {})
    strong = merged[merged["recommended_action"].map(normalize_action) == "keep"].copy()
    strong = apply_clear_keep_policy(strong)
    final_count = int(candidates_cfg.get("final_buy_count", 3))
    strong = strong.sort_values(["review_rank", "symbol"], ascending=[True, True]).head(final_count).copy()
    strong["buy_intent"] = "Strong buy"
    strong["target_exposure"] = float(
        paper_cfg(config).get("per_position_target_exposure", paper_cfg(config).get("target_gross_exposure", 0.70))
    )

    watch_buy_count = int(candidates_cfg.get("watch_buy_count", 0))
    watch_review_rank_limit_value = candidates_cfg.get("watch_review_rank_limit")
    watch_review_rank_limit = int(watch_review_rank_limit_value) if watch_review_rank_limit_value is not None else None
    watch = merged[merged["recommended_action"].map(normalize_action).isin({"watch buy", "watch_buy"})].copy()
    watch = apply_watch_buy_policy(watch, review_rank_limit=watch_review_rank_limit)
    if not strong.empty and not watch.empty:
        strong_symbols = set(str(symbol) for symbol in strong["symbol"].astype(str))
        watch = watch[~watch["symbol"].astype(str).isin(strong_symbols)].copy()
    watch = watch.sort_values(["review_rank", "symbol"], ascending=[True, True]).head(watch_buy_count).copy()
    watch["buy_intent"] = "Watch buy"
    watch["target_exposure"] = float(paper_cfg(config).get("watch_position_target_exposure", 0.10))

    buy_rows = pd.concat([strong, watch], ignore_index=True, sort=False)

    signal_date = pd.Timestamp(state.get("last_prepare_date") or trade_date.date().isoformat()).normalize()
    fetcher = build_fetcher(base_config)
    if buy_rows.empty:
        buy_date = fetcher.next_trade_date(signal_date)
    else:
        buy_date = pd.Timestamp(buy_rows["next_trade_date"].dropna().iloc[0]).normalize()
    sell_date = fetcher.next_trade_date(buy_date)
    rates = trading_cost_rates(config)
    buy_slippage_rate = rates["buy_slippage_rate"]
    fee_rate = rates["fee_rate"]

    state["pending_buys"] = [
        plan
        for plan in state.get("pending_buys", [])
        if str(plan.get("signal_date")) != signal_date.date().isoformat()
    ]
    state["buy_plan_history"] = [
        plan
        for plan in state.get("buy_plan_history", [])
        if str(plan.get("signal_date")) != signal_date.date().isoformat()
    ]
    for index, row in buy_rows.reset_index(drop=True).iterrows():
        expected_buy_price = maybe_float(row.get("buy_price"))
        if expected_buy_price is None:
            expected_buy_price = maybe_float(row.get("close"))
        expected_execution_price = (
            float(expected_buy_price * (1.0 + buy_slippage_rate)) if expected_buy_price is not None else None
        )
        plan = {
            "signal_date": signal_date.date().isoformat(),
            "buy_date": buy_date.date().isoformat(),
            "sell_date": sell_date.date().isoformat(),
            "symbol": str(row["symbol"]),
            "name": str(row.get("name", row.get("name_event", "")) or ""),
            "source": str(row.get("source", "")),
            "buy_intent": str(row.get("buy_intent", "")),
            "rank": int(index + 1),
            "review_rank": int(row.get("review_rank", index + 1)),
            "recommended_action": str(row.get("recommended_action", "")),
            "risk_level": str(row.get("risk_level", "")),
            "positive_catalyst_level": str(row.get("positive_catalyst_level", "")),
            "expected_buy_price": expected_buy_price,
            "buy_slippage_rate": buy_slippage_rate,
            "expected_buy_execution_price": expected_execution_price,
            "fee_rate": fee_rate,
            "target_exposure": maybe_float(row.get("target_exposure")),
            "execution_status": "PLANNED",
        }
        state.setdefault("pending_buys", []).append(plan)
        upsert_buy_plan_history(state, plan)

    final_plan_dir = PROJECT_ROOT / "outputs" / "paper_trading" / signal_date.strftime("%Y%m%d")
    final_plan_dir.mkdir(parents=True, exist_ok=True)
    final_plan_path = final_plan_dir / "final_buy_plan.csv"
    current_pending = [
        plan
        for plan in state.get("pending_buys", [])
        if str(plan.get("signal_date")) == signal_date.date().isoformat()
    ]
    final_plan = to_display_frame(buy_plan_frame(current_pending, include_execution_columns=False))
    final_plan.to_csv(final_plan_path, index=False, encoding="utf-8-sig")
    latest_plan_path = PROJECT_ROOT / "outputs" / "paper_trading" / "latest_final_buy_plan.csv"
    latest_plan_path.parent.mkdir(parents=True, exist_ok=True)
    final_plan.to_csv(latest_plan_path, index=False, encoding="utf-8-sig")

    state["last_finalize_date"] = trade_date.date().isoformat()
    state["last_final_buy_plan_path"] = str(final_plan_path.relative_to(PROJECT_ROOT))
    state["latest_final_buy_plan_path"] = str(latest_plan_path.relative_to(PROJECT_ROOT))
    mark_to_market(state, stock_df, signal_date)
    save_state(config, state)
    write_buy_plan_tracking_files(
        config,
        state,
        project_root=PROJECT_ROOT,
        extra_signal_dates=[signal_date.date().isoformat()],
    )
    write_account_ledger_files(config, state, project_root=PROJECT_ROOT)
    write_workbook(
        config=config,
        state=state,
        project_root=PROJECT_ROOT,
        latest_candidates=candidates,
        event_decisions=decisions,
    )

    print(f"交易日期: {trade_date.date()}")
    print(f"信号日期: {signal_date.date()}")
    print(f"买入日期: {buy_date.date()}")
    print(f"卖出日期: {sell_date.date()}")
    print(f"强买入数量: {len(strong)}")
    print(f"观察买入数量: {len(watch)}")
    print(f"最终买入数量: {len(buy_rows)}")
    print(f"最终买入计划: {final_plan_path}")
    print("买入执行跟踪: outputs/paper_trading/latest_buy_execution_tracking.csv")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = read_yaml(PROJECT_ROOT / args.config)
    if args.stage == "prepare":
        prepare_stage(args, config)
    else:
        finalize_stage(args, config)


if __name__ == "__main__":
    main()
