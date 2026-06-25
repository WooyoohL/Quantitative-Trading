from __future__ import annotations

import argparse
import json
import mimetypes
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "ui_assets"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def project_path(value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def path_info(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    exists = path.exists()
    return {
        "path": relative_path(path),
        "exists": exists,
        "size": path.stat().st_size if exists and path.is_file() else None,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)) if exists else None,
    }


def frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", force_ascii=False, date_format="iso"))


def read_csv_frame(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def compact_summary(run_dir: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "summary.json")
    if not summary:
        return {
            "run_dir": relative_path(run_dir),
            "exists": run_dir.exists(),
            "model_name": run_dir.name,
        }
    fields = [
        "model_name",
        "signal_date",
        "next_trade_date",
        "training_mode",
        "universe_size",
        "candidate_rank_count",
        "recommendation_pool_count",
        "review_top_k_count",
        "recommendation_max_latest_price",
        "right_side_filter_applied",
        "right_side_filter_before_count",
        "right_side_filter_after_count",
        "source_recommendation_label",
        "source_best_valid_ic",
        "source_test_daily_ic",
    ]
    out = {key: summary.get(key) for key in fields}
    out["run_dir"] = relative_path(run_dir)
    out["updated_at"] = path_info(run_dir / "summary.json").get("updated_at")
    return out


def load_state() -> dict[str, Any]:
    return read_json(PROJECT_ROOT / "outputs" / "paper_trading" / "state.json")


def latest_state_inference_dirs(state: dict[str, Any]) -> list[Path]:
    dirs: list[Path] = []
    for value in state.get("last_inference_dirs") or []:
        path = project_path(value)
        if path is not None and path.exists():
            dirs.append(path)
    if dirs:
        return dirs
    runs_root = PROJECT_ROOT / "outputs" / "inference_runs"
    if not runs_root.exists():
        return []
    recent = sorted((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    seen_models: set[str] = set()
    for run_dir in recent:
        model = read_json(run_dir / "summary.json").get("model_name") or run_dir.name
        if model in seen_models:
            continue
        seen_models.add(str(model))
        dirs.append(run_dir)
        if len(dirs) >= 2:
            break
    return dirs


def recent_inference_runs(limit: int = 12) -> list[dict[str, Any]]:
    runs_root = PROJECT_ROOT / "outputs" / "inference_runs"
    if not runs_root.exists():
        return []
    dirs = sorted((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    return [compact_summary(path) for path in dirs[:limit]]


def table_sources(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    paper_root = PROJECT_ROOT / "outputs" / "paper_trading"
    sources: dict[str, dict[str, Any]] = {
        "final_buy_plan": {"title": "最终买入计划", "path": paper_root / "latest_final_buy_plan.csv"},
        "candidate_universe": {"title": "当前候选池", "path": paper_root / "latest_candidate_universe.csv"},
        "filter_input": {"title": "事件筛选输入", "path": paper_root / "latest_filter_input.csv"},
        "event_decisions": {"title": "事件筛选结论", "path": paper_root / "latest_event_filter_decisions.csv"},
        "account_ledger": {"title": "账户总账", "path": paper_root / "account_ledger.csv"},
        "trade_ledger": {"title": "交易流水", "path": paper_root / "trade_ledger.csv"},
        "buy_tracking": {"title": "买入执行跟踪", "path": paper_root / "latest_buy_execution_tracking.csv"},
        "positions": {"title": "当前持仓", "records": state.get("positions", [])},
        "pending_buys": {"title": "待买计划", "records": state.get("pending_buys", [])},
        "equity_curve": {"title": "权益曲线", "records": state.get("equity_curve", [])},
    }
    for run_dir in latest_state_inference_dirs(state):
        summary = read_json(run_dir / "summary.json")
        label = str(summary.get("model_name") or run_dir.name).lower().replace(" ", "_")
        sources[f"{label}_review"] = {"title": f"{label} 复核 TopK", "path": run_dir / "review_top_k.csv"}
        sources[f"{label}_rank"] = {"title": f"{label} 候选排序", "path": run_dir / "candidate_rank.csv"}
        sources[f"{label}_predictions"] = {"title": f"{label} 推理明细", "path": run_dir / "inference_predictions.csv"}
    return sources


def safe_int(value: str | None, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except ValueError:
        parsed = default
    return max(low, min(high, parsed))


def table_payload(name: str, query: dict[str, list[str]]) -> dict[str, Any]:
    state = load_state()
    sources = table_sources(state)
    source = sources.get(name)
    if source is None:
        raise KeyError(f"Unknown table: {name}")

    if "records" in source:
        frame = pd.DataFrame(source.get("records") or [])
        info = {"path": None, "exists": True}
    else:
        path = source.get("path")
        frame = read_csv_frame(path)
        info = path_info(path)

    total_count = len(frame)
    search = (query.get("search") or [""])[0].strip()
    if search and not frame.empty:
        mask = frame.astype(str).apply(
            lambda column: column.str.contains(search, case=False, na=False, regex=False)
        ).any(axis=1)
        frame = frame[mask].copy()

    sort_column = (query.get("sort") or [""])[0]
    direction = (query.get("direction") or ["asc"])[0]
    if sort_column in frame.columns:
        frame = frame.sort_values(sort_column, ascending=direction != "desc", kind="mergesort")

    filtered_count = len(frame)
    offset = safe_int((query.get("offset") or [None])[0], default=0, low=0, high=max(filtered_count, 0))
    limit = safe_int((query.get("limit") or [None])[0], default=500, low=1, high=5000)
    page = frame.iloc[offset : offset + limit].copy()

    return {
        "name": name,
        "title": source.get("title", name),
        "columns": list(page.columns),
        "rows": frame_to_records(page),
        "count": filtered_count,
        "total_count": total_count,
        "offset": offset,
        "limit": limit,
        "source": info,
    }


def config_summary() -> dict[str, Any]:
    base_config = read_yaml(PROJECT_ROOT / "config.yaml")
    paper_config = read_yaml(PROJECT_ROOT / "paper_trading.yaml")
    universe_filters = base_config.get("universe", {}).get("filters", {})
    training_cfg = base_config.get("training", {})
    paper_cfg = paper_config.get("paper_trading", {})
    return {
        "candidate_path": base_config.get("data", {}).get("candidate_path"),
        "base_candidate_path": base_config.get("data", {}).get("base_candidate_path"),
        "recommendation_max_latest_price": universe_filters.get("max_latest_price"),
        "training_universe_max_latest_price": training_cfg.get("universe_max_latest_price"),
        "use_candidate_universe": training_cfg.get("use_candidate_universe"),
        "right_side_filter": base_config.get("strategy", {}).get("right_side_filter", {}),
        "heat_max_price": paper_cfg.get("heat", {}).get("max_price"),
        "final_buy_count": paper_cfg.get("candidates", {}).get("final_buy_count"),
        "watch_buy_count": paper_cfg.get("candidates", {}).get("watch_buy_count"),
        "retrain_weekdays": paper_cfg.get("retrain_weekdays"),
        "source_runs": paper_cfg.get("candidate_models"),
    }


def recent_log_files(limit: int = 12) -> list[dict[str, Any]]:
    if not OUTPUT_ROOT.exists():
        return []
    files = sorted(OUTPUT_ROOT.rglob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [path_info(path) for path in files[:limit] if is_inside(path, OUTPUT_ROOT)]


def overview_payload() -> dict[str, Any]:
    state = load_state()
    latest_run = read_json(PROJECT_ROOT / "outputs" / "latest_run.json")
    tables = table_sources(state)
    final_plan = table_payload("final_buy_plan", {"limit": ["100"]})
    account = table_payload("account_ledger", {"limit": ["100"]})
    trades = table_payload("trade_ledger", {"limit": ["100"]})
    equity_curve = state.get("equity_curve") or []
    latest_equity = equity_curve[-1] if equity_curve else {}
    inference_dirs = latest_state_inference_dirs(state)

    return {
        "state": state,
        "latest_run": latest_run,
        "config": config_summary(),
        "candidate_universe": state.get("candidate_universe") or {},
        "kpis": {
            "initial_cash": state.get("initial_cash"),
            "cash": state.get("cash"),
            "total_equity": latest_equity.get("total_equity"),
            "position_count": len(state.get("positions") or []),
            "pending_buy_count": len(state.get("pending_buys") or []),
            "trade_count": len(state.get("trades") or []),
            "last_prepare_date": state.get("last_prepare_date"),
            "last_finalize_date": state.get("last_finalize_date"),
        },
        "final_plan": final_plan,
        "account_ledger": account,
        "trade_ledger": trades,
        "tables": [
            {"name": name, "title": source.get("title", name), "source": path_info(source.get("path"))}
            if "path" in source
            else {"name": name, "title": source.get("title", name), "source": {"path": None, "exists": True}}
            for name, source in tables.items()
        ],
        "inference_summaries": [compact_summary(path) for path in inference_dirs],
        "recent_inference_runs": recent_inference_runs(),
        "recent_logs": recent_log_files(),
    }


ALLOWED_ACTIONS: dict[str, dict[str, Any]] = {
    "paper_prepare": {
        "label": "每日准备",
        "command": [sys.executable, "main.py", "paper-trade", "--stage", "prepare"],
        "description": "更新数据、按计划训练、生成候选池和事件筛选输入。",
    },
    "paper_prepare_skip_update_train": {
        "label": "准备：跳过数据与训练",
        "command": [
            sys.executable,
            "main.py",
            "paper-trade",
            "--stage",
            "prepare",
            "--skip-update",
            "--skip-train",
            "--allow-stale-data",
        ],
        "description": "复用当前本地数据与已有模型，重新生成候选与推理文件。",
    },
    "paper_finalize": {
        "label": "定稿买入计划",
        "command": [sys.executable, "main.py", "paper-trade", "--stage", "finalize"],
        "description": "读取事件筛选结论，生成最终买入计划并更新模拟账户文件。",
    },
    "check_data": {
        "label": "数据状态检查",
        "command": [sys.executable, "main.py", "check-data"],
        "description": "检查本地行情、候选池和基础文件状态。",
    },
}


def actions_payload() -> dict[str, Any]:
    return {
        "actions": [
            {
                "name": name,
                "label": spec["label"],
                "description": spec["description"],
                "command": " ".join(spec["command"]),
            }
            for name, spec in ALLOWED_ACTIONS.items()
        ]
    }


def write_job_meta(job: dict[str, Any]) -> None:
    log_path = Path(job["log_path"])
    meta_path = log_path.with_suffix(".json")
    meta_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")


def run_job(job_id: str, command: list[str], log_path: Path) -> None:
    started = time.time()
    with JOBS_LOCK:
        job = JOBS[job_id]
        job["status"] = "running"
        write_job_meta(job)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    return_code: int | None = None
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
        return_code = process.wait()
        handle.write(f"\n[ui] process exited with code {return_code}\n")

    with JOBS_LOCK:
        job = JOBS[job_id]
        job["status"] = "succeeded" if return_code == 0 else "failed"
        job["return_code"] = return_code
        job["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        job["duration_seconds"] = round(time.time() - started, 2)
        write_job_meta(job)


def start_job(action_name: str) -> dict[str, Any]:
    spec = ALLOWED_ACTIONS.get(action_name)
    if spec is None:
        raise KeyError(f"Unknown action: {action_name}")

    job_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    log_path = PROJECT_ROOT / "outputs" / "ui_jobs" / f"{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    job = {
        "id": job_id,
        "action": action_name,
        "label": spec["label"],
        "command": spec["command"],
        "status": "queued",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None,
        "return_code": None,
        "duration_seconds": None,
        "log_path": relative_path(log_path),
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
        write_job_meta(job)

    thread = threading.Thread(target=run_job, args=(job_id, spec["command"], log_path), daemon=True)
    thread.start()
    return job


def job_payload(job_id: str | None = None) -> dict[str, Any]:
    with JOBS_LOCK:
        if job_id:
            job = dict(JOBS.get(job_id) or {})
            return {"job": job, "log": read_log_tail(project_path(job.get("log_path")) if job else None)}
        jobs = [dict(value) for value in JOBS.values()]
    jobs.sort(key=lambda value: value.get("created_at") or "", reverse=True)
    return {"jobs": jobs}


def read_log_tail(path: Path | None, max_chars: int = 60000) -> str:
    if path is None or not path.exists() or path.suffix.lower() != ".log":
        return ""
    if not is_inside(path, OUTPUT_ROOT):
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def log_payload(query: dict[str, list[str]]) -> dict[str, Any]:
    raw = (query.get("path") or [""])[0]
    path = project_path(unquote(raw))
    return {"source": path_info(path), "text": read_log_tail(path)}


class QuantUiHandler(BaseHTTPRequestHandler):
    server_version = "QuantUi/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("[ui] " + format % args + "\n")

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        self.send_json({"error": message}, status=status)

    def send_static(self, path: Path) -> None:
        if not path.exists() or not path.is_file() or not is_inside(path, STATIC_ROOT):
            self.send_error_json(HTTPStatus.NOT_FOUND, "Not found")
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path == "/":
                self.send_static(STATIC_ROOT / "index.html")
                return
            if path.startswith("/assets/"):
                rel = unquote(path.removeprefix("/assets/"))
                self.send_static(STATIC_ROOT / rel)
                return
            if path == "/api/overview":
                self.send_json(overview_payload())
                return
            if path == "/api/actions":
                self.send_json(actions_payload())
                return
            if path == "/api/jobs":
                self.send_json(job_payload())
                return
            if path.startswith("/api/jobs/"):
                self.send_json(job_payload(path.rsplit("/", 1)[-1]))
                return
            if path == "/api/table":
                name = (query.get("name") or [""])[0]
                self.send_json(table_payload(name, query))
                return
            if path == "/api/log":
                self.send_json(log_payload(query))
                return
            self.send_error_json(HTTPStatus.NOT_FOUND, "Not found")
        except KeyError as exc:
            self.send_error_json(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:  # pragma: no cover - visible in local UI.
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/jobs":
            self.send_error_json(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            action = str(payload.get("action") or "")
            self.send_json({"job": start_job(action)}, status=HTTPStatus.ACCEPTED)
        except KeyError as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover - visible in local UI.
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local quantitative trading UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), QuantUiHandler)
    print(f"Quant UI: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
