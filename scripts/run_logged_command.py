from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a long command, write logs, and report progress at a fixed interval.")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def latest_non_empty_line(path: Path, max_bytes: int = 65536) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            content = handle.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return f"failed to read log: {exc}"
    lines = content.splitlines()
    for line in reversed(lines):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def normalize_command(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("Missing command after --.")
    return command


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    command = normalize_command(list(args.command))
    interval = max(1, int(args.interval))
    log_path = args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[LongTask] command={' '.join(command)}", flush=True)
    print(f"[LongTask] log={log_path.resolve()}", flush=True)
    print(f"[LongTask] interval_seconds={interval}", flush=True)

    started_at = time.monotonic()
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        while True:
            try:
                return_code = process.wait(timeout=interval)
            except subprocess.TimeoutExpired:
                elapsed = int(time.monotonic() - started_at)
                latest_line = latest_non_empty_line(log_path)
                if latest_line:
                    print(f"[LongTask] still_running elapsed_seconds={elapsed} latest_log_line={latest_line}", flush=True)
                else:
                    print(f"[LongTask] still_running elapsed_seconds={elapsed} latest_log_line=", flush=True)
                continue

            elapsed = int(time.monotonic() - started_at)
            latest_line = latest_non_empty_line(log_path)
            print(f"[LongTask] finished return_code={return_code} elapsed_seconds={elapsed}", flush=True)
            if latest_line:
                print(f"[LongTask] latest_log_line={latest_line}", flush=True)
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
            return


if __name__ == "__main__":
    main()
