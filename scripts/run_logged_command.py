from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a long command with output redirected to a log file.")
    parser.add_argument("--log", type=Path, required=True, help="Combined stdout/stderr log path.")
    parser.add_argument("--interval", type=int, default=600, help="Status interval in seconds.")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Working directory for the child command.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --.")
    return parser.parse_args(argv)


def tail_lines(path: Path, limit: int = 20) -> list[str]:
    if not path.exists():
        return []
    lines: deque[str] = deque(maxlen=limit)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lines.append(line.rstrip("\n"))
    return list(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("Missing child command after --.")

    log_path = args.log.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    interval = max(1, int(args.interval))
    started = time.monotonic()
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[LoggedCommand] start={started_at}")
    print(f"[LoggedCommand] cwd={args.cwd.resolve()}")
    print(f"[LoggedCommand] log={log_path}")
    print(f"[LoggedCommand] command={' '.join(command)}")

    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=args.cwd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        while True:
            try:
                return_code = process.wait(timeout=interval)
                break
            except subprocess.TimeoutExpired:
                elapsed = int(time.monotonic() - started)
                size = log_path.stat().st_size if log_path.exists() else 0
                modified = datetime.fromtimestamp(log_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                print(f"[LoggedCommand] running pid={process.pid} elapsed={elapsed}s log_bytes={size} modified={modified}")
                for line in tail_lines(log_path, limit=12):
                    print(f"[LoggedCommand][tail] {line}")

    elapsed = int(time.monotonic() - started)
    size = log_path.stat().st_size if log_path.exists() else 0
    print(f"[LoggedCommand] finished return_code={return_code} elapsed={elapsed}s log_bytes={size}")
    for line in tail_lines(log_path, limit=20):
        print(f"[LoggedCommand][tail] {line}")
    return int(return_code)


if __name__ == "__main__":
    raise SystemExit(main())
