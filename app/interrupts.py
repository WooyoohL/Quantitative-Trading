from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def exit_on_keyboard_interrupt(label: str, detail: str = "User interrupted.") -> None:
    print(f"\n[{label}] {detail}")
    raise SystemExit(130)


def run_cli(main_fn: Callable[[], T], *, label: str, detail: str = "User interrupted.") -> T:
    try:
        return main_fn()
    except KeyboardInterrupt:
        exit_on_keyboard_interrupt(label=label, detail=detail)
