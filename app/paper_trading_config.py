from __future__ import annotations

from typing import Any


def paper_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("paper_trading", {}))
