"""
JSON helpers for safe file writing.
"""

import json
from pathlib import Path
from typing import Any


def write_json(path: Path | str, data: Any) -> None:
    """
    Write JSON data to a file, ensuring the parent directory exists.

    Args:
        path: Target file path.
        data: JSON-serializable object to write.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
