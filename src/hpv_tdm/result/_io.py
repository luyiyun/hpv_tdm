from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_json_attr(group: h5py.Group, key: str, value: Any) -> None:
    group.attrs[key] = json.dumps(value, ensure_ascii=False)


def read_json_attr(group: h5py.Group, key: str) -> Any:
    return json.loads(group.attrs[key])
