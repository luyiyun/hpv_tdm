from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Self

from pydantic import BaseModel, ConfigDict


def deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class ConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> Self:
        defaults = cls().model_dump(mode="python")
        merged = deep_merge(defaults, payload)
        return cls.model_validate(merged)

    @classmethod
    def from_json_file(cls, path: str | Path) -> Self:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_json_dict(payload)

    def to_json_file(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(
                self.model_dump(mode="json"), handle, ensure_ascii=False, indent=2
            )
            handle.write("\n")
