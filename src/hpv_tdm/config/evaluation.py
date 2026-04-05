from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import ConfigBase


class EvaluationConfig(ConfigBase):
    cost_per_cecx: float = Field(
        default=7547.0,
        ge=0,
        description="每例宫颈癌病例的直接医疗成本。",
    )
    daly_nofatal: float = Field(
        default=0.52,
        ge=0,
        description="非致死宫颈癌病例的 DALY 损失权重。",
    )
    daly_fatal: float = Field(
        default=0.86,
        ge=0,
        description="致死宫颈癌病例的 DALY 损失权重。",
    )
    discount_rate: float = Field(
        default=0.03,
        ge=0,
        description="成本和健康结局折现率。",
    )
    life_table_method: Literal["prime", "textbook"] = Field(
        default="prime",
        description="寿命损失计算所采用的生命表方法。",
    )
