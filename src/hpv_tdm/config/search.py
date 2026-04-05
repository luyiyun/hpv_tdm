from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from .base import ConfigBase


class SearchConfig(ConfigBase):
    seed: int = Field(
        default=1234,
        description="Optuna 搜索的随机种子。",
    )
    n_trials: int = Field(
        default=500,
        ge=1,
        description="Optuna 总 trial 数。",
    )
    n_jobs: int = Field(
        default=1,
        ge=1,
        description=(
            "Optuna 优化时使用的并行 worker 数。"
            "传 1 表示串行；大于 1 时启用并行 trial 评估。"
        ),
    )
    incidence_threshold: float = Field(
        default=4e-5,
        ge=0,
        description="发病率约束阈值；只有达到该阈值的策略才视为可行。",
    )
    strategy: Literal[
        "one",
        "multi",
        "conti",
        "conti_one_cover",
        "contiOneCover",
    ] = Field(
        default="one",
        description="最佳接种年龄搜索策略。",
    )
    n_vacc_ages: int = Field(
        default=2,
        ge=1,
        description="`multi` 策略下同时搜索的离散接种年龄组数量。",
    )
    age_index_span: tuple[int, int] = Field(
        default=(0, 11),
        description="允许搜索的年龄组索引范围，闭区间表示。",
    )
    study_name: str = Field(
        default="hpv_tdm",
        description="Optuna study 名称。",
    )
    storage_filename: str = Field(
        default="study.db",
        description="Optuna SQLite 存储文件名。",
    )
    product_ids: list[str] | None = Field(
        default=None,
        description="允许参与搜索的疫苗产品编号列表；为空时使用模型疫苗目录全部产品。",
    )

    @field_validator("strategy", mode="before")
    @classmethod
    def normalize_strategy(cls, value: str) -> str:
        if value == "contiOneCover":
            return "conti_one_cover"
        return value
