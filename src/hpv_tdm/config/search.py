from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from .base import ConfigBase


class WeightedSumObjectiveConfig(ConfigBase):
    icur_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="单目标加权搜索中 ICUR 项的权重。",
    )
    incidence_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="单目标加权搜索中发病率项的权重。",
    )
    transform: Literal["ratio", "log_ratio"] = Field(
        default="log_ratio",
        description=(
            "单目标加权搜索中归一化后的变换方式。"
            "`ratio` 直接使用比例值，`log_ratio` 使用带符号对数压缩量级。"
        ),
    )
    icur_scale: float = Field(
        default=1_000_000.0,
        gt=0.0,
        description="单目标加权搜索中 ICUR 的参考尺度，用于归一化。",
    )
    incidence_scale: float | None = Field(
        default=None,
        description=(
            "单目标加权搜索中发病率的参考尺度；为空时自动使用 "
            "`SearchConfig.incidence_threshold`。"
        ),
    )


class SearchConfig(ConfigBase):
    objective_mode: Literal["multi_objective", "weighted_sum", "constrained"] = Field(
        default="multi_objective",
        description="搜索目标模式：多目标搜索、单目标加权搜索或约束优化。",
    )
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
    weighted_sum: WeightedSumObjectiveConfig = Field(
        default_factory=WeightedSumObjectiveConfig,
        description="单目标加权搜索所需的归一化与权重配置。",
    )

    @field_validator("strategy", mode="before")
    @classmethod
    def normalize_strategy(cls, value: str) -> str:
        if value == "contiOneCover":
            return "conti_one_cover"
        return value
