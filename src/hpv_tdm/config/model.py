from __future__ import annotations

import warnings
from typing import Literal, Self

import numpy as np
from pydantic import Field, model_validator

from .base import ConfigBase

_UNSET = object()


def _default_agebins() -> list[float]:
    return [
        0,
        1,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        18,
        19,
        20,
        25,
        27,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        float("inf"),
    ]


def _default_fertilities() -> list[float]:
    return (
        [0.0] * 8
        + [
            4.61 / 1000,
            18.62 / 1000,
            29.73 / 1000,
            74.39 / 1000,
            112.60 / 1000,
            106.95 / 1000,
            68.57 / 1000,
            26.08 / 1000,
            5.23 / 1000,
            0.44 / 1000,
        ]
        + [0.0] * 8
    )


def _default_deathes_female() -> list[float]:
    return [
        6.68 / 1000,
        0.21 / 1000,
        0.0,
        0.48 / 1000,
        0.41 / 1000,
        0.13 / 1000,
        0.0,
        0.61 / 1000,
        0.0,
        0.12 / 1000,
        0.67 / 1000,
        0.36 / 1000,
        0.50 / 1000,
        0.27 / 1000,
        0.26 / 1000,
        0.57 / 1000,
        0.75 / 1000,
        1.09 / 1000,
        1.79 / 1000,
        2.64 / 1000,
        5.97 / 1000,
        9.51 / 1000,
        18.25 / 1000,
        27.70 / 1000,
        61.45 / 1000,
        118.39 / 1000,
    ]


def _default_deathes_male() -> list[float]:
    return [
        8.49 / 1000,
        0.31 / 1000,
        0.15 / 1000,
        0.66 / 1000,
        0.76 / 1000,
        0.14 / 1000,
        0.41 / 1000,
        0.14 / 1000,
        0.61 / 1000,
        0.71 / 1000,
        0.0,
        0.57 / 1000,
        0.92 / 1000,
        0.55 / 1000,
        0.87 / 1000,
        1.51 / 1000,
        1.63 / 1000,
        2.38 / 1000,
        4.71 / 1000,
        6.31 / 1000,
        11.82 / 1000,
        17.52 / 1000,
        28.45 / 1000,
        44.76 / 1000,
        90.01 / 1000,
        160.64 / 1000,
    ]


def _default_omega_f() -> list[float]:
    return [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.11 * 0.457,
        0.11 * 0.457,
        1.36 * 0.457,
        1.67 * 0.457,
        1.65 * 0.457,
        1.40 * 0.457,
        1.16 * 0.457,
        1.13 * 0.457,
        1.06 * 0.457,
        1.02 * 0.457,
        0.96 * 0.457,
        0.93 * 0.457,
        0.83 * 0.457,
        0.63 * 0.457,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def _default_omega_m() -> list[float]:
    return [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.04 * 0.457,
        0.04 * 0.457,
        0.49 * 0.457,
        1.02 * 0.457,
        1.20 * 0.457,
        1.43 * 0.457,
        1.32 * 0.457,
        1.19 * 0.457,
        1.20 * 0.457,
        1.08 * 0.457,
        1.09 * 0.457,
        0.91 * 0.457,
        0.85 * 0.457,
        0.74 * 0.457,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def _default_d_l() -> list[float]:
    return (
        [0.0] * 8
        + [0.7 / 100 * 0.01] * 6
        + [0.6 / 100 * 0.01] * 2
        + [0.8 / 100 * 0.01] * 2
        + [1.9 / 100 * 0.01] * 2
        + [4.2 / 100 * 0.01] * 2
        + [11.6 / 100 * 0.01] * 4
    )


def _default_d_r() -> list[float]:
    return (
        [0.0] * 8
        + [13.4 / 100 * 0.01] * 6
        + [8.9 / 100 * 0.01] * 2
        + [11.0 / 100 * 0.01] * 2
        + [10.1 / 100 * 0.01] * 2
        + [17.6 / 100 * 0.01] * 2
        + [28.6 / 100 * 0.01] * 4
    )


def _default_d_d() -> list[float]:
    return (
        [0.0] * 8
        + [42.9 / 100 * 0.01] * 6
        + [41.0 / 100 * 0.01] * 2
        + [46.7 / 100 * 0.01] * 2
        + [52.7 / 100 * 0.01] * 2
        + [54.6 / 100 * 0.01] * 2
        + [70.3 / 100 * 0.01] * 4
    )


class PopulationConfig(ConfigBase):
    total_female: float = Field(
        default=713_510_000,
        gt=0,
        description="总女性人口数，用于按年龄稳定分布缩放模型中的女性人口。",
    )
    total_male: float = Field(
        default=681_870_000,
        gt=0,
        description="总男性人口数，用于按年龄稳定分布缩放模型中的男性人口。",
    )


class DemographyConfig(ConfigBase):
    agebins: list[float] = Field(
        default_factory=_default_agebins,
        description="年龄分组边界，长度应为年龄组数加 1。",
    )
    fertilities: list[float] = Field(
        default_factory=_default_fertilities,
        description="各女性年龄组的年生育率。",
    )
    deathes_female: list[float] = Field(
        default_factory=_default_deathes_female,
        description="各女性年龄组的自然死亡率。",
    )
    deathes_male: list[float] = Field(
        default_factory=_default_deathes_male,
        description="各男性年龄组的自然死亡率。",
    )
    lambda_f: float = Field(
        default=0.4681,
        ge=0,
        le=1,
        description="新生儿中女性比例。",
    )
    lambda_m: float = Field(
        default=0.5319,
        ge=0,
        le=1,
        description="新生儿中男性比例。",
    )
    q_is_zero: bool = Field(
        default=True,
        description="是否使用零人口增长假设来构建稳定年龄分布。",
    )
    rtol: float = Field(
        default=1e-5,
        gt=0,
        description="求解稳定人口分布时使用的相对误差容忍度。",
    )
    atol: float = Field(
        default=1e-5,
        gt=0,
        description="求解稳定人口分布时使用的绝对误差容忍度。",
    )


class TransmissionConfig(ConfigBase):
    epsilon_f: float = Field(
        default=0.846,
        ge=0,
        description="女性在一次有效接触中获得 HPV 感染的基准传播概率。",
    )
    epsilon_m: float = Field(
        default=0.913,
        ge=0,
        description="男性在一次有效接触中获得 HPV 感染的基准传播概率。",
    )
    omega_f: list[float] = Field(
        default_factory=_default_omega_f,
        description="女性各年龄组的性活跃度修正系数。",
    )
    omega_m: list[float] = Field(
        default_factory=_default_omega_m,
        description="男性各年龄组的性活跃度修正系数。",
    )
    partner_window: int = Field(
        default=10,
        ge=0,
        description="构建年龄混合矩阵时允许配对的最大年龄组跨度。",
    )
    partner_decline: float = Field(
        default=0.05,
        ge=0,
        description="年龄差增加时接触概率的衰减系数。",
    )
    partner_interval: tuple[int, int] = Field(
        default=(13, 60),
        description="认为存在性伴接触的年龄范围（岁）。",
    )
    phi: float = Field(
        default=1 / 3,
        ge=0,
        description="恢复后免疫衰减速率，控制恢复者回到易感状态的速度。",
    )
    beta_I: float = Field(
        default=0.15,
        ge=0,
        description="从初始感染 If 进展到持续感染 Pf 的基准速率。",
    )
    beta_P: float = Field(
        default=0.13,
        ge=0,
        description="从持续感染 Pf 进展到局部癌 LC 的基准速率。",
    )
    beta_LC: float = Field(
        default=0.10,
        ge=0,
        description="从局部癌 LC 进展到区域癌 RC 的基准速率。",
    )
    beta_RC: float = Field(
        default=0.30,
        ge=0,
        description="从区域癌 RC 进展到远处转移癌 DC 的基准速率。",
    )
    dL: list[float] = Field(
        default_factory=_default_d_l,
        description="局部癌 LC 的年龄别疾病死亡率。",
    )
    dR: list[float] = Field(
        default_factory=_default_d_r,
        description="区域癌 RC 的年龄别疾病死亡率。",
    )
    dD: list[float] = Field(
        default_factory=_default_d_d,
        description="远处转移癌 DC 的年龄别疾病死亡率。",
    )
    gamma_I: float = Field(
        default=0.35,
        ge=0,
        description="初始感染 If 自然恢复到恢复状态 R 的速率。",
    )
    gamma_P: float = Field(
        default=0.7 * (12 / 18),
        ge=0,
        description="持续感染 Pf 自然恢复到恢复状态 R 的速率。",
    )
    gamma_LC: float = Field(
        default=0.0,
        ge=0,
        description="局部癌 LC 回退到恢复状态 R 的速率。",
    )
    gamma_RC: float = Field(
        default=0.0,
        ge=0,
        description="区域癌 RC 回退到恢复状态 R 的速率。",
    )
    gamma_DC: float = Field(
        default=0.0,
        ge=0,
        description="远处转移癌 DC 回退到恢复状态 R 的速率。",
    )
    cal_cumulate: bool = Field(
        default=True,
        description="是否在仿真时同时累计记录感染、癌症、死亡和接种等事件。",
    )
    vacc_prefer: bool = Field(
        default=False,
        description="是否优先将原本会感染的易感者导向接种通道。",
    )
    verbose: bool = Field(
        default=False,
        description="是否输出更详细的求解日志。",
    )
    backend: Literal["solve_ivp", "odeint"] = Field(
        default="solve_ivp",
        description="常微分方程求解后端。",
    )
    female_initial_state: list[float] = Field(
        default_factory=lambda: [0.85, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        description=("女性初始状态比例向量，顺序为 [Sf, If, Pf, LC, RC, DC, Rf, Vf]。"),
    )
    male_initial_state: list[float] = Field(
        default_factory=lambda: [0.85, 0.15, 0.0, 0.0],
        description="男性初始状态比例向量，顺序为 [Sm, Im, Pm, Rm]。",
    )


class VaccinationProgramConfig(ConfigBase):
    product_id: str | None = Field(
        default=None,
        description="本次策略使用的疫苗产品编号；为空时回退到疫苗目录默认产品。",
    )
    coverage_by_age: list[float] | None = Field(
        default=None,
        description="各年龄组接种覆盖率；为空时视为所有年龄组均不接种。",
    )


class VaccineDoseScheduleConfig(ConfigBase):
    doses: int = Field(
        ge=1,
        description="该年龄范围内推荐接种剂次数。",
    )
    age_min: int = Field(
        ge=0,
        description="该剂次规则适用的最小年龄（含）。",
    )
    age_max: int = Field(
        ge=0,
        description="该剂次规则适用的最大年龄（含）。",
    )

    @model_validator(mode="after")
    def validate_age_range(self) -> Self:
        if self.age_max < self.age_min:
            raise ValueError("dose schedule age_max must be >= age_min")
        return self


class VaccineProductConfig(ConfigBase):
    display_name: str = Field(description="疫苗产品展示名称。")
    aggregate_efficacy: float = Field(
        ge=0,
        le=1,
        description="聚合模型使用的约化保护效力参数，代表整体宫颈癌可预防比例。",
    )
    dose_cost: float = Field(
        ge=0,
        description="单剂疫苗成本。",
    )
    dose_schedules: list[VaccineDoseScheduleConfig] = Field(
        default_factory=lambda: [
            VaccineDoseScheduleConfig(doses=2, age_min=9, age_max=14),
            VaccineDoseScheduleConfig(doses=3, age_min=15, age_max=45),
        ],
        description="按年龄范围定义的接种剂次规则列表。",
    )
    group_protection: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "SubtypeGrouped 模型使用的亚型组别保护强度。"
            "当前默认解释为：覆盖型别组取 1，未覆盖型别组取 0。"
        ),
    )

    @model_validator(mode="after")
    def validate_dose_schedules(self) -> Self:
        if not self.dose_schedules:
            raise ValueError("dose_schedules must contain at least one rule")
        sorted_rules = sorted(self.dose_schedules, key=lambda item: item.age_min)
        for previous, current in zip(sorted_rules[:-1], sorted_rules[1:]):
            if current.age_min <= previous.age_max:
                raise ValueError("dose_schedules age ranges must not overlap")
        return self


def _default_vaccine_catalog() -> dict[str, VaccineProductConfig]:
    return {
        "domestic_bivalent": VaccineProductConfig(
            display_name="Domestic bivalent HPV vaccine",
            aggregate_efficacy=0.691,
            dose_cost=986.99 / 3,
            dose_schedules=[
                VaccineDoseScheduleConfig(doses=2, age_min=9, age_max=14),
                VaccineDoseScheduleConfig(doses=3, age_min=15, age_max=45),
            ],
            group_protection={"hr_16_18": 1.0},
        ),
        "imported_bivalent": VaccineProductConfig(
            display_name="Imported bivalent HPV vaccine",
            aggregate_efficacy=0.691,
            dose_cost=1739.97 / 3,
            dose_schedules=[
                VaccineDoseScheduleConfig(doses=2, age_min=9, age_max=14),
                VaccineDoseScheduleConfig(doses=3, age_min=15, age_max=45),
            ],
            group_protection={"hr_16_18": 1.0},
        ),
        "imported_quadrivalent": VaccineProductConfig(
            display_name="Imported quadrivalent HPV vaccine",
            aggregate_efficacy=0.691,
            dose_cost=2393.98 / 3,
            dose_schedules=[
                VaccineDoseScheduleConfig(doses=3, age_min=9, age_max=45),
            ],
            group_protection={"hr_16_18": 1.0},
        ),
        "domestic_nonavalent": VaccineProductConfig(
            display_name="Domestic nonavalent HPV vaccine",
            aggregate_efficacy=0.921,
            dose_cost=499.0,
            dose_schedules=[
                VaccineDoseScheduleConfig(doses=2, age_min=9, age_max=17),
                VaccineDoseScheduleConfig(doses=3, age_min=18, age_max=45),
            ],
            group_protection={
                "hr_16_18": 1.0,
                "hr_31_33_45_52_58": 1.0,
            },
        ),
        "imported_nonavalent": VaccineProductConfig(
            display_name="Imported nonavalent HPV vaccine",
            aggregate_efficacy=0.921,
            dose_cost=3894.01 / 3,
            dose_schedules=[
                VaccineDoseScheduleConfig(doses=2, age_min=9, age_max=14),
                VaccineDoseScheduleConfig(doses=3, age_min=15, age_max=45),
            ],
            group_protection={
                "hr_16_18": 1.0,
                "hr_31_33_45_52_58": 1.0,
            },
        ),
    }


class VaccineCatalogConfig(ConfigBase):
    default_product_id: str = Field(
        default="domestic_bivalent",
        description="未显式指定时默认采用的疫苗产品编号。",
    )
    products: dict[str, VaccineProductConfig] = Field(
        default_factory=_default_vaccine_catalog,
        description="疫苗产品目录，键为产品编号，值为对应的参数配置。",
    )

    def get_product(self, product_id: str | None) -> VaccineProductConfig:
        resolved_product_id = product_id or self.default_product_id
        return self.products[resolved_product_id]


class SimulationOptionsConfig(ConfigBase):
    t_span: tuple[float, float] = Field(
        default=(0.0, 100.0),
        description="仿真时间区间，单位为年。",
    )
    n_eval: int = Field(
        default=101,
        ge=2,
        description="仿真结果采样点数量。",
    )
    init_state_path: str | None = Field(
        default=None,
        description="可选的初始状态文件路径；为空时使用模型默认初值。",
    )
    save_last_state: bool = Field(
        default=True,
        description="是否保存最后时刻状态到 `last.npy`。",
    )
    generate_plots: bool = Field(
        default=True,
        description="是否自动生成结果图像。",
    )


class AggregateModelConfig(ConfigBase):
    model_kind: Literal["aggregate"] = Field(
        default="aggregate",
        description="模型类型标记；聚合模型固定为 `aggregate`。",
    )
    population: PopulationConfig = Field(
        default_factory=PopulationConfig,
        description="人口规模配置。",
    )
    demography: DemographyConfig = Field(
        default_factory=DemographyConfig,
        description="人口学和年龄结构配置。",
    )
    transmission: TransmissionConfig = Field(
        default_factory=TransmissionConfig,
        description="传播动力学参数配置。",
    )
    vaccination: VaccinationProgramConfig = Field(
        default_factory=VaccinationProgramConfig,
        description="当前策略的接种方案配置。",
    )
    vaccine_catalog: VaccineCatalogConfig = Field(
        default_factory=VaccineCatalogConfig,
        description="疫苗产品目录配置。",
    )
    simulation: SimulationOptionsConfig = Field(
        default_factory=SimulationOptionsConfig,
        description="仿真运行与结果输出配置。",
    )

    @property
    def nages(self) -> int:
        return len(self.demography.agebins) - 1

    def resolved_product_id(self) -> str:
        return self.vaccination.product_id or self.vaccine_catalog.default_product_id

    def resolved_coverage_by_age(self) -> list[float]:
        if self.vaccination.coverage_by_age is None:
            return [0.0] * self.nages
        return list(self.vaccination.coverage_by_age)

    def with_vaccination(
        self,
        *,
        product_id: str | None | object = _UNSET,
        coverage_by_age: list[float] | None | object = _UNSET,
    ) -> Self:
        payload = {}
        if product_id is not _UNSET:
            payload["product_id"] = product_id
        if coverage_by_age is not _UNSET:
            payload["coverage_by_age"] = coverage_by_age
        vaccination = self.vaccination.model_copy(update=payload)
        return self.model_copy(update={"vaccination": vaccination})

    @model_validator(mode="after")
    def validate_lengths(self) -> Self:
        nages = self.nages
        if len(self.demography.fertilities) != nages:
            raise ValueError("fertilities length must match agebins")
        if len(self.demography.deathes_female) != nages:
            raise ValueError("deathes_female length must match agebins")
        if len(self.demography.deathes_male) != nages:
            raise ValueError("deathes_male length must match agebins")
        if len(self.transmission.omega_f) != nages:
            raise ValueError("omega_f length must match agebins")
        if len(self.transmission.omega_m) != nages:
            raise ValueError("omega_m length must match agebins")
        if len(self.transmission.dL) != nages:
            raise ValueError("dL length must match agebins")
        if len(self.transmission.dR) != nages:
            raise ValueError("dR length must match agebins")
        if len(self.transmission.dD) != nages:
            raise ValueError("dD length must match agebins")
        if len(self.transmission.female_initial_state) != 8:
            raise ValueError("female_initial_state length must be 8")
        if len(self.transmission.male_initial_state) != 4:
            raise ValueError("male_initial_state length must be 4")
        if self.vaccination.coverage_by_age is not None:
            if len(self.vaccination.coverage_by_age) != nages:
                raise ValueError("coverage_by_age length must match agebins")
        self._warn_if_vaccination_outside_allowed_age()
        return self

    def _warn_if_vaccination_outside_allowed_age(self) -> None:
        coverage = self.resolved_coverage_by_age()
        if not any(value > 0 for value in coverage):
            return
        product = self.vaccine_catalog.get_product(self.resolved_product_id())
        lower_bounds = np.asarray(self.demography.agebins[:-1], dtype=float)
        upper_bounds = np.asarray(self.demography.agebins[1:], dtype=float)
        allowed = np.zeros(self.nages, dtype=bool)
        for rule in product.dose_schedules:
            allowed |= (lower_bounds >= rule.age_min) & (lower_bounds <= rule.age_max)

        invalid_indices = [
            index
            for index, (is_allowed, value) in enumerate(zip(allowed, coverage))
            if (not is_allowed) and value > 0
        ]
        if not invalid_indices:
            return

        age_labels = []
        for index in invalid_indices:
            lower = lower_bounds[index]
            upper = upper_bounds[index]
            lower_text = str(int(lower)) if float(lower).is_integer() else str(lower)
            if upper == float("inf"):
                age_labels.append(f"[{lower_text}, inf)")
            else:
                upper_text = (
                    str(int(upper))
                    if float(upper).is_integer()
                    else str(upper)
                )
                age_labels.append(f"[{lower_text}, {upper_text})")

        warnings.warn(
            (
                f"vaccination coverage is set for age groups outside the allowed "
                f"dose_schedules of product {self.resolved_product_id()!r}: "
                f"{', '.join(age_labels)}"
            ),
            stacklevel=2,
        )


class SubtypeGroupConfig(ConfigBase):
    label: str = Field(description="亚型组展示名称。")
    initial_weight: float = Field(
        default=1.0,
        gt=0,
        description="初始感染状态在该亚型组中的分配权重。",
    )
    persistence_multiplier: float = Field(
        default=1.0,
        gt=0,
        description="该亚型组从初始感染 If 进展到持续感染 Pf 的相对倍率。",
    )
    cancer_progression_multiplier: float = Field(
        default=1.0,
        gt=0,
        description=("该亚型组从持续感染到癌症相关阶段（Pf->LC->RC->DC）的相对倍率。"),
    )


def _default_subtype_groups() -> dict[str, SubtypeGroupConfig]:
    # 默认值改为全国口径：
    # 1. 初始感染权重来自 BMC Medicine 2025 全国汇总分析中各高危型别的
    #    型别流行率，并按当前三组合并后重新归一化：
    #    16/18 = (2.15 + 0.74) / 19.65 = 0.147
    #    31/33/45/52/58 = (0.83 + 0.91 + 0.41 + 4.40 + 2.65) / 19.65 = 0.468
    #    other hr = 0.385
    # 2. persistence_multiplier 使用中国 CIN1/CIN2/CIN3 组别分布的等权平均
    #    相对全国感染分布的富集比构造，并保证以感染分布加权平均后为 1。
    # 3. cancer_progression_multiplier 使用中国宫颈癌组别分布相对上述
    #    precursor 等权平均分布的富集比构造，并保证以 precursor 分布加权平均后为 1。
    return {
        "hr_16_18": SubtypeGroupConfig(
            label="HPV 16/18",
            initial_weight=0.147,
            persistence_multiplier=2.616,
            cancer_progression_multiplier=1.768,
        ),
        "hr_31_33_45_52_58": SubtypeGroupConfig(
            label="HPV 31/33/45/52/58",
            initial_weight=0.468,
            persistence_multiplier=0.931,
            cancer_progression_multiplier=0.492,
        ),
        "hr_other": SubtypeGroupConfig(
            label="Other high-risk HPV",
            initial_weight=0.385,
            persistence_multiplier=0.467,
            cancer_progression_multiplier=0.587,
        ),
    }


class SubtypeTransmissionConfig(TransmissionConfig):
    female_initial_state: list[float] = Field(
        default_factory=lambda: [0.85, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        description=(
            "SubtypeGrouped 模型使用的女性初始状态比例向量，顺序为 "
            "[Sf, If, Pf, LC, RC, DC, Rf, Vf]。其中 If/Pf/LC/RC/DC/Rf "
            "会按亚型组初始权重拆分到各组。"
        ),
    )


class SubtypeGroupedModelConfig(AggregateModelConfig):
    model_kind: Literal["subtype_grouped"] = Field(
        default="subtype_grouped",
        description="模型类型标记；亚型组模型固定为 `subtype_grouped`。",
    )
    transmission: SubtypeTransmissionConfig = Field(
        default_factory=SubtypeTransmissionConfig,
        description="SubtypeGrouped 模型的传播参数配置。",
    )
    subtype_groups: dict[str, SubtypeGroupConfig] = Field(
        default_factory=_default_subtype_groups,
        description="高危 HPV 亚型组配置。",
    )

    @model_validator(mode="after")
    def validate_subtypes(self) -> Self:
        if not self.subtype_groups:
            raise ValueError("subtype_groups must not be empty")
        initial_weight_sum = sum(
            group.initial_weight for group in self.subtype_groups.values()
        )
        if initial_weight_sum <= 0:
            raise ValueError("subtype initial weights must be positive")
        return self
