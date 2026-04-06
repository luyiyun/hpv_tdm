# AGENTS

## 项目目标

本项目用于支持基于年龄-性别结构传播动力学模型的 HPV 疫苗接种策略分析，包括：

- 单次接种策略下的传播动力学模拟
- 基于经济学指标的结果评价
- 基于 Optuna 的贝叶斯优化搜索
- 基于标准化结果文件的论文图表与补充分析

## 当前代码组织

- `simulate.py`
  - 包外脚本入口
  - 负责读取 `ModelConfig` / `EvaluationConfig`
  - 像普通使用者一样实例化 `Model`、`Evaluator` 并运行
- `search.py`
  - 包外脚本入口
  - 负责读取 `ModelConfig` / `EvaluationConfig` / `SearchConfig`
  - 像普通使用者一样实例化 `Model`、`Evaluator`、`Searcher` 并运行
- `src/hpv_tdm/config`
  - Pydantic 配置对象
  - 公开入口收敛为：
    - `AggregateModelConfig`
    - `SubtypeGroupedModelConfig`
    - `EvaluationConfig`
    - `SearchConfig`
- `src/hpv_tdm/model`
  - 传播动力学模型实现
  - 当前包含：
    - `BaseHPVTransmissionModel`
    - `AgeSexAggregateHPVModel`
    - `AgeSexSubtypeGroupedHPVModel`
  - 内含人口学、生命表和性接触矩阵的内部组件
- `src/hpv_tdm/evaluator`
  - `Evaluator`
  - 只负责“如何计算评价指标”
- `src/hpv_tdm/search`
  - `Searcher`
  - 只负责“如何做贝叶斯优化搜索”
- `src/hpv_tdm/result`
  - `SimulationResult`
  - `EvaluationResult`
  - `SearchResult`
  - 负责汇总、绘图和 HDF5 读写
- `src/hpv_tdm/utils`
  - 少量通用工具函数
- `experiments`
  - 继续承担论文图表与补充分析职责
  - 原则上直接消费标准化结果文件，而不是依赖运行时对象

## 公开接口原则

包的公开接口只保留 4 组对象：

- `ModelConfig` + `Model`
- `EvaluationConfig` + `Evaluator`
- `SearchConfig` + `Searcher`
- `SimulationResult` / `EvaluationResult` / `SearchResult`

不再保留更高一级的 orchestration 接口，例如：

- `ScenarioConfig`
- `services`
- `cli`

## 当前模型说明

### `AgeSexAggregateHPVModel`

- 年龄-性别结构聚合模型
- 不区分 HPV 亚型
- 用于作为对照模型

### `AgeSexSubtypeGroupedHPVModel`

- 年龄-性别结构分高危亚型组模型
- 当前默认只纳入宫颈癌相关高危型
- 默认亚型组：
  - `hr_16_18`
  - `hr_31_33_45_52_58`
  - `hr_other`
- 当前第一版不纳入 `6/11`
- 当前第一版不显式建模共感染
- 当前与 aggregate 模型一致，女性初始状态向量统一采用 8 维显式语义：
  - `[Sf, If, Pf, LC, RC, DC, Rf, Vf]`
  - subtype 模型会再将 `If/Pf/LC/RC/DC/Rf` 按亚型组权重拆分
- 亚型参数当前明确拆分为：
  - `initial_weight`
    - 用于初始感染状态在各亚型组间的分配
  - `persistence_multiplier`
    - 仅作用于 `If -> Pf`
  - `cancer_progression_multiplier`
    - 作用于 `Pf -> LC -> RC -> DC`
- `group_protection` 在 subtype 模型中的默认语义为结构性 `0/1`
  - 覆盖亚型组取 `1`
  - 未覆盖亚型组取 `0`
- 参数默认值和来源说明见 `docs/subtype_parameter_defaults.md`

## 配置约定

- 所有对外配置统一使用 Pydantic
- Python 侧通过构造函数直接覆盖默认值
  - 例如 `SearchConfig(n_trials=500)`
  - 例如 `AggregateModelConfig(simulation={"n_eval": 51})`
- JSON 侧通过配置类的类方法载入
  - `from_json_file`
  - `from_json_dict`
  - `to_json_file`
- JSON 合并规则固定为：
  - 对象递归覆盖
  - 列表整体替换
  - 标量直接替换

## 结果文件约定

### 仿真

- `simulation_result.h5`
- `evaluation_result.h5`
- `model_config.json`
- `evaluation_config.json`
- `last.npy`

### 搜索

- `study.db`
- `search_result.h5`
- `best_simulation.h5`
- `best_evaluation.h5`
- `model_config.json`
- `evaluation_config.json`
- `search_config.json`
- `best_trial.json`

## 代码规范

### 错误处理

- 尽量不要使用大范围 `try/except`
- 优先让错误直接暴露
- 参数合法性优先通过 Pydantic 和显式校验保证

### 参数管理

- 不再使用 Hydra
- 参数统一使用 Pydantic + JSON
- 默认值集中在 config 层维护
- 不要在脚本、模型、评价器、搜索器里重复散落默认参数

### 结构设计

- 优先使用组合组织代码
- 仅在确实需要统一接口时使用继承
- 模型类只接受其对应的 config
- 结果类负责保存、回读、汇总和绘图
- 搜索年龄策略作为 `SearchConfig.strategy` 的分支逻辑存在于 `Searcher` 中
- 不再单独新增公开的 Strategy 类

### 导入规范

- `src/hpv_tdm` 包内部统一使用相对导入
- 只有包外脚本和外部使用者使用 `from hpv_tdm import ...`

### 代码质量

- 使用 `uv` 作为唯一推荐的依赖与运行入口
- 充分使用 Ruff：
  - `ruff check`
  - `ruff format`
  - `ruff check --select I`
- 保持代码文件干净、可读、可维护

### 实现习惯

- 如果某个函数只在一个地方使用，优先内联到使用位置
- 对关键步骤补充简短中文注释
- 避免没有实际收益的抽象层
- 如果用户手动修改了部分代码或配置，不要将这些修改回退；应在理解当前工作区状态后，与这些修改兼容地继续实现需求

## 顶层工作流约定

- `simulate.py`
  - 对应“根据某个接种策略做传播动力学模拟”
  - 结果输出到 `results/`
- `search.py`
  - 对应“通过贝叶斯优化搜索最优接种策略”
  - 结果输出到 `results/`
- `experiments`
  - 对应“在已有结果基础上做进一步分析和论文图表绘制”

这一顶层工作流后续继续保留。

## 常用命令

```bash
uv sync --all-extras --dev
uv run simulate.py --model-config conf/simulate.json --evaluation-config conf/evaluation.json
uv run search.py --model-config conf/simulate.json --evaluation-config conf/evaluation.json --search-config conf/search.json
uv run pytest
uv run ruff check .
uv run ruff format .
```
