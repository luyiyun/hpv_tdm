# HPV TDM Toolkit

`hpv_tdm` 是一个面向科研与政策分析的 HPV 传播动力学模拟与接种策略优化工具包。它保留了科研项目常见的脚本工作流，但包本身只暴露 4 组核心接口：

- `ModelConfig` + `Model`
- `EvaluationConfig` + `Evaluator`
- `SearchConfig` + `Searcher`
- `SimulationResult` / `EvaluationResult` / `SearchResult`

当前仓库同时支持两类模型：

- `AgeSexAggregateHPVModel`
  - 年龄-性别结构聚合模型
  - 不区分 HPV 亚型
  - 适合作为对照模型
- `AgeSexSubtypeGroupedHPVModel`
  - 年龄-性别结构分高危亚型组模型
  - 默认纳入 3 组宫颈癌相关高危型：
    - `hr_16_18`
    - `hr_31_33_45_52_58`
    - `hr_other`
 - 当前第一版不显式建模共感染
  - 当前第一版不纳入 `6/11`
  - `SubtypeGrouped` 默认将亚型相关参数拆成：
    - 初始感染权重 `initial_weight`
    - 新发感染权重 `infection_weight`
    - 持续感染倍率 `persistence_multiplier`
    - 癌症进展倍率 `cancer_progression_multiplier`

两类模型当前都使用显式 8 维女性初始状态向量：

- `[Sf, If, Pf, LC, RC, DC, Rf, Vf]`

其中 subtype 模型会把 `If/Pf/LC/RC/DC/Rf` 这些总量初始比例再按亚型组权重拆分到各组感染链中。

关于 subtype 模型新增参数的默认值、设定逻辑和引用来源，见
[docs/subtype_parameter_defaults.md](/home/rongzw/projects/hpv_tdm/docs/subtype_parameter_defaults.md)。

## 安装

项目使用 `uv` 管理依赖。

```bash
uv sync --all-extras --dev
```

常用命令：

```bash
uv run simulate.py --model-config conf/simulate.json --evaluation-config conf/evaluation.json --output-dir results/example_simulation
uv run search.py --model-config conf/simulate.json --evaluation-config conf/evaluation.json --search-config conf/search.json --time-horizon 60 --output-dir results/example_search
uv run pytest
uv run ruff check .
uv run ruff format .
```

## 项目结构

- `simulate.py`
  - 包外脚本入口
  - 像普通使用者一样调用 `hpv_tdm` 的公开接口
- `search.py`
  - 包外脚本入口
  - 像普通使用者一样调用 `hpv_tdm` 的公开接口
- `src/hpv_tdm/config`
  - Pydantic 配置对象
- `src/hpv_tdm/model`
  - 传播动力学模型和内部人口学/接触矩阵组件
- `src/hpv_tdm/evaluator`
  - 经济学评价逻辑
- `src/hpv_tdm/search`
  - 贝叶斯优化搜索逻辑
- `src/hpv_tdm/result`
  - 结果对象、汇总表格、绘图和 HDF5 读写
- `experiments`
  - 基于标准化结果文件做论文图表和补充分析
- `results`
  - 仿真和搜索结果输出目录

## 1. 直接做传播动力学模拟

如果只想根据某个国家或地区的参数做传播动力学模拟，只需要：

1. 构造对应的 `ModelConfig`
2. 用该 config 实例化对应的 `Model`
3. 调用 `simulate()`

```python
from hpv_tdm import AggregateModelConfig, AgeSexAggregateHPVModel

config = AggregateModelConfig(
    population={"total_female": 50_000_000, "total_male": 48_000_000},
    vaccination={
        "product_id": "bivalent",
        "coverage_by_age": [0.0] * 7 + [0.85] + [0.0] * 18,
    },
    simulation={
        "t_span": (0.0, 80.0),
        "n_eval": 81,
    },
)

model = AgeSexAggregateHPVModel(config)
sim_result = model.simulate()
print(sim_result.summary_table())
sim_result.plot_incidence(save_path="results/demo_simulation/incidence.png")
sim_result.to_hdf("results/demo_simulation/simulation_result.h5")
```

`SimulationResult` 提供：

- `summary_table()`
- `plot_incidence()`
- `plot_mortality()`
- `to_hdf()`
- `from_hdf()`

如果需要访问详细模拟状态，可以直接用属性：

- `sim_result.time`
- `sim_result.state`
- `sim_result.cumulative`

## 2. 对模拟结果做经济学评价

如果已经拿到了 `SimulationResult`，可以继续做经济学评价：

```python
from hpv_tdm import EvaluationConfig, Evaluator

evaluator = Evaluator(EvaluationConfig(discount_rate=0.03))
eval_result = evaluator.evaluate(sim_result)
print(eval_result.summary_table())
eval_result.plot_cost(save_path="results/demo_simulation/cost.png")
eval_result.to_hdf("results/demo_simulation/evaluation_result.h5")
```

如果要做相对评价，例如 ICUR、避免病例、避免死亡、避免 DALY，需要再传入一个参考场景：

```python
reference_result = AgeSexAggregateHPVModel(
    config.with_vaccination(product_id=None, coverage_by_age=[0.0] * config.nages)
).simulate()
incremental_result = evaluator.evaluate(sim_result, reference_result)
print(incremental_result.summary_table())
```

`EvaluationResult` 提供：

- `summary_table()`
- `plot_incidence()`
- `plot_mortality()`
- `plot_cost()`
- `plot_daly()`
- `plot_icur()`
- `to_hdf()`
- `from_hdf()`

## 3. 搜索最优接种策略

如果要搜索最优接种策略，需要：

1. 构造 `SearchConfig`
2. 实例化 `Searcher`
3. 调用 `search(model, evaluator)`

```python
from hpv_tdm import (
    EvaluationConfig,
    Evaluator,
    SearchConfig,
    Searcher,
    SubtypeGroupedModelConfig,
    AgeSexSubtypeGroupedHPVModel,
)

model_config = SubtypeGroupedModelConfig(
    simulation={"t_span": (0.0, 50.0), "n_eval": 51},
)
model = AgeSexSubtypeGroupedHPVModel(model_config)
evaluator = Evaluator(EvaluationConfig(discount_rate=0.03))
searcher = Searcher(
    SearchConfig(
        n_trials=100,
        strategy="one",
    )
)

search_result = searcher.search(model, evaluator, output_dir="results/demo_search")
search_result.save("results/demo_search")
print(search_result.summary_table())
```

`SearchConfig.strategy` 直接控制搜索接种年龄的策略：

- `one`
  - 搜索单个接种年龄组
- `multi`
  - 搜索多个离散年龄组
- `conti`
  - 搜索连续年龄段，并分别搜索各年龄覆盖率
- `conti_one_cover`
  - 搜索连续年龄段，但整段共用一个覆盖率

`SearchResult` 提供：

- `summary_table()`
- `plot_history()`
- `plot_pareto()`
- `save(directory)`
- `from_dir()`

默认保存内容包括：

- `study.db`
- `search_result.h5`
- `best_simulation.h5`
- `best_evaluation.h5`
- `search_config.json`
- `best_trial.json`

## 4. 配置的两种使用方式

### Python 侧直接构造

Pydantic 配置类支持在构造时局部覆盖默认值：

```python
from hpv_tdm import AggregateModelConfig, SearchConfig

model_config = AggregateModelConfig(
    simulation={"n_eval": 31},
    vaccination={"coverage_by_age": [0.0] * 26},
)
search_config = SearchConfig(n_trials=200, strategy="conti_one_cover")
```

所有公开配置字段都带有中文 `description`，可以通过 Pydantic 字段元数据直接查看参数说明。

### JSON 侧部分覆盖

每个公开配置类都支持：

- `from_json_file`
- `from_json_dict`
- `to_json_file`

JSON 只需要提供想覆盖的字段，未提供的字段会保留默认值。

```python
from hpv_tdm import SubtypeGroupedModelConfig

config = SubtypeGroupedModelConfig.from_json_file("conf/simulate.json")
```

JSON 合并规则固定为：

- 对象字段递归覆盖
- 列表字段整体替换
- 标量字段直接替换

## 5. 脚本工作流

虽然包本身没有再提供更高层的 `Scenario`/`service` 接口，但仓库仍保留两个研究脚本入口：

### 单次仿真

```bash
uv run simulate.py \
  --model-config conf/simulate.json \
  --evaluation-config conf/evaluation.json \
  --output-dir results/example_simulation
```

### 策略搜索

```bash
uv run search.py \
  --model-config conf/simulate.json \
  --evaluation-config conf/evaluation.json \
  --search-config conf/search.json \
  --time-horizon 60 \
  --output-dir results/example_search
```

这两个脚本本质上只是包外示例工程，它们和普通使用者一样，只调用公开 API。

## 6. 结果文件

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
- `best_model_config.json`
- `model_config.json`
- `evaluation_config.json`
- `search_config.json`
- `best_trial.json`

## 7. 扩展方式

未来如果要扩展：

- 新传播动力学模型
  - 新增对应的 `ModelConfig`
  - 新增对应的 `Model`
  - 保持 `Model(config).simulate()` 接口即可
- 新经济学评价逻辑
  - 在 `Evaluator` 中扩展指标计算
  - 在 `EvaluationResult` 中补充汇总和绘图
- 新优化策略
  - 在 `Searcher` 内部增加新的 `strategy` 分支
  - 不需要暴露新的 strategy class

## 开发规范

项目内部约定见 [AGENTS.md](AGENTS.md)。
