# HPV TDM Toolkit

[English README](README.md)

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
  - 当前不纳入 `6/11`
  - 当前不显式建模个体层面的共感染组合
  - 共享真实女性/男性人口分母，并引入共享女性 persistent-risk pool `Pany`
  - 在癌症事件层采用 competing-risk / single-cause attribution 近似
  - `SubtypeGrouped` 默认将亚型相关参数拆成：
    - 初始感染权重 `initial_weight`
    - 持续感染倍率 `persistence_multiplier`
    - 癌症进展倍率 `cancer_progression_multiplier`

两类模型当前都使用显式 8 维女性初始状态向量：

- `[Sf, If, Pf, LC, RC, DC, Rf, Vf]`

其中 subtype 模型会把 `If/Pf/LC/RC/DC/Rf` 这些总量初始比例再按亚型组权重拆分到各组感染链中。

关于 subtype 模型新增参数的默认值、设定逻辑和引用来源，见
[docs/subtype_parameter_defaults.md](docs/subtype_parameter_defaults.md)。

## 安装

项目使用 `uv` 管理依赖。

```bash
uv sync --all-extras --dev
```

常用命令：

```bash
uv run find_params.py --model-config conf/simulate.json --params-config conf/find_params.json --output-dir results/find-params
uv run simulate.py --model-config conf/simulate.json --evaluation-config conf/evaluation.json --output-dir results/example_simulation
uv run search.py --model-config conf/simulate.json --evaluation-config conf/evaluation.json --search-config conf/search.json --time-horizon 60 --output-dir results/example_search
uv run summary.py all --results-root results --output-dir summary
uv run pytest
uv run ruff check .
uv run ruff format .
```

## 项目结构

- `simulate.py`
  - 包外脚本入口
  - 负责读取 `ModelConfig` / `EvaluationConfig`
  - 像普通使用者一样实例化 `Model`、`Evaluator` 并运行
- `search.py`
  - 包外脚本入口
  - 负责读取 `ModelConfig` / `EvaluationConfig` / `SearchConfig`
  - 像普通使用者一样实例化 `Model`、`Evaluator`、`Searcher` 并运行
- `find_params.py`
  - 包外脚本入口
  - 通过长时程模拟 + Optuna 联合校准 subtype-grouped 模型中的初始状态与亚型参数
  - 当前只支持 `subtype_grouped` 模型
- `summary.py`
  - 包外脚本入口
  - 直接消费标准化结果文件，生成论文表格、主文图和补充分析输出
- `src/hpv_tdm/config`
  - Pydantic 配置对象
  - 公开入口收敛为 `AggregateModelConfig`、`SubtypeGroupedModelConfig`、`EvaluationConfig`、`SearchConfig`
- `src/hpv_tdm/model`
  - 传播动力学模型实现
  - 内含人口学、生命表和性接触矩阵的内部组件
- `src/hpv_tdm/evaluator`
  - `Evaluator`
  - 只负责“如何计算评价指标”
- `src/hpv_tdm/search`
  - `Searcher`
  - 只负责“如何做贝叶斯优化搜索”
- `src/hpv_tdm/result`
  - 结果对象、汇总表格、绘图和 HDF5 读写
- `experiments`
  - 基于标准化结果文件做论文图表和补充分析
- `results`
  - `simulate.py`、`search.py`、`find_params.py` 等脚本的结果输出目录
- `summary`
  - `summary.py` 默认输出目录

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

虽然包本身没有再提供更高层的 `Scenario`/`service` 接口，但仓库仍保留完整的研究脚本工作流。对于 subtype 模型的正式分析，通常会按下面顺序组织：

1. 用 `find_params.py` 联合校准 subtype 模型的初始状态和亚型参数；
2. 用 `simulate.py` 运行单个接种场景，并在需要时同时输出经济学评价；
3. 用 `search.py` 在给定 horizon 下搜索最优接种策略；
4. 用 `summary.py` 基于标准化结果文件生成论文表格、图和补充分析。

`docs/subtype_model_find_params_search_guide.md` 对 subtype 模型、参数校准逻辑和正式搜索流程做了更完整的方法学说明。

### `find_params.py`：联合校准 subtype 参数与初始状态

这个脚本面向 `subtype_grouped` 模型，通过长时程模拟 + Optuna 联合搜索以下参数：

- `initial_infectious_ratio`
- `subtype_groups.*.initial_weight`
- `subtype_groups.*.persistence_multiplier`
- `subtype_groups.*.cancer_progression_multiplier`

当前目标函数同时考虑总宫颈癌发病率误差、感染亚型占比误差、癌症亚型占比误差和末期发病率趋势约束。脚本在生成候选参数时会显式忽略传入 `model-config` 中已有的 `simulation.init_state_path`，避免复用旧稳态文件污染校准结果。

```bash
uv run find_params.py \
  --model-config conf/simulate.json \
  --params-config conf/find_params.json \
  --output-dir results/find-params
```

### `simulate.py`：单次仿真与可选评价

这个脚本对应“根据某个接种策略做传播动力学模拟”。它会读取模型配置并运行一次仿真；如果同时提供 `--evaluation-config`，还会继续计算经济学评价。

```bash
uv run simulate.py \
  --model-config conf/simulate.json \
  --evaluation-config conf/evaluation.json \
  --output-dir results/example_simulation
```

### `search.py`：最优接种策略搜索

这个脚本对应“通过贝叶斯优化搜索最优接种策略”。它会读取模型、评价和搜索配置，调用 `Searcher` 进行多目标搜索。正式分析中通常会在不同 horizon 下分别运行，例如 30、40、50、60、80、100 年。

```bash
uv run search.py \
  --model-config conf/simulate.json \
  --evaluation-config conf/evaluation.json \
  --search-config conf/search.json \
  --time-horizon 60 \
  --output-dir results/example_search
```

### `summary.py`：基于标准化结果文件做汇总分析

这个脚本直接消费 `results/` 下的标准化结果目录，生成论文表格、主文图、补充图，以及预算影响、共付和多方分担分析结果。常用子命令包括：

- `tab1`
- `tabs1`
- `sensitivity`
- `budget`
- `copay`
- `triparty`
- `fig2`
- `fig3`
- `all`

```bash
uv run summary.py all \
  --results-root results \
  --output-dir summary
```

这些脚本本质上都只是包外工作流入口，它们和普通使用者一样，只调用 `hpv_tdm` 的公开 API。

## 6. 结果文件

### 仿真

- `simulation_result.h5`
- `evaluation_result.h5`
- `model_config.json`
- `evaluation_config.json`
- `last.npy`

### 参数校准

- `study.db`
- `find_params_config.json`
- `initial_state.npy`
- `calibrated_model_config.json`
- `model_config_with_init_state.json`
- `calibration_simulation_result.h5`
- `best_trial.json`
- `summary.json`

### 搜索

- `study.db`
- `search_result.h5`
- `best_simulation.h5`
- `best_evaluation.h5`
- `model_config.json`
- `evaluation_config.json`
- `search_config.json`
- `best_trial.json`

### 汇总分析

`summary.py` 的输出会随子命令变化，常见文件包括：

- `table1.xlsx` / `table1.csv` / `table1.md`
- `table_s1.xlsx` / `table_s1.csv` / `table_s1.md`
- `table_s3.xlsx` / `table_s3.csv` / `table_s3.md`
- `sensitivity_s3.json` / `sensitivity_s3.csv`
- `budget_impact.xlsx`
- `copay_impact.xlsx`
- `triparty_impact.xlsx`
- `figure_2.png`
- `figure_3.png`
- `figure_s1.png` 到 `figure_s6.png`

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
