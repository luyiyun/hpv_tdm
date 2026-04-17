# HPV TDM Toolkit

[中文说明 / Chinese Version](README.zh.md)

`hpv_tdm` is an HPV transmission dynamics and vaccination-strategy optimization toolkit for research and policy analysis. The repository keeps a script-oriented research workflow, while the package itself exposes only 4 groups of public interfaces:

- `ModelConfig` + `Model`
- `EvaluationConfig` + `Evaluator`
- `SearchConfig` + `Searcher`
- `SimulationResult` / `EvaluationResult` / `SearchResult`

The repository currently supports two model families:

- `AgeSexAggregateHPVModel`
  - Age-sex structured aggregate model
  - Does not distinguish HPV subtypes
  - Useful as a baseline or comparison model
- `AgeSexSubtypeGroupedHPVModel`
  - Age-sex structured high-risk subtype-grouped model
  - Includes 3 cervical-cancer-related high-risk groups by default:
    - `hr_16_18`
    - `hr_31_33_45_52_58`
    - `hr_other`
  - `6/11` is not included at the moment
  - Individual-level co-infection combinations are not modeled explicitly
  - Shares the true female/male population denominator and introduces a shared female persistent-risk pool `Pany`
  - Uses a competing-risk / single-cause attribution approximation at the cancer-event layer
  - `SubtypeGrouped` splits subtype-specific parameters into:
    - initial infection weight `initial_weight`
    - persistence multiplier `persistence_multiplier`
    - cancer progression multiplier `cancer_progression_multiplier`

Both model families currently use the explicit 8-dimensional female initial-state vector:

- `[Sf, If, Pf, LC, RC, DC, Rf, Vf]`

In the subtype model, the initial totals in `If/Pf/LC/RC/DC/Rf` are further distributed across subtype groups according to their weights.

For default values, rationale, and source notes for subtype-specific parameters, see [docs/subtype_parameter_defaults.md](docs/subtype_parameter_defaults.md).

## Installation

The project uses `uv` for dependency management.

```bash
uv sync --all-extras --dev
```

Common commands:

```bash
uv run find_params.py --model-config conf/simulate.json --params-config conf/find_params.json --output-dir results/find-params
uv run simulate.py --model-config conf/simulate.json --evaluation-config conf/evaluation.json --output-dir results/example_simulation
uv run search.py --model-config conf/simulate.json --evaluation-config conf/evaluation.json --search-config conf/search.json --time-horizon 60 --output-dir results/example_search
uv run summary.py all --results-root results --output-dir summary
uv run pytest
uv run ruff check .
uv run ruff format .
```

## Project Layout

- `simulate.py`
  - External script entry point
  - Reads `ModelConfig` / `EvaluationConfig`
  - Instantiates `Model` and `Evaluator` through the public API and runs them like a normal user would
- `search.py`
  - External script entry point
  - Reads `ModelConfig` / `EvaluationConfig` / `SearchConfig`
  - Instantiates `Model`, `Evaluator`, and `Searcher` through the public API and runs them
- `find_params.py`
  - External script entry point
  - Jointly calibrates the subtype-grouped model's initial state and subtype parameters through long-horizon simulation + Optuna
  - Currently supports only the `subtype_grouped` model
- `summary.py`
  - External script entry point
  - Consumes standardized result files directly and generates paper tables, main figures, and supplementary analysis outputs
- `src/hpv_tdm/config`
  - Pydantic config objects
  - Public entry points are consolidated as `AggregateModelConfig`, `SubtypeGroupedModelConfig`, `EvaluationConfig`, and `SearchConfig`
- `src/hpv_tdm/model`
  - Transmission model implementations
  - Includes internal demographic, life-table, and sexual-contact components
- `src/hpv_tdm/evaluator`
  - `Evaluator`
  - Responsible only for how evaluation metrics are calculated
- `src/hpv_tdm/search`
  - `Searcher`
  - Responsible only for how Bayesian optimization is carried out
- `src/hpv_tdm/result`
  - Result objects, summary tables, plotting, and HDF5 I/O
- `experiments`
  - Paper figures and supplementary analyses based on standardized result files
- `results`
  - Output directory for `simulate.py`, `search.py`, `find_params.py`, and related scripts
- `summary`
  - Default output directory for `summary.py`

## 1. Run a Transmission Simulation

If you only want to run a transmission simulation for a specific country or region, you only need to:

1. Build the appropriate `ModelConfig`
2. Instantiate the corresponding `Model`
3. Call `simulate()`

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

`SimulationResult` provides:

- `summary_table()`
- `plot_incidence()`
- `plot_mortality()`
- `to_hdf()`
- `from_hdf()`

If you need detailed simulation states, you can access them directly through:

- `sim_result.time`
- `sim_result.state`
- `sim_result.cumulative`

## 2. Evaluate Simulation Results

Once you have a `SimulationResult`, you can continue with economic evaluation:

```python
from hpv_tdm import EvaluationConfig, Evaluator

evaluator = Evaluator(EvaluationConfig(discount_rate=0.03))
eval_result = evaluator.evaluate(sim_result)
print(eval_result.summary_table())
eval_result.plot_cost(save_path="results/demo_simulation/cost.png")
eval_result.to_hdf("results/demo_simulation/evaluation_result.h5")
```

If you need incremental evaluation such as ICUR, cases averted, deaths averted, or DALYs averted, pass a reference scenario as well:

```python
reference_result = AgeSexAggregateHPVModel(
    config.with_vaccination(product_id=None, coverage_by_age=[0.0] * config.nages)
).simulate()
incremental_result = evaluator.evaluate(sim_result, reference_result)
print(incremental_result.summary_table())
```

`EvaluationResult` provides:

- `summary_table()`
- `plot_incidence()`
- `plot_mortality()`
- `plot_cost()`
- `plot_daly()`
- `plot_icur()`
- `to_hdf()`
- `from_hdf()`

## 3. Search for an Optimal Vaccination Strategy

To search for an optimal vaccination strategy, you need to:

1. Build a `SearchConfig`
2. Instantiate `Searcher`
3. Call `search(model, evaluator)`

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

`SearchConfig.strategy` directly controls the vaccination-age search mode:

- `one`
  - Search a single vaccination age group
- `multi`
  - Search multiple discrete age groups
- `conti`
  - Search a continuous age band and optimize coverage for each age separately
- `conti_one_cover`
  - Search a continuous age band with one shared coverage level

`SearchResult` provides:

- `summary_table()`
- `plot_history()`
- `plot_pareto()`
- `save(directory)`
- `from_dir()`

The default saved outputs include:

- `study.db`
- `search_result.h5`
- `best_simulation.h5`
- `best_evaluation.h5`
- `search_config.json`
- `best_trial.json`

## 4. Two Ways to Use Configs

### Build Them Directly in Python

Pydantic config classes support partial overrides at construction time:

```python
from hpv_tdm import AggregateModelConfig, SearchConfig

model_config = AggregateModelConfig(
    simulation={"n_eval": 31},
    vaccination={"coverage_by_age": [0.0] * 26},
)
search_config = SearchConfig(n_trials=200, strategy="conti_one_cover")
```

Public config fields currently keep Chinese `description` metadata, which can still be inspected directly through Pydantic field metadata.

### Load Partial Overrides from JSON

Each public config class supports:

- `from_json_file`
- `from_json_dict`
- `to_json_file`

In JSON, you only need to provide the fields you want to override. Unspecified fields keep their defaults.

```python
from hpv_tdm import SubtypeGroupedModelConfig

config = SubtypeGroupedModelConfig.from_json_file("conf/simulate.json")
```

The JSON merge rules are fixed:

- object fields are merged recursively
- list fields are replaced as a whole
- scalar fields are overridden directly

## 5. Script Workflow

Although the package no longer exposes a higher-level `Scenario` / `service` layer, the repository still keeps a complete research-script workflow. For the subtype model, the formal analysis is usually organized in the following order:

1. Use `find_params.py` to jointly calibrate the subtype model's initial state and subtype parameters
2. Use `simulate.py` to run a single vaccination scenario, optionally followed by economic evaluation
3. Use `search.py` to search for optimal strategies under a given horizon
4. Use `summary.py` to generate paper tables, figures, and supplementary analyses from standardized result files

For a fuller methodological explanation of the subtype model, calibration logic, and formal search workflow, see [docs/subtype_model_find_params_search_guide.md](docs/subtype_model_find_params_search_guide.md).

### `find_params.py`: Joint Calibration of Subtype Parameters and Initial State

This script is designed for the `subtype_grouped` model and jointly searches the following parameters through long-horizon simulation + Optuna:

- `initial_infectious_ratio`
- `subtype_groups.*.initial_weight`
- `subtype_groups.*.persistence_multiplier`
- `subtype_groups.*.cancer_progression_multiplier`

The current objective function simultaneously considers total cervical cancer incidence error, infection subtype-share error, cancer subtype-share error, and a terminal incidence-trend constraint. When generating candidate parameters, the script explicitly ignores any existing `simulation.init_state_path` in the incoming `model-config`, so that old steady-state files do not contaminate calibration.

```bash
uv run find_params.py \
  --model-config conf/simulate.json \
  --params-config conf/find_params.json \
  --output-dir results/find-params
```

### `simulate.py`: Single Simulation with Optional Evaluation

This script corresponds to "run a transmission simulation for one vaccination strategy". It reads the model configuration and runs one simulation. If `--evaluation-config` is also provided, it continues with economic evaluation.

```bash
uv run simulate.py \
  --model-config conf/simulate.json \
  --evaluation-config conf/evaluation.json \
  --output-dir results/example_simulation
```

### `search.py`: Optimal Vaccination Strategy Search

This script corresponds to "search for the optimal vaccination strategy through Bayesian optimization". It reads the model, evaluation, and search configs, then calls `Searcher` for multi-objective search. In formal analyses, it is usually run separately for different horizons such as 30, 40, 50, 60, 80, and 100 years.

```bash
uv run search.py \
  --model-config conf/simulate.json \
  --evaluation-config conf/evaluation.json \
  --search-config conf/search.json \
  --time-horizon 60 \
  --output-dir results/example_search
```

### `summary.py`: Summary Analyses from Standardized Result Files

This script consumes standardized result directories under `results/` and generates paper tables, main figures, supplementary figures, and financing analyses such as budget impact, co-payment, and multi-party contribution scenarios. Common subcommands include:

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

These scripts are all outer workflow entry points and use only the public `hpv_tdm` API, just like an external user would.

## 6. Output Files

### Simulation

- `simulation_result.h5`
- `evaluation_result.h5`
- `model_config.json`
- `evaluation_config.json`
- `last.npy`

### Parameter Calibration

- `study.db`
- `find_params_config.json`
- `initial_state.npy`
- `calibrated_model_config.json`
- `model_config_with_init_state.json`
- `calibration_simulation_result.h5`
- `best_trial.json`
- `summary.json`

### Search

- `study.db`
- `search_result.h5`
- `best_simulation.h5`
- `best_evaluation.h5`
- `model_config.json`
- `evaluation_config.json`
- `search_config.json`
- `best_trial.json`

### Summary Analyses

Outputs from `summary.py` depend on the subcommand. Common files include:

- `table1.xlsx` / `table1.csv` / `table1.md`
- `table_s1.xlsx` / `table_s1.csv` / `table_s1.md`
- `table_s3.xlsx` / `table_s3.csv` / `table_s3.md`
- `sensitivity_s3.json` / `sensitivity_s3.csv`
- `budget_impact.xlsx`
- `copay_impact.xlsx`
- `triparty_impact.xlsx`
- `figure_2.png`
- `figure_3.png`
- `figure_s1.png` to `figure_s6.png`

## 7. How to Extend

If you want to extend the project in the future:

- Add a new transmission model
  - add the corresponding `ModelConfig`
  - add the corresponding `Model`
  - keep the `Model(config).simulate()` interface
- Add new economic evaluation logic
  - extend metric calculation in `Evaluator`
  - add summary and plotting support in `EvaluationResult`
- Add a new optimization strategy
  - add a new `strategy` branch inside `Searcher`
  - no need to expose a new public strategy class

## Development Conventions

Project-specific development rules are documented in [AGENTS.md](AGENTS.md).
