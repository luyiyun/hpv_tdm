from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hpv_tdm import (
    AgeSexAggregateHPVModel,
    AgeSexSubtypeGroupedHPVModel,
    AggregateModelConfig,
    EvaluationConfig,
    Evaluator,
    SubtypeGroupedModelConfig,
)


def _load_model_config(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    model_kind = payload.get("model_kind", "aggregate")
    if model_kind == "aggregate":
        return AggregateModelConfig.from_json_dict(payload)
    if model_kind == "subtype_grouped":
        return SubtypeGroupedModelConfig.from_json_dict(payload)
    raise ValueError(f"unsupported model kind: {model_kind}")


def _build_model(model_config):
    if model_config.model_kind == "aggregate":
        return AgeSexAggregateHPVModel(model_config)
    return AgeSexSubtypeGroupedHPVModel(model_config)


def _override_time_horizon(model_config, time_horizon: float | None):
    if time_horizon is None:
        return model_config
    payload = model_config.model_dump(mode="python")
    payload["simulation"]["t_span"] = (
        float(payload["simulation"]["t_span"][0]),
        float(time_horizon),
    )
    if model_config.model_kind == "aggregate":
        return AggregateModelConfig.model_validate(payload)
    return SubtypeGroupedModelConfig.model_validate(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HPV transmission simulation.")
    parser.add_argument(
        "--model-config", required=True, help="Path to model config JSON."
    )
    parser.add_argument(
        "--evaluation-config",
        help="Optional path to evaluation config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/simulate_run",
        help="Directory used to save simulation outputs.",
    )
    parser.add_argument(
        "--time-horizon",
        type=float,
        help="Optional simulation horizon override in years.",
    )
    args = parser.parse_args()

    model_config = _load_model_config(args.model_config)
    model_config = _override_time_horizon(model_config, args.time_horizon)
    model = _build_model(model_config)
    simulation_result = model.simulate()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config.to_json_file(output_dir / "model_config.json")
    simulation_result.to_hdf(output_dir / "simulation_result.h5")
    if model_config.simulation.save_last_state:
        np.save(output_dir / "last.npy", simulation_result.state[-1].reshape(-1))
    if model_config.simulation.generate_plots:
        simulation_result.plot_incidence(save_path=output_dir / "incidence.png")
        simulation_result.plot_mortality(save_path=output_dir / "mortality.png")

    print(simulation_result.summary_table().to_string(index=False))

    if args.evaluation_config is None:
        return

    evaluation_config = EvaluationConfig.from_json_file(args.evaluation_config)
    evaluator = Evaluator(evaluation_config)
    evaluation_result = evaluator.evaluate(simulation_result)
    evaluation_config.to_json_file(output_dir / "evaluation_config.json")
    evaluation_result.to_hdf(output_dir / "evaluation_result.h5")
    if model_config.simulation.generate_plots:
        evaluation_result.plot_cost(save_path=output_dir / "cost.png")
        evaluation_result.plot_daly(save_path=output_dir / "daly.png")
    print(evaluation_result.summary_table().to_string(index=False))


if __name__ == "__main__":
    main()
