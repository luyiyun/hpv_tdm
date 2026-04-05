from __future__ import annotations

import argparse
import json
from pathlib import Path

from hpv_tdm import (
    AgeSexAggregateHPVModel,
    AgeSexSubtypeGroupedHPVModel,
    AggregateModelConfig,
    EvaluationConfig,
    Evaluator,
    SearchConfig,
    Searcher,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Search HPV vaccination strategies.")
    parser.add_argument(
        "--model-config", required=True, help="Path to model config JSON."
    )
    parser.add_argument(
        "--evaluation-config",
        required=True,
        help="Path to evaluation config JSON.",
    )
    parser.add_argument(
        "--search-config", required=True, help="Path to search config JSON."
    )
    parser.add_argument(
        "--time-horizon",
        type=float,
        help="Optional final simulation year used during search.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/search_run",
        help="Directory used to save search outputs.",
    )
    args = parser.parse_args()

    model_config = _load_model_config(args.model_config)
    if args.time_horizon is not None:
        simulation = model_config.simulation.model_copy(
            update={
                "t_span": (model_config.simulation.t_span[0], float(args.time_horizon))
            }
        )
        model_config = model_config.model_copy(update={"simulation": simulation})
    evaluation_config = EvaluationConfig.from_json_file(args.evaluation_config)
    search_config = SearchConfig.from_json_file(args.search_config)

    model = _build_model(model_config)
    evaluator = Evaluator(evaluation_config)
    searcher = Searcher(search_config)
    search_result = searcher.search(model, evaluator, output_dir=args.output_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config.to_json_file(output_dir / "model_config.json")
    evaluation_config.to_json_file(output_dir / "evaluation_config.json")
    search_result.save(output_dir)
    search_result.plot_history(save_path=output_dir / "search_history.png")
    search_result.plot_pareto(save_path=output_dir / "pareto_front.png")
    if search_result.best_simulation is not None:
        search_result.best_simulation.plot_incidence(
            save_path=output_dir / "best_incidence.png"
        )
        search_result.best_simulation.plot_mortality(
            save_path=output_dir / "best_mortality.png"
        )
    if search_result.best_evaluation is not None:
        search_result.best_evaluation.plot_cost(save_path=output_dir / "best_cost.png")
        search_result.best_evaluation.plot_daly(save_path=output_dir / "best_daly.png")
        search_result.best_evaluation.plot_icur(save_path=output_dir / "best_icur.png")

    print(search_result.summary_table().to_string(index=False))
    if search_result.best_evaluation is not None:
        print(search_result.best_evaluation.summary_table().to_string(index=False))


if __name__ == "__main__":
    main()
