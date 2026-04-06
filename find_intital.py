from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hpv_tdm import (
    AgeSexAggregateHPVModel,
    AgeSexSubtypeGroupedHPVModel,
    AggregateModelConfig,
    BaseHPVTransmissionModel,
    SubtypeGroupedModelConfig,
)


def _load_model_config(
    path: str | Path,
) -> AggregateModelConfig | SubtypeGroupedModelConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    model_kind = payload.get("model_kind", "aggregate")
    if model_kind == "aggregate":
        return AggregateModelConfig.from_json_dict(payload)
    if model_kind == "subtype_grouped":
        return SubtypeGroupedModelConfig.from_json_dict(payload)
    raise ValueError(f"unsupported model kind: {model_kind}")


def _build_model(
    config: AggregateModelConfig | SubtypeGroupedModelConfig,
) -> BaseHPVTransmissionModel:
    if config.model_kind == "aggregate":
        return AgeSexAggregateHPVModel(config)
    return AgeSexSubtypeGroupedHPVModel(config)


def _with_seed_ratio(
    config: AggregateModelConfig | SubtypeGroupedModelConfig,
    *,
    infectious_ratio: float,
    disable_vaccination: bool,
) -> AggregateModelConfig | SubtypeGroupedModelConfig:
    payload = config.model_dump(mode="python")
    payload["transmission"]["female_initial_state"] = [
        1.0 - infectious_ratio,
        infectious_ratio,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    payload["transmission"]["male_initial_state"] = [
        1.0 - infectious_ratio,
        infectious_ratio,
        0.0,
        0.0,
    ]
    if disable_vaccination:
        payload["vaccination"] = {
            "product_id": None,
            "coverage_by_age": [0.0] * config.nages,
        }
    return config.__class__.model_validate(payload)


def _with_init_state_path(
    config: AggregateModelConfig | SubtypeGroupedModelConfig,
    *,
    init_state_path: Path,
) -> AggregateModelConfig | SubtypeGroupedModelConfig:
    payload = config.model_dump(mode="python")
    payload["simulation"]["init_state_path"] = str(init_state_path.resolve())
    return config.__class__.model_validate(payload)


def _terminal_incidence(
    model: BaseHPVTransmissionModel,
    *,
    infectious_ratio: float,
    base_config: AggregateModelConfig | SubtypeGroupedModelConfig,
    time_horizon: float,
    n_eval: int,
    disable_vaccination: bool,
) -> tuple[float, Any]:
    candidate_config = _with_seed_ratio(
        base_config,
        infectious_ratio=infectious_ratio,
        disable_vaccination=disable_vaccination,
    )
    model.set_config(candidate_config)
    result = model.simulate(
        t_span=(0.0, time_horizon),
        n_eval=n_eval,
        verbose=False,
    )

    female_population = model.total_female_population(result.state)
    final_incidence = np.divide(
        model.incidence_matrix(result.state).sum(axis=1),
        female_population.sum(axis=1),
        out=np.zeros(result.time.shape[0], dtype=float),
        where=female_population.sum(axis=1) > 0,
    )
    return float(final_incidence[-1]), result


def _summarize_state(
    model: BaseHPVTransmissionModel,
    final_state: np.ndarray,
) -> dict[str, dict[str, float]]:
    state_matrix = final_state.reshape(model.nrooms, model.nages)
    state_totals = state_matrix.sum(axis=1)
    state_spec = model.state_spec
    male_start = state_spec.index("Sm")
    female_total = float(state_totals[:male_start].sum())
    male_total = float(state_totals[male_start:].sum())

    female_share = {
        state_spec[index]: float(state_totals[index] / female_total)
        for index in range(male_start)
        if female_total > 0
    }
    male_share = {
        state_spec[index]: float(state_totals[index] / male_total)
        for index in range(male_start, len(state_spec))
        if male_total > 0
    }
    return {
        "female": female_share,
        "male": male_share,
    }


def _build_search_trace(rows: list[dict[str, float]]) -> str:
    header = [
        "phase",
        "iteration",
        "infectious_ratio",
        "final_incidence",
        "abs_error",
    ]
    lines = [",".join(header)]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row["phase"]),
                    str(int(row["iteration"])),
                    f"{row['infectious_ratio']:.12g}",
                    f"{row['final_incidence']:.12g}",
                    f"{row['abs_error']:.12g}",
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _find_bracket(
    coarse_results: list[tuple[float, float]],
    target_incidence: float,
) -> tuple[float, float] | None:
    for left, right in zip(coarse_results[:-1], coarse_results[1:]):
        left_error = left[1] - target_incidence
        right_error = right[1] - target_incidence
        if left_error == 0:
            return left[0], left[0]
        if left_error * right_error <= 0:
            return left[0], right[0]
    return None


def _write_diagnostic_payload(
    output_dir: Path,
    *,
    target_incidence: float,
    coarse_results: list[tuple[float, float]],
    rows: list[dict[str, float]],
    model_kind: str,
    disable_vaccination: bool,
    time_horizon: float,
    n_eval: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "search_trace.csv").write_text(
        _build_search_trace(rows),
        encoding="utf-8",
    )
    incidences = [incidence for _, incidence in coarse_results]
    payload = {
        "model_kind": model_kind,
        "target_incidence": float(target_incidence),
        "target_incidence_per_100k": float(target_incidence * 100_000.0),
        "coarse_final_incidence_min": float(min(incidences)),
        "coarse_final_incidence_max": float(max(incidences)),
        "coarse_final_incidence_min_per_100k": float(min(incidences) * 100_000.0),
        "coarse_final_incidence_max_per_100k": float(max(incidences) * 100_000.0),
        "disable_vaccination_during_calibration": disable_vaccination,
        "time_horizon": float(time_horizon),
        "n_eval": int(n_eval),
    }
    with (output_dir / "diagnostic_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Search an initial infectious ratio so that the long-run cervical cancer "
            "incidence approaches a target value, then save the terminal full state "
            "as an initialization file."
        )
    )
    parser.add_argument(
        "--model-config",
        required=True,
        help="Path to model config JSON.",
    )
    parser.add_argument(
        "--target-incidence",
        type=float,
        help=(
            "Target final cervical cancer incidence in model units "
            "(per woman per year)."
        ),
    )
    parser.add_argument(
        "--target-incidence-per-100k",
        type=float,
        help="Target final cervical cancer incidence per 100,000 women per year.",
    )
    parser.add_argument(
        "--time-horizon",
        type=float,
        default=200.0,
        help="Long-run simulation horizon used for calibration, in years.",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=201,
        help="Number of evaluation points used during calibration simulations.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=11,
        help="Number of coarse grid points used before local refinement.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=25,
        help="Maximum number of bisection refinement iterations.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-7,
        help="Absolute tolerance on the final incidence error.",
    )
    parser.add_argument(
        "--max-infectious-ratio",
        type=float,
        default=0.5,
        help="Upper bound of the searched initial infectious ratio.",
    )
    parser.add_argument(
        "--keep-vaccination",
        action="store_true",
        help=(
            "Keep the vaccination settings in model-config during calibration. "
            "By default vaccination is disabled so that the calibrated state "
            "represents a baseline natural-history equilibrium."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/find_initial_state",
        help="Directory used to save calibration outputs.",
    )
    args = parser.parse_args()

    if args.target_incidence is None and args.target_incidence_per_100k is None:
        raise ValueError(
            "one of --target-incidence or --target-incidence-per-100k is required"
        )
    if args.target_incidence is not None and args.target_incidence_per_100k is not None:
        raise ValueError(
            "only one of --target-incidence or --target-incidence-per-100k can be used"
        )
    if args.grid_size < 2:
        raise ValueError("--grid-size must be at least 2")
    if not 0 < args.max_infectious_ratio <= 1:
        raise ValueError("--max-infectious-ratio must be in (0, 1]")

    target_incidence = (
        args.target_incidence
        if args.target_incidence is not None
        else args.target_incidence_per_100k / 100_000.0
    )
    disable_vaccination = not args.keep_vaccination
    output_dir = Path(args.output_dir)

    original_config = _load_model_config(args.model_config)
    calibration_config = original_config
    model = _build_model(calibration_config)

    coarse_rows: list[dict[str, float]] = []
    coarse_results: list[tuple[float, float]] = []
    best_result: tuple[float, float, Any] | None = None

    for index, infectious_ratio in enumerate(
        np.linspace(0.0, args.max_infectious_ratio, args.grid_size)
    ):
        final_incidence, result = _terminal_incidence(
            model,
            infectious_ratio=float(infectious_ratio),
            base_config=calibration_config,
            time_horizon=args.time_horizon,
            n_eval=args.n_eval,
            disable_vaccination=disable_vaccination,
        )
        abs_error = abs(final_incidence - target_incidence)
        coarse_rows.append(
            {
                "phase": "grid",
                "iteration": index,
                "infectious_ratio": float(infectious_ratio),
                "final_incidence": final_incidence,
                "abs_error": abs_error,
            }
        )
        coarse_results.append((float(infectious_ratio), final_incidence))
        if best_result is None or abs_error < best_result[0]:
            best_result = (abs_error, float(infectious_ratio), result)

    bracket = _find_bracket(coarse_results, target_incidence)
    if bracket is None:
        _write_diagnostic_payload(
            output_dir,
            target_incidence=target_incidence,
            coarse_results=coarse_results,
            rows=coarse_rows,
            model_kind=original_config.model_kind,
            disable_vaccination=disable_vaccination,
            time_horizon=args.time_horizon,
            n_eval=args.n_eval,
        )
        incidences = [incidence for _, incidence in coarse_results]
        raise ValueError(
            "target incidence is unreachable by only varying the initial "
            "infectious ratio under the current model parameters. "
            f"Coarse long-run incidence range: [{min(incidences):.8g}, "
            f"{max(incidences):.8g}] "
            f"([{min(incidences) * 100_000.0:.4f}, "
            f"{max(incidences) * 100_000.0:.4f}] per 100k). "
            f"Target: {target_incidence:.8g} "
            f"({target_incidence * 100_000.0:.4f} per 100k). "
            "See search_trace.csv and diagnostic_summary.json for details."
        )

    refine_rows: list[dict[str, float]] = []
    if bracket is not None and bracket[0] != bracket[1]:
        low, high = bracket
        low_incidence = next(
            incidence for ratio, incidence in coarse_results if ratio == low
        )
        for iteration in range(args.max_iter):
            mid = (low + high) / 2.0
            final_incidence, result = _terminal_incidence(
                model,
                infectious_ratio=mid,
                base_config=calibration_config,
                time_horizon=args.time_horizon,
                n_eval=args.n_eval,
                disable_vaccination=disable_vaccination,
            )
            abs_error = abs(final_incidence - target_incidence)
            refine_rows.append(
                {
                    "phase": "bisection",
                    "iteration": iteration,
                    "infectious_ratio": mid,
                    "final_incidence": final_incidence,
                    "abs_error": abs_error,
                }
            )
            if abs_error < best_result[0]:
                best_result = (abs_error, mid, result)
            if abs_error <= args.tol:
                break

            if (low_incidence - target_incidence) * (
                final_incidence - target_incidence
            ) <= 0:
                high = mid
            else:
                low = mid
                low_incidence = final_incidence

    if best_result is None:
        raise RuntimeError("failed to evaluate any candidate initial state")

    _, best_ratio, best_simulation = best_result
    best_final_state = best_simulation.state[-1].reshape(-1)
    best_model = best_simulation.get_model()
    best_incidence = float(
        np.divide(
            best_model.incidence_matrix(best_simulation.state).sum(axis=1),
            best_model.total_female_population(best_simulation.state).sum(axis=1),
            out=np.zeros(best_simulation.time.shape[0], dtype=float),
            where=best_model.total_female_population(best_simulation.state).sum(axis=1)
            > 0,
        )[-1]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    init_state_path = output_dir / "initial_state.npy"
    np.save(init_state_path, best_final_state)

    calibration_model_config = _with_seed_ratio(
        calibration_config,
        infectious_ratio=best_ratio,
        disable_vaccination=disable_vaccination,
    )
    calibration_model_config.to_json_file(output_dir / "calibration_model_config.json")

    ready_model_config = _with_init_state_path(
        original_config,
        init_state_path=init_state_path,
    )
    ready_model_config.to_json_file(output_dir / "model_config_with_init_state.json")
    best_simulation.to_hdf(output_dir / "calibration_simulation_result.h5")

    summary = {
        "model_kind": original_config.model_kind,
        "target_incidence": float(target_incidence),
        "target_incidence_per_100k": float(target_incidence * 100_000.0),
        "matched_incidence": best_incidence,
        "matched_incidence_per_100k": float(best_incidence * 100_000.0),
        "absolute_error": float(abs(best_incidence - target_incidence)),
        "selected_infectious_ratio": float(best_ratio),
        "time_horizon": float(args.time_horizon),
        "n_eval": int(args.n_eval),
        "disable_vaccination_during_calibration": disable_vaccination,
        "initial_state_path": str(init_state_path.resolve()),
        "room_share": _summarize_state(best_model, best_final_state),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    (output_dir / "search_trace.csv").write_text(
        _build_search_trace(coarse_rows + refine_rows),
        encoding="utf-8",
    )

    print("Calibration summary")
    print(f"  model_kind: {original_config.model_kind}")
    print(
        "  target_incidence: "
        f"{target_incidence:.8g} ({target_incidence * 100_000.0:.4f} per 100k)"
    )
    print(
        "  matched_incidence: "
        f"{best_incidence:.8g} ({best_incidence * 100_000.0:.4f} per 100k)"
    )
    print(f"  selected_infectious_ratio: {best_ratio:.8g}")
    print(f"  initial_state_path: {init_state_path.resolve()}")
    print(
        "  ready_model_config: "
        f"{(output_dir / 'model_config_with_init_state.json').resolve()}"
    )


if __name__ == "__main__":
    main()
