from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook

from hpv_tdm import (
    AgeSexAggregateHPVModel,
    AgeSexSubtypeGroupedHPVModel,
    AggregateModelConfig,
    EvaluationConfig,
    EvaluationResult,
    Evaluator,
    SearchResult,
    SimulationResult,
    SubtypeGroupedModelConfig,
)
from hpv_tdm.result._plot import (
    PRODUCT_COLORS,
    apply_nature_style,
    apply_scientific_format,
)

USD_TO_RMB_2019 = 6.90
BUDGET_BEGIN_YEAR = 2019
BUDGET_UNIT_LABEL = "10,000 yuan"
BUDGET_UNIT_DIVISOR = 10_000


@dataclass(frozen=True)
class SensitivityParameter:
    runtime_key: str
    table_label: str
    plot_label: str
    change_display: str
    value_unit: str


SENSITIVITY_PARAMETERS: tuple[SensitivityParameter, ...] = (
    SensitivityParameter(
        runtime_key="epsilon_f",
        table_label=(
            "The probability that women are infected when they come into contact "
            "with HPV partners"
        ),
        plot_label="Female infection\nprob.",
        change_display="0.05",
        value_unit="plain",
    ),
    SensitivityParameter(
        runtime_key="epsilon_m",
        table_label="The probability that a man will be infected by an HPV partner",
        plot_label="Male infection\nprob.",
        change_display="0.05",
        value_unit="plain",
    ),
    SensitivityParameter(
        runtime_key="discount_rate",
        table_label="Bank rate",
        plot_label="Bank rate",
        change_display="0.02",
        value_unit="plain",
    ),
    SensitivityParameter(
        runtime_key="dose_cost",
        table_label="Per capita cost of vaccine procurement",
        plot_label="Vaccine cost",
        change_display="0.2",
        value_unit="rmb",
    ),
    SensitivityParameter(
        runtime_key="dose_cost",
        table_label=(
            "Per capita transportation and management costs and per capita "
            "service costs"
        ),
        plot_label="Service cost",
        change_display="0.2",
        value_unit="rmb",
    ),
    SensitivityParameter(
        runtime_key="cost_per_cecx",
        table_label="Cost of cervical cancer treatment (lifetime)",
        plot_label="CC treatment\ncost",
        change_display="0.2",
        value_unit="rmb",
    ),
    SensitivityParameter(
        runtime_key="daly_fatal",
        table_label="DALYs for cancer diagnosis",
        plot_label="Diagnosis\nDALY",
        change_display="0.1",
        value_unit="plain",
    ),
    SensitivityParameter(
        runtime_key="daly_fatal",
        table_label="DALYs for advanced cancer",
        plot_label="Advanced\nDALY",
        change_display="0.1",
        value_unit="plain",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready summary tables and figures."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    tabs1_parser = subparsers.add_parser(
        "tabs1",
        help="Generate Supplementary Table s1 from the selected model configuration.",
    )
    tabs1_parser.add_argument(
        "--model-config",
        default="results/find_params_4/model_config_with_init_state.json",
        help="Model configuration file used to derive Supplementary Table s1.",
    )
    tabs1_parser.add_argument(
        "--output-dir",
        default="summary",
        help="Directory used to store the generated summary files.",
    )

    sensitivity_parser = subparsers.add_parser(
        "sensitivity",
        help="Compute and persist ICUR one-way sensitivity analysis payloads.",
    )
    sensitivity_parser.add_argument(
        "--results-glob",
        default="search-*y",
        help="Glob pattern under results/ used to discover scenario directories.",
    )
    sensitivity_parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory containing search result folders.",
    )
    sensitivity_parser.add_argument(
        "--output-dir",
        default="summary",
        help="Directory used to store the generated sensitivity payload files.",
    )

    budget_parser = subparsers.add_parser(
        "budget",
        help="Compute budget impact analysis and write a multi-sheet workbook.",
    )
    budget_parser.add_argument(
        "--results-glob",
        default="search-*y",
        help="Glob pattern under results/ used to discover scenario directories.",
    )
    budget_parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory containing search result folders.",
    )
    budget_parser.add_argument(
        "--output-dir",
        default="summary",
        help="Directory used to store the generated budget workbook.",
    )

    for command_name, help_text in (
        (
            "tab1",
            "Summarize the selected optimal vaccination strategies across scenarios.",
        ),
        (
            "figs1",
            "Generate Supplementary Figure s1 by combining Pareto fronts "
            "across scenarios.",
        ),
        (
            "figs2",
            "Generate Supplementary Figure s2 by combining search-history "
            "panels across scenarios.",
        ),
        (
            "tabs3",
            "Generate Supplementary Table s3 with ICUR one-way sensitivity analysis.",
        ),
        (
            "figs3",
            "Generate Supplementary Figure s3 with ICUR tornado plots.",
        ),
        (
            "figs4",
            "Generate Supplementary Figure s4 by combining budget impact panels.",
        ),
    ):
        figure_parser = subparsers.add_parser(command_name, help=help_text)
        figure_parser.add_argument(
            "--results-glob",
            default="search-*y",
            help="Glob pattern under results/ used to discover scenario directories.",
        )
        figure_parser.add_argument(
            "--results-root",
            default="results",
            help="Root directory containing search result folders.",
        )
        figure_parser.add_argument(
            "--output-dir",
            default="summary",
            help="Directory used to store the generated summary files.",
        )
        if command_name in {"tabs3", "figs3"}:
            figure_parser.add_argument(
                "--sensitivity-path",
                default=None,
                help=(
                    "Path to a precomputed sensitivity payload JSON file. "
                    "Defaults to <output-dir>/sensitivity_s3.json."
                ),
            )
        if command_name == "figs4":
            figure_parser.add_argument(
                "--budget-path",
                default="summary/budget_impact.xlsx",
                help="Path to the budget impact workbook generated by `budget`.",
            )

    return parser.parse_args()


def _scenario_years_from_name(directory: Path) -> int:
    stem = directory.name
    match = re.match(r"^search-(\d+)y(?:-.*)?$", stem)
    if match is None:
        raise ValueError(f"unexpected scenario directory name: {directory}")
    return int(match.group(1))


def _age_span_label(age_span: str, agebins: list[float]) -> str:
    start_text, stop_text = age_span.split(":")
    start = int(start_text)
    stop = int(stop_text)
    lower = agebins[start]
    upper = agebins[stop + 1]
    lower_label = int(lower) if float(lower).is_integer() else lower
    upper_label = int(upper) if float(upper).is_integer() else upper
    return f"[{lower_label}, {upper_label})"


def _product_label(product_id: str) -> str:
    mapping = {
        "bivalent": "Bivalent vaccine",
        "quadrivalent": "Quadrivalent vaccine",
        "nonavalent": "Nonavalent vaccine",
    }
    return mapping.get(product_id, product_id)


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _adjust_excel_widths(path: Path) -> None:
    workbook = load_workbook(path)
    worksheet = workbook.active
    for column_cells in worksheet.columns:
        text_lengths = [
            len(str(cell.value)) for cell in column_cells if cell.value is not None
        ]
        width = max(text_lengths, default=0) + 2
        worksheet.column_dimensions[column_cells[0].column_letter].width = min(
            max(width, 12), 36
        )
    workbook.save(path)


def _load_model_config(path: Path) -> AggregateModelConfig | SubtypeGroupedModelConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    model_kind = payload.get("model_kind", "aggregate")
    if model_kind == "aggregate":
        return AggregateModelConfig.from_json_dict(payload)
    if model_kind == "subtype_grouped":
        return SubtypeGroupedModelConfig.from_json_dict(payload)
    raise ValueError(f"unsupported model kind: {model_kind}")


def _load_evaluation_config(path: Path) -> EvaluationConfig:
    return EvaluationConfig.from_json_file(path)


def _build_model(
    config: AggregateModelConfig | SubtypeGroupedModelConfig,
) -> AgeSexAggregateHPVModel | AgeSexSubtypeGroupedHPVModel:
    if isinstance(config, AggregateModelConfig) and not isinstance(
        config,
        SubtypeGroupedModelConfig,
    ):
        return AgeSexAggregateHPVModel(config)
    return AgeSexSubtypeGroupedHPVModel(config)


def _format_table_value(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.5f}".rstrip("0").rstrip(".")


def _table_age_label(lower: float, upper: float) -> str:
    lower_text = str(int(lower)) if float(lower).is_integer() else str(lower)
    if upper == float("inf"):
        return f"[{lower_text}, inf)"
    upper_text = str(int(upper)) if float(upper).is_integer() else str(upper)
    return f"[{lower_text}, {upper_text})"


def _build_tabs1_dataframe(
    model_config: AggregateModelConfig | SubtypeGroupedModelConfig,
) -> pd.DataFrame:
    demography = model_config.demography
    transmission = model_config.transmission
    agebins = demography.agebins
    rows: list[dict[str, str]] = []
    for idx in range(len(agebins) - 1):
        rows.append(
            {
                "Age group": _table_age_label(agebins[idx], agebins[idx + 1]),
                "Number of sexual partners per year for women": _format_table_value(
                    transmission.omega_f[idx]
                ),
                "Number of sexual partners per year for men": _format_table_value(
                    transmission.omega_m[idx]
                ),
                "Mortality of local cancer": _format_table_value(transmission.dL[idx]),
                "Mortality rate of region cancer": _format_table_value(
                    transmission.dR[idx]
                ),
                "Mortality of distant cancer": _format_table_value(
                    transmission.dD[idx]
                ),
                "Fertility rate": _format_table_value(demography.fertilities[idx]),
                "Death rate of women": _format_table_value(
                    demography.deathes_female[idx]
                ),
                "Death rate of men": _format_table_value(demography.deathes_male[idx]),
            }
        )
    return pd.DataFrame.from_records(rows)


def _write_table_outputs(
    dataframe: pd.DataFrame, output_dir: Path, stem: str
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / f"{stem}.xlsx"
    csv_path = output_dir / f"{stem}.csv"
    markdown_path = output_dir / f"{stem}.md"

    dataframe.to_excel(excel_path, index=False)
    dataframe.to_csv(csv_path, index=False)
    markdown_lines = []
    headers = list(dataframe.columns)
    markdown_lines.append("| " + " | ".join(headers) + " |")
    markdown_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in dataframe.itertuples(index=False, name=None):
        markdown_lines.append("| " + " | ".join(str(value) for value in row) + " |")
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    _adjust_excel_widths(excel_path)
    return excel_path, csv_path, markdown_path


def _load_tab1_row(search_dir: Path) -> dict[str, object]:
    with (search_dir / "best_trial.json").open("r", encoding="utf-8") as handle:
        best_trial = json.load(handle)
    with (search_dir / "model_config.json").open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)

    evaluation = EvaluationResult.from_hdf(search_dir / "best_evaluation.h5")

    years = _scenario_years_from_name(search_dir)
    agebins = model_config["demography"]["agebins"]
    target_product_id = best_trial["params"]["target_product_id"]
    target_age_span = best_trial["params"]["target_age_span"]
    coverage = float(best_trial["params"]["coverage"])

    return {
        "Scenario": f"{years} Years",
        "Time horizon (years)": years,
        "Target age group": _age_span_label(target_age_span, agebins),
        "Vaccine type": _product_label(target_product_id),
        "Coverage": _format_percent(coverage),
        "ICUR": float(best_trial["values"][0]),
        "Final incidence (/100k)": float(evaluation.incidence[-1] * 1e5),
        "Final mortality (/100k)": float(evaluation.mortality[-1] * 1e5),
        "Averted cervical cancer cases": float(evaluation.avoid_cecx.sum(axis=1)[-1]),
        "Averted cervical cancer deaths": float(
            evaluation.avoid_cecx_deaths.sum(axis=1)[-1]
        ),
        "Averted DALYs": float(evaluation.avoid_daly[-1]),
    }


def _build_tab1_dataframe(results_root: Path, results_glob: str) -> pd.DataFrame:
    search_dirs = sorted(
        results_root.glob(results_glob),
        key=_scenario_years_from_name,
    )
    if not search_dirs:
        raise ValueError(
            "no search directories matched pattern "
            f"{results_glob!r} under {results_root}"
        )
    records = [_load_tab1_row(directory) for directory in search_dirs]
    return pd.DataFrame.from_records(records)


def _discover_search_dirs(results_root: Path, results_glob: str) -> list[Path]:
    search_dirs = sorted(
        results_root.glob(results_glob),
        key=_scenario_years_from_name,
    )
    if not search_dirs:
        raise ValueError(
            "no search directories matched pattern "
            f"{results_glob!r} under {results_root}"
        )
    return search_dirs


def _validate_sensitivity_dir(search_dir: Path) -> None:
    required = (
        "best_model_config.json",
        "evaluation_config.json",
        "best_trial.json",
    )
    missing = [name for name in required if not (search_dir / name).exists()]
    if missing:
        raise ValueError(
            f"search result directory {search_dir} is missing required files: "
            f"{', '.join(missing)}"
        )


def _build_reference_config(
    model_config: AggregateModelConfig | SubtypeGroupedModelConfig,
) -> AggregateModelConfig | SubtypeGroupedModelConfig:
    return model_config.with_vaccination(
        product_id=None,
        coverage_by_age=[0.0] * model_config.nages,
    )


def _replace_transmission_value(
    model_config: AggregateModelConfig | SubtypeGroupedModelConfig,
    key: str,
    value: float,
) -> AggregateModelConfig | SubtypeGroupedModelConfig:
    transmission = model_config.transmission.model_copy(update={key: value})
    return model_config.model_copy(update={"transmission": transmission})


def _replace_product_dose_cost(
    model_config: AggregateModelConfig | SubtypeGroupedModelConfig,
    dose_cost: float,
) -> AggregateModelConfig | SubtypeGroupedModelConfig:
    product_id = model_config.resolved_product_id()
    product = model_config.vaccine_catalog.get_product(product_id)
    products = dict(model_config.vaccine_catalog.products)
    products[product_id] = product.model_copy(update={"dose_cost": dose_cost})
    vaccine_catalog = model_config.vaccine_catalog.model_copy(
        update={"products": products}
    )
    return model_config.model_copy(update={"vaccine_catalog": vaccine_catalog})


def _replace_evaluation_value(
    evaluation_config: EvaluationConfig,
    key: str,
    value: float,
) -> EvaluationConfig:
    return evaluation_config.model_copy(update={key: value})


def _format_numeric(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"


def _simulate_and_evaluate(
    model_config: AggregateModelConfig | SubtypeGroupedModelConfig,
    evaluation_config: EvaluationConfig,
) -> float:
    vaccinated_model = _build_model(model_config)
    vaccinated_simulation = vaccinated_model.simulate()
    reference_model = _build_model(_build_reference_config(model_config))
    reference_simulation = reference_model.simulate()
    evaluator = Evaluator(evaluation_config)
    evaluation = evaluator.evaluate(vaccinated_simulation, reference_simulation)
    return float(evaluation.icur[-1])


def _evaluate_from_existing_simulations(
    vaccinated_simulation,
    reference_simulation,
    evaluation_config: EvaluationConfig,
) -> float:
    evaluator = Evaluator(evaluation_config)
    evaluation = evaluator.evaluate(vaccinated_simulation, reference_simulation)
    return float(evaluation.icur[-1])


def _parameter_display_values(
    parameter: SensitivityParameter,
    model_config: AggregateModelConfig | SubtypeGroupedModelConfig,
    evaluation_config: EvaluationConfig,
) -> tuple[float, float, float]:
    if parameter.runtime_key == "epsilon_f":
        original = float(model_config.transmission.epsilon_f)
        return original, original * 0.95, original * 1.05
    if parameter.runtime_key == "epsilon_m":
        original = float(model_config.transmission.epsilon_m)
        return original, original * 0.95, original * 1.05
    if parameter.runtime_key == "discount_rate":
        original = float(evaluation_config.discount_rate)
        return original, 0.01, 0.05
    if parameter.runtime_key == "dose_cost":
        original = float(
            model_config.vaccine_catalog.get_product(
                model_config.resolved_product_id()
            ).dose_cost
        )
        return original, original * 0.8, original * 1.2
    if parameter.runtime_key == "cost_per_cecx":
        original = float(evaluation_config.cost_per_cecx)
        return original, original * 0.8, original * 1.2
    if parameter.runtime_key == "daly_fatal":
        original = float(evaluation_config.daly_fatal)
        return original, original * 0.9, original * 1.1
    raise ValueError(f"unsupported sensitivity parameter: {parameter.runtime_key}")


def _format_parameter_value(value: float, unit: str) -> str:
    display = value * USD_TO_RMB_2019 if unit == "rmb" else value
    if unit == "plain" and abs(display) < 1:
        return f"{display:.4f}".rstrip("0").rstrip(".")
    return _format_numeric(display, decimals=2)


def _compute_sensitivity_payloads(
    results_root: Path,
    results_glob: str,
) -> tuple[list[tuple[int, Path]], list[dict[str, object]]]:
    search_dirs = _discover_search_dirs(results_root, results_glob)
    scenarios = [
        (_scenario_years_from_name(directory), directory)
        for directory in search_dirs
    ]
    rows: list[dict[str, object]] = []
    for years, search_dir in scenarios:
        _validate_sensitivity_dir(search_dir)
        model_config = _load_model_config(search_dir / "best_model_config.json")
        evaluation_config = _load_evaluation_config(
            search_dir / "evaluation_config.json"
        )

        vaccinated_model = _build_model(model_config)
        vaccinated_simulation = vaccinated_model.simulate()
        reference_model = _build_model(_build_reference_config(model_config))
        reference_simulation = reference_model.simulate()
        baseline_icur = _evaluate_from_existing_simulations(
            vaccinated_simulation,
            reference_simulation,
            evaluation_config,
        )

        runtime_results: dict[str, dict[str, float]] = {
            "baseline": {"original": baseline_icur}
        }

        epsilon_f = float(model_config.transmission.epsilon_f)
        epsilon_f_results = {}
        for bound_name, bound_value in {
            "lower": epsilon_f * 0.95,
            "upper": epsilon_f * 1.05,
        }.items():
            perturbed_model_config = _replace_transmission_value(
                model_config,
                "epsilon_f",
                bound_value,
            )
            epsilon_f_results[bound_name] = _simulate_and_evaluate(
                perturbed_model_config,
                evaluation_config,
            )
        runtime_results["epsilon_f"] = epsilon_f_results

        epsilon_m = float(model_config.transmission.epsilon_m)
        epsilon_m_results = {}
        for bound_name, bound_value in {
            "lower": epsilon_m * 0.95,
            "upper": epsilon_m * 1.05,
        }.items():
            perturbed_model_config = _replace_transmission_value(
                model_config,
                "epsilon_m",
                bound_value,
            )
            epsilon_m_results[bound_name] = _simulate_and_evaluate(
                perturbed_model_config,
                evaluation_config,
            )
        runtime_results["epsilon_m"] = epsilon_m_results

        dose_cost = float(
            model_config.vaccine_catalog.get_product(model_config.resolved_product_id()).dose_cost
        )
        dose_cost_results = {}
        for bound_name, bound_value in {
            "lower": dose_cost * 0.8,
            "upper": dose_cost * 1.2,
        }.items():
            perturbed_model_config = _replace_product_dose_cost(
                model_config,
                bound_value,
            )
            vaccinated_model_perturbed = _build_model(perturbed_model_config)
            vaccinated_simulation_perturbed = vaccinated_model_perturbed.simulate()
            dose_cost_results[bound_name] = _evaluate_from_existing_simulations(
                vaccinated_simulation_perturbed,
                reference_simulation,
                evaluation_config,
            )
        runtime_results["dose_cost"] = dose_cost_results

        discount_results = {}
        for bound_name, bound_value in {"lower": 0.01, "upper": 0.05}.items():
            perturbed_evaluation_config = _replace_evaluation_value(
                evaluation_config,
                "discount_rate",
                bound_value,
            )
            discount_results[bound_name] = _evaluate_from_existing_simulations(
                vaccinated_simulation,
                reference_simulation,
                perturbed_evaluation_config,
            )
        runtime_results["discount_rate"] = discount_results

        cost_per_cecx = float(evaluation_config.cost_per_cecx)
        cost_results = {}
        for bound_name, bound_value in {
            "lower": cost_per_cecx * 0.8,
            "upper": cost_per_cecx * 1.2,
        }.items():
            perturbed_evaluation_config = _replace_evaluation_value(
                evaluation_config,
                "cost_per_cecx",
                bound_value,
            )
            cost_results[bound_name] = _evaluate_from_existing_simulations(
                vaccinated_simulation,
                reference_simulation,
                perturbed_evaluation_config,
            )
        runtime_results["cost_per_cecx"] = cost_results

        daly_fatal = float(evaluation_config.daly_fatal)
        daly_results = {}
        for bound_name, bound_value in {
            "lower": daly_fatal * 0.9,
            "upper": daly_fatal * 1.1,
        }.items():
            perturbed_evaluation_config = _replace_evaluation_value(
                evaluation_config,
                "daly_fatal",
                bound_value,
            )
            daly_results[bound_name] = _evaluate_from_existing_simulations(
                vaccinated_simulation,
                reference_simulation,
                perturbed_evaluation_config,
            )
        runtime_results["daly_fatal"] = daly_results

        baseline_icur_rmb = baseline_icur * USD_TO_RMB_2019
        for parameter in SENSITIVITY_PARAMETERS:
            original_value, lower_value, upper_value = _parameter_display_values(
                parameter,
                model_config,
                evaluation_config,
            )
            lower_icur_rmb = (
                runtime_results[parameter.runtime_key]["lower"] * USD_TO_RMB_2019
            )
            upper_icur_rmb = (
                runtime_results[parameter.runtime_key]["upper"] * USD_TO_RMB_2019
            )
            rows.append(
                {
                    "scenario_years": years,
                    "parameter": parameter.table_label,
                    "plot_label": parameter.plot_label,
                    "change_display": parameter.change_display,
                    "original_value_display": _format_parameter_value(
                        original_value,
                        parameter.value_unit,
                    ),
                    "lower_value_display": _format_parameter_value(
                        lower_value,
                        parameter.value_unit,
                    ),
                    "upper_value_display": _format_parameter_value(
                        upper_value,
                        parameter.value_unit,
                    ),
                    "baseline_icur_rmb": baseline_icur_rmb,
                    "lower_icur_rmb": lower_icur_rmb,
                    "upper_icur_rmb": upper_icur_rmb,
                    "absolute_change_rmb": max(
                        abs(lower_icur_rmb - baseline_icur_rmb),
                        abs(upper_icur_rmb - baseline_icur_rmb),
                    ),
                    "lower_delta_rmb": lower_icur_rmb - baseline_icur_rmb,
                    "upper_delta_rmb": upper_icur_rmb - baseline_icur_rmb,
                }
            )
    return scenarios, rows


def _sensitivity_payload_path(path: str | Path | None, output_dir: Path) -> Path:
    if path is not None:
        return Path(path)
    return output_dir / "sensitivity_s3.json"


def _write_sensitivity_outputs(
    scenarios: list[tuple[int, Path]],
    rows: list[dict[str, object]],
    output_dir: Path,
    *,
    payload_path: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenarios": [
            {"years": years, "directory": str(directory)}
            for years, directory in scenarios
        ],
        "rows": rows,
        "usd_to_rmb_2019": USD_TO_RMB_2019,
    }
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    longform_path = output_dir / "sensitivity_s3.csv"
    pd.DataFrame.from_records(rows).to_csv(longform_path, index=False)
    return payload_path, longform_path


def _load_sensitivity_payload(
    path: Path,
) -> tuple[list[tuple[int, Path]], list[dict[str, object]]]:
    if not path.exists():
        raise ValueError(
            f"sensitivity payload not found: {path}. "
            "Run `uv run summary.py sensitivity` first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenarios = [
        (int(item["years"]), Path(item["directory"])) for item in payload["scenarios"]
    ]
    rows = list(payload["rows"])
    return scenarios, rows


def _adjust_excel_widths_all_sheets(path: Path) -> None:
    workbook = load_workbook(path)
    for worksheet in workbook.worksheets:
        for column_cells in worksheet.columns:
            text_lengths = [
                len(str(cell.value)) for cell in column_cells if cell.value is not None
            ]
            width = max(text_lengths, default=0) + 2
            worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                max(width, 12), 36
            )
    workbook.save(path)


def _validate_budget_dir(search_dir: Path) -> None:
    required = (
        "best_model_config.json",
        "evaluation_config.json",
        "best_simulation.h5",
    )
    missing = [name for name in required if not (search_dir / name).exists()]
    if missing:
        raise ValueError(
            f"search result directory {search_dir} is missing required files: "
            f"{', '.join(missing)}"
        )


def _budget_workbook_path(output_dir: Path) -> Path:
    return output_dir / "budget_impact.xlsx"


def _interpolate_cumulative_to_year_grid(
    time: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    horizon = int(round(float(time[-1])))
    year_grid = np.arange(0, horizon + 1, dtype=float)
    interpolated = np.vstack(
        [
            np.interp(year_grid, time, values[:, age_index])
            for age_index in range(values.shape[1])
        ]
    ).T
    return year_grid, interpolated


def _compute_budget_component_frame(
    sim_result: SimulationResult,
    evaluation_config: EvaluationConfig,
    *,
    vacc_insured: float = 1.0,
    vacc_reimburse: float = 1.0,
    minor_insured: float = 0.869,
    cecx_diag: float = 0.81,
    cecx_treatment: float = 0.9,
    cecx_insured: tuple[float, float] = (0.7569, 0.2431),
    cecx_reimburse: tuple[float, float] = (0.479, 0.6211),
    exchange_rate: float = USD_TO_RMB_2019,
    n_minor: int = 10,
) -> pd.DataFrame:
    model = sim_result.get_model()
    cumulative_cecx = np.asarray(
        model.cumulative_cecx(sim_result.cumulative),
        dtype=float,
    )
    cumulative_vaccinated = np.asarray(
        model.cumulative_vaccinated(sim_result.cumulative),
        dtype=float,
    )

    year_grid, cumulative_cecx_yearly = _interpolate_cumulative_to_year_grid(
        np.asarray(sim_result.time, dtype=float),
        cumulative_cecx,
    )
    _, cumulative_vaccinated_yearly = _interpolate_cumulative_to_year_grid(
        np.asarray(sim_result.time, dtype=float),
        cumulative_vaccinated,
    )

    annual_cecx = cumulative_cecx_yearly[1:] - cumulative_cecx_yearly[:-1]
    annual_vaccinated = (
        cumulative_vaccinated_yearly[1:] - cumulative_vaccinated_yearly[:-1]
    )
    annual_treated = annual_cecx * cecx_diag * cecx_treatment

    discount = np.power(
        1 / (1 + evaluation_config.discount_rate),
        np.arange(annual_cecx.shape[0]),
    )

    product_id = model.config.resolved_product_id()
    if product_id is None:
        vaccine_cost_per_age = np.zeros(model.nages, dtype=float)
    else:
        product = model.config.vaccine_catalog.get_product(product_id)
        vaccine_cost_per_age = model.vaccination_cost_per_age(
            dose_cost=product.dose_cost,
            doses_under_15=product.doses_under_15,
            doses_over_15=product.doses_over_15,
        )

    cost_vacc_age = (
        annual_vaccinated * vaccine_cost_per_age[None, :] * discount[:, None]
    )
    cost_cecx_age = annual_treated * evaluation_config.cost_per_cecx * discount[:, None]

    cost_vacc = cost_vacc_age.sum(axis=1)
    cost_cecx_minor = cost_cecx_age[:, :n_minor].sum(axis=1)
    cost_cecx_adult = cost_cecx_age[:, n_minor:].sum(axis=1)

    cost_vacc_insurance = cost_vacc * vacc_insured * vacc_reimburse
    cost_cecx_minor_insurance = cost_cecx_minor * minor_insured * cecx_reimburse[0]
    cost_cecx_adult_insurance = sum(
        cost_cecx_adult * insure_rate * reimburse_rate
        for insure_rate, reimburse_rate in zip(cecx_insured, cecx_reimburse)
    )
    cost_cecx_insured = cost_cecx_minor_insurance + cost_cecx_adult_insurance
    cost_insurance = cost_vacc_insurance + cost_cecx_insured

    years = year_grid[1:].astype(int) + BUDGET_BEGIN_YEAR - 1
    payload = {
        "Year": years,
        "Vaccination count": annual_vaccinated.sum(axis=1),
        "Treated cervical cancer cases": annual_treated.sum(axis=1),
        "Vaccination fund": cost_vacc_insurance * exchange_rate / BUDGET_UNIT_DIVISOR,
        "Treatment fund": cost_cecx_insured * exchange_rate / BUDGET_UNIT_DIVISOR,
        "Total fund": cost_insurance * exchange_rate / BUDGET_UNIT_DIVISOR,
    }
    return pd.DataFrame(payload)


def _build_budget_sheet_dataframe(search_dir: Path) -> tuple[int, pd.DataFrame]:
    _validate_budget_dir(search_dir)
    years = _scenario_years_from_name(search_dir)
    model_config = _load_model_config(search_dir / "best_model_config.json")
    evaluation_config = _load_evaluation_config(search_dir / "evaluation_config.json")

    vaccinated_simulation = SimulationResult.from_hdf(search_dir / "best_simulation.h5")
    reference_model = _build_model(_build_reference_config(model_config))
    reference_simulation = reference_model.simulate()

    vaccinated_df = _compute_budget_component_frame(
        vaccinated_simulation,
        evaluation_config,
    )
    reference_df = _compute_budget_component_frame(
        reference_simulation,
        evaluation_config,
    )

    merged = vaccinated_df.merge(
        reference_df,
        on="Year",
        how="inner",
        suffixes=(" - Optimal strategy", " - No vaccination"),
    )
    merged = merged.sort_values("Year").reset_index(drop=True)
    merged["Treatment fund increment"] = (
        merged["Treatment fund - Optimal strategy"]
        - merged["Treatment fund - No vaccination"]
    )
    merged["Total fund increment"] = (
        merged["Total fund - Optimal strategy"] - merged["Total fund - No vaccination"]
    )
    ordered = merged[
        [
            "Year",
            "Treatment fund - No vaccination",
            "Treatment fund - Optimal strategy",
            "Treatment fund increment",
            "Total fund - No vaccination",
            "Total fund - Optimal strategy",
            "Total fund increment",
            "Vaccination fund - Optimal strategy",
            "Vaccination count - Optimal strategy",
            "Treated cervical cancer cases - No vaccination",
            "Treated cervical cancer cases - Optimal strategy",
        ]
    ]
    return years, ordered


def _build_budget_readme_dataframe() -> pd.DataFrame:
    rows = [
        ("Workbook purpose", "Budget impact analysis for optimal strategies"),
        ("Currency output", "RMB"),
        ("Unit", BUDGET_UNIT_LABEL),
        ("Exchange rate", f"1 USD = {USD_TO_RMB_2019:.2f} RMB"),
        ("Base year", str(BUDGET_BEGIN_YEAR)),
        ("vacc_insured", "1.0"),
        ("vacc_reimburse", "1.0"),
        ("minor_insured", "0.869"),
        ("cecx_diag", "0.81"),
        ("cecx_treatment", "0.9"),
        ("cecx_insured", "(0.7569, 0.2431)"),
        ("cecx_reimburse", "(0.479, 0.6211)"),
        ("discount_rate", "0.03"),
    ]
    return pd.DataFrame(rows, columns=["Item", "Value"])


def _write_budget_workbook(
    output_dir: Path,
    sheet_payloads: list[tuple[int, pd.DataFrame]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = _budget_workbook_path(output_dir)
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for years, dataframe in sheet_payloads:
            dataframe.to_excel(writer, sheet_name=f"{years}y", index=False)
        _build_budget_readme_dataframe().to_excel(
            writer,
            sheet_name="README",
            index=False,
        )
    _adjust_excel_widths_all_sheets(workbook_path)
    return workbook_path


def _load_budget_sheets(
    budget_path: Path,
) -> list[tuple[int, pd.DataFrame]]:
    if not budget_path.exists():
        raise ValueError(
            f"budget workbook not found: {budget_path}. "
            "Run `uv run summary.py budget` first."
        )
    workbook = pd.read_excel(budget_path, sheet_name=None)
    sheets: list[tuple[int, pd.DataFrame]] = []
    for name, dataframe in workbook.items():
        match = re.match(r"^(\d+)y$", name)
        if match is None:
            continue
        sheets.append((int(match.group(1)), dataframe))
    if not sheets:
        raise ValueError(f"no scenario sheets found in {budget_path}")
    return sorted(sheets, key=lambda item: item[0])


def _build_tabs3_dataframe(
    scenarios: list[tuple[int, Path]],
    rows: list[dict[str, object]],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for parameter in SENSITIVITY_PARAMETERS:
        matching_first = next(
            row
            for row in rows
            if row["scenario_years"] == scenarios[0][0]
            and row["parameter"] == parameter.table_label
        )
        record: dict[str, object] = {
            "Parameter": parameter.table_label,
            "The rate of change": matching_first["change_display"],
            "Original value": matching_first["original_value_display"],
            "Lower limit value": matching_first["lower_value_display"],
            "Upper limit value": matching_first["upper_value_display"],
        }
        for years, _ in scenarios:
            matching = next(
                row
                for row in rows
                if row["scenario_years"] == years
                and row["parameter"] == parameter.table_label
            )
            prefix = f"{years}y"
            record[f"{prefix} Original ICUR"] = _format_numeric(
                matching["baseline_icur_rmb"]
            )
            record[f"{prefix} Lower limit ICUR"] = _format_numeric(
                matching["lower_icur_rmb"]
            )
            record[f"{prefix} Upper limit ICUR"] = _format_numeric(
                matching["upper_icur_rmb"]
            )
            record[f"{prefix} Absolute change in ICUR"] = _format_numeric(
                matching["absolute_change_rmb"]
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _load_search_results(
    results_root: Path, results_glob: str
) -> list[tuple[int, Path, SearchResult]]:
    search_dirs = _discover_search_dirs(results_root, results_glob)
    return [
        (
            _scenario_years_from_name(directory),
            directory,
            SearchResult.from_dir(directory),
        )
        for directory in search_dirs
    ]


def _subplot_panel_label(index: int) -> str:
    return chr(ord("A") + index)


def _build_figs1(
    output_dir: Path, search_results: list[tuple[int, Path, SearchResult]]
) -> Path:
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(8.4, 8.4),
        sharex=True,
        sharey=False,
    )
    axes_flat = axes.flatten()
    x_values: list[float] = []
    for _, _, result in search_results:
        completed_trials = [
            trial for trial in result.study.trials if trial.state.name == "COMPLETE"
        ]
        x_values.extend(
            result._trial_incidence(trial) * 1e5 for trial in completed_trials
        )
    x_min, x_max = min(x_values), max(x_values)
    x_pad = (x_max - x_min) * 0.05 if x_max > x_min else max(x_max * 0.05, 0.1)

    for index, ((years, _, result), axis) in enumerate(zip(search_results, axes_flat)):
        completed_trials = [
            trial for trial in result.study.trials if trial.state.name == "COMPLETE"
        ]
        product_ids = sorted(
            {
                str(trial.params.get("target_product_id", "unknown"))
                for trial in completed_trials
            }
        )
        pareto_numbers = {trial.number for trial in result.study.best_trials}
        for product_id in product_ids:
            product_trials = [
                trial
                for trial in completed_trials
                if trial.params.get("target_product_id") == product_id
            ]
            if not product_trials:
                continue
            axis.scatter(
                [result._trial_incidence(trial) * 1e5 for trial in product_trials],
                [result._trial_icur(trial) for trial in product_trials],
                color=PRODUCT_COLORS.get(product_id, "#6C757D"),
                alpha=0.7,
                s=16,
                linewidths=0,
            )
            highlight_trials = [
                trial for trial in product_trials if trial.number in pareto_numbers
            ]
            if highlight_trials:
                axis.scatter(
                    [
                        result._trial_incidence(trial) * 1e5
                        for trial in highlight_trials
                    ],
                    [result._trial_icur(trial) for trial in highlight_trials],
                    color=PRODUCT_COLORS.get(product_id, "#6C757D"),
                    marker="*",
                    s=56,
                    edgecolors="#202020",
                    linewidths=0.4,
                )
                axis.axvline(
                    x=4, color="#202020", linestyle="--", linewidth=0.8, alpha=0.9
                )
        axis.set_title(f"{years}-year horizon", fontsize=10, pad=4)
        axis.text(
            0.02,
            0.98,
            _subplot_panel_label(index),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )
        axis.set_xlim(x_min - x_pad, x_max + x_pad)
        # axis.set_ylim(max(0, y_min - y_pad), y_max + y_pad)
        apply_scientific_format(axis, x=False, y=True)
    for axis in axes_flat[:4]:
        axis.tick_params(labelbottom=False)
    for axis in [axes_flat[1], axes_flat[3], axes_flat[5]]:
        axis.tick_params(labelleft=False)
    fig.supxlabel("Final cervical cancer incidence (/100,000 women)", fontsize=11)
    fig.supylabel("ICUR", fontsize=11)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=PRODUCT_COLORS[product_id],
            markeredgecolor="none",
            markersize=6,
            label=_product_label(product_id),
        )
        for product_id in ["bivalent", "quadrivalent", "nonavalent"]
    ]
    handles.extend(
        [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="#444444",
                markeredgecolor="none",
                markersize=6,
                label="All trials",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="*",
                linestyle="",
                markerfacecolor="#444444",
                markeredgecolor="#202020",
                markersize=9,
                label="Pareto front",
            ),
        ]
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=8.5,
    )
    apply_nature_style(fig, axes_flat)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "figure_s1.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _build_figs2(
    output_dir: Path, search_results: list[tuple[int, Path, SearchResult]]
) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(8.6, 8.6), sharex=False)
    axes_flat = axes.flatten()
    right_axes = []
    for index, ((years, _, result), axis) in enumerate(zip(search_results, axes_flat)):
        completed_trials = [
            trial for trial in result.study.trials if trial.state.name == "COMPLETE"
        ]
        trial_numbers = [trial.number for trial in completed_trials]
        icur_values = [result._trial_icur(trial) for trial in completed_trials]
        incidence_values = [
            result._trial_incidence(trial) * 1e5 for trial in completed_trials
        ]
        right_axis = axis.twinx()
        right_axes.append(right_axis)
        axis.plot(
            trial_numbers,
            icur_values,
            color="#2C7FB8",
            linewidth=1.1,
            alpha=0.95,
        )
        right_axis.plot(
            trial_numbers,
            incidence_values,
            color="#D1495B",
            linewidth=1.1,
            alpha=0.90,
        )
        axis.set_title(f"{years}-year horizon", fontsize=10, pad=4)
        axis.text(
            0.02,
            0.98,
            _subplot_panel_label(index),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )
        axis.tick_params(axis="y", colors="#2C7FB8")
        right_axis.tick_params(axis="y", colors="#D1495B")
        apply_scientific_format(axis, x=False, y=True)
        apply_scientific_format(right_axis, x=False, y=True)
        apply_nature_style(fig, axis)
        right_axis.spines["top"].set_visible(False)
        right_axis.spines["left"].set_visible(False)
        right_axis.spines["right"].set_linewidth(1.0)
        right_axis.grid(False)
        if index % 2 == 0:
            right_axis.tick_params(labelright=False)
    for axis in axes_flat[:4]:
        axis.tick_params(labelbottom=False)
    for axis in axes_flat[::2]:
        axis.set_ylabel("ICUR", color="#2C7FB8", fontsize=9)
    for right_axis in right_axes[1::2]:
        right_axis.set_ylabel(
            "Final incidence (/100,000 women)", color="#D1495B", fontsize=9
        )
    fig.supxlabel("Trial", fontsize=11)
    handles = [
        plt.Line2D([0], [0], color="#2C7FB8", linewidth=1.6, label="ICUR"),
        plt.Line2D(
            [0],
            [0],
            color="#D1495B",
            linewidth=1.6,
            label="Final cervical cancer incidence (/100,000 women)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "figure_s2.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _build_figs3(
    output_dir: Path,
    scenarios: list[tuple[int, Path]],
    rows: list[dict[str, object]],
) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(13.2, 10.6), sharex=False, sharey=False)
    axes_flat = axes.flatten()
    lower_color = "#2C7FB8"
    upper_color = "#D1495B"

    for index, ((years, _), axis) in enumerate(zip(scenarios, axes_flat)):
        scenario_rows = [row for row in rows if row["scenario_years"] == years]
        scenario_rows = sorted(
            scenario_rows,
            key=lambda row: float(row["absolute_change_rmb"]),
            reverse=True,
        )
        y_positions = list(range(len(scenario_rows)))
        axis.barh(
            [y - 0.16 for y in y_positions],
            [row["lower_delta_rmb"] for row in scenario_rows],
            height=0.28,
            color=lower_color,
            alpha=0.85,
        )
        axis.barh(
            [y + 0.16 for y in y_positions],
            [row["upper_delta_rmb"] for row in scenario_rows],
            height=0.28,
            color=upper_color,
            alpha=0.85,
        )
        axis.axvline(0, color="#202020", linewidth=0.9, linestyle="--", alpha=0.9)
        axis.set_yticks(y_positions)
        axis.set_yticklabels(
            [row["plot_label"] for row in scenario_rows],
            fontsize=8.4,
            linespacing=1.05,
        )
        axis.tick_params(axis="y", pad=3)
        axis.invert_yaxis()
        axis.set_title(f"{years}-year horizon", fontsize=10, pad=4)
        axis.text(
            0.02,
            0.98,
            _subplot_panel_label(index),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )
        apply_scientific_format(axis, x=True, y=False)

    fig.supxlabel("ΔICUR (RMB / DALY)", fontsize=11)
    handles = [
        plt.Line2D([0], [0], color=lower_color, linewidth=6, label="Lower limit"),
        plt.Line2D([0], [0], color=upper_color, linewidth=6, label="Upper limit"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=8.5,
    )
    apply_nature_style(fig, axes_flat)
    fig.tight_layout(rect=(0.09, 0, 1, 0.96))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "figure_s3.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _build_figs4(
    output_dir: Path,
    budget_sheets: list[tuple[int, pd.DataFrame]],
) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(12.6, 9.6), sharex=False, sharey=False)
    axes_flat = axes.flatten()
    treatment_color = "#2C7FB8"
    total_color = "#D1495B"

    for index, ((years, dataframe), axis) in enumerate(zip(budget_sheets, axes_flat)):
        axis.plot(
            dataframe["Year"],
            dataframe["Treatment fund - No vaccination"],
            color=treatment_color,
            linestyle="--",
            linewidth=1.8,
        )
        axis.plot(
            dataframe["Year"],
            dataframe["Treatment fund - Optimal strategy"],
            color=treatment_color,
            linestyle="-",
            linewidth=1.8,
        )
        axis.plot(
            dataframe["Year"],
            dataframe["Total fund - No vaccination"],
            color=total_color,
            linestyle="--",
            linewidth=1.8,
        )
        axis.plot(
            dataframe["Year"],
            dataframe["Total fund - Optimal strategy"],
            color=total_color,
            linestyle="-",
            linewidth=1.8,
        )
        axis.set_title(f"{years}-year horizon", fontsize=10, pad=4)
        axis.text(
            0.02,
            0.98,
            _subplot_panel_label(index),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )
        if index >= 4:
            axis.set_xlabel("Calendar year", fontsize=9)
        if index % 2 == 0:
            axis.set_ylabel(f"Expenditure ({BUDGET_UNIT_LABEL})", fontsize=9)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=treatment_color,
            linestyle="--",
            linewidth=1.8,
            label="Treatment fund - No vaccination",
        ),
        plt.Line2D(
            [0],
            [0],
            color=treatment_color,
            linestyle="-",
            linewidth=1.8,
            label="Treatment fund - Optimal strategy",
        ),
        plt.Line2D(
            [0],
            [0],
            color=total_color,
            linestyle="--",
            linewidth=1.8,
            label="Total fund - No vaccination",
        ),
        plt.Line2D(
            [0],
            [0],
            color=total_color,
            linestyle="-",
            linewidth=1.8,
            label="Total fund - Optimal strategy",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=8.5,
    )
    apply_nature_style(fig, axes_flat)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "figure_s4.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _run_tab1(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    dataframe = _build_tab1_dataframe(results_root, args.results_glob)
    excel_path, csv_path, markdown_path = _write_table_outputs(
        dataframe, output_dir, "table1"
    )
    print(f"Wrote {excel_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")


def _run_tabs1(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    model_config = _load_model_config(Path(args.model_config))
    dataframe = _build_tabs1_dataframe(model_config)
    excel_path, csv_path, markdown_path = _write_table_outputs(
        dataframe, output_dir, "table_s1"
    )
    print(f"Wrote {excel_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")


def _run_figs1(args: argparse.Namespace) -> None:
    search_results = _load_search_results(Path(args.results_root), args.results_glob)
    output_path = _build_figs1(Path(args.output_dir), search_results)
    print(f"Wrote {output_path}")


def _run_figs2(args: argparse.Namespace) -> None:
    search_results = _load_search_results(Path(args.results_root), args.results_glob)
    output_path = _build_figs2(Path(args.output_dir), search_results)
    print(f"Wrote {output_path}")


def _run_sensitivity(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    scenarios, rows = _compute_sensitivity_payloads(
        Path(args.results_root),
        args.results_glob,
    )
    payload_path = _sensitivity_payload_path(None, output_dir)
    json_path, csv_path = _write_sensitivity_outputs(
        scenarios,
        rows,
        output_dir,
        payload_path=payload_path,
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def _run_tabs3(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    payload_path = _sensitivity_payload_path(args.sensitivity_path, output_dir)
    scenarios, rows = _load_sensitivity_payload(payload_path)
    dataframe = _build_tabs3_dataframe(scenarios, rows)
    excel_path, csv_path, markdown_path = _write_table_outputs(
        dataframe, output_dir, "table_s3"
    )
    print(f"Wrote {excel_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")


def _run_figs3(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    payload_path = _sensitivity_payload_path(args.sensitivity_path, output_dir)
    scenarios, rows = _load_sensitivity_payload(payload_path)
    output_path = _build_figs3(
        output_dir,
        scenarios,
        rows,
    )
    print(f"Wrote {output_path}")


def _run_budget(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    search_dirs = _discover_search_dirs(Path(args.results_root), args.results_glob)
    sheet_payloads = [
        _build_budget_sheet_dataframe(search_dir) for search_dir in search_dirs
    ]
    workbook_path = _write_budget_workbook(output_dir, sheet_payloads)
    print(f"Wrote {workbook_path}")


def _run_figs4(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    budget_sheets = _load_budget_sheets(Path(args.budget_path))
    output_path = _build_figs4(output_dir, budget_sheets)
    print(f"Wrote {output_path}")


def main() -> None:
    args = _parse_args()
    if args.command == "tab1":
        _run_tab1(args)
        return
    if args.command == "tabs1":
        _run_tabs1(args)
        return
    if args.command == "figs1":
        _run_figs1(args)
        return
    if args.command == "figs2":
        _run_figs2(args)
        return
    if args.command == "sensitivity":
        _run_sensitivity(args)
        return
    if args.command == "tabs3":
        _run_tabs3(args)
        return
    if args.command == "figs3":
        _run_figs3(args)
        return
    if args.command == "budget":
        _run_budget(args)
        return
    if args.command == "figs4":
        _run_figs4(args)
        return
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
