from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import load_workbook

from hpv_tdm import (
    AggregateModelConfig,
    EvaluationResult,
    SearchResult,
    SubtypeGroupedModelConfig,
)
from hpv_tdm.result._plot import (
    PRODUCT_COLORS,
    apply_nature_style,
    apply_scientific_format,
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
    y_values: list[float] = []
    for _, _, result in search_results:
        completed_trials = [
            trial for trial in result.study.trials if trial.state.name == "COMPLETE"
        ]
        x_values.extend(
            result._trial_incidence(trial) * 1e5 for trial in completed_trials
        )
        y_values.extend(result._trial_icur(trial) for trial in completed_trials)
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_pad = (x_max - x_min) * 0.05 if x_max > x_min else max(x_max * 0.05, 0.1)
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else max(y_max * 0.05, 1.0)

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
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
