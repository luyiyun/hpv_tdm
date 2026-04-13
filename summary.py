from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from hpv_tdm import EvaluationResult


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready summary tables and figures."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    tab1_parser = subparsers.add_parser(
        "tab1",
        help="Summarize the selected optimal vaccination strategies across scenarios.",
    )
    tab1_parser.add_argument(
        "--results-glob",
        default="search-*y-find-params-4-multiobj",
        help="Glob pattern under results/ used to discover scenario directories.",
    )
    tab1_parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory containing search result folders.",
    )
    tab1_parser.add_argument(
        "--output-dir",
        default="summary",
        help="Directory used to store the generated summary files.",
    )

    return parser.parse_args()


def _scenario_years_from_name(directory: Path) -> int:
    stem = directory.name
    if not stem.startswith("search-") or "y-" not in stem:
        raise ValueError(f"unexpected scenario directory name: {directory}")
    years_text = stem.split("search-", maxsplit=1)[1].split("y-", maxsplit=1)[0]
    return int(years_text)


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


def _write_tab1_outputs(dataframe: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / "table1.xlsx"
    csv_path = output_dir / "table1.csv"
    markdown_path = output_dir / "table1.md"

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


def _run_tab1(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    dataframe = _build_tab1_dataframe(results_root, args.results_glob)
    _write_tab1_outputs(dataframe, output_dir)
    print(f"Wrote {output_dir / 'table1.xlsx'}")
    print(f"Wrote {output_dir / 'table1.csv'}")
    print(f"Wrote {output_dir / 'table1.md'}")


def main() -> None:
    args = _parse_args()
    if args.command == "tab1":
        _run_tab1(args)
        return
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
