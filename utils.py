from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple, Union
import os

TimeLike = Union[datetime, int, float, str]


@dataclass
class ReportData:
    num_cities: int
    start_time: TimeLike
    end_time: TimeLike
    best_solution: List[Tuple[float, float]]
    best_generation: int
    best_fitness: float
    mutation_probability: float
    random_population_percent: float
    initial_population_method: str
    output_path: Optional[str] = None


_REPORT_DATA: Optional[ReportData] = None


def set_report_data(report_data: ReportData) -> None:
    """Store report data to be used by generate_report()."""
    global _REPORT_DATA
    _REPORT_DATA = report_data


def _to_datetime(value: TimeLike) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _format_time(value: TimeLike) -> str:
    dt_value = _to_datetime(value)
    if dt_value is not None:
        return dt_value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def generate_report() -> str:
    """
    Generate an execution report.

    Required fields:
    - num_cities
    - start_time
    - end_time
    - best_solution
    - best_generation
    - initial_population_method
    """
    if _REPORT_DATA is None:
        raise ValueError("Report data not set. Call set_report_data(...) before generate_report().")

    start_dt = _to_datetime(_REPORT_DATA.start_time)
    end_dt = _to_datetime(_REPORT_DATA.end_time)
    total_seconds = None
    if start_dt is not None and end_dt is not None:
        total_seconds = (end_dt - start_dt).total_seconds()

    total_str = f"{total_seconds:.2f}" if total_seconds is not None else "N/A"

    report_lines = [
        "Relatorio da execucao",
        f"Numero de cidades: {_REPORT_DATA.num_cities}",
        f"Inicio: {_format_time(_REPORT_DATA.start_time)}",
        f"Fim: {_format_time(_REPORT_DATA.end_time)}",
        f"Tempo total (s): {total_str}",
        f"Melhor fitness: {_REPORT_DATA.best_fitness:.2f}",
        f"Melhor solucao: {_REPORT_DATA.best_solution}",
        f"Geracao da melhor solucao: {_REPORT_DATA.best_generation}",
        f"Probabilidade de mutacao: {_REPORT_DATA.mutation_probability:.2f}",
        f"Percentual populacao random: {_REPORT_DATA.random_population_percent:.2f}",
        f"Populacao inicial: {_REPORT_DATA.initial_population_method}",
    ]

    report_text = "\n".join(report_lines)

    if _REPORT_DATA.output_path:
        output_dir = os.path.dirname(_REPORT_DATA.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(_REPORT_DATA.output_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(report_text)

    return report_text
