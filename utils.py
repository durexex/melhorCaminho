from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import os

TimeLike = Union[datetime, int, float, str]


@dataclass
class VehicleStats:
    vehicle_id: int
    cities: List[Tuple[float, float]]
    distance: float
    weight: float
    volume: float
    depot_returns: int


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
    num_vehicles: int = 1
    capacity_weight: Optional[float] = None
    capacity_volume: Optional[float] = None
    vehicle_stats: List[VehicleStats] = field(default_factory=list)


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

    if _REPORT_DATA.num_vehicles > 1 or _REPORT_DATA.vehicle_stats:
        report_lines.append("")
        report_lines.append("--- Frota VRP ---")
        report_lines.append(f"Veiculos configurados: {_REPORT_DATA.num_vehicles}")
        cap_w = _REPORT_DATA.capacity_weight
        cap_v = _REPORT_DATA.capacity_volume
        report_lines.append(f"Capacidade peso: {f'{cap_w:.1f} kg' if cap_w is not None else 'desativada'}")
        report_lines.append(f"Capacidade volume: {f'{cap_v:.2f} m3' if cap_v is not None else 'desativada'}")

        if _REPORT_DATA.vehicle_stats:
            active = [vs for vs in _REPORT_DATA.vehicle_stats if vs.cities]
            idle = _REPORT_DATA.num_vehicles - len(active)
            total_weight = sum(vs.weight for vs in active)
            total_dist = sum(vs.distance for vs in active)

            report_lines.append(f"Veiculos ativos: {len(active)}")
            if idle > 0:
                report_lines.append(f"Veiculos ociosos: {idle}")
            if active:
                avg_weight = total_weight / len(active)
                avg_dist = total_dist / len(active)
                report_lines.append(f"Carga media por veiculo: {avg_weight:.2f} kg")
                report_lines.append(f"Distancia media por veiculo: {avg_dist:.2f}")

            report_lines.append("")
            for vs in _REPORT_DATA.vehicle_stats:
                report_lines.append(f"  Veiculo {vs.vehicle_id}:")
                report_lines.append(f"    Cidades: {len(vs.cities)}")
                report_lines.append(f"    Distancia: {vs.distance:.2f}")
                report_lines.append(f"    Peso: {vs.weight:.2f} kg")
                report_lines.append(f"    Volume: {vs.volume:.4f} m3")
                report_lines.append(f"    Retornos ao deposito: {vs.depot_returns}")

    report_text = "\n".join(report_lines)

    if _REPORT_DATA.output_path:
        output_dir = os.path.dirname(_REPORT_DATA.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(_REPORT_DATA.output_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(report_text)

    return report_text
