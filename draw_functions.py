# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D


_ROUTE_PALETTE = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


def _normalize_color(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    return (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)


def _close_path(path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not path:
        return []
    if path[0] == path[-1]:
        return path
    return [*path, path[0]]


def draw_plot(x: list, y: list, x_label: str = "Generation", y_label: str = "Fitness"):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.plot(x, y, color=_normalize_color((0, 0, 255)))
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def draw_cities(ax, cities_locations: List[Tuple[int, int]], rgb_color: Tuple[int, int, int], node_radius: int) -> None:
    if not cities_locations:
        return
    xs, ys = zip(*cities_locations)
    ax.scatter(xs, ys, s=node_radius * node_radius, c=[_normalize_color(rgb_color)], zorder=3)


def _priority_color(index: int) -> Tuple[int, int, int]:
    palette = [
        (216, 27, 96),
        (251, 140, 0),
        (30, 136, 229),
        (67, 160, 71),
        (106, 27, 154),
        (0, 172, 193),
    ]
    if index < len(palette):
        return palette[index]
    cmap = cm.get_cmap("Set2")
    r, g, b, _ = cmap(index % 8)
    return (int(r * 255), int(g * 255), int(b * 255))


def _priority_color_map(priority_order: List[str]) -> Dict[str, Tuple[int, int, int]]:
    return {priority_id: _priority_color(index) for index, priority_id in enumerate(priority_order)}


def build_priority_legend_items(
    priority_order: List[str],
    priority_labels: Optional[Dict[str, str]] = None,
) -> List[Dict[str, object]]:
    color_map = _priority_color_map(priority_order)
    return [
        {
            "id": priority_id,
            "label": (priority_labels or {}).get(priority_id, priority_id),
            "color": color_map[priority_id],
        }
        for priority_id in priority_order
    ]


def _draw_priority_cities(
    ax,
    cities_locations: List[Tuple[int, int]],
    city_priority_ids: Optional[List[str]],
    priority_labels: Optional[Dict[str, str]],
    priority_order: Optional[List[str]],
    node_radius: int,
):
    if not cities_locations:
        return {}

    if not city_priority_ids:
        return {}

    color_priority_order = list(priority_order or [])
    for priority_id in city_priority_ids:
        if priority_id and priority_id not in color_priority_order:
            color_priority_order.append(priority_id)

    color_map = _priority_color_map(color_priority_order)
    grouped: Dict[str, List[Tuple[int, int]]] = {
        priority_id: [] for priority_id in color_priority_order
    }

    for index, city in enumerate(cities_locations):
        priority_id = city_priority_ids[index] if index < len(city_priority_ids) else None
        if not priority_id:
            priority_id = "_sem_prioridade"
        if priority_id not in grouped:
            grouped[priority_id] = []
            color_map[priority_id] = _priority_color(len(color_map))
        grouped[priority_id].append(city)

    for priority_id, group in grouped.items():
        if not group:
            continue
        draw_cities(ax, group, color_map[priority_id], node_radius)

    return {
        priority_id: {
            "color": color_map[priority_id],
            "label": (priority_labels or {}).get(priority_id, "Sem prioridade" if priority_id == "_sem_prioridade" else priority_id),
        }
        for priority_id in grouped.keys()
    }


def _draw_reference_city(
    ax,
    reference_city: Tuple[int, int],
    node_radius: int,
):
    ax.scatter(
        [reference_city[0]],
        [reference_city[1]],
        s=(node_radius + 5) * (node_radius + 5),
        facecolors="none",
        edgecolors="black",
        linewidths=1.8,
        zorder=5,
    )
    ax.scatter(
        [reference_city[0]],
        [reference_city[1]],
        s=(node_radius + 1) * (node_radius + 1),
        marker="x",
        c="black",
        linewidths=2.0,
        zorder=6,
    )


def _build_priority_legend_handles(priority_legend_items: Dict[str, Dict[str, object]]):
    handles = []
    for item in priority_legend_items.values():
        color = _normalize_color(item["color"])
        label = str(item["label"])
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=8,
                label=label,
            )
        )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="x",
            color="black",
            linestyle="None",
            markersize=8,
            markeredgewidth=2,
            label="Cidade inicial",
        )
    )
    return handles


def draw_paths(ax, path: List[Tuple[int, int]], rgb_color: Tuple[int, int, int], width: int = 1, label: Optional[str] = None) -> None:
    if not path:
        return
    closed = _close_path(list(path))
    xs, ys = zip(*closed)
    ax.plot(xs, ys, color=_normalize_color(rgb_color), linewidth=width, zorder=2, label=label)


def _route_color(index: int) -> Tuple[int, int, int]:
    """Return a distinct RGB color for the given route index."""
    if index < len(_ROUTE_PALETTE):
        return _ROUTE_PALETTE[index]
    cmap = cm.get_cmap("tab20")
    r, g, b, _ = cmap(index % 20)
    return (int(r * 255), int(g * 255), int(b * 255))


def build_solution_figure(
    cities_locations: List[Tuple[int, int]],
    best_path: Optional[List[Tuple[int, int]]] = None,
    candidate_path: Optional[List[Tuple[int, int]]] = None,
    routes: Optional[List[List[Tuple[int, int]]]] = None,
    node_radius: int = 10,
    city_color: Tuple[int, int, int] = (255, 0, 0),
    city_priority_ids: Optional[List[str]] = None,
    priority_labels: Optional[Dict[str, str]] = None,
    priority_order: Optional[List[str]] = None,
    best_color: Tuple[int, int, int] = (0, 0, 255),
    candidate_color: Tuple[int, int, int] = (128, 128, 128),
    reference_city: Optional[Tuple[int, int]] = None,
    reference_radius: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    x_offset: Optional[int] = None,
    invert_y: bool = True,
):
    """Build a matplotlib figure for the solution.

    When *routes* is provided (list of sub-routes from VRP), each route is
    drawn with a distinct color and the legend identifies vehicles.
    Falls back to the legacy single-path rendering via *best_path*.
    """
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
    priority_legend_items = _draw_priority_cities(
        ax,
        cities_locations,
        city_priority_ids,
        priority_labels,
        priority_order,
        node_radius,
    )
    if not priority_legend_items:
        draw_cities(ax, cities_locations, city_color, node_radius)
    if reference_city is not None:
        radius = reference_radius if reference_radius is not None else max(node_radius + 2, 2)
        _draw_reference_city(ax, reference_city, radius)

    if routes and len(routes) > 0:
        route_handles = []
        for idx, route in enumerate(routes):
            if not route:
                continue
            color = _route_color(idx)
            draw_paths(ax, route, color, width=2, label=f"Veiculo {idx + 1}")
            route_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=_normalize_color(color),
                    linewidth=2,
                    label=f"Veiculo {idx + 1}",
                )
            )
        if route_handles:
            route_legend = ax.legend(
                handles=route_handles,
                loc="upper left",
                fontsize="small",
                framealpha=0.85,
                title="Rotas",
            )
            ax.add_artist(route_legend)
    else:
        if candidate_path:
            draw_paths(ax, candidate_path, candidate_color, width=1)
        if best_path:
            draw_paths(ax, best_path, best_color, width=2)

    if priority_legend_items:
        priority_legend = ax.legend(
            handles=_build_priority_legend_handles(priority_legend_items),
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize="small",
            framealpha=0.85,
            title="Prioridades",
        )
        ax.add_artist(priority_legend)

    if width is not None and height is not None:
        x_min = x_offset if x_offset is not None else 0
        ax.set_xlim(x_min, width)
        ax.set_ylim(0, height)

    if invert_y:
        ax.invert_yaxis()

    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    return fig
