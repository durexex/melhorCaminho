# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


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


def draw_paths(ax, path: List[Tuple[int, int]], rgb_color: Tuple[int, int, int], width: int = 1) -> None:
    if not path:
        return
    closed = _close_path(list(path))
    xs, ys = zip(*closed)
    ax.plot(xs, ys, color=_normalize_color(rgb_color), linewidth=width, zorder=2)


def build_solution_figure(
    cities_locations: List[Tuple[int, int]],
    best_path: List[Tuple[int, int]],
    candidate_path: Optional[List[Tuple[int, int]]] = None,
    node_radius: int = 10,
    city_color: Tuple[int, int, int] = (255, 0, 0),
    best_color: Tuple[int, int, int] = (0, 0, 255),
    candidate_color: Tuple[int, int, int] = (128, 128, 128),
    reference_city: Optional[Tuple[int, int]] = None,
    reference_color: Tuple[int, int, int] = (0, 255, 0),
    reference_radius: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    x_offset: Optional[int] = None,
    invert_y: bool = True,
):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
    draw_cities(ax, cities_locations, city_color, node_radius)
    if reference_city is not None:
        radius = reference_radius if reference_radius is not None else max(node_radius + 2, 2)
        draw_cities(ax, [reference_city], reference_color, radius)
    if candidate_path:
        draw_paths(ax, candidate_path, candidate_color, width=1)
    if best_path:
        draw_paths(ax, best_path, best_color, width=2)

    if width is not None and height is not None:
        x_min = x_offset if x_offset is not None else 0
        ax.set_xlim(x_min, width)
        ax.set_ylim(0, height)

    if invert_y:
        ax.invert_yaxis()

    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    fig.tight_layout()
    return fig
