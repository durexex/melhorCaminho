# -*- coding: utf-8 -*-

import random
from typing import Callable, List, Tuple

from genetic_algorithm import generate_random_population, calculate_fitness


def tournament_selection(
    population: List[List[Tuple[float, float]]],
    fitness_fn: Callable[[List[Tuple[float, float]]], float],
    tournament_size: int = 3,
    minimize: bool = True,
) -> Tuple[List[Tuple[float, float]], float]:
    if tournament_size < 1:
        raise ValueError("tournament_size must be >= 1")
    if tournament_size > len(population):
        raise ValueError("tournament_size cannot exceed population size")

    contenders = random.sample(population, k=tournament_size)
    scored = [(fitness_fn(individual), individual) for individual in contenders]
    best = min(scored, key=lambda item: item[0]) if minimize else max(scored, key=lambda item: item[0])
    return best[1], best[0]


if __name__ == "__main__":
    random.seed(42)

    n_cities = 8
    population_size = 10
    cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(n_cities)]

    population = generate_random_population(cities, population_size)

    parent1, fitness1 = tournament_selection(population, calculate_fitness, tournament_size=3, minimize=True)
    parent2, fitness2 = tournament_selection(population, calculate_fitness, tournament_size=3, minimize=True)

    print("Selected parent 1 fitness:", round(fitness1, 2))
    print("Selected parent 2 fitness:", round(fitness2, 2))
