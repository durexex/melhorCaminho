# -*- coding: utf-8 -*-
"""
VRP Benchmark: compares GA execution time and solution quality
for 1, 3, and 5 vehicles on the ATT-48 dataset.

Usage:
    python benchmark_vrp.py
"""

import random
import time
from typing import List, Tuple

import genetic_algorithm as ga
from benchmark_att48 import att_48_cities_locations

N_GENERATIONS = 500
POPULATION_SIZE = 100
MUTATION_PROBABILITY = 0.5
VEHICLE_CONFIGS = [1, 3, 5]
CAPACITY_WEIGHT = 200.0
CAPACITY_VOLUME = 8.0
DEMAND_WEIGHT_RANGE = (5.0, 30.0)
DEMAND_VOLUME_RANGE = (0.1, 1.5)
SEED = 42


def _generate_demands(
    cities: List[Tuple[float, float]],
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    demands = []
    for _ in cities:
        w = round(rng.uniform(*DEMAND_WEIGHT_RANGE), 2)
        v = round(rng.uniform(*DEMAND_VOLUME_RANGE), 4)
        demands.append({"weight": w, "volume": v})
    return demands


def _run_ga(
    cities: List[Tuple[float, float]],
    num_vehicles: int,
    demands: List[dict],
) -> dict:
    """Run the GA and return timing / quality metrics."""
    ga.clear_asymmetric_costs()
    ga.clear_car_autonomy()
    ga.clear_delivery_priorities()
    ga.clear_vehicle_params()
    ga.clear_city_demands()

    depot = cities[0]

    if num_vehicles == 1:
        ga.set_vehicle_params(num_vehicles=1)
    else:
        ga.set_city_demands(cities, demands)
        ga.set_vehicle_params(
            num_vehicles=num_vehicles,
            capacity_weight=CAPACITY_WEIGHT,
            capacity_volume=CAPACITY_VOLUME,
            depot=depot,
        )

    random.seed(SEED)
    population = ga.generate_random_population(cities, POPULATION_SIZE)

    best_fitness_ever = None
    best_solution_ever = None
    best_gen = 0

    t0 = time.perf_counter()

    for gen in range(1, N_GENERATIONS + 1):
        fitnesses = [ga.calculate_fitness(ind) for ind in population]
        population, fitnesses = ga.sort_population(population, fitnesses)

        if best_fitness_ever is None or fitnesses[0] < best_fitness_ever:
            best_fitness_ever = fitnesses[0]
            best_solution_ever = population[0]
            best_gen = gen

        new_pop = [population[0]]
        while len(new_pop) < POPULATION_SIZE:
            p1, p2 = random.choices(population[:20], k=2)
            child = ga.order_crossover(p1, p2)
            child = ga.mutate(child, MUTATION_PROBABILITY, False)
            new_pop.append(child)

        population = new_pop

    elapsed = time.perf_counter() - t0

    num_routes = 1
    if num_vehicles > 1 and best_solution_ever:
        num_routes = len(ga.split_routes(best_solution_ever))

    return {
        "num_vehicles": num_vehicles,
        "best_fitness": best_fitness_ever,
        "best_generation": best_gen,
        "elapsed_s": elapsed,
        "num_routes": num_routes,
    }


def main():
    cities = att_48_cities_locations
    demands = _generate_demands(cities, SEED)

    print("=" * 70)
    print("VRP Benchmark — ATT-48 dataset")
    print(f"Generations: {N_GENERATIONS}  |  Population: {POPULATION_SIZE}")
    print(f"Capacity: {CAPACITY_WEIGHT} kg / {CAPACITY_VOLUME} m3")
    print(f"Cities: {len(cities)}")
    print("=" * 70)

    results = []
    for nv in VEHICLE_CONFIGS:
        label = "TSP (baseline)" if nv == 1 else f"VRP {nv} vehicles"
        print(f"\nRunning {label} ...")
        result = _run_ga(cities, nv, demands)
        results.append(result)
        print(
            f"  Fitness: {result['best_fitness']:.2f}  |  "
            f"Gen: {result['best_generation']}  |  "
            f"Time: {result['elapsed_s']:.3f}s  |  "
            f"Routes: {result['num_routes']}"
        )

    print("\n" + "=" * 70)
    print(f"{'Config':<20} {'Fitness':>12} {'Time (s)':>10} {'Routes':>8} {'Gen':>6}")
    print("-" * 70)
    baseline_time = results[0]["elapsed_s"]
    for r in results:
        label = "TSP" if r["num_vehicles"] == 1 else f"VRP-{r['num_vehicles']}v"
        ratio = r["elapsed_s"] / baseline_time if baseline_time > 0 else 0
        print(
            f"{label:<20} {r['best_fitness']:>12.2f} "
            f"{r['elapsed_s']:>9.3f}s {r['num_routes']:>8} {r['best_generation']:>6}"
        )
    print("=" * 70)

    baseline_time = results[0]["elapsed_s"]
    print("\nPerformance ratios (relative to TSP baseline):")
    for r in results:
        label = "TSP" if r["num_vehicles"] == 1 else f"VRP-{r['num_vehicles']}v"
        ratio = r["elapsed_s"] / baseline_time if baseline_time > 0 else 0
        print(f"  {label}: {ratio:.2f}x")

    max_time = max(r["elapsed_s"] for r in results)
    if max_time < 120:
        print(f"\nAll configs completed within 2 minutes (max: {max_time:.1f}s). PASS")
    else:
        print(f"\nWARNING: slowest config took {max_time:.1f}s (> 120s threshold)")


if __name__ == "__main__":
    main()
