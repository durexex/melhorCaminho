import pytest

import genetic_algorithm as ga
from demo_tournament import tournament_selection


def setup_function():
    ga.clear_asymmetric_costs()
    ga.clear_car_autonomy()
    ga.clear_delivery_priorities()
    ga.clear_vehicle_params()
    ga.clear_city_demands()


def test_calculate_distance():
    assert ga.calculate_distance((0, 0), (3, 4)) == 5


def test_fitness_without_priorities():
    cities = [(0, 0), (3, 0), (3, 4)]
    fitness = ga.calculate_fitness(cities)
    assert fitness == pytest.approx(12.0)


def test_priority_penalty_applied():
    cities = [(0, 0), (3, 0)]
    rules = {
        "critical": {
            "weight_multiplier": 2.0,
            "penalty_per_km": 1.5,
        },
        "regular": {
            "weight_multiplier": 1.0,
            "penalty_per_km": 1.0,
        },
    }
    ga.set_delivery_priorities(cities, ["regular", "critical"], rules, default_priority_id="regular")
    fitness = ga.calculate_fitness(cities)
    assert fitness == pytest.approx(12.0)


def test_priority_fallback_for_unknown_id():
    cities = [(0, 0), (3, 0)]
    rules = {
        "critical": {
            "weight_multiplier": 2.0,
            "penalty_per_km": 1.5,
        },
        "regular": {
            "weight_multiplier": 1.0,
            "penalty_per_km": 1.0,
        },
    }
    ga.set_delivery_priorities(cities, ["unknown", "critical"], rules, default_priority_id="regular")
    fitness = ga.calculate_fitness(cities)
    assert fitness == pytest.approx(12.0)


def test_emergency_priority_is_served_earlier():
    cities = [(0, 0), (10, 0), (1, 0)]
    rules = {
        "emergency": {
            "weight_multiplier": 3.0,
            "penalty_per_km": 2.0,
            "max_delay_min": 5,
            "latest_position": 1,
            "sequence_penalty": 100.0,
        },
        "regular": {
            "weight_multiplier": 1.0,
            "penalty_per_km": 1.0,
        },
    }
    ga.set_delivery_priorities(
        cities,
        ["regular", "regular", "emergency"],
        rules,
        default_priority_id="regular",
    )

    emergency_first = ga.calculate_fitness([(0, 0), (1, 0), (10, 0)])
    emergency_late = ga.calculate_fitness([(0, 0), (10, 0), (1, 0)])

    assert emergency_first < emergency_late


def test_postpartum_time_window_penalizes_late_arrival():
    cities = [(0, 0), (0, 50), (10, 0)]
    rules = {
        "postpartum": {
            "weight_multiplier": 2.0,
            "penalty_per_km": 1.5,
            "time_window_start_min": 5,
            "time_window_end_min": 15,
            "window_penalty_multiplier": 10.0,
        },
        "regular": {
            "weight_multiplier": 1.0,
            "penalty_per_km": 1.0,
        },
    }
    ga.set_delivery_priorities(
        cities,
        ["regular", "regular", "postpartum"],
        rules,
        default_priority_id="regular",
    )

    inside_window = ga.calculate_fitness([(0, 0), (10, 0), (0, 50)])
    outside_window = ga.calculate_fitness([(0, 0), (0, 50), (10, 0)])

    assert inside_window < outside_window


def test_autonomy_invalid_route_penalized():
    cities = [(0, 0), (100, 0)]
    ga.set_car_autonomy(10, reference_city=cities[0])
    fitness = ga.calculate_fitness(cities)
    assert fitness >= ga._INVALID_ROUTE_PENALTY


def test_atsp_cost_matrix_used():
    cities = [(0, 0), (1, 0), (2, 0)]
    costs = [
        [0.0, 1.0, 5.0],
        [2.0, 0.0, 3.0],
        [4.0, 6.0, 0.0],
    ]
    ga.set_asymmetric_costs(cities, costs)
    fitness = ga.calculate_fitness(cities)
    assert fitness == pytest.approx(8.0)


def test_tournament_selection_minimize():
    population = [
        [(3, 0)],
        [(1, 0)],
        [(4, 0)],
        [(2, 0)],
    ]

    def fitness_fn(route):
        return route[0][0]

    import random

    random.seed(42)
    contenders = random.sample(population, k=2)
    expected = min([(fitness_fn(r), r) for r in contenders], key=lambda item: item[0])

    random.seed(42)
    selected, score = tournament_selection(population, fitness_fn, tournament_size=2, minimize=True)
    assert score == expected[0]
    assert selected == expected[1]


def test_order_crossover_permutation():
    parent1 = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
    parent2 = [(5, 0), (4, 0), (3, 0), (2, 0), (1, 0)]

    import random

    random.seed(7)
    child = ga.order_crossover(parent1, parent2)
    assert len(child) == len(parent1)
    assert set(child) == set(parent1)


def test_mutate_swap_adjacent():
    original = [(0, 0), (1, 0)]
    mutated = ga.mutate(original, mutation_probability=1.0, just_swap=True)
    assert mutated == [(1, 0), (0, 0)]


def test_mutate_inversion_changes_route():
    original = [(0, 0), (1, 0), (2, 0), (3, 0)]
    import random

    random.seed(3)
    mutated = ga.mutate(original, mutation_probability=1.0, just_swap=False)
    assert set(mutated) == set(original)
    assert mutated != original
