import pytest
import math
import time
import random

import genetic_algorithm as ga


@pytest.fixture(autouse=True)
def _reset_ga_state():
    """Reset all GA global state before each test."""
    ga.clear_asymmetric_costs()
    ga.clear_car_autonomy()
    ga.clear_delivery_priorities()
    ga.clear_vehicle_params()
    ga.clear_city_demands()


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_cities(n, spacing=100):
    """Generate n cities in a straight line from (0,0)."""
    return [(i * spacing, 0) for i in range(n)]


def _uniform_demands(cities, weight=10.0, volume=0.5):
    return [{"weight": weight, "volume": volume} for _ in cities]


# ===========================================================================
# 4.1 - Unit tests for split_routes
# ===========================================================================

class TestSplitRoutesWeight:
    """Verify split_routes breaks correctly when weight capacity is exceeded."""

    def test_balances_across_vehicles_when_all_fit(self):
        cities = _make_cities(5)
        depot = cities[0]
        demands = _uniform_demands(cities, weight=10.0, volume=0.1)

        ga.set_city_demands(cities, demands)
        ga.set_vehicle_params(num_vehicles=3, capacity_weight=100.0, depot=depot)

        routes = ga.split_routes(cities)
        non_depot = [c for c in cities if c != depot]
        assert len(routes) == 3
        all_cities = [c for r in routes for c in r]
        assert set(all_cities) == set(non_depot)

    def test_single_vehicle_single_route_when_all_fit(self):
        cities = _make_cities(5)
        depot = cities[0]
        demands = _uniform_demands(cities, weight=10.0, volume=0.1)

        ga.set_city_demands(cities, demands)
        ga.set_vehicle_params(num_vehicles=1, capacity_weight=100.0, depot=depot)

        routes = ga.split_routes(cities)
        non_depot = [c for c in cities if c != depot]
        assert len(routes) == 1
        assert routes[0] == non_depot

    def test_splits_when_weight_exceeded(self):
        cities = _make_cities(6)
        depot = cities[0]
        demands = _uniform_demands(cities, weight=30.0, volume=0.01)

        ga.set_city_demands(cities, demands)
        ga.set_vehicle_params(num_vehicles=5, capacity_weight=50.0, depot=depot)

        routes = ga.split_routes(cities)
        assert len(routes) > 1
        all_cities = [c for route in routes for c in route]
        expected = [c for c in cities if c != depot]
        assert set(all_cities) == set(expected)

    def test_each_subroute_respects_weight(self):
        cities = _make_cities(10)
        depot = cities[0]
        demands = _uniform_demands(cities, weight=15.0, volume=0.01)

        ga.set_city_demands(cities, demands)
        ga.set_vehicle_params(num_vehicles=10, capacity_weight=40.0, depot=depot)

        routes = ga.split_routes(cities)
        for route in routes:
            total = sum(
                (ga.get_city_demand(c) or {}).get("weight", 0) for c in route
            )
            assert total <= 40.0 + 1e-9


class TestSplitRoutesVolume:
    """Verify split_routes breaks correctly when volume capacity is exceeded."""

    def test_splits_when_volume_exceeded(self):
        cities = _make_cities(6)
        depot = cities[0]
        demands = _uniform_demands(cities, weight=1.0, volume=3.0)

        ga.set_city_demands(cities, demands)
        ga.set_vehicle_params(num_vehicles=5, capacity_volume=5.0, depot=depot)

        routes = ga.split_routes(cities)
        assert len(routes) > 1

    def test_each_subroute_respects_volume(self):
        cities = _make_cities(10)
        depot = cities[0]
        demands = _uniform_demands(cities, weight=0.1, volume=2.5)

        ga.set_city_demands(cities, demands)
        ga.set_vehicle_params(num_vehicles=10, capacity_volume=5.0, depot=depot)

        routes = ga.split_routes(cities)
        for route in routes:
            total = sum(
                (ga.get_city_demand(c) or {}).get("volume", 0) for c in route
            )
            assert total <= 5.0 + 1e-9


class TestSplitRoutesCombined:
    """Volume AND weight constraints together."""

    def test_whichever_limit_hits_first(self):
        cities = _make_cities(6)
        depot = cities[0]
        demands = [{"weight": 25.0, "volume": 4.0} for _ in cities]

        ga.set_city_demands(cities, demands)
        ga.set_vehicle_params(
            num_vehicles=10,
            capacity_weight=100.0,
            capacity_volume=5.0,
            depot=depot,
        )

        routes = ga.split_routes(cities)
        assert len(routes) > 1
        for route in routes:
            tw = sum((ga.get_city_demand(c) or {}).get("weight", 0) for c in route)
            tv = sum((ga.get_city_demand(c) or {}).get("volume", 0) for c in route)
            assert tw <= 100.0 + 1e-9
            assert tv <= 5.0 + 1e-9


class TestSplitRoutesEdgeCases:
    """Edge cases for the split procedure."""

    def test_empty_individual(self):
        ga.set_vehicle_params(num_vehicles=2, capacity_weight=50.0)
        routes = ga.split_routes([])
        assert routes == [[]]

    def test_single_city(self):
        city = (100, 100)
        ga.set_city_demands([city], [{"weight": 10.0, "volume": 1.0}])
        ga.set_vehicle_params(num_vehicles=1, capacity_weight=50.0, depot=city)
        routes = ga.split_routes([city])
        assert routes == [[]]

    def test_no_capacity_balances_across_vehicles(self):
        cities = _make_cities(5)
        ga.set_vehicle_params(num_vehicles=3, depot=cities[0])
        routes = ga.split_routes(cities)
        non_depot = [c for c in cities if c != cities[0]]
        assert len(routes) == 3
        all_cities = [c for r in routes for c in r]
        assert set(all_cities) == set(non_depot)

    def test_more_vehicles_than_cities_caps_at_city_count(self):
        cities = _make_cities(4)
        depot = cities[0]
        ga.set_city_demands(cities, _uniform_demands(cities, weight=1.0, volume=0.01))
        ga.set_vehicle_params(num_vehicles=10, capacity_weight=100.0, depot=depot)
        routes = ga.split_routes(cities)
        non_depot = [c for c in cities if c != depot]
        assert len(routes) == len(non_depot)
        for route in routes:
            assert len(route) == 1

    def test_balanced_routes_are_roughly_equal_size(self):
        cities = _make_cities(13)
        depot = cities[0]
        ga.set_city_demands(cities, _uniform_demands(cities, weight=1.0, volume=0.01))
        ga.set_vehicle_params(num_vehicles=4, capacity_weight=1000.0, depot=depot)
        routes = ga.split_routes(cities)
        assert len(routes) == 4
        sizes = [len(r) for r in routes]
        assert max(sizes) - min(sizes) <= 1

    def test_vrp_inactive_returns_full_list(self):
        cities = _make_cities(5)
        routes = ga.split_routes(cities)
        assert len(routes) == 1
        assert routes[0] == list(cities)


# ===========================================================================
# 4.1 - Fitness with VRP: vehicle count penalty
# ===========================================================================

class TestVRPFitnessPenalty:
    """Verify massive penalty when routes exceed NUM_VEHICLES."""

    def test_penalty_when_too_many_routes(self):
        cities = _make_cities(10)
        depot = cities[0]
        demands = _uniform_demands(cities, weight=20.0, volume=0.01)

        ga.set_city_demands(cities, demands)
        ga.set_vehicle_params(
            num_vehicles=1, capacity_weight=30.0, depot=depot,
        )

        fitness = ga.calculate_fitness(cities)
        assert fitness >= ga._INVALID_ROUTE_PENALTY

    def test_no_penalty_when_vehicles_sufficient(self):
        cities = _make_cities(6)
        depot = cities[0]
        demands = _uniform_demands(cities, weight=20.0, volume=0.01)

        ga.set_city_demands(cities, demands)
        ga.set_vehicle_params(
            num_vehicles=10, capacity_weight=50.0, depot=depot,
        )

        fitness = ga.calculate_fitness(cities)
        assert fitness < ga._INVALID_ROUTE_PENALTY


# ===========================================================================
# 4.1 - Autonomy inside VRP subroutes
# ===========================================================================

class TestVRPAutonomy:
    """Autonomy constraints should still apply within VRP subroutes."""

    def test_autonomy_invalid_inside_subroute(self):
        cities = [(0, 0), (1000, 0), (2000, 0)]
        depot = cities[0]
        demands = _uniform_demands(cities, weight=1.0, volume=0.01)

        ga.set_city_demands(cities, demands)
        ga.set_vehicle_params(num_vehicles=5, capacity_weight=500.0, depot=depot)
        ga.set_car_autonomy(50.0, reference_city=depot)

        fitness = ga.calculate_fitness(cities)
        assert fitness >= ga._INVALID_ROUTE_PENALTY

    def test_autonomy_valid_inside_subroute(self):
        cities = [(0, 0), (10, 0), (20, 0)]
        depot = cities[0]
        demands = _uniform_demands(cities, weight=1.0, volume=0.01)

        ga.set_city_demands(cities, demands)
        ga.set_vehicle_params(num_vehicles=5, capacity_weight=500.0, depot=depot)
        ga.set_car_autonomy(500.0, reference_city=depot)

        fitness = ga.calculate_fitness(cities)
        assert fitness < ga._INVALID_ROUTE_PENALTY


# ===========================================================================
# 4.1 - Impossible demand (city demand > vehicle capacity)
# ===========================================================================

class TestImpossibleDemand:
    """System should handle gracefully when a single city's demand > vehicle cap."""

    def test_city_demand_exceeds_weight_capacity(self):
        cities = [(0, 0), (100, 0)]
        depot = cities[0]
        demands = [{"weight": 1.0, "volume": 0.01}, {"weight": 999.0, "volume": 0.01}]

        ga.set_city_demands(cities, demands)
        ga.set_vehicle_params(num_vehicles=5, capacity_weight=50.0, depot=depot)

        routes = ga.split_routes(cities)
        all_cities = [c for r in routes for c in r]
        assert (100, 0) in all_cities

    def test_city_demand_exceeds_volume_capacity(self):
        cities = [(0, 0), (100, 0)]
        depot = cities[0]
        demands = [{"weight": 1.0, "volume": 0.01}, {"weight": 1.0, "volume": 999.0}]

        ga.set_city_demands(cities, demands)
        ga.set_vehicle_params(num_vehicles=5, capacity_volume=5.0, depot=depot)

        routes = ga.split_routes(cities)
        all_cities = [c for r in routes for c in r]
        assert (100, 0) in all_cities


# ===========================================================================
# 4.1 - evaluate_vrp_routes structure
# ===========================================================================

class TestEvaluateVRPRoutes:
    """evaluate_vrp_routes should return depot-bracketed routes."""

    def test_routes_start_and_end_at_depot(self):
        cities = _make_cities(6)
        depot = cities[0]
        demands = _uniform_demands(cities, weight=30.0, volume=0.01)

        ga.set_city_demands(cities, demands)
        ga.set_vehicle_params(num_vehicles=10, capacity_weight=50.0, depot=depot)

        full_routes = ga.evaluate_vrp_routes(cities)
        for route in full_routes:
            assert route[0] == depot
            assert route[-1] == depot

    def test_all_cities_present_in_routes(self):
        cities = _make_cities(8)
        depot = cities[0]
        demands = _uniform_demands(cities, weight=20.0, volume=0.01)

        ga.set_city_demands(cities, demands)
        ga.set_vehicle_params(num_vehicles=10, capacity_weight=50.0, depot=depot)

        full_routes = ga.evaluate_vrp_routes(cities)
        all_cities = set()
        for route in full_routes:
            for c in route:
                if c != depot:
                    all_cities.add(c)
        expected = set(c for c in cities if c != depot)
        assert all_cities == expected


# ===========================================================================
# 4.2 - Retrocompatibility with pure TSP
# ===========================================================================

class TestTSPRetrocompatibility:
    """When NUM_VEHICLES=1 and no capacity, behavior must be identical to TSP."""

    def test_fitness_matches_euclidean_cycle(self):
        cities = [(0, 0), (3, 0), (3, 4)]
        ga.clear_vehicle_params()
        ga.clear_city_demands()

        fitness = ga.calculate_fitness(cities)
        d01 = math.sqrt(9)
        d12 = math.sqrt(16)
        d20 = math.sqrt(9 + 16)
        assert fitness == pytest.approx(d01 + d12 + d20)

    def test_fitness_with_autonomy_same_as_before(self):
        cities = [(0, 0), (10, 0), (20, 0), (30, 0)]
        ga.set_car_autonomy(500.0, reference_city=cities[0])

        fitness_tsp = ga.calculate_fitness(cities)
        assert fitness_tsp < ga._INVALID_ROUTE_PENALTY
        assert fitness_tsp > 0

    def test_fitness_with_atsp_same_as_before(self):
        cities = [(0, 0), (1, 0), (2, 0)]
        costs = [
            [0.0, 1.0, 5.0],
            [2.0, 0.0, 3.0],
            [4.0, 6.0, 0.0],
        ]
        ga.set_asymmetric_costs(cities, costs)

        fitness = ga.calculate_fitness(cities)
        assert fitness == pytest.approx(1.0 + 3.0 + 4.0)

    def test_vrp_single_vehicle_no_cap_same_as_tsp(self):
        """NUM_VEHICLES=1 without capacity should NOT activate VRP split."""
        cities = [(0, 0), (3, 0), (3, 4)]
        ga.set_vehicle_params(num_vehicles=1)

        assert not ga._is_vrp_active()
        fitness = ga.calculate_fitness(cities)
        d01 = 3.0
        d12 = 4.0
        d20 = 5.0
        assert fitness == pytest.approx(d01 + d12 + d20)

    def test_crossover_preserves_permutation(self):
        cities = _make_cities(8)
        random.seed(42)
        parent1 = random.sample(cities, len(cities))
        parent2 = random.sample(cities, len(cities))
        child = ga.order_crossover(parent1, parent2)
        assert len(child) == len(cities)
        assert set(child) == set(cities)

    def test_mutation_preserves_permutation(self):
        cities = _make_cities(8)
        mutated = ga.mutate(cities, mutation_probability=1.0, just_swap=False)
        assert len(mutated) == len(cities)
        assert set(mutated) == set(cities)
