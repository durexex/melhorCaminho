import random
import math
import copy 
from typing import List, Tuple, Optional


_ASYMMETRIC_COSTS: Optional[List[List[float]]] = None
_CITY_INDEX: Optional[dict[Tuple[float, float], int]] = None
_CAR_AUTONOMY: Optional[float] = None
_REFERENCE_CITY: Optional[Tuple[float, float]] = None
_INVALID_ROUTE_PENALTY = 1e12
_CITY_PRIORITY: Optional[dict[Tuple[float, float], str]] = None
_PRIORITY_RULES: Optional[dict[str, dict]] = None
_DEFAULT_PRIORITY_ID: Optional[str] = None
_PRIORITY_REFERENCE_CITY: Optional[Tuple[float, float]] = None

_NUM_VEHICLES: int = 1
_VEHICLE_CAPACITY_WEIGHT: Optional[float] = None
_VEHICLE_CAPACITY_VOLUME: Optional[float] = None
_CITY_DEMANDS: Optional[dict[Tuple[float, float], dict]] = None
_VRP_DEPOT: Optional[Tuple[float, float]] = None


def set_asymmetric_costs(cities_location: List[Tuple[float, float]], costs: List[List[float]]) -> None:
    """
    Configure asymmetric costs for ATSP.

    Parameters:
    - cities_location (List[Tuple[float, float]]): List of city coordinates.
    - costs (List[List[float]]): Square matrix where costs[i][j] is the cost from city i to j.
    """
    n = len(cities_location)
    if n != len(costs) or any(len(row) != n for row in costs):
        raise ValueError("Asymmetric cost matrix must be square and match number of cities.")

    global _ASYMMETRIC_COSTS, _CITY_INDEX
    _ASYMMETRIC_COSTS = costs
    _CITY_INDEX = {city: i for i, city in enumerate(cities_location)}


def clear_asymmetric_costs() -> None:
    """Disable asymmetric costs and revert to Euclidean distance."""
    global _ASYMMETRIC_COSTS, _CITY_INDEX
    _ASYMMETRIC_COSTS = None
    _CITY_INDEX = None


def set_car_autonomy(autonomy: Optional[float], reference_city: Optional[Tuple[float, float]] = None) -> None:
    """
    Configure car autonomy behavior.

    Parameters:
    - autonomy (Optional[float]): Max distance the car can travel before refueling.
      Use None to disable autonomy constraints.
    - reference_city (Optional[Tuple[float, float]]): Fixed reference city to refuel.
      If None, the first city of each route is used as reference.
    """
    global _CAR_AUTONOMY, _REFERENCE_CITY
    _CAR_AUTONOMY = autonomy
    _REFERENCE_CITY = reference_city


def clear_car_autonomy() -> None:
    """Disable autonomy constraints."""
    global _CAR_AUTONOMY, _REFERENCE_CITY
    _CAR_AUTONOMY = None
    _REFERENCE_CITY = None


def set_delivery_priorities(
    cities_location: List[Tuple[float, float]],
    city_priorities,
    priority_rules: dict,
    default_priority_id: Optional[str] = None,
) -> None:
    """
    Configure delivery priorities per city.

    Parameters:
    - cities_location (List[Tuple[float, float]]): List of city coordinates.
    - city_priorities: list aligned with cities_location or dict (index->priority_id or city->priority_id).
    - priority_rules (dict): Rules per priority id (weight_multiplier, penalty_per_km, etc).
    - default_priority_id (Optional[str]): Fallback priority id.
    """
    if not cities_location or not priority_rules:
        clear_delivery_priorities()
        return

    priority_ids = set(priority_rules.keys())
    if default_priority_id is None or default_priority_id not in priority_ids:
        default_priority_id = next(iter(priority_ids))

    mapping: dict[Tuple[float, float], str] = {}
    if isinstance(city_priorities, dict):
        for i, city in enumerate(cities_location):
            priority_id = city_priorities.get(i, city_priorities.get(city, default_priority_id))
            if priority_id not in priority_ids:
                priority_id = default_priority_id
            mapping[city] = priority_id
    elif isinstance(city_priorities, (list, tuple)):
        for i, city in enumerate(cities_location):
            if i < len(city_priorities):
                priority_id = city_priorities[i]
            else:
                priority_id = default_priority_id
            if priority_id not in priority_ids:
                priority_id = default_priority_id
            mapping[city] = priority_id
    else:
        mapping = {city: default_priority_id for city in cities_location}

    global _CITY_PRIORITY, _PRIORITY_RULES, _DEFAULT_PRIORITY_ID, _PRIORITY_REFERENCE_CITY
    _CITY_PRIORITY = mapping
    _PRIORITY_RULES = priority_rules
    _DEFAULT_PRIORITY_ID = default_priority_id
    _PRIORITY_REFERENCE_CITY = cities_location[0] if cities_location else None


def clear_delivery_priorities() -> None:
    """Disable delivery priorities."""
    global _CITY_PRIORITY, _PRIORITY_RULES, _DEFAULT_PRIORITY_ID, _PRIORITY_REFERENCE_CITY
    _CITY_PRIORITY = None
    _PRIORITY_RULES = None
    _DEFAULT_PRIORITY_ID = None
    _PRIORITY_REFERENCE_CITY = None


def set_vehicle_params(
    num_vehicles: int = 1,
    capacity_weight: Optional[float] = None,
    capacity_volume: Optional[float] = None,
    depot: Optional[Tuple[float, float]] = None,
) -> None:
    global _NUM_VEHICLES, _VEHICLE_CAPACITY_WEIGHT, _VEHICLE_CAPACITY_VOLUME, _VRP_DEPOT
    _NUM_VEHICLES = max(1, num_vehicles)
    _VEHICLE_CAPACITY_WEIGHT = capacity_weight
    _VEHICLE_CAPACITY_VOLUME = capacity_volume
    _VRP_DEPOT = depot


def clear_vehicle_params() -> None:
    global _NUM_VEHICLES, _VEHICLE_CAPACITY_WEIGHT, _VEHICLE_CAPACITY_VOLUME, _VRP_DEPOT
    _NUM_VEHICLES = 1
    _VEHICLE_CAPACITY_WEIGHT = None
    _VEHICLE_CAPACITY_VOLUME = None
    _VRP_DEPOT = None


def set_city_demands(
    cities_location: List[Tuple[float, float]],
    demands: List[dict],
) -> None:
    """
    Store per-city demand data (weight/volume) for use in VRP fitness evaluation.

    Parameters:
    - cities_location: List of city coordinates.
    - demands: List of dicts with keys 'weight' and 'volume', aligned with cities_location.
    """
    global _CITY_DEMANDS
    _CITY_DEMANDS = {}
    for city, demand in zip(cities_location, demands):
        _CITY_DEMANDS[city] = demand


def clear_city_demands() -> None:
    global _CITY_DEMANDS
    _CITY_DEMANDS = None


def get_city_demand(city: Tuple[float, float]) -> Optional[dict]:
    if _CITY_DEMANDS is None:
        return None
    return _CITY_DEMANDS.get(city)


def _get_depot(individual: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Resolve the depot city for VRP. Precedence: explicit VRP depot > autonomy reference > first city."""
    if _VRP_DEPOT is not None:
        return _VRP_DEPOT
    if _REFERENCE_CITY is not None:
        return _REFERENCE_CITY
    return individual[0] if individual else None


def _is_vrp_active() -> bool:
    """VRP mode is active when capacity constraints are set or multiple vehicles configured."""
    return (
        _NUM_VEHICLES > 1
        or _VEHICLE_CAPACITY_WEIGHT is not None
        or _VEHICLE_CAPACITY_VOLUME is not None
    )


def split_routes(
    individual: List[Tuple[float, float]],
) -> List[List[Tuple[float, float]]]:
    """Split a chromosome into sub-routes for VRP.

    Phase 1 (Capacity Split): walks the chromosome and starts a new route
    whenever the next city would exceed weight or volume capacity.

    Phase 2 (Balance): if the number of routes after phase 1 is less than
    NUM_VEHICLES, the largest routes are subdivided so that all configured
    vehicles are utilised, improving workload distribution.

    Returns a list of sub-routes (each a list of cities, excluding the depot).
    """
    if not individual:
        return [[]]

    if not _is_vrp_active():
        return [list(individual)]

    depot = _get_depot(individual)
    cities = [c for c in individual if c != depot]

    if not cities:
        return [[]]

    has_capacity = (
        _VEHICLE_CAPACITY_WEIGHT is not None or _VEHICLE_CAPACITY_VOLUME is not None
    )

    # --- Phase 1: capacity-aware greedy split ---
    if has_capacity:
        routes: List[List[Tuple[float, float]]] = []
        current_route: List[Tuple[float, float]] = []
        current_weight = 0.0
        current_volume = 0.0

        for city in cities:
            demand = get_city_demand(city) or {"weight": 0, "volume": 0}
            city_weight = demand.get("weight", 0)
            city_volume = demand.get("volume", 0)

            would_exceed_weight = (
                _VEHICLE_CAPACITY_WEIGHT is not None
                and current_weight + city_weight > _VEHICLE_CAPACITY_WEIGHT
            )
            would_exceed_volume = (
                _VEHICLE_CAPACITY_VOLUME is not None
                and current_volume + city_volume > _VEHICLE_CAPACITY_VOLUME
            )

            if (would_exceed_weight or would_exceed_volume) and current_route:
                routes.append(current_route)
                current_route = []
                current_weight = 0.0
                current_volume = 0.0

            current_route.append(city)
            current_weight += city_weight
            current_volume += city_volume

        if current_route:
            routes.append(current_route)

        if not routes:
            routes = [[]]
    else:
        routes = [cities]

    # --- Phase 2: balance across NUM_VEHICLES ---
    if _NUM_VEHICLES > 1:
        while len(routes) < _NUM_VEHICLES:
            longest_idx = max(range(len(routes)), key=lambda i: len(routes[i]))
            longest = routes[longest_idx]
            if len(longest) <= 1:
                break
            mid = len(longest) // 2
            routes[longest_idx] = longest[:mid]
            routes.insert(longest_idx + 1, longest[mid:])

    return routes


def _cost_between(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    if _ASYMMETRIC_COSTS is not None and _CITY_INDEX is not None:
        try:
            return _ASYMMETRIC_COSTS[_CITY_INDEX[point1]][_CITY_INDEX[point2]]
        except KeyError:
            return calculate_distance(point1, point2)
    return calculate_distance(point1, point2)


def _priority_rule_for_city(city: Tuple[float, float]) -> Optional[dict]:
    if _CITY_PRIORITY is None or _PRIORITY_RULES is None or _DEFAULT_PRIORITY_ID is None:
        return None
    priority_id = _CITY_PRIORITY.get(city, _DEFAULT_PRIORITY_ID)
    return _PRIORITY_RULES.get(priority_id)


def _coerce_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _priority_factor_for_city(city: Tuple[float, float]) -> float:
    rule = _priority_rule_for_city(city)
    if not rule:
        return 1.0
    try:
        weight = float(rule.get("weight_multiplier", 1.0))
    except (TypeError, ValueError):
        weight = 1.0
    try:
        penalty = float(rule.get("penalty_per_km", 1.0))
    except (TypeError, ValueError):
        penalty = 1.0
    return weight * penalty


def _resolve_priority_route(path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not path:
        return []

    reference_city = None
    if _REFERENCE_CITY is not None and _REFERENCE_CITY in path:
        reference_city = _REFERENCE_CITY
    elif _PRIORITY_REFERENCE_CITY is not None and _PRIORITY_REFERENCE_CITY in path:
        reference_city = _PRIORITY_REFERENCE_CITY

    if reference_city is None:
        return list(path)

    rotated = _rotate_to_reference(path, reference_city)
    return rotated if rotated is not None else list(path)


def _priority_penalty_for_sequence(
    route_cities: List[Tuple[float, float]],
    start_city: Tuple[float, float],
) -> float:
    if _CITY_PRIORITY is None or _PRIORITY_RULES is None or _DEFAULT_PRIORITY_ID is None:
        return 0.0
    if not route_cities:
        return 0.0

    delay_penalty_multiplier = 25.0
    window_penalty_multiplier = 18.0
    position_penalty_multiplier = 60.0

    penalty_total = 0.0
    elapsed_cost = 0.0
    current = start_city

    for visit_index, city in enumerate(route_cities, start=1):
        leg_cost = _cost_between(current, city)
        elapsed_cost += leg_cost

        factor = max(1.0, _priority_factor_for_city(city))
        rule = _priority_rule_for_city(city) or {}

        if factor > 1.0:
            penalty_total += leg_cost * (factor - 1.0)

        max_delay_min = _coerce_float(rule.get("max_delay_min"))
        if max_delay_min is not None and elapsed_cost > max_delay_min:
            penalty_total += (
                (elapsed_cost - max_delay_min)
                * factor
                * _coerce_float(rule.get("delay_penalty_multiplier"), delay_penalty_multiplier)
            )

        latest_position = _coerce_int(rule.get("latest_position"))
        if latest_position is not None and latest_position > 0 and visit_index > latest_position:
            penalty_total += (
                (visit_index - latest_position)
                * factor
                * _coerce_float(rule.get("position_penalty"), position_penalty_multiplier)
            )

        sequence_penalty = _coerce_float(rule.get("sequence_penalty"), 0.0) or 0.0
        if sequence_penalty > 0:
            penalty_total += (visit_index - 1) * sequence_penalty

        window_start = _coerce_float(rule.get("time_window_start_min"))
        window_end = _coerce_float(rule.get("time_window_end_min"))
        if window_start is not None and elapsed_cost < window_start:
            penalty_total += (
                (window_start - elapsed_cost)
                * factor
                * _coerce_float(rule.get("window_penalty_multiplier"), window_penalty_multiplier)
            )
        if window_end is not None and elapsed_cost > window_end:
            penalty_total += (
                (elapsed_cost - window_end)
                * factor
                * _coerce_float(rule.get("window_penalty_multiplier"), window_penalty_multiplier)
            )

        temperature_penalty_per_km = _coerce_float(rule.get("temperature_penalty_per_km"), 0.0) or 0.0
        if temperature_penalty_per_km > 0:
            penalty_total += elapsed_cost * temperature_penalty_per_km

        service_time_min = _coerce_float(rule.get("service_time_min"), 0.0) or 0.0
        if service_time_min > 0:
            elapsed_cost += service_time_min

        current = city

    return penalty_total


def _priority_penalty(path: List[Tuple[float, float]]) -> float:
    rotated = _resolve_priority_route(path)
    if len(rotated) <= 1:
        return 0.0
    return _priority_penalty_for_sequence(rotated[1:], rotated[0])


def _rotate_to_reference(path: List[Tuple[float, float]], reference_city: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
    if not path:
        return []
    try:
        idx = path.index(reference_city)
    except ValueError:
        return None
    if idx == 0:
        return path
    return path[idx:] + path[:idx]


def _route_cost_with_autonomy(path: List[Tuple[float, float]], autonomy: float) -> float:
    if not path:
        return 0
    if autonomy <= 0:
        return _INVALID_ROUTE_PENALTY

    reference = _REFERENCE_CITY if _REFERENCE_CITY is not None else path[0]
    rotated = _rotate_to_reference(path, reference)
    if rotated is None:
        return _INVALID_ROUTE_PENALTY

    total = 0.0
    remaining = autonomy
    current = reference

    for next_city in rotated[1:]:
        cost_to_next = _cost_between(current, next_city)
        cost_next_to_ref = _cost_between(next_city, reference)

        if cost_to_next + cost_next_to_ref > autonomy:
            return _INVALID_ROUTE_PENALTY

        if current != reference and remaining < cost_to_next + cost_next_to_ref:
            cost_back = _cost_between(current, reference)
            if cost_back > remaining:
                return _INVALID_ROUTE_PENALTY
            total += cost_back
            remaining = autonomy
            current = reference
            cost_to_next = _cost_between(current, next_city)
            cost_next_to_ref = _cost_between(next_city, reference)
            if cost_to_next + cost_next_to_ref > autonomy:
                return _INVALID_ROUTE_PENALTY

        if remaining < cost_to_next:
            return _INVALID_ROUTE_PENALTY

        total += cost_to_next
        remaining -= cost_to_next
        current = next_city

    if current != reference:
        cost_back = _cost_between(current, reference)
        if cost_back > remaining:
            return _INVALID_ROUTE_PENALTY
        total += cost_back

    return total

def generate_random_population(cities_location: List[Tuple[float, float]], population_size: int) -> List[List[Tuple[float, float]]]:
    """
    Generate a random population of routes for a given set of cities.

    Parameters:
    - cities_location (List[Tuple[float, float]]): A list of tuples representing the locations of cities,
      where each tuple contains the latitude and longitude.
    - population_size (int): The size of the population, i.e., the number of routes to generate.

    Returns:
    List[List[Tuple[float, float]]]: A list of routes, where each route is represented as a list of city locations.
    """
    return [random.sample(cities_location, len(cities_location)) for _ in range(population_size)]

def generate__population_using_Nearest_Neighbours(
    cities_location: List[Tuple[float, float]],
    population_size: int,
) -> List[List[Tuple[float, float]]]:
    """
    Generate a population using a nearest neighbour heuristic.

    Each route starts from a random city, then repeatedly visits the nearest unvisited city.
    """
    if population_size <= 0 or not cities_location:
        return []

    population = []
    for _ in range(population_size):
        unvisited = cities_location.copy()
        current = random.choice(unvisited)
        route = [current]
        unvisited.remove(current)

        while unvisited:
            next_city = min(unvisited, key=lambda city: calculate_distance(current, city))
            route.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        population.append(route)

    return population


def generate__population_using_greddy_approach(
    cities_location: List[Tuple[float, float]],
    population_size: int,
) -> List[List[Tuple[float, float]]]:
    """
    Generate a population using a greedy multi-fragment heuristic.

    This approach builds a tour by repeatedly adding the shortest available edge
    while avoiding early cycles and keeping each city degree <= 2.
    """
    if population_size <= 0 or not cities_location:
        return []

    n = len(cities_location)
    if n <= 2:
        return [random.sample(cities_location, n) for _ in range(population_size)]

    def greedy_multifragment() -> Optional[List[int]]:
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = calculate_distance(cities_location[i], cities_location[j])
                # Small jitter to diversify ties across the population
                dist += random.random() * 1e-6
                edges.append((dist, i, j))

        edges.sort(key=lambda x: x[0])

        parent = list(range(n))
        rank = [0] * n
        degree = [0] * n
        chosen_edges = []

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        for _, i, j in edges:
            if degree[i] >= 2 or degree[j] >= 2:
                continue

            fi, fj = find(i), find(j)
            if fi == fj and len(chosen_edges) < n - 1:
                continue

            chosen_edges.append((i, j))
            degree[i] += 1
            degree[j] += 1

            if fi != fj:
                union(fi, fj)

            if len(chosen_edges) == n:
                break

        if len(chosen_edges) != n:
            return None

        adjacency = [[] for _ in range(n)]
        for i, j in chosen_edges:
            adjacency[i].append(j)
            adjacency[j].append(i)

        if any(len(neigh) != 2 for neigh in adjacency):
            return None

        start = random.randrange(n)
        route = [start]
        prev = -1
        current = start
        for _ in range(n - 1):
            next_node = adjacency[current][0] if adjacency[current][0] != prev else adjacency[current][1]
            route.append(next_node)
            prev, current = current, next_node

        return route

    population = []
    for _ in range(population_size):
        route_indices = greedy_multifragment()
        if route_indices is None:
            route = random.sample(cities_location, n)
        else:
            route = [cities_location[i] for i in route_indices]
        population.append(route)

    return population


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    - point1 (Tuple[float, float]): The coordinates of the first point.
    - point2 (Tuple[float, float]): The coordinates of the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def _subroute_cost(
    route_cities: List[Tuple[float, float]],
    depot: Tuple[float, float],
) -> float:
    """Calculate cost of a sub-route: depot -> cities -> depot.

    Handles autonomy within the sub-route: if the vehicle cannot reach the
    next city and return to depot with remaining fuel, it refuels at depot first.
    """
    if not route_cities:
        return 0.0

    total = 0.0
    current = depot
    autonomy = _CAR_AUTONOMY
    remaining = autonomy

    for city in route_cities:
        cost_to_city = _cost_between(current, city)
        cost_city_to_depot = _cost_between(city, depot)

        if autonomy is not None:
            cost_depot_to_city = _cost_between(depot, city)
            if cost_depot_to_city + cost_city_to_depot > autonomy:
                return _INVALID_ROUTE_PENALTY

            if current != depot and remaining < cost_to_city + cost_city_to_depot:
                cost_back = _cost_between(current, depot)
                if cost_back > remaining:
                    return _INVALID_ROUTE_PENALTY
                total += cost_back
                remaining = autonomy
                current = depot
                cost_to_city = _cost_between(depot, city)

            if remaining < cost_to_city:
                return _INVALID_ROUTE_PENALTY

            remaining -= cost_to_city
        total += cost_to_city
        current = city

    cost_back = _cost_between(current, depot)
    if autonomy is not None and remaining is not None and cost_back > remaining:
        return _INVALID_ROUTE_PENALTY
    total += cost_back

    return total


def _subroute_priority_penalty(
    route_cities: List[Tuple[float, float]],
    depot: Tuple[float, float],
) -> float:
    """Priority penalty for a sub-route (depot -> cities -> depot)."""
    return _priority_penalty_for_sequence(route_cities, depot)


def _calculate_vrp_fitness(path: List[Tuple[float, float]]) -> float:
    """Fitness for VRP mode: split + per-subroute cost + vehicle count penalty."""
    routes = split_routes(path)
    depot = _get_depot(path)

    if depot is None:
        return _INVALID_ROUTE_PENALTY

    total_cost = 0.0

    for route in routes:
        if not route:
            continue
        cost = _subroute_cost(route, depot)
        if cost >= _INVALID_ROUTE_PENALTY:
            return _INVALID_ROUTE_PENALTY
        total_cost += cost
        total_cost += _subroute_priority_penalty(route, depot)

    if len(routes) > _NUM_VEHICLES:
        total_cost += _INVALID_ROUTE_PENALTY

    return total_cost


def evaluate_vrp_routes(
    individual: List[Tuple[float, float]],
) -> List[List[Tuple[float, float]]]:
    """Return full VRP route structure for visualization.

    Each route includes depot at start and end: [depot, city1, ..., cityN, depot].
    When VRP is not active, returns a single route wrapping all cities.
    """
    routes = split_routes(individual)
    depot = _get_depot(individual)

    if depot is None:
        return [list(individual)]

    full_routes = []
    for route in routes:
        if route:
            full_routes.append([depot] + route + [depot])

    return full_routes if full_routes else [[depot]]


def calculate_fitness(path: List[Tuple[float, float]]) -> float:
    """
    Calculate the fitness of a given path.

    When VRP mode is active (capacity constraints or multiple vehicles), the path
    is split into sub-routes via Greedy Split and each sub-route is costed
    individually (depot -> cities -> depot), with a massive penalty if the
    number of routes exceeds the configured vehicle count.

    When VRP is inactive, the original TSP logic is used:
    - ATSP directional cost matrix if configured.
    - Autonomy-aware routing with refueling at the reference city.
    - Priority-weighted penalties.
    - Euclidean circular tour distance as fallback.

    Parameters:
    - path (List[Tuple[float, float]]): A list of tuples representing the path,
      where each tuple contains the coordinates of a point.

    Returns:
    float: The total distance/cost of the path.
    """
    if _is_vrp_active():
        return _calculate_vrp_fitness(path)

    if _CAR_AUTONOMY is not None:
        base_cost = _route_cost_with_autonomy(path, _CAR_AUTONOMY)
        if base_cost >= _INVALID_ROUTE_PENALTY:
            return base_cost
        return base_cost + _priority_penalty(path)

    distance = 0.0
    n = len(path)
    if n == 0:
        return 0

    if _ASYMMETRIC_COSTS is not None and _CITY_INDEX is not None:
        costs = _ASYMMETRIC_COSTS
        index = _CITY_INDEX
        try:
            indices = [index[city] for city in path]
        except KeyError:
            indices = None

        if indices is not None:
            for i in range(n):
                distance += costs[indices[i]][indices[(i + 1) % n]]
            return distance + _priority_penalty(path)

    for i in range(n):
        distance += calculate_distance(path[i], path[(i + 1) % n])

    return distance + _priority_penalty(path)


def order_crossover(parent1: List[Tuple[float, float]], parent2: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Perform order crossover (OX) between two parent sequences to create a child sequence.

    Parameters:
    - parent1 (List[Tuple[float, float]]): The first parent sequence.
    - parent2 (List[Tuple[float, float]]): The second parent sequence.

    Returns:
    List[Tuple[float, float]]: The child sequence resulting from the order crossover.
    """
    length = len(parent1)

    # Choose two random indices for the crossover
    start_index = random.randint(0, length - 1)
    end_index = random.randint(start_index + 1, length)

    # Initialize the child with a copy of the substring from parent1
    child = parent1[start_index:end_index]

    # Fill in the remaining positions with genes from parent2
    remaining_positions = [i for i in range(length) if i < start_index or i >= end_index]
    remaining_genes = [gene for gene in parent2 if gene not in child]

    for position, gene in zip(remaining_positions, remaining_genes):
        child.insert(position, gene)

    return child

### demonstration: crossover test code
# Example usage:
# parent1 = [(1, 1), (2, 2), (3, 3), (4,4), (5,5), (6, 6)]
# parent2 = [(6, 6), (5, 5), (4, 4), (3, 3),  (2, 2), (1, 1)]

# # parent1 = [1, 2, 3, 4, 5, 6]
# # parent2 = [6, 5, 4, 3, 2, 1]


# child = order_crossover(parent1, parent2)
# print("Parent 1:", [0, 1, 2, 3, 4, 5, 6, 7, 8])
# print("Parent 1:", parent1)
# print("Parent 2:", parent2)
# print("Child   :", child)


# # Example usage:
# population = generate_random_population(5, 10)

# print(calculate_fitness(population[0]))


# population = [(random.randint(0, 100), random.randint(0, 100))
#           for _ in range(3)]

def mutate(solution:  List[Tuple[float, float]], mutation_probability: float, 
           just_swap) ->  List[Tuple[float, float]]:
    """
    Mutate a solution by inverting a segment of the sequence with a given mutation probability.

    Parameters:
    - solution (List[int]): The solution sequence to be mutated.
    - mutation_probability (float): The probability of mutation for each individual in the solution.

    Returns:
    List[int]: The mutated solution sequence.
   
    # Aplica mutação na rota com probabilidade mutation_probability:
    # se just_swap=True, troca duas cidades adjacentes aleatórias;
    # caso contrário, inverte um segmento aleatório da rota.
    # Retorna a rota (mutada ou original) sem alterar a entrada.
    """

    mutated_solution = copy.deepcopy(solution)

    # Check if mutation should occur    
    if random.random() < mutation_probability:
        
        # Ensure there are at least two cities to perform a swap
        if len(solution) < 2:
            return solution
    
        if just_swap:
        # Select a random index (excluding the last index) for swapping
            index = random.randint(0, len(solution) - 2)
            # Swap the cities at the selected index and the next index
            mutated_solution[index], mutated_solution[index + 1] = solution[index + 1], solution[index]           
        else:
            i, j = sorted(random.sample(range(len(mutated_solution)), 2))
            # Inverte o segmento entre i e j
            mutated_solution[i:j+1] = reversed(mutated_solution[i:j+1])
        
    return mutated_solution

### Demonstration: mutation test code    
# # Example usage:
# original_solution = [(1, 1), (2, 2), (3, 3), (4, 4)]
# mutation_probability = 1

# mutated_solution = mutate(original_solution, mutation_probability)
# print("Original Solution:", original_solution)
# print("Mutated Solution:", mutated_solution)


def sort_population(population: List[List[Tuple[float, float]]], fitness: List[float]) -> Tuple[List[List[Tuple[float, float]]], List[float]]:
    """
    Sort a population based on fitness values.

    Parameters:
    - population (List[List[Tuple[float, float]]]): The population of solutions, where each solution is represented as a list.
    - fitness (List[float]): The corresponding fitness values for each solution in the population.

    Returns:
    Tuple[List[List[Tuple[float, float]]], List[float]]: A tuple containing the sorted population and corresponding sorted fitness values.
    """
    # Combine lists into pairs
    combined_lists = list(zip(population, fitness))

    # Sort based on the values of the fitness list
    sorted_combined_lists = sorted(combined_lists, key=lambda x: x[1])

    # Separate the sorted pairs back into individual lists
    sorted_population, sorted_fitness = zip(*sorted_combined_lists)

    return sorted_population, sorted_fitness


if __name__ == '__main__':
    N_CITIES = 10
    
    POPULATION_SIZE = 100
    N_GENERATIONS = 100
    MUTATION_PROBABILITY = 0.3
    cities_locations = [(random.randint(0, 100), random.randint(0, 100))
              for _ in range(N_CITIES)]
    
    # CREATE INITIAL POPULATION
    population = generate_random_population(cities_locations, POPULATION_SIZE)

    # Lists to store best fitness and generation for plotting
    best_fitness_values = []
    best_solutions = []
    
    for generation in range(N_GENERATIONS):
  
        
        population_fitness = [calculate_fitness(individual) for individual in population]    
        
        population, population_fitness = sort_population(population,  population_fitness)
        
        best_fitness = calculate_fitness(population[0])
        best_solution = population[0]
           
        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)    

        print(f"Generation {generation}: Best fitness = {best_fitness}")

        new_population = [population[0]]  # Keep the best individual: ELITISM
        
        while len(new_population) < POPULATION_SIZE:
            
            # SELECTION
            parent1, parent2 = random.choices(population[:10], k=2)  # Select parents from the top 10 individuals
            
            # CROSSOVER
            child1 = order_crossover(parent1, parent2)
            
            ## MUTATION
            child1 = mutate(child1, MUTATION_PROBABILITY, False)
            
            new_population.append(child1)
            
    
        print('generation: ', generation)
        population = new_population
    
def generate__population_using_convex_hull(
    cities_location: List[Tuple[float, float]],
    population_size: int,
) -> List[List[Tuple[float, float]]]:
    """
    Generate a population using a convex hull + insertion heuristic.

    Steps:
    1) Build the convex hull of the cities.
    2) Insert remaining cities into the tour by cheapest insertion.
    3) Add small randomization to diversify the population.
    """
    if population_size <= 0 or not cities_location:
        return []

    n = len(cities_location)
    if n <= 2:
        return [random.sample(cities_location, n) for _ in range(population_size)]

    def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        points_sorted = sorted(set(points))
        if len(points_sorted) <= 1:
            return points_sorted

        lower: List[Tuple[float, float]] = []
        for p in points_sorted:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper: List[Tuple[float, float]] = []
        for p in reversed(points_sorted):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        return lower[:-1] + upper[:-1]

    def rotate_route(route: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not route:
            return route
        k = random.randrange(len(route))
        return route[k:] + route[:k]

    def insert_cheapest(route: List[Tuple[float, float]], city: Tuple[float, float]) -> List[Tuple[float, float]]:
        if len(route) < 2:
            return route + [city]

        best_index = 0
        best_delta = float("inf")
        m = len(route)
        for i in range(m):
            a = route[i]
            b = route[(i + 1) % m]
            delta = calculate_distance(a, city) + calculate_distance(city, b) - calculate_distance(a, b)
            delta += random.random() * 1e-6
            if delta < best_delta:
                best_delta = delta
                best_index = i + 1

        return route[:best_index] + [city] + route[best_index:]

    hull = convex_hull(cities_location)
    population = []

    for _ in range(population_size):
        route = hull[:]

        if random.random() < 0.5:
            route.reverse()
        route = rotate_route(route)

        remaining = [c for c in cities_location if c not in route]
        random.shuffle(remaining)

        for city in remaining:
            route = insert_cheapest(route, city)

        population.append(route)

    return population
