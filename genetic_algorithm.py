import random
import math
import copy 
from typing import List, Tuple, Optional


_ASYMMETRIC_COSTS: Optional[List[List[float]]] = None
_CITY_INDEX: Optional[dict[Tuple[float, float], int]] = None


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


#TODO: Apagar se não mais usado
# default_problems = {
# 5: [(733, 251), (706, 87), (546, 97), (562, 49), (576, 253)],
# 10:[(470, 169), (602, 202), (754, 239), (476, 233), (468, 301), (522, 29), (597, 171), (487, 325), (746, 232), (558, 136)],
# 12:[(728, 67), (560, 160), (602, 312), (712, 148), (535, 340), (720, 354), (568, 300), (629, 260), (539, 46), (634, 343), (491, 135), (768, 161)],
# 15:[(512, 317), (741, 72), (552, 50), (772, 346), (637, 12), (589, 131), (732, 165), (605, 15), (730, 38), (576, 216), (589, 381), (711, 387), (563, 228), (494, 22), (787, 288)]
# }

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


def calculate_fitness(path: List[Tuple[float, float]]) -> float:
    """
    Calculate the fitness of a given path.

    If asymmetric costs are configured (ATSP), the directional cost matrix is used.
    Otherwise, the total Euclidean distance of the path is returned.

    Parameters:
    - path (List[Tuple[float, float]]): A list of tuples representing the path,
      where each tuple contains the coordinates of a point.

    Returns:
    float: The total distance/cost of the path.
    """
    distance = 0
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
            return distance

    for i in range(n):
        distance += calculate_distance(path[i], path[(i + 1) % n])

    return distance


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
