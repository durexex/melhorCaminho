import pygame
from pygame.locals import *
import random
import itertools
from genetic_algorithm import mutate, order_crossover, generate_random_population, calculate_fitness, sort_population, generate__population_using_Nearest_Neighbours, default_problems
from demo_tournament import tournament_selection
from draw_functions import draw_paths, draw_plot, draw_cities
import sys
import numpy as np
import pygame
from benchmark_att48 import *


# Define constant values
# pygame
WIDTH, HEIGHT = 1980, 1080
NODE_RADIUS = 10
FPS = 30
PLOT_X_OFFSET = 450

if len(sys.argv) > 1:
    GERAR_CIDADES = int(sys.argv[1]) == 0
    MAX_GENERATION_ALLOWED = int(sys.argv[2])    
else:
    GERAR_CIDADES = True
    MAX_GENERATION_ALLOWED = 1000

# GA
N_CITIES = 15
POPULATION_SIZE = 100
N_GENERATIONS = None
MUTATION_PROBABILITY = 0.5

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


# Initialize problem
# Using Random cities generation
if GERAR_CIDADES:
    cities_locations = [
        (random.randint(NODE_RADIUS + PLOT_X_OFFSET, WIDTH - NODE_RADIUS), random.randint(NODE_RADIUS, HEIGHT - NODE_RADIUS))
        for _ in range(N_CITIES)
    ]

    with open("cities_locations.txt", "w", encoding="utf-8") as cities_file:
        for x, y in cities_locations:
            cities_file.write(f"{x},{y}\n")
else:
    with open("cities_locations.txt", "r", encoding="utf-8") as cities_file:
        cities_locations = []
        for line in cities_file:
            line = line.strip()
            if not line:
                continue
            x_str, y_str = line.split(",", 1)
            cities_locations.append((int(x_str), int(y_str)))



# # Using Deault Problems: 10, 12 or 15
# WIDTH, HEIGHT = 800, 400
# cities_locations = default_problems[15]


# Using att48 benchmark
# WIDTH, HEIGHT = 1500, 800
# att_cities_locations = np.array(att_48_cities_locations)
# max_x = max(point[0] for point in att_cities_locations)
# max_y = max(point[1] for point in att_cities_locations)
# scale_x = (WIDTH - PLOT_X_OFFSET - NODE_RADIUS) / max_x
# scale_y = HEIGHT / max_y
# cities_locations = [(int(point[0] * scale_x + PLOT_X_OFFSET),
#                      int(point[1] * scale_y)) for point in att_cities_locations]
# target_solution = [cities_locations[i-1] for i in att_48_cities_order]
# fitness_target_solution = calculate_fitness(target_solution)
# print(f"Best Solution: {fitness_target_solution}")
# ----- Using att48 benchmark


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TSP Solver using Pygame")
clock = pygame.time.Clock()
generation_counter = itertools.count(start=1)  # Start the counter at 1


# Create Initial Population
# TODO:- use some heuristic like Nearest Neighbour our Convex Hull to initialize
population = generate_random_population(cities_locations, int(round(POPULATION_SIZE * 0.9, 0)))
population = generate__population_using_Nearest_Neighbours(cities_locations, int(round(POPULATION_SIZE * 0.1, 0)))
best_fitness_values = []
best_solutions = []


# Main game loop
running = True
while running:
        
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    generation = next(generation_counter)

    if generation == MAX_GENERATION_ALLOWED:
        running = False    

    screen.fill(WHITE)

    population_fitness = [calculate_fitness(
        individual) for individual in population]

    population, population_fitness = sort_population(
        population,  population_fitness)

    best_fitness = calculate_fitness(population[0])
    best_solution = population[0]

    best_fitness_values.append(best_fitness)
    best_solutions.append(best_solution)

    draw_plot(screen, list(range(len(best_fitness_values))),
              best_fitness_values, y_label="Fitness - Distance (pxls)")

    draw_cities(screen, cities_locations, RED, NODE_RADIUS)
    draw_paths(screen, best_solution, BLUE, width=3)
    draw_paths(screen, population[1], rgb_color=(128, 128, 128), width=1)

    print(f"Generation {generation}: Best fitness = {round(best_fitness, 2)}")

    new_population = [population[0]]  # Keep the best individual: ELITISM

    while len(new_population) < (POPULATION_SIZE  * 0.5):

        # selection
        # simple selection based on first 10 best solutions
        # parent1, parent2 = random.choices(population[:10], k=2)

        # solution based on fitness probability
        probability = 1 / np.array(population_fitness)
        parent1, parent2 = random.choices(population, weights=probability, k=2)

        # child1 = order_crossover(parent1, parent2)
        child1 = order_crossover(parent1, parent2)

        child1 = mutate(child1, MUTATION_PROBABILITY)

        new_population.append(child1)

    while len(new_population) < (POPULATION_SIZE  * 0.5):

        # selection
        # parent1, parent2 = tournament_selection(population[:10], k=2)

        # tournament selection
        parent1, _ = tournament_selection(population, calculate_fitness, tournament_size=3, minimize=True)
        parent2, _ = tournament_selection(population, calculate_fitness, tournament_size=3, minimize=True)

        # child1 = order_crossover(parent1, parent2)
        child1 = order_crossover(parent1, parent2)

        child1 = mutate(child1, MUTATION_PROBABILITY)

        new_population.append(child1)

    population = new_population
    
    pygame.display.flip()
    clock.tick(FPS)


# TODO: save the best individual in a file if it is better than the one saved.

# exit software
pygame.quit()
sys.exit()
