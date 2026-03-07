import random
from genetic_algorithm import *
from demo_tournament import tournament_selection
from draw_functions import draw_plot, build_solution_figure
from utils import ReportData, set_report_data, generate_report
import numpy as np
from benchmark_att48 import *
from datetime import datetime
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="TSP Solver", layout="wide")
st.title("TSP Solver - Algoritmo Genetico")

# Define constant values
WIDTH, HEIGHT = 1980, 1080
NODE_RADIUS = 10
PLOT_X_OFFSET = 450
RENDER_EVERY = 5

ATSP_ENABLED = True
ASYMMETRY_FACTOR = 0.3
ASYMMETRY_SEED = 42
ASYMMETRIC_COSTS_FILE = "cities_asynmetric_cost_matrix.txt"
CITIES_LOCATION_FILE = "cities_locations.txt"


def build_asymmetric_cost_matrix(cities, asymmetry_factor=0.3, seed=None):
    rng = random.Random(seed) if seed is not None else random
    n = len(cities)
    costs = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            base = calculate_distance(cities[i], cities[j])
            multiplier = 1.0 + asymmetry_factor * (rng.random() * 2 - 1)
            if multiplier < 0.1:
                multiplier = 0.1
            costs[i][j] = base * multiplier

    with open(ASYMMETRIC_COSTS_FILE, "w", encoding="utf-8") as costs_file:
        for row in costs:
            costs_file.write(",".join(f"{value}" for value in row) + "\n")

    return costs


def load_asymmetric_cost_matrix(path, expected_size):
    try:
        with open(path, "r", encoding="utf-8") as costs_file:
            matrix = []
            for line in costs_file:
                line = line.strip()
                if not line:
                    continue
                row = [float(value) for value in line.split(",") if value]
                matrix.append(row)
        if not matrix:
            return None
        if len(matrix) != expected_size or any(len(row) != expected_size for row in matrix):
            return None
        return matrix
    except FileNotFoundError:
        return None

# Define criacao de cidades
#   True indica que as cidades serao geradas aleatoriamente
#   False indica que as cidades virao de arquivo texto pre definido
#
#  Este parametro trabalha em conjunto com o parametro NUMBER_OF_CITIES, caso as cidades sejam geradas
#  NUMBER_OF_CITIES indicara a quantidade de cidades que serao geradas
#  Se definido como False, sera necessario ter um arquivo chamado cities_locations.txt onde cada linha
#  indica uma cidade (1457,614) - sendo a latitude e a longitude
GERAR_CIDADES = False

# Total de cidades que serao geradas (Somente opera se GERAR_CIDADES = True)
NUMBER_OF_CITIES = 20

# Total de geracoes que serao testadas
MAX_GENERATION_ALLOWED = 50

# Indica como serao geradas as populacoes. Se True 100% sera aleatoriamente, se False somente o percentual
# indicado em RANDOM_POPULATION_PERCENT sera gerada aleatoriamente e o restante pelo algoritmo indicado
# em GENERATE_POLPULATION_USING_NEAREST_NEIGHBOURS ou GENERATE_POLPULATION_USING_GREEDY_APPROACH
ONLY_RANDOM_POPULATION = False

# Total de populacao gerada de forma aleatoria. Se ONLY_RANDOM_POPULATION = for False o valor aqui definido
# e desprezado e sera utilizado o valor 1 (100%)
# Este parametro pode variar de 0 a 1
RANDOM_POPULATION_PERCENT = 0.9
if RANDOM_POPULATION_PERCENT < 0:
    RANDOM_POPULATION_PERCENT = 0
elif RANDOM_POPULATION_PERCENT > 1:
    RANDOM_POPULATION_PERCENT = 1

# Estes 2 parametros a seguir indicam o metodo alternativo para geracao de populacao
# Somente funcionam se ONLY_RANDOM_POPULATION for False
# A quantidade de populacao gerada por eles sera 1 - RANDOM_POPULATION_PERCENT
# Funcionam de forma exclusiva (ou um ou outro).
# Se GENERATE_POLPULATION_USING_NEAREST_NEIGHBOURS For True qualquer valor colocado em
# GENERATE_POLPULATION_USING_GREEDY_APPROACH sera desprezado

GENERATE_POLPULATION_USING_NEAREST_NEIGHBOURS = False
GENERATE_POLPULATION_USING_GREEDY_APPROACH = False
GENERATE_POLPULATION_USING_CONVEX_HULL = True

# Tamanho da populacao - para entendimento Populacao neste caso sao as rotas procuradas
POPULATION_SIZE = 100

# Probabilidade de gerar mutacao
MUTATION_PROBABILITY = 0.5
# Ao fazer a mutacao define se ira somente fazer a troca dos segmentos ou inverte-los
JUST_SWAP = True

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


# 1. Guardar hora de inicio
hora_inicio = datetime.now()

# Initialize problem
# Using Random cities generation
if GERAR_CIDADES:
    cities_locations = [
        (random.randint(NODE_RADIUS + PLOT_X_OFFSET, WIDTH - NODE_RADIUS), random.randint(NODE_RADIUS, HEIGHT - NODE_RADIUS))
        for _ in range(NUMBER_OF_CITIES)
    ]

    with open(CITIES_LOCATION_FILE, "w", encoding="utf-8") as cities_file:
        for x, y in cities_locations:
            cities_file.write(f"{x},{y}\n")

    if ATSP_ENABLED:
        asymmetric_costs = load_asymmetric_cost_matrix(
            ASYMMETRIC_COSTS_FILE,
            expected_size=len(cities_locations),
        )
        if asymmetric_costs is None:
            asymmetric_costs = build_asymmetric_cost_matrix(
                cities_locations,
                asymmetry_factor=ASYMMETRY_FACTOR,
                seed=ASYMMETRY_SEED,
            )
        set_asymmetric_costs(cities_locations, asymmetric_costs)
else:
    with open(CITIES_LOCATION_FILE, "r", encoding="utf-8") as cities_file:
        cities_locations = []
        for line in cities_file:
            line = line.strip()
            if not line:
                continue
            x_str, y_str = line.split(",", 1)
            cities_locations.append((int(x_str), int(y_str)))
    
    if ATSP_ENABLED:
        asymmetric_costs = load_asymmetric_cost_matrix(
            ASYMMETRIC_COSTS_FILE,
            expected_size=len(cities_locations),
        )
        if asymmetric_costs is None:
            asymmetric_costs = build_asymmetric_cost_matrix(
                cities_locations,
                asymmetry_factor=ASYMMETRY_FACTOR,
                seed=ASYMMETRY_SEED,
            )
        set_asymmetric_costs(cities_locations, asymmetric_costs)


# Create Initial Population
if ONLY_RANDOM_POPULATION:
    RANDOM_POPULATION_PERCENT = 1

population = generate_random_population(cities_locations, int(round(POPULATION_SIZE * RANDOM_POPULATION_PERCENT, 0)))
initial_population_method = "random"

if ONLY_RANDOM_POPULATION == False:
    if GENERATE_POLPULATION_USING_NEAREST_NEIGHBOURS:
        population = generate__population_using_Nearest_Neighbours(cities_locations, int(round(POPULATION_SIZE * (1 - RANDOM_POPULATION_PERCENT), 0)))
        initial_population_method = "nearest_neighbours"
    elif GENERATE_POLPULATION_USING_GREEDY_APPROACH:
        population = generate__population_using_greddy_approach(cities_locations, int(round(POPULATION_SIZE * (1 - RANDOM_POPULATION_PERCENT), 0)))
        initial_population_method = "greedy"
    elif GENERATE_POLPULATION_USING_CONVEX_HULL:
        generate__population_using_convex_hull(cities_locations, int(round(POPULATION_SIZE * (1 - RANDOM_POPULATION_PERCENT), 0)))
        initial_population_method = "convex hull"

best_fitness_values = []
best_solutions = []
# para logar qual geracao acho melhor valor
choosen_generation = 0
old_best_solution = 0

status_placeholder = st.empty()
plot_col, map_col = st.columns([1, 2])
fitness_placeholder = plot_col.empty()
map_placeholder = map_col.empty()
progress = st.progress(0)


# Main loop
for generation in range(1, MAX_GENERATION_ALLOWED + 1):
    population_fitness = [calculate_fitness(individual) for individual in population]
    population, population_fitness = sort_population(population, population_fitness)

    best_solution = population[0]
    best_fitness = population_fitness[0]

    best_fitness_values.append(best_fitness)
    best_solutions.append(best_solution)

    if old_best_solution == 0:
        old_best_solution = best_solution

    if best_solution < old_best_solution:
        old_best_solution = best_solution
        choosen_generation = generation

    # Keep the best individual: ELITISM
    new_population = [population[0]]

    while len(new_population) < (POPULATION_SIZE * 0.5):
        # solution based on fitness probability
        probability = 1 / np.array(population_fitness)
        parent1, parent2 = random.choices(population, weights=probability, k=2)

        child1 = order_crossover(parent1, parent2)
        child1 = mutate(child1, MUTATION_PROBABILITY, JUST_SWAP)

        new_population.append(child1)

    while len(new_population) < (POPULATION_SIZE * 0.5):
        # tournament selection
        parent1, _ = tournament_selection(population, calculate_fitness, tournament_size=3, minimize=True)
        parent2, _ = tournament_selection(population, calculate_fitness, tournament_size=3, minimize=True)

        child1 = order_crossover(parent1, parent2)
        child1 = mutate(child1, MUTATION_PROBABILITY, JUST_SWAP)

        new_population.append(child1)

    population = new_population

    if generation == 1 or generation == MAX_GENERATION_ALLOWED or generation % RENDER_EVERY == 0:
        status_placeholder.markdown(
            f"**Geracao:** {generation}  |  **Melhor fitness:** {best_fitness:.2f}"
        )

        fitness_fig = draw_plot(list(range(len(best_fitness_values))), best_fitness_values, y_label="Fitness - Distance (pxls)")
        fitness_placeholder.pyplot(fitness_fig, width="stretch")
        plt.close(fitness_fig)

        candidate_path = population[1] if len(population) > 1 else None
        solution_fig = build_solution_figure(
            cities_locations,
            best_solution,
            candidate_path=candidate_path,
            node_radius=NODE_RADIUS,
            width=WIDTH,
            height=HEIGHT,
            x_offset=PLOT_X_OFFSET,
        )
        map_placeholder.pyplot(solution_fig, width="stretch")
        plt.close(solution_fig)

        progress.progress(int(generation / MAX_GENERATION_ALLOWED * 100))


hora_final = datetime.now()
report_output_path = f"reports\\{hora_final.strftime('%Y%m%d%H%M%S')}.txt"

if best_fitness_values:
    best_index, _ = min(enumerate(best_fitness_values), key=lambda x: x[1])
    best_solution_report = best_solutions[best_index]
    best_generation_report = best_index + 1
    best_fitness_report = best_fitness_values[best_index]
else:
    best_solution_report = []
    best_generation_report = 0
    best_fitness_report = 0.0

set_report_data(ReportData(
    num_cities=len(cities_locations),
    start_time=hora_inicio,
    end_time=hora_final,
    best_solution=best_solution_report,
    best_generation=best_generation_report,
    best_fitness=best_fitness_report,
    mutation_probability=MUTATION_PROBABILITY,
    random_population_percent=RANDOM_POPULATION_PERCENT,
    initial_population_method=initial_population_method,
    output_path=report_output_path,
))

report_text = generate_report()
st.text(report_text)
