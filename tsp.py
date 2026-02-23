import pygame
from pygame.locals import *
import random
import itertools
from genetic_algorithm import *
from demo_tournament import tournament_selection
from draw_functions import draw_paths, draw_plot, draw_cities
from utils import ReportData, set_report_data, generate_report
import sys
import numpy as np
import pygame
from benchmark_att48 import *
from datetime import datetime
import time


# Define constant values
# pygame
WIDTH, HEIGHT = 1980, 1080
NODE_RADIUS = 10
FPS = 30
PLOT_X_OFFSET = 450

# Define criação de cidades
#   True indica que as cidades serão geradas aleatoriamente
#   False indica que as cidades virão de arquivo texto pré definido
#
#  Este parametro trabalha em conjunto com o parâmetro NUMBER_OF_CITIES, caso as cidades sejam geradas
#  NUMBER_OF_CITIES indicará a quantidade de cidades que serão geradas
#  Se definido como False, será necessário ter um arquivo chamado cities_locations.txt onde cada linha
#  indica uma cidade (1457,614) - sendo a latitude e a longitude
GERAR_CIDADES = False

# Total de cidades que serão geradas (Somente opera se GERAR_CIDADES = True)
NUMBER_OF_CITIES = 20

# Total de gerações que serão testadas
MAX_GENERATION_ALLOWED = 500

# Indica como seão geradas as populações. Se True 100% será aleatoriamente, se False somente o percentual
# indicado em RANDOM_POPULATION_PERCENT será gerada aleatoriamente e o restante pelo algorítmo indicado
# em GENERATE_POLPULATION_USING_NEAREST_NEIGHBOURS ou GENERATE_POLPULATION_USING_GREEDY_APPROACH 
ONLY_RANDOM_POPULATION=False

# Total de população gerada de forma aleatória. Se ONLY_RANDOM_POPULATION = for False o valor aqui definido 
# é desprezado e será utilizado o valor 1 (100%)
# Este parâmetro pode variar de 0 a 1
RANDOM_POPULATION_PERCENT = 0.9
if RANDOM_POPULATION_PERCENT < 0:
    RANDOM_POPULATION_PERCENT = 0
elif RANDOM_POPULATION_PERCENT > 1:
    RANDOM_POPULATION_PERCENT = 1

# Estes 2 parâmetros a seguir indicam o método alternativo para geração de população
# Somente funcionam se ONLY_RANDOM_POPULATION for False
# A quantidade de população gerada por eles será 1 - RANDOM_POPULATION_PERCENT 
# Funcionam de forma exclusiva (ou um ou outro). 
# Se GENERATE_POLPULATION_USING_NEAREST_NEIGHBOURS For True qualquer valor colocado em 
# GENERATE_POLPULATION_USING_GREEDY_APPROACH será desprezado

GENERATE_POLPULATION_USING_NEAREST_NEIGHBOURS = False
GENERATE_POLPULATION_USING_GREEDY_APPROACH = False
GENERATE_POLPULATION_USING_CONVEX_HULL=True

# Tamanho da população - para entendimento População neste caso são as rotas procuradas
POPULATION_SIZE = 100

# Probabilidade de gerar mutação 
MUTATION_PROBABILITY = 0.5
# Ao fazer a mutação define se irá somente fazer a troca dos segmentos ou invertÊ-los
JUST_SWAP = True

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


# 1. Guardar hora de início
hora_inicio = datetime.now()

# Initialize problem
# Using Random cities generation
if GERAR_CIDADES:
    cities_locations = [
        (random.randint(NODE_RADIUS + PLOT_X_OFFSET, WIDTH - NODE_RADIUS), random.randint(NODE_RADIUS, HEIGHT - NODE_RADIUS))
        for _ in range(NUMBER_OF_CITIES)
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
# para logar qual geração acho melhor valor
choosen_generation = 0
old_best_solution = 0

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

    population_fitness = [calculate_fitness(individual) for individual in population]

    population, population_fitness = sort_population(population,  population_fitness)

    
    best_fitness = calculate_fitness(population[0])
    best_solution = population[0]

    best_fitness_values.append(best_fitness)
    best_solutions.append(best_solution)

    if old_best_solution == 0:
        old_best_solution = best_solution
    
    if best_solution < old_best_solution:
        old_best_solution = best_solution
        choosen_generation = generation
    

    draw_plot(screen, list(range(len(best_fitness_values))),
              best_fitness_values, y_label="Fitness - Distance (pxls)")

    draw_cities(screen, cities_locations, RED, NODE_RADIUS)
    draw_paths(screen, best_solution, BLUE, width=3)
    draw_paths(screen, population[1], rgb_color=(128, 128, 128), width=1)

    print(f"Generation {generation}: Best fitness = {round(best_fitness, 2)}")

    # Keep the best individual: ELITISM
    new_population = [population[0]]  

    while len(new_population) < (POPULATION_SIZE  * 0.5):

        # selection
        # simple selection based on first 10 best solutions
        # parent1, parent2 = random.choices(population[:10], k=2)

        # solution based on fitness probability
        probability = 1 / np.array(population_fitness)
        parent1, parent2 = random.choices(population, weights=probability, k=2)

        # child1 = order_crossover(parent1, parent2)
        child1 = order_crossover(parent1, parent2)

        child1 = mutate(child1, MUTATION_PROBABILITY, JUST_SWAP)

        new_population.append(child1)

    while len(new_population) < (POPULATION_SIZE  * 0.5):

        # selection
        # parent1, parent2 = tournament_selection(population[:10], k=2)

        # tournament selection
        parent1, _ = tournament_selection(population, calculate_fitness, tournament_size=3, minimize=True)
        parent2, _ = tournament_selection(population, calculate_fitness, tournament_size=3, minimize=True)

        # child1 = order_crossover(parent1, parent2)
        child1 = order_crossover(parent1, parent2)

        child1 = mutate(child1, MUTATION_PROBABILITY, JUST_SWAP)

        new_population.append(child1)

    population = new_population
    
    pygame.display.flip()
    clock.tick(FPS)


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

print(generate_report())

# exit software
pygame.quit()
sys.exit()



# TODO:

###
# Restrições de rota

# TSP assimétrico (ATSP): custo A->B diferente de B->A para “sentido único”.
# Arestas proibidas: algumas conexões não podem existir (vias fechadas).
# Penalidade por uso: desestimular certas rotas com custo extra.
# Restrições de tempo

# Time windows: cada cidade tem horário permitido [início, fim].
# Tempo de serviço: tempo fixo parado em cada cidade.
# Tempo dependente: tempo de viagem depende do horário (trânsito).
# Qualidade da solução

# Local search: 2-opt, 3-opt, or-opt pós‑crossover para refinar.
# Heurística híbrida: inicializar parte da população com greedy/convex hull + random.
# Evolução

# Mutação adaptativa: aumenta quando a população estagna.
# Elitismo controlado: preservar top k mas evitar convergência prematura.
# Diversidade: penalizar rotas muito parecidas.
# Seleção e crossover

# Crossover especializado: Edge Recombination, PMX, OX2.
# Seleção por torneio ajustável: tamanho do torneio muda ao longo das gerações.
# Critérios de parada

# Estagnação: para se não melhora após N gerações.
# Orçamento de tempo: interrompe por tempo máximo.
# Se quiser, digo quais mudanças cabem melhor no seu código atual e implemento a primeira. Para direcionar:

# Você quer sentido único (custo A->B ≠ B->A) ou rotas proibidas?
# Quer time windows rígidas ou apenas penalidade por atraso?
# O objetivo continua minimizar distância, ou passa a minimizar tempo total?
