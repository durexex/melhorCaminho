import random
import os
from pathlib import Path
from dotenv import load_dotenv
from genetic_algorithm import *
from demo_tournament import tournament_selection
from draw_functions import draw_plot, build_solution_figure
from utils import ReportData, set_report_data, generate_report
import numpy as np
from benchmark_att48 import *
from datetime import datetime
import streamlit as st
import matplotlib.pyplot as plt


def _normalize_env_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in ("none", "null"):
            return None
    return value


def _parse_int(value, default=None):
    value = _normalize_env_value(value)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return default


def _parse_float(value, default=None):
    value = _normalize_env_value(value)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_bool(value, default=False):
    value = _normalize_env_value(value)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("1", "true", "t", "yes", "y", "on"):
        return True
    if value in ("0", "false", "f", "no", "n", "off"):
        return False
    return default


def _parse_str(value, default=None):
    value = _normalize_env_value(value)
    if value is None:
        return default
    return str(value)


load_dotenv(Path(__file__).with_name(".env"))

st.set_page_config(page_title="TSP Solver", layout="wide")
st.title("TSP Solver - Algoritmo Genetico")


def _coerce_config_value(value, template):
    if template is None:
        return _normalize_env_value(value)
    if isinstance(template, bool):
        return _parse_bool(value, default=template)
    if isinstance(template, int) and not isinstance(template, bool):
        return _parse_int(value, default=template)
    if isinstance(template, float):
        return _parse_float(value, default=template)
    return _parse_str(value, default=template)

_config_defaults = {
    "PLOT_X_OFFSET": _parse_int(os.getenv("PLOT_X_OFFSET")),
    "WIDTH": _parse_int(os.getenv("WIDTH")),
    "HEIGHT": _parse_int(os.getenv("HEIGHT")),
    "NODE_RADIUS": _parse_int(os.getenv("NODE_RADIUS")),
    "RENDER_EVERY": _parse_int(os.getenv("RENDER_EVERY")),
    "CAR_AUTONOMY": _parse_float(os.getenv("CAR_AUTONOMY")),
    "ATSP_ENABLED": _parse_bool(os.getenv("ATSP_ENABLED")),
    "ASYMMETRY_FACTOR": _parse_float(os.getenv("ASYMMETRY_FACTOR")),
    "ASYMMETRY_SEED": _parse_int(os.getenv("ASYMMETRY_SEED")),
    "ASYMMETRIC_COSTS_FILE": _parse_str(os.getenv("ASYMMETRIC_COSTS_FILE")),
    "CITIES_LOCATION_FILE": _parse_str(os.getenv("CITIES_LOCATION_FILE")),
    "GERAR_CIDADES": _parse_bool(os.getenv("GERAR_CIDADES")),
    "NUMBER_OF_CITIES": _parse_int(os.getenv("NUMBER_OF_CITIES"), default=10),
    "MAX_GENERATION_ALLOWED": _parse_int(os.getenv("MAX_GENERATION_ALLOWED")),
    "ONLY_RANDOM_POPULATION": _parse_bool(os.getenv("ONLY_RANDOM_POPULATION")),
    "RANDOM_POPULATION_PERCENT": _parse_float(os.getenv("RANDOM_POPULATION_PERCENT"), default=0),
    "GENERATE_POPULATION_USING_NEAREST_NEIGHBOURS": _parse_bool(
        os.getenv("GENERATE_POPULATION_USING_NEAREST_NEIGHBOURS")
    ),
    "GENERATE_POPULATION_USING_GREEDY_APPROACH": _parse_bool(
        os.getenv("GENERATE_POPULATION_USING_GREEDY_APPROACH")
    ),
    "GENERATE_POPULATION_USING_CONVEX_HULL": _parse_bool(
        os.getenv("GENERATE_POPULATION_USING_CONVEX_HULL")
    ),
    "POPULATION_SIZE": _parse_int(os.getenv("POPULATION_SIZE")),
    "MUTATION_PROBABILITY": _parse_float(os.getenv("MUTATION_PROBABILITY")),
    "INCREASE_MUTATION_PROBABILITY_IF_STAGNATION": _parse_bool(
        os.getenv("INCREASE_MUTATION_PROBABILITY_IF_STAGNATION")
    ),
    "STAGNATION_GENERATIONS": _parse_int(os.getenv("STAGNATION_GENERATIONS")),
    "MUTATION_PROBABILITY_MAX": _parse_float(os.getenv("MUTATION_PROBABILITY_MAX")),
    "MUTATION_PROBABILITY_STEP": _parse_float(os.getenv("MUTATION_PROBABILITY_STEP")),
    "JUST_SWAP": _parse_bool(os.getenv("JUST_SWAP")),
}

_config_order = [
    "PLOT_X_OFFSET",
    "WIDTH",
    "HEIGHT",
    "NODE_RADIUS",
    "RENDER_EVERY",
    "CAR_AUTONOMY",
    "ATSP_ENABLED",
    "ASYMMETRY_FACTOR",
    "ASYMMETRY_SEED",
    "ASYMMETRIC_COSTS_FILE",
    "CITIES_LOCATION_FILE",
    "GERAR_CIDADES",
    "NUMBER_OF_CITIES",
    "MAX_GENERATION_ALLOWED",
    "ONLY_RANDOM_POPULATION",
    "RANDOM_POPULATION_PERCENT",
    "GENERATE_POPULATION_USING_NEAREST_NEIGHBOURS",
    "GENERATE_POPULATION_USING_GREEDY_APPROACH",
    "GENERATE_POPULATION_USING_CONVEX_HULL",
    "POPULATION_SIZE",
    "MUTATION_PROBABILITY",
    "INCREASE_MUTATION_PROBABILITY_IF_STAGNATION",
    "STAGNATION_GENERATIONS",
    "MUTATION_PROBABILITY_MAX",
    "MUTATION_PROBABILITY_STEP",
    "JUST_SWAP",
]

_config = dict(_config_defaults)
if "config_overrides" in st.session_state:
    _config.update(st.session_state.config_overrides)

_plot_x_offset = _config["PLOT_X_OFFSET"]
_width = _config["WIDTH"]
_height = _config["HEIGHT"]
_node_radius = _config["NODE_RADIUS"]
_render_every = _config["RENDER_EVERY"]

# Autonomia maxima do carro (distancia em pixels). Use None para desativar.
_car_autonomy = _config["CAR_AUTONOMY"]

_atsp_enabled = _config["ATSP_ENABLED"]
_asymmetry_factor = _config["ASYMMETRY_FACTOR"]
_asymmetry_seed = _config["ASYMMETRY_SEED"]
_asymmetric_costs_file = _config["ASYMMETRIC_COSTS_FILE"]
_cities_location_file = _config["CITIES_LOCATION_FILE"]

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

    with open(_asymmetric_costs_file, "w", encoding="utf-8") as costs_file:
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

_gerar_cidades = _config["GERAR_CIDADES"]
_number_of_cities = _config["NUMBER_OF_CITIES"]
_max_generation_allowed = _config["MAX_GENERATION_ALLOWED"]
_only_random_population = _config["ONLY_RANDOM_POPULATION"]

_random_population_percent = _config["RANDOM_POPULATION_PERCENT"]
if _random_population_percent < 0:
    _random_population_percent = 0
elif _random_population_percent > 1:
    _random_population_percent = 1
_config["RANDOM_POPULATION_PERCENT"] = _random_population_percent

_generate_population_using_nearest_neighbours = _config[
    "GENERATE_POPULATION_USING_NEAREST_NEIGHBOURS"
]
_generate_population_using_greedy_approach = _config[
    "GENERATE_POPULATION_USING_GREEDY_APPROACH"
]
_generate_population_using_convex_hull = _config[
    "GENERATE_POPULATION_USING_CONVEX_HULL"
]

_population_size = _config["POPULATION_SIZE"]
_mutation_probability = _config["MUTATION_PROBABILITY"]
_increase_mutation_probability_if_stagnation = _config[
    "INCREASE_MUTATION_PROBABILITY_IF_STAGNATION"
]
_stagnation_generations = _config["STAGNATION_GENERATIONS"]
_mutation_probability_max = _config["MUTATION_PROBABILITY_MAX"]
_mutation_probability_step = _config["MUTATION_PROBABILITY_STEP"]
_just_swap = _config["JUST_SWAP"]

# 1. Guardar hora de inicio
hora_inicio = datetime.now()

# Se for gerar cidades ele verifica a quantidade a ser gerada em NUMBER_OF_CITIES e ao final guarda um 
# arquivo texto definido em CITIES_LOCATION_FILE, na sequencia verifica se deve usar matriz de assimatria 
# em ATSP_ENABLED e gera a matriz e também guarda num arquivo texto definido em  ASYMMETRIC_COSTS_FILE
if _gerar_cidades:
    cities_locations = [
        (random.randint(_node_radius + _plot_x_offset, _width - _node_radius), random.randint(_node_radius, _height - _node_radius))
        for _ in range(_number_of_cities)
    ]

    with open(_cities_location_file, "w", encoding="utf-8") as cities_file:
        for x, y in cities_locations:
            cities_file.write(f"{x},{y}\n")

    if _atsp_enabled:
        asymmetric_costs = load_asymmetric_cost_matrix(
            _asymmetric_costs_file,
            expected_size=len(cities_locations),
        )
        if asymmetric_costs is None:
            asymmetric_costs = build_asymmetric_cost_matrix(
                cities_locations,
                asymmetry_factor = _asymmetry_factor,
                seed = _asymmetry_seed,
            )
        set_asymmetric_costs(cities_locations, asymmetric_costs)
else:
    # aqui ele irá ler os 2 arquivos CITIES_LOCATION_FILE e ASYMMETRIC_COSTS_FILE.
    with open(_cities_location_file, "r", encoding="utf-8") as cities_file:
        cities_locations = []
        for line in cities_file:
            line = line.strip()
            if not line:
                continue
            x_str, y_str = line.split(",", 1)
            cities_locations.append((int(x_str), int(y_str)))
    
    if _atsp_enabled:
        asymmetric_costs = load_asymmetric_cost_matrix(
            _asymmetric_costs_file,
            expected_size=len(cities_locations),
        )
        if asymmetric_costs is None:
            asymmetric_costs = build_asymmetric_cost_matrix(
                cities_locations,
                asymmetry_factor = _asymmetry_factor,
                seed = _asymmetry_factor,
            )
        set_asymmetric_costs(cities_locations, asymmetric_costs)

if _car_autonomy is not None:
    set_car_autonomy(_car_autonomy, reference_city=cities_locations[0])
else:
    clear_car_autonomy()


# Criação da população inicial, pode ser totalmente randomica ou um pouco randomico e o restante para chegar 
# em 100% pode ser: NEAREST_NEIGHBOURS, GREEDY_APPROACH ou CONVEX_HULL 
if _only_random_population:
    _random_population_percent = 1

population = generate_random_population(cities_locations, int(round(_population_size * _random_population_percent, 0)))
initial_population_method = "random"

if _only_random_population == False:
    if _generate_population_using_nearest_neighbours:
        population = generate__population_using_Nearest_Neighbours(cities_locations, int(round(_population_size * (1 - _random_population_percent), 0)))
        initial_population_method = "nearest_neighbours"
    elif _generate_population_using_greedy_approach:
        population = generate__population_using_greddy_approach(cities_locations, int(round(_population_size * (1 - _random_population_percent), 0)))
        initial_population_method = "greedy"
    elif _generate_population_using_convex_hull:
        generate__population_using_convex_hull(cities_locations, int(round(_population_size * (1 - _random_population_percent), 0)))
        initial_population_method = "convex hull"

best_fitness_values = []
best_solutions = []

current_mutation_probability = _mutation_probability
best_fitness_so_far = None
stagnation_count = 0
# para logar qual geracao acho melhor valor
choosen_generation = 0
old_best_solution = 0

status_placeholder = st.empty()
plot_col, map_col = st.columns([1, 2])
fitness_placeholder = plot_col.empty()
map_tab, heatmap_tab, settings_tab = map_col.tabs(["Mapa", "Mapa de calor", "Configuracoes"])
map_placeholder = map_tab.empty()
progress = st.progress(0)

ref_city = cities_locations[0] if cities_locations else None
with st.sidebar:
    st.subheader("Parametros de rota")
    if _car_autonomy is None:
        st.write("Autonomia do carro: desativada")
    else:
        st.write(f"Autonomia do carro: {_car_autonomy:.2f} px")
    if ref_city is None:
        st.write("Cidade de referencia: N/A")
    else:
        st.write(f"Cidade de referencia: {ref_city}")

with settings_tab:
    st.subheader("Configuracoes do .env")
    st.caption("Edite os valores e clique em Aplicar para recarregar.")
    edited_values = {}
    with st.form("env_config_form"):
        cols = st.columns(2)
        for index, key in enumerate(_config_order):
            value = _config.get(key)
            default = _config_defaults.get(key)
            col = cols[index % 2]
            with col:
                if isinstance(default, bool):
                    edited_values[key] = st.checkbox(key, value=bool(value))
                else:
                    display_value = "" if value is None else str(value)
                    edited_values[key] = st.text_input(key, value=display_value)
        submitted = st.form_submit_button("Aplicar configuracoes")
    if submitted:
        overrides = {}
        for key, value in edited_values.items():
            overrides[key] = _coerce_config_value(value, _config_defaults.get(key))
        st.session_state.config_overrides = overrides
        st.experimental_rerun()

if _atsp_enabled and os.path.exists(_asymmetric_costs_file):
    with heatmap_tab:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(asymmetric_costs, cmap="viridis")
        fig.colorbar(im, ax=ax)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

# Main loop
for generation in range(1, _max_generation_allowed + 1):
    population_fitness = [calculate_fitness(individual) for individual in population]
    population, population_fitness = sort_population(population, population_fitness)

    best_solution = population[0]
    best_fitness = population_fitness[0]

    best_fitness_values.append(best_fitness)
    best_solutions.append(best_solution)

    if best_fitness_so_far is None or best_fitness < best_fitness_so_far:
        best_fitness_so_far = best_fitness
        stagnation_count = 0
        current_mutation_probability = _mutation_probability
    else:
        if _increase_mutation_probability_if_stagnation:
            stagnation_count += 1
            if stagnation_count >= _stagnation_generations:
                current_mutation_probability = min(
                    _mutation_probability_max,
                    current_mutation_probability + _mutation_probability_step,
                )
                stagnation_count = 0

    if old_best_solution == 0:
        old_best_solution = best_solution

    if best_solution < old_best_solution:
        old_best_solution = best_solution
        choosen_generation = generation

    # Keep the best individual: ELITISM
    new_population = [population[0]]

    while len(new_population) < (_population_size * 0.5):
        # solution based on fitness probability
        probability = 1 / np.array(population_fitness)
        parent1, parent2 = random.choices(population, weights=probability, k=2)

        child1 = order_crossover(parent1, parent2)
        child1 = mutate(child1, current_mutation_probability, _just_swap)

        new_population.append(child1)

    while len(new_population) < (_population_size * 0.5):
        # tournament selection
        parent1, _ = tournament_selection(population, calculate_fitness, tournament_size=3, minimize=True)
        parent2, _ = tournament_selection(population, calculate_fitness, tournament_size=3, minimize=True)

        child1 = order_crossover(parent1, parent2)
        child1 = mutate(child1, current_mutation_probability, _just_swap)

        new_population.append(child1)

    population = new_population

    if generation == 1 or generation == _max_generation_allowed or generation % _render_every == 0:
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
            node_radius=_node_radius,
            reference_city=cities_locations[0] if cities_locations else None,
            width=_width,
            height=_height,
            x_offset=_plot_x_offset,
        )
        map_placeholder.pyplot(solution_fig, width="stretch")
        plt.close(solution_fig)

        progress.progress(int(generation / _max_generation_allowed * 100))


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
    mutation_probability=current_mutation_probability,
    random_population_percent=_random_population_percent,
    initial_population_method=initial_population_method,
    output_path=report_output_path,
))

report_text = generate_report()
st.text(report_text)
