import random
import os
import csv
import io
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
st.markdown(
    """
<style>
:root {
  --bg: #0b0b0c;
  --panel: #111216;
  --panel-2: #1b1d23;
  --panel-3: #22252c;
  --border: #2a2d36;
  --text: #f3f4f6;
  --muted: #a6adbb;
  --accent: #ff4b4b;
}
div[data-testid="stAppViewContainer"],
section.main > div {
  background: var(--bg);
  color: var(--text);
}
div[data-testid="stHeader"] { background: transparent; }
div[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
}
.priority-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 18px 14px 18px;
}
.priorities-wrap {
  margin-left: 0;
  padding-left: 0;
}
.priority-id {
  font-weight: 600;
  margin: 8px 0 12px 0;
}
.priority-label {
  color: var(--muted);
  font-size: 0.85rem;
  margin-bottom: 6px;
}
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextArea"] textarea {
  background: var(--panel-2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}
div[data-testid="stNumberInput"] button {
  color: var(--text) !important;
}
button[kind="primary"] {
  background: var(--panel-3) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
}
</style>
    """,
    unsafe_allow_html=True,
)


def _safe_rerun():
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_fn is not None:
        rerun_fn()


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


def _build_priority_rules(priority_definitions):
    rules = {}
    for item in priority_definitions:
        priority_id = item.get("id")
        if not priority_id:
            continue
        rules[priority_id] = {
            "id": priority_id,
            "label": item.get("label") or priority_id,
            "description": item.get("description") or "",
            "weight_multiplier": _parse_float(item.get("weight_multiplier"), default=1.0),
            "penalty_per_km": _parse_float(item.get("penalty_per_km"), default=1.0),
            "max_delay_min": _parse_int(item.get("max_delay_min"), default=0),
        }
    return rules


def _parse_city_priority_csv(uploaded_file, cities_locations, priority_rules):
    overrides = {}
    errors = []
    if uploaded_file is None:
        return overrides, errors

    try:
        text = uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        text = uploaded_file.getvalue().decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        errors.append("CSV sem cabecalho.")
        return overrides, errors

    coord_index = {city: idx for idx, city in enumerate(cities_locations)}
    priority_ids = set(priority_rules.keys())

    for row_number, row in enumerate(reader, start=2):
        if not any((value or "").strip() for value in row.values()):
            continue

        priority = (
            row.get("priority")
            or row.get("prioridade")
            or row.get("priority_id")
            or row.get("prioridade_id")
            or ""
        ).strip()
        if priority and priority not in priority_ids:
            errors.append(f"Linha {row_number}: prioridade desconhecida '{priority}'.")
            continue

        idx = None
        for key in ("index", "idx", "city_index", "cidade_index"):
            raw = row.get(key)
            if raw is not None and str(raw).strip() != "":
                try:
                    idx = int(float(raw))
                except ValueError:
                    errors.append(f"Linha {row_number}: index invalido '{raw}'.")
                break

        if idx is None:
            x_val = row.get("x") or row.get("city_x") or row.get("lon") or row.get("longitude")
            y_val = row.get("y") or row.get("city_y") or row.get("lat") or row.get("latitude")
            if (x_val is None or y_val is None) and row.get("city"):
                parts = row.get("city").split(",", 1)
                if len(parts) == 2:
                    x_val, y_val = parts[0].strip(), parts[1].strip()
            if x_val is not None and y_val is not None:
                try:
                    coord = (int(float(x_val)), int(float(y_val)))
                except ValueError:
                    coord = (float(x_val), float(y_val))
                idx = coord_index.get(coord)
                if idx is None:
                    try:
                        coord = (float(x_val), float(y_val))
                    except ValueError:
                        coord = None
                    if coord is not None:
                        idx = coord_index.get(coord)

        if idx is None:
            errors.append(f"Linha {row_number}: cidade nao encontrada.")
            continue
        if idx < 0 or idx >= len(cities_locations):
            errors.append(f"Linha {row_number}: index fora do intervalo ({idx}).")
            continue

        if not priority:
            priority = next(iter(priority_ids), None)
        overrides[idx] = priority

    return overrides, errors


def _build_city_priority_csv(cities_locations, city_overrides, default_priority_id):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["index", "x", "y", "priority"])
    for idx, city in enumerate(cities_locations):
        priority_id = city_overrides.get(idx, default_priority_id)
        writer.writerow([idx, city[0], city[1], priority_id])
    return output.getvalue()

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

DELIVERY_PRIORITIES = [
    {
        "id": "critical_meds",
        "label": "Medicamentos criticos",
        "description": "Urgencia alta, penalidade maior por atrasos.",
        "weight_multiplier": 2.0,
        "penalty_per_km": 1.5,
        "max_delay_min": 30,
    },
    {
        "id": "regular_supplies",
        "label": "Insumos regulares",
        "description": "Urgencia normal, penalidade padrao.",
        "weight_multiplier": 1.0,
        "penalty_per_km": 1.0,
        "max_delay_min": 240,
    },
]

_priority_definitions = st.session_state.get("priority_overrides", DELIVERY_PRIORITIES)
_priority_rules = _build_priority_rules(_priority_definitions)
_priority_ids = list(_priority_rules.keys())
_default_priority_id = _priority_ids[0] if _priority_ids else None

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

_city_priority_overrides = st.session_state.get("city_priority_overrides", {})
_city_priorities = []
for i, city in enumerate(cities_locations):
    priority_id = _city_priority_overrides.get(i, _default_priority_id)
    if priority_id not in _priority_rules:
        priority_id = _default_priority_id
    _city_priorities.append(priority_id)

if _priority_rules and _default_priority_id is not None:
    set_delivery_priorities(
        cities_locations,
        _city_priorities,
        _priority_rules,
        default_priority_id=_default_priority_id,
    )
else:
    clear_delivery_priorities()

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

status_placeholder = None
fitness_placeholder = None
map_placeholder = None
progress = None

map_tab, heatmap_tab, settings_tab, priorities_tab = st.tabs(
    ["Mapa", "Mapa de calor", "Configuracoes", "Prioridades"]
)

with map_tab:
    status_placeholder = st.empty()
    plot_col, map_col = st.columns([1, 2])
    fitness_placeholder = plot_col.empty()
    map_placeholder = map_col.empty()
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
    if "run_ga" not in st.session_state:
        st.session_state.run_ga = False
    if st.button("Play", type="primary"):
        st.session_state.run_ga = True

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
        _safe_rerun()

with priorities_tab:
    st.markdown("<div class=\"priorities-wrap\">", unsafe_allow_html=True)
    st.subheader("Prioridades de entrega")
    st.caption("Edite as prioridades e aplique para atualizar a funcao fitness.")
    if not _priority_ids:
        st.warning("Nenhuma prioridade configurada.")
    else:
        updated_definitions = []
        st.markdown("<div class=\"priority-card\">", unsafe_allow_html=True)
        with st.form("priority_definitions_form"):
            for item in _priority_definitions:
                priority_id = item.get("id")
                if not priority_id:
                    continue
                st.markdown(f"<div class=\"priority-id\">{priority_id}</div>", unsafe_allow_html=True)
                label_col, weight_col, penalty_col, delay_col = st.columns(4)
                with label_col:
                    st.markdown("<div class=\"priority-label\">Label</div>", unsafe_allow_html=True)
                    label = st.text_input(
                        "Label",
                        value=str(item.get("label") or priority_id),
                        key=f"priority_label_{priority_id}",
                        label_visibility="collapsed",
                    )
                with weight_col:
                    st.markdown("<div class=\"priority-label\">Peso</div>", unsafe_allow_html=True)
                    weight_multiplier = st.number_input(
                        "Peso",
                        value=float(item.get("weight_multiplier") or 1.0),
                        step=0.1,
                        key=f"priority_weight_{priority_id}",
                        label_visibility="collapsed",
                    )
                with penalty_col:
                    st.markdown("<div class=\"priority-label\">Penalidade por km</div>", unsafe_allow_html=True)
                    penalty_per_km = st.number_input(
                        "Penalidade por km",
                        value=float(item.get("penalty_per_km") or 1.0),
                        step=0.1,
                        key=f"priority_penalty_{priority_id}",
                        label_visibility="collapsed",
                    )
                with delay_col:
                    st.markdown("<div class=\"priority-label\">Max atraso (min)</div>", unsafe_allow_html=True)
                    max_delay_min = st.number_input(
                        "Max atraso (min)",
                        value=int(item.get("max_delay_min") or 0),
                        step=5,
                        key=f"priority_delay_{priority_id}",
                        label_visibility="collapsed",
                    )
                st.markdown("<div class=\"priority-label\">Descricao</div>", unsafe_allow_html=True)
                description = st.text_area(
                    "Descricao",
                    value=str(item.get("description") or ""),
                    key=f"priority_desc_{priority_id}",
                    label_visibility="collapsed",
                    height=70,
                )
                updated_definitions.append(
                    {
                        "id": priority_id,
                        "label": label,
                        "description": description,
                        "weight_multiplier": weight_multiplier,
                        "penalty_per_km": penalty_per_km,
                        "max_delay_min": max_delay_min,
                    }
                )
            priorities_submitted = st.form_submit_button("Aplicar prioridades")
        st.markdown("</div>", unsafe_allow_html=True)
        if priorities_submitted:
            st.session_state.priority_overrides = updated_definitions
            _safe_rerun()

    st.divider()
    st.subheader("Prioridade por cidade")
    if not cities_locations:
        st.info("Nenhuma cidade carregada.")
    else:
        csv_file = st.file_uploader(
            "CSV de prioridades por cidade",
            type=["csv"],
            help="Colunas suportadas: index/idx/city_index ou x,y,priority.",
            key="priority_csv_upload",
        )
        st.caption("Exemplo de CSV:")
        st.code(
            "index,priority\n0,critical_meds\n1,regular_supplies",
            language="csv",
        )
        if st.button("Aplicar CSV", type="secondary"):
            if csv_file is None:
                st.info("Selecione um arquivo CSV.")
            else:
                csv_overrides, csv_errors = _parse_city_priority_csv(
                    csv_file,
                    cities_locations,
                    _priority_rules,
                )
                if csv_errors:
                    st.warning("Erros no CSV:\n- " + "\n- ".join(csv_errors))
                if csv_overrides:
                    merged_overrides = dict(_city_priority_overrides)
                    merged_overrides.update(csv_overrides)
                    st.session_state.city_priority_overrides = merged_overrides
                    _safe_rerun()
                elif not csv_errors:
                    st.info("Nenhuma linha valida encontrada no CSV.")

        with st.expander("Editar manualmente por cidade"):
            updated_city_overrides = {}
            cols = st.columns(2)
            for index, city in enumerate(cities_locations):
                current_priority = _city_priority_overrides.get(index, _default_priority_id)
                if current_priority not in _priority_rules:
                    current_priority = _default_priority_id
                option_index = _priority_ids.index(current_priority)
                col = cols[index % 2]
                with col:
                    selected = st.selectbox(
                        f"Cidade {index} ({city[0]}, {city[1]})",
                        options=_priority_ids,
                        index=option_index,
                        format_func=lambda pid: f"{pid} - {_priority_rules[pid]['label']}",
                        key=f"city_priority_{index}",
                    )
                    updated_city_overrides[index] = selected

            csv_payload = _build_city_priority_csv(
                cities_locations,
                updated_city_overrides,
                _default_priority_id,
            )
            apply_col, download_col = st.columns([1, 1])
            with apply_col:
                cities_submitted = st.button("Aplicar prioridades por cidade")
            with download_col:
                st.download_button(
                    "Gravar CSV prioridades",
                    data=csv_payload,
                    file_name="prioridades_cidades.csv",
                    mime="text/csv",
                )
            if cities_submitted:
                st.session_state.city_priority_overrides = updated_city_overrides
                _safe_rerun()

        priority_summary = {}
        for priority_id in _priority_ids:
            priority_summary[priority_id] = 0
        for priority_id in _city_priorities:
            if priority_id in priority_summary:
                priority_summary[priority_id] += 1
        summary_rows = [
            {
                "Prioridade": f"{pid} - {_priority_rules[pid]['label']}",
                "Cidades": count,
            }
            for pid, count in priority_summary.items()
        ]
        st.table(summary_rows)

        city_rows = [
            {
                "Cidade": index,
                "X": city[0],
                "Y": city[1],
                "Prioridade": _city_priorities[index],
            }
            for index, city in enumerate(cities_locations)
        ]
        st.dataframe(city_rows, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

if _atsp_enabled and os.path.exists(_asymmetric_costs_file):
    with heatmap_tab:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(asymmetric_costs, cmap="viridis")
        fig.colorbar(im, ax=ax)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

# Main loop
if not st.session_state.run_ga:
    st.info("Clique em Play para iniciar o processamento.")
    st.stop()

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
