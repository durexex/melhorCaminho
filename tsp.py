import random
import csv
import io
import os
from pathlib import Path
from dotenv import load_dotenv
from genetic_algorithm import *
from demo_tournament import tournament_selection
from draw_functions import draw_plot, build_solution_figure, build_priority_legend_items
from utils import ReportData, VehicleStats, set_report_data, generate_report
from priority_utils import parse_city_priority_csv, build_city_priority_csv
from llm_service import LLMService
from pdf_service import create_pdf_report
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
.priority-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 6px 0 14px 0;
}
.priority-legend-item {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 999px;
  font-size: 0.9rem;
}
.priority-legend-swatch {
  width: 12px;
  height: 12px;
  border-radius: 999px;
  display: inline-block;
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


def _friendly_llm_error_message(exc: Exception) -> str:
    cause = getattr(exc, "__cause__", None) or exc
    cause_type = type(cause).__name__.lower()
    msg = str(cause).lower()

    if "authentication" in cause_type or "api_key" in msg or "invalid api key" in msg:
        return (
            "Chave da API invalida ou expirada. "
            "Verifique a variavel OPENAI_API_KEY no arquivo .env."
        )
    if "ratelimit" in cause_type or "rate_limit" in msg or "rate limit" in msg:
        return (
            "Limite de requisicoes excedido. "
            "Aguarde alguns minutos e tente novamente."
        )
    if "model_not_found" in msg or "does not exist" in msg:
        current_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        return (
            f"O modelo '{current_model}' nao esta disponivel para sua chave de API. "
            "Altere a variavel LLM_MODEL no arquivo .env para um modelo acessivel "
            "(ex: gpt-4o-mini, gpt-3.5-turbo)."
        )
    if "connection" in cause_type or "connect" in msg:
        return (
            "Erro de conexao com a API da OpenAI. "
            "Verifique sua conexao com a internet e tente novamente."
        )
    return f"Erro inesperado: {exc}"


@st.cache_resource
def _get_llm_service():
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if not api_key or api_key == "sk-your-api-key-here":
        return None
    return LLMService(api_key=api_key, model=model)


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
            "latest_position": _parse_int(item.get("latest_position"), default=None),
            "service_time_min": _parse_float(item.get("service_time_min"), default=0.0),
            "time_window_start_min": _parse_float(item.get("time_window_start_min"), default=None),
            "time_window_end_min": _parse_float(item.get("time_window_end_min"), default=None),
            "temperature_penalty_per_km": _parse_float(
                item.get("temperature_penalty_per_km"), default=0.0
            ),
            "sequence_penalty": _parse_float(item.get("sequence_penalty"), default=0.0),
        }
    return rules


_PRIORITY_RULE_FIELDS = (
    "label",
    "description",
    "weight_multiplier",
    "penalty_per_km",
    "max_delay_min",
    "latest_position",
    "service_time_min",
    "time_window_start_min",
    "time_window_end_min",
    "temperature_penalty_per_km",
    "sequence_penalty",
)


def _normalize_priority_definitions(priority_definitions):
    incoming = {}
    for item in priority_definitions or []:
        priority_id = item.get("id")
        if priority_id:
            incoming[priority_id] = item

    normalized = []
    for default_item in DELIVERY_PRIORITIES:
        merged = dict(default_item)
        override = incoming.get(default_item["id"], {})
        for key in _PRIORITY_RULE_FIELDS:
            if key in override and override.get(key) not in (None, ""):
                merged[key] = override[key]
        normalized.append(merged)
    return normalized


def _format_time_window(rule):
    start = rule.get("time_window_start_min")
    end = rule.get("time_window_end_min")
    if start is None and end is None:
        return "Sem janela"
    start_txt = "-" if start is None else f"{int(start)} min"
    end_txt = "-" if end is None else f"{int(end)} min"
    return f"{start_txt} -> {end_txt}"


def _format_priority_service_hint(rule):
    parts = [f"SLA {int(rule.get('max_delay_min', 0))} min"]

    latest_position = rule.get("latest_position")
    if latest_position:
        parts.append(f"ate parada {int(latest_position)}")

    service_time = rule.get("service_time_min")
    if service_time:
        parts.append(f"protocolo {int(service_time)} min")

    if rule.get("temperature_penalty_per_km"):
        parts.append("cadeia fria")

    if rule.get("time_window_start_min") is not None or rule.get("time_window_end_min") is not None:
        parts.append(_format_time_window(rule))

    return " | ".join(parts)


def _render_priority_legend(priority_ids, priority_rules):
    if not priority_ids:
        return

    legend_items = build_priority_legend_items(
        priority_ids,
        priority_labels={pid: priority_rules[pid]["label"] for pid in priority_ids if pid in priority_rules},
    )

    chips = []
    for item in legend_items:
        r, g, b = item["color"]
        chips.append(
            (
                "<span class=\"priority-legend-item\">"
                f"<span class=\"priority-legend-swatch\" style=\"background: rgb({r}, {g}, {b});\"></span>"
                f"{item['label']}"
                "</span>"
            )
        )

    chips.append(
        "<span class=\"priority-legend-item\"><strong>X</strong> Cidade inicial</span>"
    )

    st.markdown("**Legenda de prioridades**", unsafe_allow_html=True)
    st.markdown(
        f"<div class=\"priority-legend\">{''.join(chips)}</div>",
        unsafe_allow_html=True,
    )



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
    "NUM_VEHICLES": _parse_int(os.getenv("NUM_VEHICLES"), default=1),
    "VEHICLE_CAPACITY_WEIGHT_KG": _parse_float(os.getenv("VEHICLE_CAPACITY_WEIGHT_KG")),
    "VEHICLE_CAPACITY_VOLUME_M3": _parse_float(os.getenv("VEHICLE_CAPACITY_VOLUME_M3")),
    "DEMAND_WEIGHT_MIN": _parse_float(os.getenv("DEMAND_WEIGHT_MIN"), default=1.0),
    "DEMAND_WEIGHT_MAX": _parse_float(os.getenv("DEMAND_WEIGHT_MAX"), default=50.0),
    "DEMAND_VOLUME_MIN": _parse_float(os.getenv("DEMAND_VOLUME_MIN"), default=0.01),
    "DEMAND_VOLUME_MAX": _parse_float(os.getenv("DEMAND_VOLUME_MAX"), default=2.0),
    "CITIES_DEMAND_FILE": _parse_str(os.getenv("CITIES_DEMAND_FILE"), default="cities_demand.csv"),
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
    "CITIES_DEMAND_FILE",
]

_vrp_config_keys = [
    "NUM_VEHICLES",
    "VEHICLE_CAPACITY_WEIGHT_KG",
    "VEHICLE_CAPACITY_VOLUME_M3",
    "DEMAND_WEIGHT_MIN",
    "DEMAND_WEIGHT_MAX",
    "DEMAND_VOLUME_MIN",
    "DEMAND_VOLUME_MAX",
]

DELIVERY_PRIORITIES = [
    {
        "id": "emergencia_obstetrica",
        "label": "Emergencias obstetricas",
        "description": "Prioridade maxima. Deve ser atendida no inicio da rota e com o menor atraso possivel.",
        "weight_multiplier": 3.8,
        "penalty_per_km": 3.0,
        "max_delay_min": 20,
        "latest_position": 2,
        "service_time_min": 8.0,
        "time_window_start_min": None,
        "time_window_end_min": None,
        "temperature_penalty_per_km": 0.0,
        "sequence_penalty": 140.0,
    },
    {
        "id": "violencia_domestica",
        "label": "Casos de violencia domestica",
        "description": "Protocolos especiais. O algoritmo antecipa a visita e reserva mais tempo operacional por atendimento.",
        "weight_multiplier": 3.2,
        "penalty_per_km": 2.6,
        "max_delay_min": 35,
        "latest_position": 4,
        "service_time_min": 20.0,
        "time_window_start_min": None,
        "time_window_end_min": None,
        "temperature_penalty_per_km": 0.0,
        "sequence_penalty": 110.0,
    },
    {
        "id": "medicamentos_hormonais",
        "label": "Medicamentos hormonais",
        "description": "Carga com temperatura controlada. Penaliza trajetos longos ate a entrega para preservar cadeia fria.",
        "weight_multiplier": 2.6,
        "penalty_per_km": 2.2,
        "max_delay_min": 60,
        "latest_position": 6,
        "service_time_min": 10.0,
        "time_window_start_min": None,
        "time_window_end_min": None,
        "temperature_penalty_per_km": 3.5,
        "sequence_penalty": 75.0,
    },
    {
        "id": "atendimento_pos_parto",
        "label": "Atendimento pos-parto",
        "description": "Janela de atendimento especifica. A rota procura encaixar a visita dentro da faixa operacional planejada.",
        "weight_multiplier": 2.0,
        "penalty_per_km": 1.8,
        "max_delay_min": 120,
        "latest_position": 8,
        "service_time_min": 15.0,
        "time_window_start_min": 45.0,
        "time_window_end_min": 120.0,
        "temperature_penalty_per_km": 0.0,
        "sequence_penalty": 55.0,
    },
]

_priority_definitions = _normalize_priority_definitions(
    st.session_state.get("priority_overrides", DELIVERY_PRIORITIES)
)
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

def _generate_city_demands(cities_locs, city_prios, prio_rules, cfg):
    """Generate random demand (weight/volume) per city, modulated by priority.

    Higher-priority items (higher weight_multiplier) produce lighter/smaller
    packages, matching the PRD requirement that critical meds are lighter.
    """
    w_min = cfg["DEMAND_WEIGHT_MIN"]
    w_max = cfg["DEMAND_WEIGHT_MAX"]
    v_min = cfg["DEMAND_VOLUME_MIN"]
    v_max = cfg["DEMAND_VOLUME_MAX"]
    w_range = w_max - w_min
    v_range = v_max - v_min

    demands = []
    for i in range(len(cities_locs)):
        prio_id = city_prios[i] if i < len(city_prios) else None
        rule = prio_rules.get(prio_id, {}) if prio_id and prio_rules else {}

        wm = float(rule.get("weight_multiplier", 1.0)) if rule else 1.0
        fraction = min(1.0, max(0.2, 1.0 / wm))

        eff_w_max = w_min + fraction * w_range
        eff_v_max = v_min + fraction * v_range

        weight = round(random.uniform(w_min, eff_w_max), 2)
        volume = round(random.uniform(v_min, eff_v_max), 4)
        demands.append({"weight": weight, "volume": volume})
    return demands


def _save_demands_csv(filepath, cities_locs, city_prios, demands):
    """Persist city demand data to CSV for reproducibility."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "x", "y", "priority", "weight_kg", "volume_m3"])
        for i, city in enumerate(cities_locs):
            prio = city_prios[i] if i < len(city_prios) else ""
            d = demands[i] if i < len(demands) else {"weight": 0, "volume": 0}
            writer.writerow([i, city[0], city[1], prio, d["weight"], d["volume"]])


def _load_demands_csv(filepath, expected_count):
    """Load demand data from CSV. Returns list of dicts or None on failure."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            demands = []
            for row in reader:
                weight = float(row.get("weight_kg", 0))
                volume = float(row.get("volume_m3", 0))
                demands.append({"weight": round(weight, 2), "volume": round(volume, 4)})
            if len(demands) != expected_count:
                return None
            return demands
    except (FileNotFoundError, ValueError, KeyError):
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

_num_vehicles = _config["NUM_VEHICLES"]
_vehicle_capacity_weight = _config["VEHICLE_CAPACITY_WEIGHT_KG"]
_vehicle_capacity_volume = _config["VEHICLE_CAPACITY_VOLUME_M3"]
_demand_weight_min = _config["DEMAND_WEIGHT_MIN"]
_demand_weight_max = _config["DEMAND_WEIGHT_MAX"]
_demand_volume_min = _config["DEMAND_VOLUME_MIN"]
_demand_volume_max = _config["DEMAND_VOLUME_MAX"]
_cities_demand_file = _config["CITIES_DEMAND_FILE"]

# 1. Guardar hora de inicio
hora_inicio = datetime.now()

# Se for gerar cidades ele verifica a quantidade a ser gerada em NUMBER_OF_CITIES e ao final guarda um 
# arquivo texto definido em CITIES_LOCATION_FILE, na sequencia verifica se deve usar matriz de assimatria 
# em ATSP_ENABLED e gera a matriz e também guarda num arquivo texto definido em  ASYMMETRIC_COSTS_FILE
if _gerar_cidades:
    _cached = st.session_state.get("_cached_cities")
    if _cached is not None and len(_cached) == _number_of_cities:
        cities_locations = list(_cached)
    else:
        cities_locations = [
            (random.randint(_node_radius + _plot_x_offset, _width - _node_radius), random.randint(_node_radius, _height - _node_radius))
            for _ in range(_number_of_cities)
        ]
        st.session_state["_cached_cities"] = list(cities_locations)
        st.session_state.pop("_cached_demands", None)
        st.session_state.pop("_cached_asymmetric_costs", None)

    with open(_cities_location_file, "w", encoding="utf-8") as cities_file:
        for x, y in cities_locations:
            cities_file.write(f"{x},{y}\n")

    if _atsp_enabled:
        _cached_ac = st.session_state.get("_cached_asymmetric_costs")
        if _cached_ac is not None and len(_cached_ac) == len(cities_locations):
            asymmetric_costs = _cached_ac
        else:
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
            st.session_state["_cached_asymmetric_costs"] = asymmetric_costs
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

if _gerar_cidades:
    _cached_dem = st.session_state.get("_cached_demands")
    if _cached_dem is not None and len(_cached_dem) == len(cities_locations):
        _city_demands = _cached_dem
    else:
        _city_demands = _generate_city_demands(
            cities_locations, _city_priorities, _priority_rules, _config,
        )
        st.session_state["_cached_demands"] = list(_city_demands)
    _save_demands_csv(_cities_demand_file, cities_locations, _city_priorities, _city_demands)
else:
    _city_demands = _load_demands_csv(_cities_demand_file, len(cities_locations))
    if _city_demands is None:
        _city_demands = _generate_city_demands(
            cities_locations, _city_priorities, _priority_rules, _config,
        )
        _save_demands_csv(_cities_demand_file, cities_locations, _city_priorities, _city_demands)

set_city_demands(cities_locations, _city_demands)
set_vehicle_params(
    num_vehicles=_num_vehicles,
    capacity_weight=_vehicle_capacity_weight,
    capacity_volume=_vehicle_capacity_volume,
    depot=cities_locations[0] if cities_locations else None,
)

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

map_tab, heatmap_tab, settings_tab, priorities_tab, ai_tab = st.tabs(
    ["Mapa", "Mapa de calor", "Configuracoes", "Prioridades", "Assistente IA"]
)

with map_tab:
    status_placeholder = st.empty()
    _render_priority_legend(_priority_ids, _priority_rules)
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

    st.divider()
    st.subheader("VRP / Capacidade")
    st.write(f"Veiculos: **{_num_vehicles}**")
    if _vehicle_capacity_weight is not None:
        st.write(f"Peso max: **{_vehicle_capacity_weight:.1f} kg**")
    else:
        st.write("Peso max: **desativado**")
    if _vehicle_capacity_volume is not None:
        st.write(f"Volume max: **{_vehicle_capacity_volume:.2f} m3**")
    else:
        st.write("Volume max: **desativado**")
    st.caption("Edite na aba Configuracoes.")

    st.divider()
    if "run_ga" not in st.session_state:
        st.session_state.run_ga = False
    if st.button("Play", type="primary"):
        st.session_state.run_ga = True

with settings_tab:
    # --- VRP / Frota ---
    st.subheader("VRP / Frota e Capacidade")
    st.caption("Configure veiculos e capacidade de carga. Clique Aplicar VRP para recarregar.")
    _vrp_edited = {}
    with st.form("vrp_config_form"):
        vrp_c1, vrp_c2, vrp_c3 = st.columns(3)
        with vrp_c1:
            _vrp_edited["NUM_VEHICLES"] = st.number_input(
                "Numero de veiculos",
                min_value=1,
                max_value=20,
                value=int(_num_vehicles),
                step=1,
                key="vrp_num_vehicles",
            )
        with vrp_c2:
            _cap_w_disabled = st.checkbox(
                "Desativar limite de peso",
                value=(_vehicle_capacity_weight is None),
                key="vrp_cap_weight_off",
            )
            if _cap_w_disabled:
                _vrp_edited["VEHICLE_CAPACITY_WEIGHT_KG"] = None
                st.number_input(
                    "Capacidade peso (kg)",
                    min_value=1.0,
                    value=500.0,
                    step=10.0,
                    disabled=True,
                    key="vrp_cap_weight_disabled",
                )
            else:
                _vrp_edited["VEHICLE_CAPACITY_WEIGHT_KG"] = st.number_input(
                    "Capacidade peso (kg)",
                    min_value=1.0,
                    value=float(_vehicle_capacity_weight or 500.0),
                    step=10.0,
                    key="vrp_cap_weight",
                )
        with vrp_c3:
            _cap_v_disabled = st.checkbox(
                "Desativar limite de volume",
                value=(_vehicle_capacity_volume is None),
                key="vrp_cap_vol_off",
            )
            if _cap_v_disabled:
                _vrp_edited["VEHICLE_CAPACITY_VOLUME_M3"] = None
                st.number_input(
                    "Capacidade volume (m3)",
                    min_value=0.01,
                    value=10.0,
                    step=0.5,
                    disabled=True,
                    key="vrp_cap_vol_disabled",
                )
            else:
                _vrp_edited["VEHICLE_CAPACITY_VOLUME_M3"] = st.number_input(
                    "Capacidade volume (m3)",
                    min_value=0.01,
                    value=float(_vehicle_capacity_volume or 10.0),
                    step=0.5,
                    key="vrp_cap_vol",
                )

        st.markdown("**Faixas de demanda por cidade**")
        dem_c1, dem_c2, dem_c3, dem_c4 = st.columns(4)
        with dem_c1:
            _vrp_edited["DEMAND_WEIGHT_MIN"] = st.number_input(
                "Peso min (kg)", min_value=0.0,
                value=float(_config.get("DEMAND_WEIGHT_MIN", 1.0)),
                step=1.0, key="vrp_dw_min",
            )
        with dem_c2:
            _vrp_edited["DEMAND_WEIGHT_MAX"] = st.number_input(
                "Peso max (kg)", min_value=0.1,
                value=float(_config.get("DEMAND_WEIGHT_MAX", 50.0)),
                step=5.0, key="vrp_dw_max",
            )
        with dem_c3:
            _vrp_edited["DEMAND_VOLUME_MIN"] = st.number_input(
                "Volume min (m3)", min_value=0.0,
                value=float(_config.get("DEMAND_VOLUME_MIN", 0.01)),
                step=0.01, format="%.2f", key="vrp_dv_min",
            )
        with dem_c4:
            _vrp_edited["DEMAND_VOLUME_MAX"] = st.number_input(
                "Volume max (m3)", min_value=0.01,
                value=float(_config.get("DEMAND_VOLUME_MAX", 2.0)),
                step=0.5, key="vrp_dv_max",
            )

        vrp_submitted = st.form_submit_button("Aplicar VRP")
    if vrp_submitted:
        overrides = st.session_state.get("config_overrides", dict(_config))
        for k, v in _vrp_edited.items():
            overrides[k] = v
        st.session_state.config_overrides = overrides
        _safe_rerun()

    st.divider()

    # --- Parametros gerais ---
    st.subheader("Parametros gerais")
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
        overrides = st.session_state.get("config_overrides", dict(_config))
        for key, value in edited_values.items():
            overrides[key] = _coerce_config_value(value, _config_defaults.get(key))
        st.session_state.config_overrides = overrides
        _safe_rerun()

with priorities_tab:
    st.markdown("<div class=\"priorities-wrap\">", unsafe_allow_html=True)
    st.subheader("Prioridades de entrega")
    st.caption(
        "Catalogo clinico com 4 classes fixas. A fitness agora considera urgencia, ordem de atendimento, "
        "SLA maximo, janela de tempo e penalidade de cadeia fria."
    )
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
                order_col, service_col, temp_col, seq_col = st.columns(4)
                with order_col:
                    st.markdown("<div class=\"priority-label\">Ultima parada</div>", unsafe_allow_html=True)
                    latest_position = st.number_input(
                        "Ultima parada",
                        value=int(item.get("latest_position") or 0),
                        min_value=0,
                        step=1,
                        key=f"priority_latest_position_{priority_id}",
                        label_visibility="collapsed",
                    )
                with service_col:
                    st.markdown("<div class=\"priority-label\">Tempo protocolo (min)</div>", unsafe_allow_html=True)
                    service_time_min = st.number_input(
                        "Tempo protocolo (min)",
                        value=float(item.get("service_time_min") or 0.0),
                        min_value=0.0,
                        step=1.0,
                        key=f"priority_service_time_{priority_id}",
                        label_visibility="collapsed",
                    )
                with temp_col:
                    st.markdown("<div class=\"priority-label\">Penalidade cadeia fria</div>", unsafe_allow_html=True)
                    temperature_penalty_per_km = st.number_input(
                        "Penalidade cadeia fria",
                        value=float(item.get("temperature_penalty_per_km") or 0.0),
                        min_value=0.0,
                        step=0.1,
                        key=f"priority_temperature_penalty_{priority_id}",
                        label_visibility="collapsed",
                    )
                with seq_col:
                    st.markdown("<div class=\"priority-label\">Penalidade por posicao</div>", unsafe_allow_html=True)
                    sequence_penalty = st.number_input(
                        "Penalidade por posicao",
                        value=float(item.get("sequence_penalty") or 0.0),
                        min_value=0.0,
                        step=5.0,
                        key=f"priority_sequence_penalty_{priority_id}",
                        label_visibility="collapsed",
                    )
                window_start_col, window_end_col = st.columns(2)
                with window_start_col:
                    st.markdown("<div class=\"priority-label\">Janela inicio (min)</div>", unsafe_allow_html=True)
                    time_window_start_min = st.number_input(
                        "Janela inicio (min)",
                        value=float(item.get("time_window_start_min") or 0.0),
                        min_value=0.0,
                        step=5.0,
                        key=f"priority_window_start_{priority_id}",
                        label_visibility="collapsed",
                    )
                with window_end_col:
                    st.markdown("<div class=\"priority-label\">Janela fim (min)</div>", unsafe_allow_html=True)
                    time_window_end_min = st.number_input(
                        "Janela fim (min)",
                        value=float(item.get("time_window_end_min") or 0.0),
                        min_value=0.0,
                        step=5.0,
                        key=f"priority_window_end_{priority_id}",
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
                        "latest_position": latest_position or None,
                        "service_time_min": service_time_min,
                        "time_window_start_min": time_window_start_min or None,
                        "time_window_end_min": time_window_end_min or None,
                        "temperature_penalty_per_km": temperature_penalty_per_km,
                        "sequence_penalty": sequence_penalty,
                    }
                )
            priorities_submitted = st.form_submit_button("Aplicar prioridades")
        st.markdown("</div>", unsafe_allow_html=True)
        if priorities_submitted:
            st.session_state.priority_overrides = _normalize_priority_definitions(updated_definitions)
            _safe_rerun()

    st.divider()
    st.subheader("Prioridade por cidade")
    st.caption(
        "Associe cada cidade a uma das quatro classes clinicas. Essa classificacao entra diretamente "
        "na ordem da rota e nas penalidades da fitness."
    )
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
            "index,priority\n0,emergencia_obstetrica\n1,medicamentos_hormonais\n2,atendimento_pos_parto",
            language="csv",
        )
        if st.button("Aplicar CSV", type="secondary"):
            if csv_file is None:
                st.info("Selecione um arquivo CSV.")
            else:
                csv_overrides, csv_errors = parse_city_priority_csv(
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
                        format_func=lambda pid: _priority_rules[pid]["label"],
                        key=f"city_priority_{index}",
                    )
                    updated_city_overrides[index] = selected

            csv_payload = build_city_priority_csv(
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
                "Prioridade": _priority_rules[pid]["label"],
                "Cidades": count,
                "Regra operacional": _format_priority_service_hint(_priority_rules[pid]),
            }
            for pid, count in priority_summary.items()
        ]
        st.table(summary_rows)

        st.divider()
        st.subheader("Demandas por cidade")

        _total_demand_weight = sum(
            (d.get("weight", 0) if d else 0) for d in _city_demands
        )
        _total_demand_volume = sum(
            (d.get("volume", 0) if d else 0) for d in _city_demands
        )
        dem_cols = st.columns(4)
        dem_cols[0].metric("Peso total", f"{_total_demand_weight:.2f} kg")
        dem_cols[1].metric("Volume total", f"{_total_demand_volume:.4f} m3")
        if _vehicle_capacity_weight is not None and _num_vehicles > 0:
            fleet_cap_w = _vehicle_capacity_weight * _num_vehicles
            pct_w = (_total_demand_weight / fleet_cap_w * 100) if fleet_cap_w > 0 else 0
            dem_cols[2].metric("Cap. peso frota", f"{fleet_cap_w:.1f} kg", f"{pct_w:.0f}% usado")
        else:
            dem_cols[2].metric("Cap. peso frota", "ilimitada")
        if _vehicle_capacity_volume is not None and _num_vehicles > 0:
            fleet_cap_v = _vehicle_capacity_volume * _num_vehicles
            pct_v = (_total_demand_volume / fleet_cap_v * 100) if fleet_cap_v > 0 else 0
            dem_cols[3].metric("Cap. volume frota", f"{fleet_cap_v:.2f} m3", f"{pct_v:.0f}% usado")
        else:
            dem_cols[3].metric("Cap. volume frota", "ilimitada")

        city_rows = []
        for index, city in enumerate(cities_locations):
            demand = _city_demands[index] if index < len(_city_demands) else {}
            city_rows.append({
                "Cidade": index,
                "X": city[0],
                "Y": city[1],
                "Prioridade": _priority_rules[_city_priorities[index]]["label"],
                "Peso (kg)": demand.get("weight", 0),
                "Volume (m3)": demand.get("volume", 0),
            })
        st.dataframe(city_rows, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

if _atsp_enabled and os.path.exists(_asymmetric_costs_file):
    with heatmap_tab:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(asymmetric_costs, cmap="viridis")
        fig.colorbar(im, ax=ax)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

with ai_tab:
    _route_data = st.session_state.get("last_route_data")

    if _route_data is None:
        st.info(
            "Nenhuma rota otimizada disponivel. "
            "Execute a otimizacao primeiro clicando em **Play**."
        )
    else:
        _llm_svc = _get_llm_service()

        if _llm_svc is None:
            st.warning(
                "Chave da API OpenAI nao configurada. "
                "Defina `OPENAI_API_KEY` no arquivo `.env` para habilitar o Assistente IA."
            )
        else:
            nav_col, eff_col = st.columns(2)

            with nav_col:
                if st.button(
                    "Gerar Instrucoes de Navegacao",
                    key="btn_gen_nav",
                    use_container_width=True,
                ):
                    with st.spinner("Gerando instrucoes de navegacao..."):
                        try:
                            result = _llm_svc.generate_navigation_instructions(_route_data)
                            st.session_state["nav_instructions_result"] = result
                            st.success("Instrucoes geradas com sucesso!")
                        except Exception as exc:
                            st.error(_friendly_llm_error_message(exc))

            with eff_col:
                if st.button(
                    "Gerar Relatorio de Eficiencia",
                    key="btn_gen_eff",
                    use_container_width=True,
                ):
                    with st.spinner("Gerando relatorio de eficiencia..."):
                        try:
                            result = _llm_svc.generate_efficiency_report(_route_data)
                            st.session_state["efficiency_report_result"] = result
                            st.success("Relatorio gerado com sucesso!")
                        except Exception as exc:
                            st.error(_friendly_llm_error_message(exc))

            if "nav_instructions_result" in st.session_state:
                st.divider()
                st.subheader("Instrucoes de Navegacao")
                st.markdown(st.session_state["nav_instructions_result"])
                nav_pdf = create_pdf_report(
                    title="Instrucoes de Navegacao",
                    content=st.session_state["nav_instructions_result"],
                    filename="instrucoes_navegacao.pdf",
                )
                st.download_button(
                    label="Exportar Instrucoes (PDF)",
                    data=nav_pdf,
                    file_name="instrucoes_navegacao.pdf",
                    mime="application/pdf",
                    key="dl_nav_pdf",
                )

            if "efficiency_report_result" in st.session_state:
                st.divider()
                st.subheader("Relatorio de Eficiencia")
                st.markdown(st.session_state["efficiency_report_result"])
                eff_pdf = create_pdf_report(
                    title="Relatorio de Eficiencia",
                    content=st.session_state["efficiency_report_result"],
                    filename="relatorio_eficiencia.pdf",
                )
                st.download_button(
                    label="Exportar Relatorio (PDF)",
                    data=eff_pdf,
                    file_name="relatorio_eficiencia.pdf",
                    mime="application/pdf",
                    key="dl_eff_pdf",
                )

            st.divider()
            st.subheader("Chat Interativo")

            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []

            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if chat_prompt := st.chat_input(
                "Faca uma pergunta sobre a rota...",
                key="ai_chat_input",
            ):
                st.session_state.chat_messages.append(
                    {"role": "user", "content": chat_prompt}
                )
                with st.chat_message("user"):
                    st.markdown(chat_prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Pensando..."):
                        try:
                            chat_reply = _llm_svc.chat_response(
                                chat_prompt, _route_data
                            )
                            st.markdown(chat_reply)
                        except Exception as exc:
                            chat_reply = _friendly_llm_error_message(exc)
                            st.error(chat_reply)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": chat_reply}
                )

if not st.session_state.run_ga:
    if "last_route_data" not in st.session_state:
        st.info("Clique em Play para iniciar o processamento.")
    elif "ga_report_text" in st.session_state:
        with map_tab:
            if "ga_status_text" in st.session_state:
                status_placeholder.markdown(st.session_state["ga_status_text"])
            if "ga_best_fitness_values" in st.session_state:
                _saved_fv = st.session_state["ga_best_fitness_values"]
                _fit_fig = draw_plot(
                    list(range(len(_saved_fv))),
                    _saved_fv,
                    y_label="Fitness - Distance (pxls)",
                )
                fitness_placeholder.pyplot(_fit_fig, width="stretch")
                plt.close(_fit_fig)
            if "ga_best_solution" in st.session_state:
                _saved_vrp_routes = st.session_state.get("ga_vrp_routes")
                if _saved_vrp_routes:
                    _sol_fig = build_solution_figure(
                        cities_locations,
                        routes=_saved_vrp_routes,
                        node_radius=_node_radius,
                        city_priority_ids=_city_priorities,
                        priority_labels={pid: data["label"] for pid, data in _priority_rules.items()},
                        priority_order=_priority_ids,
                        reference_city=cities_locations[0] if cities_locations else None,
                        width=_width,
                        height=_height,
                        x_offset=_plot_x_offset,
                    )
                else:
                    _sol_fig = build_solution_figure(
                        cities_locations,
                        st.session_state["ga_best_solution"],
                        candidate_path=None,
                        node_radius=_node_radius,
                        city_priority_ids=_city_priorities,
                        priority_labels={pid: data["label"] for pid, data in _priority_rules.items()},
                        priority_order=_priority_ids,
                        reference_city=cities_locations[0] if cities_locations else None,
                        width=_width,
                        height=_height,
                        x_offset=_plot_x_offset,
                    )
                map_placeholder.pyplot(_sol_fig, width="stretch")
                plt.close(_sol_fig)
            progress.progress(100)
        st.text(st.session_state["ga_report_text"])
    st.stop()

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

        _vrp_active = _num_vehicles > 1 or _vehicle_capacity_weight is not None or _vehicle_capacity_volume is not None
        if _vrp_active:
            _best_routes = evaluate_vrp_routes(best_solution)
            solution_fig = build_solution_figure(
                cities_locations,
                routes=_best_routes,
                node_radius=_node_radius,
                city_priority_ids=_city_priorities,
                priority_labels={pid: data["label"] for pid, data in _priority_rules.items()},
                priority_order=_priority_ids,
                reference_city=cities_locations[0] if cities_locations else None,
                width=_width,
                height=_height,
                x_offset=_plot_x_offset,
            )
        else:
            candidate_path = population[1] if len(population) > 1 else None
            solution_fig = build_solution_figure(
                cities_locations,
                best_path=best_solution,
                candidate_path=candidate_path,
                node_radius=_node_radius,
                city_priority_ids=_city_priorities,
                priority_labels={pid: data["label"] for pid, data in _priority_rules.items()},
                priority_order=_priority_ids,
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

_vrp_active_report = _num_vehicles > 1 or _vehicle_capacity_weight is not None or _vehicle_capacity_volume is not None
_report_vehicle_stats: list = []
if _vrp_active_report and best_solution_report:
    _final_routes = evaluate_vrp_routes(best_solution_report)
    _depot = cities_locations[0] if cities_locations else None
    for idx, route in enumerate(_final_routes):
        cities_in_route = [c for c in route if c != _depot] if _depot else route
        route_weight = sum(
            (get_city_demand(c) or {}).get("weight", 0) for c in cities_in_route
        )
        route_volume = sum(
            (get_city_demand(c) or {}).get("volume", 0) for c in cities_in_route
        )
        route_dist = 0.0
        for i in range(len(route) - 1):
            route_dist += calculate_distance(route[i], route[i + 1])
        depot_returns = max(0, sum(1 for c in route if c == _depot) - 1) if _depot else 0
        _report_vehicle_stats.append(VehicleStats(
            vehicle_id=idx + 1,
            cities=cities_in_route,
            distance=route_dist,
            weight=route_weight,
            volume=route_volume,
            depot_returns=depot_returns,
        ))

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
    num_vehicles=_num_vehicles,
    capacity_weight=_vehicle_capacity_weight,
    capacity_volume=_vehicle_capacity_volume,
    vehicle_stats=_report_vehicle_stats,
))

_priorities_for_llm = {}
for _i, _pid in enumerate(_city_priorities):
    _rule = _priority_rules.get(_pid, {})
    _priorities_for_llm[_i] = (
        f"{_pid} ({_rule.get('label', _pid)}) | {_format_priority_service_hint(_rule)}"
    )

_demands_for_llm = {}
for _i, _city in enumerate(cities_locations):
    _d = _city_demands[_i] if _i < len(_city_demands) else {}
    _demands_for_llm[_i] = {"weight_kg": _d.get("weight", 0), "volume_m3": _d.get("volume", 0)}

_vehicles_for_llm = []
if _vrp_active_report and _report_vehicle_stats:
    for _vs in _report_vehicle_stats:
        _city_idxs = [cities_locations.index(c) for c in _vs.cities if c in cities_locations]
        _vehicles_for_llm.append({
            "vehicle_id": _vs.vehicle_id,
            "cities": _city_idxs,
            "distance": round(_vs.distance, 2),
            "weight_kg": round(_vs.weight, 2),
            "volume_m3": round(_vs.volume, 4),
            "depot_returns": _vs.depot_returns,
        })

st.session_state["last_route_data"] = {
    "cities": list(cities_locations),
    "sequence": list(best_solution_report),
    "total_distance": best_fitness_report,
    "num_cities": len(cities_locations),
    "priorities": _priorities_for_llm,
    "demands": _demands_for_llm,
    "num_vehicles": _num_vehicles,
    "capacity_weight_kg": _vehicle_capacity_weight,
    "capacity_volume_m3": _vehicle_capacity_volume,
    "vehicles": _vehicles_for_llm,
}

report_text = generate_report()
st.text(report_text)

st.session_state["ga_report_text"] = report_text
st.session_state["ga_status_text"] = (
    f"**Geracao:** {_max_generation_allowed}  |  "
    f"**Melhor fitness:** {best_fitness_report:.2f}"
)
st.session_state["ga_best_fitness_values"] = list(best_fitness_values)
st.session_state["ga_best_solution"] = list(best_solution_report)
st.session_state["ga_vrp_routes"] = [list(r) for r in _final_routes] if _vrp_active_report and best_solution_report else None

st.session_state.run_ga = False
_safe_rerun()
