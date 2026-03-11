import csv
import io
from typing import Dict, List, Tuple, Optional, Any


def _read_uploaded_text(uploaded_file) -> Optional[str]:
    if uploaded_file is None:
        return None

    if isinstance(uploaded_file, str):
        return uploaded_file

    data = None
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    elif hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
    elif isinstance(uploaded_file, (bytes, bytearray)):
        data = uploaded_file

    if data is None:
        return None

    if isinstance(data, str):
        return data

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def parse_city_priority_csv(
    uploaded_file,
    cities_locations: List[Tuple[float, float]],
    priority_rules: Dict[str, Any],
):
    overrides: Dict[int, str] = {}
    errors: List[str] = []

    text = _read_uploaded_text(uploaded_file)
    if text is None:
        return overrides, errors

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
                coord = _normalize_coord(x_val, y_val)
                idx = coord_index.get(coord)
                if idx is None:
                    coord = _normalize_coord(x_val, y_val, as_int=False)
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


def _normalize_coord(x_val, y_val, as_int: bool = True):
    try:
        if as_int:
            return (int(float(x_val)), int(float(y_val)))
        return (float(x_val), float(y_val))
    except ValueError:
        return None


def build_city_priority_csv(
    cities_locations: List[Tuple[float, float]],
    city_overrides: Dict[int, str],
    default_priority_id: Optional[str],
):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["index", "x", "y", "priority"])
    for idx, city in enumerate(cities_locations):
        priority_id = city_overrides.get(idx, default_priority_id)
        writer.writerow([idx, city[0], city[1], priority_id])
    return output.getvalue()
