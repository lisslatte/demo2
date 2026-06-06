from __future__ import annotations

import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, time, timedelta
from pathlib import Path

import joblib
import pandas as pd
import pydeck as pdk
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
FLY_DIR = PROJECT_ROOT / "fly"
MODEL_DIR = Path(os.getenv("MODEL_DIR", FLY_DIR / "flight_delay_model_artifacts")).expanduser()
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODEL_DIR / "best_delay_model_pipeline.joblib")).expanduser()
METADATA_PATH = Path(os.getenv("METADATA_PATH", MODEL_DIR / "best_delay_model_metadata.json")).expanduser()
THRESHOLDS_PATH = Path(os.getenv("THRESHOLDS_PATH", MODEL_DIR / "best_model_thresholds.csv")).expanduser()
AIRPORTS_PATH = Path(os.getenv("AIRPORTS_PATH", PROJECT_ROOT / "airports.csv")).expanduser()

if str(FLY_DIR) not in sys.path:
    sys.path.insert(0, str(FLY_DIR))


POPULAR_AIRPORTS = [
    "ATL",
    "BOS",
    "BWI",
    "CLT",
    "DCA",
    "DEN",
    "DFW",
    "DTW",
    "EWR",
    "FLL",
    "IAD",
    "IAH",
    "JFK",
    "LAS",
    "LAX",
    "LGA",
    "MCO",
    "MDW",
    "MIA",
    "MSP",
    "ORD",
    "PHL",
    "PHX",
    "SAN",
    "SEA",
    "SFO",
]

AIRLINES = {
    "American Airlines": "AA",
    "Alaska Airlines": "AS",
    "JetBlue": "B6",
    "Delta Air Lines": "DL",
    "Frontier Airlines": "F9",
    "Allegiant Air": "G4",
    "Hawaiian Airlines": "HA",
    "Spirit Airlines": "NK",
    "United Airlines": "UA",
    "Southwest Airlines": "WN",
    "SkyWest Airlines": "OO",
    "Republic Airways": "YX",
}

AIRPORT_FALLBACK = {
    "ATL": ("Atlanta Hartsfield-Jackson", 33.6367, -84.4281),
    "BOS": ("Boston Logan", 42.3656, -71.0096),
    "CLT": ("Charlotte Douglas", 35.2140, -80.9431),
    "DEN": ("Denver", 39.8561, -104.6737),
    "DFW": ("Dallas/Fort Worth", 32.8998, -97.0403),
    "JFK": ("New York JFK", 40.6413, -73.7781),
    "LAS": ("Las Vegas Harry Reid", 36.0840, -115.1537),
    "LAX": ("Los Angeles", 33.9416, -118.4085),
    "MIA": ("Miami", 25.7959, -80.2870),
    "ORD": ("Chicago O'Hare", 41.9742, -87.9073),
    "PHX": ("Phoenix Sky Harbor", 33.4352, -112.0101),
    "SEA": ("Seattle-Tacoma", 47.4502, -122.3088),
    "SFO": ("San Francisco", 37.6213, -122.3790),
}

WEATHER_PRESETS = {
    "Clear": {
        "temperature_2m": 24.0,
        "relative_humidity_2m": 45.0,
        "precipitation": 0.0,
        "snow_depth": 0.0,
        "surface_pressure": 1018.0,
        "cloud_cover": 8.0,
        "wind_speed_10m": 8.0,
        "wind_gusts_10m": 12.0,
        "wind_direction": 250,
    },
    "Cloudy": {
        "temperature_2m": 18.0,
        "relative_humidity_2m": 68.0,
        "precipitation": 0.2,
        "snow_depth": 0.0,
        "surface_pressure": 1013.0,
        "cloud_cover": 75.0,
        "wind_speed_10m": 16.0,
        "wind_gusts_10m": 24.0,
        "wind_direction": 230,
    },
    "Rain": {
        "temperature_2m": 17.0,
        "relative_humidity_2m": 86.0,
        "precipitation": 6.0,
        "snow_depth": 0.0,
        "surface_pressure": 1007.0,
        "cloud_cover": 92.0,
        "wind_speed_10m": 24.0,
        "wind_gusts_10m": 38.0,
        "wind_direction": 210,
    },
    "Storm": {
        "temperature_2m": 23.0,
        "relative_humidity_2m": 88.0,
        "precipitation": 18.0,
        "snow_depth": 0.0,
        "surface_pressure": 997.0,
        "cloud_cover": 98.0,
        "wind_speed_10m": 45.0,
        "wind_gusts_10m": 76.0,
        "wind_direction": 205,
    },
    "Snow": {
        "temperature_2m": -4.0,
        "relative_humidity_2m": 82.0,
        "precipitation": 5.0,
        "snow_depth": 0.25,
        "surface_pressure": 1005.0,
        "cloud_cover": 95.0,
        "wind_speed_10m": 24.0,
        "wind_gusts_10m": 42.0,
        "wind_direction": 20,
    },
    "Fog": {
        "temperature_2m": 9.0,
        "relative_humidity_2m": 97.0,
        "precipitation": 0.2,
        "snow_depth": 0.0,
        "surface_pressure": 1011.0,
        "cloud_cover": 100.0,
        "wind_speed_10m": 5.0,
        "wind_gusts_10m": 8.0,
        "wind_direction": 160,
    },
    "Windy": {
        "temperature_2m": 20.0,
        "relative_humidity_2m": 55.0,
        "precipitation": 0.5,
        "snow_depth": 0.0,
        "surface_pressure": 1009.0,
        "cloud_cover": 45.0,
        "wind_speed_10m": 44.0,
        "wind_gusts_10m": 68.0,
        "wind_direction": 275,
    },
}

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HOURLY_FIELDS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "snow_depth",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
]
try:
    AVIATIONSTACK_API_KEY = (os.getenv("AVIATIONSTACK_API_KEY") or st.secrets.get("AVIATIONSTACK_API_KEY", "")).strip()
except (FileNotFoundError, KeyError):
    AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY", "").strip()
AVIATIONSTACK_FLIGHTS_URL = "https://api.aviationstack.com/v1/flights"
AIRLINE_AUTO_LABEL = "Auto estimate"

TRAFFIC_PRESETS = {
    "Quiet": {"traffic_level": 0.25, "origin_departures": 250, "dest_arrivals": 250},
    "Normal": {"traffic_level": 0.45, "origin_departures": 550, "dest_arrivals": 550},
    "Busy": {"traffic_level": 0.70, "origin_departures": 900, "dest_arrivals": 900},
    "Very busy": {"traffic_level": 0.92, "origin_departures": 1250, "dest_arrivals": 1250},
}

RECENT_DELAY_PRESETS = {
    "Low": 0.08,
    "Some": 0.18,
    "Many": 0.35,
}

THRESHOLD_MODE_ORDER = ["Recall first", "Balanced", "Precision first"]
THRESHOLD_MODE_TARGETS = {
    "Recall first": 0.15,
    "Balanced": 0.21,
    "Precision first": 0.35,
}
THRESHOLD_MODE_DESCRIPTIONS = {
    "Recall first": "Catches more possible delays, with more false alarms.",
    "Balanced": "Keeps recall and precision in a middle operating range.",
    "Precision first": "Raises confidence before flagging a delay, with fewer false alarms.",
}

NETWORK_CRITICAL_ORIGINS = {"ASE", "BWI", "DAL", "DEN", "DFW", "FLL", "LAS", "MCO", "MDW", "MIA", "EWR"}
NETWORK_CRITICAL_ROUTES = {
    "DEN-OAK",
    "BWI-HOU",
    "BWI-SJU",
    "MDW-LAX",
    "BWI-BUF",
    "DFW-SMF",
    "DFW-MCO",
    "FLL-BOS",
}


st.set_page_config(page_title="Flight Delay Business Decision Prototype", layout="wide")


st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1180px;
        padding-top: 1.8rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 16px;
    }
    .risk-card {
        background: #ffffff;
        border: 1px solid #dbe4ee;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .risk-percent {
        font-size: 56px;
        line-height: 1;
        font-weight: 800;
        color: #0f766e;
        margin: 4px 0 10px 0;
    }
    .risk-label {
        display: inline-block;
        border-radius: 999px;
        padding: 6px 12px;
        font-weight: 700;
        background: #e0f2fe;
        color: #075985;
    }
    .action-box {
        background: #f8fafc;
        border-left: 4px solid #0f766e;
        padding: 12px 14px;
        margin-top: 14px;
    }
    .context-card {
        background: #ffffff;
        border: 1px solid #dbe4ee;
        border-radius: 8px;
        padding: 14px 16px;
        min-height: 110px;
    }
    .context-card .label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .context-card .value {
        color: #172033;
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .context-card .detail {
        color: #475569;
        font-size: 0.9rem;
    }
    .decision-card {
        background: #ffffff;
        border: 1px solid #dbe4ee;
        border-radius: 8px;
        padding: 16px;
        min-height: 150px;
    }
    .decision-card h4 {
        margin: 0 0 8px 0;
        font-size: 1rem;
    }
    .decision-card ul {
        margin: 8px 0 0 18px;
        padding: 0;
    }
    .decision-card li {
        margin: 6px 0;
    }
    .priority-high {
        color: #991b1b;
        font-weight: 800;
    }
    .priority-medium {
        color: #92400e;
        font-weight: 800;
    }
    .priority-normal {
        color: #166534;
        font-weight: 800;
    }
    .journey-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 8px 0 16px 0;
    }
    .journey-step {
        background: #ffffff;
        border: 1px solid #dbe4ee;
        border-radius: 8px;
        padding: 14px 16px;
    }
    .journey-step .kicker {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .journey-step .headline {
        color: #0f172a;
        font-size: 1.3rem;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .journey-step .note {
        color: #475569;
        font-size: 0.92rem;
    }
    .map-note {
        color: #475569;
        margin: 0 0 12px 0;
    }
    @media (max-width: 800px) {
        .journey-strip {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_artifact() -> dict:
    artifact = joblib.load(MODEL_PATH)
    repair_sklearn_compatibility(artifact)
    return artifact


def rebuild_split_model_artifact(model_path: Path) -> bool:
    """Rebuild the split model file used to stay under GitHub browser upload limits."""
    if model_path.exists():
        return True

    parts = sorted(model_path.parent.glob(f"{model_path.name}.part-*"))
    if not parts:
        return False

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as output:
        for part in parts:
            output.write(part.read_bytes())
    return model_path.exists()


def repair_sklearn_compatibility(artifact: dict) -> None:
    pipeline = artifact.get("pipeline")
    if pipeline is None or not hasattr(pipeline, "named_steps"):
        return

    for step in pipeline.named_steps.values():
        if step.__class__.__name__ == "SimpleImputer" and not hasattr(step, "_fill_dtype"):
            step._fill_dtype = getattr(step, "statistics_", pd.Series([0.0])).dtype


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_threshold_table() -> pd.DataFrame:
    if not THRESHOLDS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(THRESHOLDS_PATH)


@st.cache_data(show_spinner=False)
def load_airports() -> pd.DataFrame:
    if not AIRPORTS_PATH.exists():
        rows = [
            {"iata_code": code, "name": values[0], "latitude_deg": values[1], "longitude_deg": values[2]}
            for code, values in AIRPORT_FALLBACK.items()
        ]
        return pd.DataFrame(rows)

    cols = ["iata_code", "name", "latitude_deg", "longitude_deg", "scheduled_service"]
    airports = pd.read_csv(AIRPORTS_PATH, usecols=cols)
    airports = airports[
        airports["iata_code"].isin(POPULAR_AIRPORTS)
        & airports["latitude_deg"].notna()
        & airports["longitude_deg"].notna()
    ].copy()
    airports["name"] = airports["name"].fillna(airports["iata_code"])
    return airports.drop_duplicates("iata_code").sort_values("iata_code")


@st.cache_data(show_spinner=False)
def load_airport_catalog() -> pd.DataFrame:
    fallback_rows = [
        {"iata_code": code, "name": values[0], "latitude_deg": values[1], "longitude_deg": values[2]}
        for code, values in AIRPORT_FALLBACK.items()
    ]
    if not AIRPORTS_PATH.exists():
        return pd.DataFrame(fallback_rows)

    cols = ["iata_code", "name", "latitude_deg", "longitude_deg"]
    airports = pd.read_csv(AIRPORTS_PATH, usecols=cols)
    airports = airports[
        airports["iata_code"].notna()
        & airports["latitude_deg"].notna()
        & airports["longitude_deg"].notna()
    ].copy()
    airports["iata_code"] = airports["iata_code"].astype(str).str.upper()
    airports["name"] = airports["name"].fillna(airports["iata_code"])
    return airports.drop_duplicates("iata_code")


class WeatherFetchError(Exception):
    pass


class FlightLookupError(Exception):
    pass


def airport_location(airport_code: str, airports: pd.DataFrame) -> tuple[float, float]:
    matches = airports[airports["iata_code"] == airport_code]
    if matches.empty:
        catalog = load_airport_catalog()
        matches = catalog[catalog["iata_code"] == airport_code]
    if matches.empty:
        fallback = AIRPORT_FALLBACK.get(airport_code)
        if fallback is None:
            raise WeatherFetchError(f"No coordinates found for {airport_code}.")
        return float(fallback[1]), float(fallback[2])

    row = matches.iloc[0]
    return float(row["latitude_deg"]), float(row["longitude_deg"])


def weather_endpoint_for(flight_date: date) -> str:
    today = date.today()
    if flight_date < today:
        return OPEN_METEO_ARCHIVE_URL
    if flight_date <= today + timedelta(days=15):
        return OPEN_METEO_FORECAST_URL
    raise WeatherFetchError("Open-Meteo forecast data is available for roughly the next 15 days.")


def nearest_hour_index(times: list[str], target: datetime) -> int:
    parsed_times = [datetime.fromisoformat(value) for value in times]
    return min(range(len(parsed_times)), key=lambda index: abs(parsed_times[index] - target))


def weather_condition_name(weather: dict[str, float]) -> str:
    temperature = weather["temperature_2m"]
    precipitation = weather["precipitation"]
    cloud_cover = weather["cloud_cover"]
    humidity = weather["relative_humidity_2m"]
    wind_speed = weather["wind_speed_10m"]
    wind_gust = weather["wind_gusts_10m"]
    snow_depth = weather["snow_depth"]

    if snow_depth > 0.03 or (temperature <= 1 and precipitation >= 0.5):
        return "Snow"
    if wind_gust >= 60 or precipitation >= 12:
        return "Storm"
    if precipitation >= 1:
        return "Rain"
    if humidity >= 94 and cloud_cover >= 85 and wind_speed <= 10:
        return "Fog"
    if wind_speed >= 35:
        return "Windy"
    if cloud_cover >= 60:
        return "Cloudy"
    return "Clear"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_open_meteo_weather(
    airport_code: str,
    latitude: float,
    longitude: float,
    flight_date: date,
    departure_hour: int,
) -> dict[str, object]:
    endpoint = weather_endpoint_for(flight_date)
    params = {
        "latitude": f"{latitude:.5f}",
        "longitude": f"{longitude:.5f}",
        "hourly": ",".join(OPEN_METEO_HOURLY_FIELDS),
        "start_date": flight_date.isoformat(),
        "end_date": flight_date.isoformat(),
        "timezone": "auto",
        "wind_speed_unit": "kmh",
    }
    url = f"{endpoint}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise WeatherFetchError(f"Weather API request failed for {airport_code}.") from exc

    if payload.get("error"):
        raise WeatherFetchError(str(payload.get("reason", "Weather API returned an error.")))

    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        raise WeatherFetchError(f"No hourly weather data returned for {airport_code}.")

    target = datetime.combine(flight_date, time(hour=int(departure_hour)))
    index = nearest_hour_index(times, target)
    values = {}
    fallback = WEATHER_PRESETS["Cloudy"]
    for field in OPEN_METEO_HOURLY_FIELDS:
        key = "wind_direction" if field == "wind_direction_10m" else field
        series = hourly.get(field) or []
        value = series[index] if index < len(series) else None
        values[key] = float(value) if value is not None else float(fallback[key])

    return {
        "airport": airport_code,
        "source": "Open-Meteo forecast" if endpoint == OPEN_METEO_FORECAST_URL else "Open-Meteo archive",
        "timestamp": times[index],
        "condition": weather_condition_name(values),
        "values": values,
    }


def extract_nested(data: dict, path: list[str]) -> str:
    value: object = data
    for key in path:
        if not isinstance(value, dict):
            return ""
        value = value.get(key)
    return str(value or "").strip()


def parse_api_time(value: str) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%H:%M")
    except ValueError:
        return value[-8:-3] if len(value) >= 16 else ""


@st.cache_data(ttl=300, show_spinner=False)
def fetch_flight_by_number(flight_number: str) -> dict[str, str]:
    if not AVIATIONSTACK_API_KEY:
        raise FlightLookupError("Flight-number lookup needs AVIATIONSTACK_API_KEY to be set.")

    params = {
        "access_key": AVIATIONSTACK_API_KEY,
        "flight_iata": flight_number.strip().upper().replace(" ", ""),
        "limit": "5",
    }
    url = f"{AVIATIONSTACK_FLIGHTS_URL}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise FlightLookupError("Flight lookup API request failed.") from exc

    if payload.get("error"):
        error = payload["error"]
        raise FlightLookupError(str(error.get("message") or error.get("info") or "Flight lookup API returned an error."))

    flights = payload.get("data") or []
    if not flights:
        raise FlightLookupError("No live or recent flight was found for that flight number.")

    flight = flights[0]
    departure_time = extract_nested(flight, ["departure", "scheduled"]) or extract_nested(flight, ["departure", "estimated"])
    arrival_time = extract_nested(flight, ["arrival", "scheduled"]) or extract_nested(flight, ["arrival", "estimated"])

    return {
        "origin": extract_nested(flight, ["departure", "iata"]),
        "dest": extract_nested(flight, ["arrival", "iata"]),
        "airline": extract_nested(flight, ["airline", "iata"]),
        "flight_number": extract_nested(flight, ["flight", "iata"]),
        "tail_number": extract_nested(flight, ["aircraft", "registration"]),
        "departure_time": parse_api_time(departure_time),
        "arrival_time": parse_api_time(arrival_time),
        "status": extract_nested(flight, ["flight_status"]),
        "source": "Aviationstack",
    }


def flight_details_from_payload(flight: dict, source: str) -> dict[str, str]:
    departure_time = extract_nested(flight, ["departure", "scheduled"]) or extract_nested(flight, ["departure", "estimated"])
    arrival_time = extract_nested(flight, ["arrival", "scheduled"]) or extract_nested(flight, ["arrival", "estimated"])

    return {
        "origin": extract_nested(flight, ["departure", "iata"]),
        "dest": extract_nested(flight, ["arrival", "iata"]),
        "airline": extract_nested(flight, ["airline", "iata"]),
        "flight_number": extract_nested(flight, ["flight", "iata"]),
        "tail_number": extract_nested(flight, ["aircraft", "registration"]),
        "departure_time": parse_api_time(departure_time),
        "arrival_time": parse_api_time(arrival_time),
        "status": extract_nested(flight, ["flight_status"]),
        "source": source,
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_flight_by_tail_best_effort(tail_number: str) -> dict[str, str]:
    if not AVIATIONSTACK_API_KEY:
        raise FlightLookupError("Aircraft registration lookup needs AVIATIONSTACK_API_KEY to be set.")

    target = tail_number.strip().upper().replace(" ", "")
    if not target:
        raise FlightLookupError("Enter a flight number or aircraft tail number.")

    for offset in (0, 100, 200):
        params = {
            "access_key": AVIATIONSTACK_API_KEY,
            "limit": "100",
            "offset": str(offset),
        }
        url = f"{AVIATIONSTACK_FLIGHTS_URL}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=12) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise FlightLookupError("Aircraft registration lookup API request failed.") from exc

        if payload.get("error"):
            error = payload["error"]
            raise FlightLookupError(str(error.get("message") or error.get("info") or "Flight lookup API returned an error."))

        for flight in payload.get("data") or []:
            registration = extract_nested(flight, ["aircraft", "registration"]).upper().replace(" ", "")
            if registration == target:
                return flight_details_from_payload(flight, "Aviationstack live feed registration match")

    raise FlightLookupError(
        "No current flight was found for that aircraft registration in Aviationstack's live feed. "
        "Try route/date/time, or use a flight number such as AA100."
    )


def looks_like_flight_number(value: str) -> bool:
    compact = value.strip().upper().replace(" ", "")
    return len(compact) >= 3 and compact[:2].isalnum() and compact[2:].isdigit()


def fetch_flight_by_identifier(identifier: str) -> dict[str, str]:
    value = identifier.strip().upper().replace(" ", "")
    if not value:
        raise FlightLookupError("Enter a flight number or aircraft tail number.")

    if looks_like_flight_number(value):
        try:
            return fetch_flight_by_number(value)
        except FlightLookupError:
            return fetch_flight_by_tail_best_effort(value)

    return fetch_flight_by_tail_best_effort(value)


def airline_name_from_code(code: str) -> str:
    for name, airline_code in AIRLINES.items():
        if airline_code == code:
            return name
    return code


def estimate_airline(origin: str, dest: str) -> tuple[str, str]:
    route = {origin, dest}
    if "ATL" in route:
        return "DL", "Estimated because Atlanta is a major Delta hub."
    if route & {"DFW", "CLT", "MIA", "PHL", "PHX", "DCA"}:
        return "AA", "Estimated from major American Airlines hub presence."
    if route & {"DEN", "EWR", "IAD", "IAH", "SFO", "ORD"}:
        return "UA", "Estimated from major United hub presence."
    if route & {"BWI", "MDW", "LAS", "MCO"}:
        return "WN", "Estimated from major Southwest route presence."
    if route & {"SEA", "SAN"}:
        return "AS", "Estimated from Alaska Airlines west-coast network presence."
    return "AA", "Estimated default airline because no airline was entered."


def resolve_airline(airline_choice: str, origin: str, dest: str, lookup_airline: str = "") -> tuple[str, str]:
    lookup_airline = lookup_airline.upper().strip()
    if lookup_airline in AIRLINES.values():
        return lookup_airline, f"Loaded from flight lookup: {airline_name_from_code(lookup_airline)}."
    if airline_choice != AIRLINE_AUTO_LABEL:
        return AIRLINES[airline_choice], "Entered by user."
    return estimate_airline(origin, dest)


def estimate_operational_context(
    origin: str,
    dest: str,
    flight_date: date,
    departure_time: time,
    weather: dict[str, float],
) -> dict[str, str | float]:
    hour = int(departure_time.hour)
    busy_airports = {"ATL", "DFW", "DEN", "ORD", "LAX", "JFK", "CLT", "LAS", "MCO", "SFO", "EWR"}
    very_busy_airports = {"ATL", "DFW", "DEN", "ORD", "LAX", "JFK"}

    if origin in very_busy_airports and hour in {7, 8, 9, 16, 17, 18, 19}:
        traffic_name = "Very busy"
    elif origin in busy_airports or dest in busy_airports or hour in {7, 8, 9, 16, 17, 18, 19}:
        traffic_name = "Busy"
    elif hour < 6 or hour > 22:
        traffic_name = "Quiet"
    else:
        traffic_name = "Normal"

    weather_pressure = 0
    if weather["precipitation"] >= 8 or weather["snow_depth"] >= 0.05:
        weather_pressure += 2
    elif weather["precipitation"] >= 1:
        weather_pressure += 1
    if weather["wind_gusts_10m"] >= 55 or weather["wind_speed_10m"] >= 35:
        weather_pressure += 2
    elif weather["wind_speed_10m"] >= 24:
        weather_pressure += 1
    if weather["cloud_cover"] >= 90 and weather["relative_humidity_2m"] >= 90:
        weather_pressure += 1
    if traffic_name in {"Busy", "Very busy"}:
        weather_pressure += 1
    if holiday_flags(flight_date)["IS_NEAR_HOLIDAY"]:
        weather_pressure += 1

    if weather_pressure >= 4:
        recent_delay_name = "Many"
    elif weather_pressure >= 2:
        recent_delay_name = "Some"
    else:
        recent_delay_name = "Low"

    return {
        "traffic_name": traffic_name,
        "recent_delay_name": recent_delay_name,
        "traffic_source": f"Estimated from {origin}/{dest} hub size and {hour:02d}:00 departure time.",
        "recent_delay_source": "Estimated from weather severity, peak-time pressure, and holiday proximity.",
    }


def airport_label_map(airports: pd.DataFrame) -> dict[str, str]:
    labels = {}
    for _, row in airports.iterrows():
        code = str(row["iata_code"])
        labels[f"{code} - {row['name']}"] = code
    return labels


def threshold_modes(default_threshold: float) -> dict[str, dict[str, float | str | None]]:
    table = load_threshold_table()
    modes = {}
    for name in THRESHOLD_MODE_ORDER:
        target = default_threshold if name == "Balanced" else THRESHOLD_MODE_TARGETS[name]
        mode = {
            "threshold": float(target),
            "precision": None,
            "recall": None,
            "f1": None,
            "false_positive_rate": None,
            "description": THRESHOLD_MODE_DESCRIPTIONS[name],
        }
        if not table.empty and "threshold" in table.columns:
            row = table.iloc[(table["threshold"] - target).abs().idxmin()]
            for key in ["threshold", "precision", "recall", "f1", "false_positive_rate"]:
                if key in row:
                    mode[key] = float(row[key])
        modes[name] = mode
    return modes


def percent_text(value: float | None, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{decimals}%}"


def haversine_miles(origin: str, dest: str, airports: pd.DataFrame) -> float:
    lookup = airports.set_index("iata_code")
    if origin not in lookup.index or dest not in lookup.index:
        return 500.0
    o = lookup.loc[origin]
    d = lookup.loc[dest]
    lat1 = math.radians(float(o["latitude_deg"]))
    lon1 = math.radians(float(o["longitude_deg"]))
    lat2 = math.radians(float(d["latitude_deg"]))
    lon2 = math.radians(float(d["longitude_deg"]))
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(a))


def distance_group(distance_miles: float) -> int:
    return int(max(1, min(11, math.ceil(distance_miles / 250))))


def dep_time_slot(hour: int) -> str:
    if 5 <= hour <= 11:
        return "Morning"
    if 12 <= hour <= 16:
        return "Afternoon"
    if 17 <= hour <= 21:
        return "Evening"
    return "Night"


def season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def holiday_flags(flight_date: date) -> dict[str, int]:
    fixed_holidays = {(1, 1), (7, 4), (11, 11), (12, 24), (12, 25), (12, 31)}
    month_day = (flight_date.month, flight_date.day)
    near_holiday = int(
        month_day in fixed_holidays
        or (flight_date.month == 11 and 20 <= flight_date.day <= 30)
        or (flight_date.month == 12 and 20 <= flight_date.day <= 31)
    )
    return {
        "IS_HOLIDAY": int(month_day in fixed_holidays),
        "IS_NEAR_HOLIDAY": near_holiday,
        "IS_PEAK_TRAVEL": int(flight_date.month in (6, 7, 8, 11, 12)),
        "IS_SUMMER_BREAK": int(flight_date.month in (6, 7, 8)),
        "IS_WINTER_BREAK": int(flight_date.month in (12, 1)),
        "IS_SPRING_BREAK": int(flight_date.month == 3),
        "IS_SCHOOL_BREAK": int(flight_date.month in (3, 6, 7, 8, 12, 1)),
    }


def build_feature_row(
    flight_date: date,
    departure_time: time,
    airline_code: str,
    origin: str,
    dest: str,
    weather: dict[str, float],
    traffic_name: str,
    recent_delay_name: str,
    airports: pd.DataFrame,
) -> pd.DataFrame:
    hour = int(departure_time.hour)
    distance = haversine_miles(origin, dest, airports)
    traffic = TRAFFIC_PRESETS[traffic_name]
    wind_rad = math.radians(weather["wind_direction"])
    row = {
        "Year": flight_date.year,
        "Quarter": int(math.ceil(flight_date.month / 3)),
        "Month": flight_date.month,
        "DayofMonth": flight_date.day,
        "DayOfWeek": int(flight_date.isoweekday()),
        "CRSDepHour": hour,
        "is_peak_hour": int(hour in (7, 8, 9, 16, 17, 18, 19)),
        "Distance": round(distance, 1),
        "DistanceGroup": distance_group(distance),
        "Origin": origin,
        "Dest": dest,
        "Operating_Airline": airline_code,
        "ROUTE": f"{origin}-{dest}",
        "Origin_freq": traffic["origin_departures"],
        "Dest_freq": traffic["dest_arrivals"],
        "IS_WEEKEND": int(flight_date.weekday() >= 5),
        "prev_delay": RECENT_DELAY_PRESETS[recent_delay_name],
        "traffic_level": traffic["traffic_level"],
        "temperature_2m": weather["temperature_2m"],
        "relative_humidity_2m": weather["relative_humidity_2m"],
        "precipitation": weather["precipitation"],
        "snow_depth": weather["snow_depth"],
        "surface_pressure": weather["surface_pressure"],
        "cloud_cover": weather["cloud_cover"],
        "wind_speed_10m": weather["wind_speed_10m"],
        "wind_gusts_10m": weather["wind_gusts_10m"],
        "wind_dir_sin": math.sin(wind_rad),
        "wind_dir_cos": math.cos(wind_rad),
    }
    row.update(holiday_flags(flight_date))

    for code in ["9E", "AA", "AS", "B6", "C5", "DL", "F9", "G4", "G7", "HA", "MQ", "NK", "OH", "OO", "PT", "QX", "UA", "WN", "YV", "YX", "ZW"]:
        row[f"Operating_Airline_{code}"] = int(code == airline_code)

    for slot in ["Afternoon", "Evening", "Morning", "Night"]:
        row[f"DEP_TIME_SLOT_{slot}"] = int(slot == dep_time_slot(hour))

    for name in ["Fall", "Spring", "Summer", "Winter"]:
        row[f"SEASON_{name}"] = int(name == season(flight_date.month))

    return pd.DataFrame([row])


def score_frame(artifact: dict, frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    probabilities = artifact["pipeline"].predict_proba(frame.copy())[:, 1]
    result = frame.copy()
    result["delay_probability"] = probabilities
    result["predicted_delay"] = (probabilities >= threshold).astype(int)
    return result


def risk_label(probability: float, threshold: float) -> str:
    low_cutoff = min(0.15, threshold * 0.7)
    high_cutoff = min(0.75, threshold + 0.15)
    if probability < low_cutoff:
        return "Low risk"
    if probability < threshold:
        return "Moderate risk"
    if probability < high_cutoff:
        return "Delay likely"
    return "High risk"


def action_text(label: str) -> str:
    if label in {"Delay likely", "High risk"}:
        return "Check flight updates early, allow extra time, and watch for gate or crew schedule changes."
    if label == "Moderate risk":
        return "The flight has some delay pressure. Keep alerts on and re-check closer to departure."
    return "The flight looks reasonably stable, but keep normal flight alerts switched on."


def risk_tier(probability: float, threshold: float) -> str:
    if probability >= max(0.35, threshold + 0.12):
        return "High"
    if probability >= threshold:
        return "Elevated"
    if probability >= max(0.12, threshold * 0.7):
        return "Watch"
    return "Low"


def is_network_critical(origin: str, dest: str, hour: int, traffic_name: str) -> bool:
    route = f"{origin}-{dest}"
    return (
        route in NETWORK_CRITICAL_ROUTES
        or origin in NETWORK_CRITICAL_ORIGINS
        or traffic_name in {"Busy", "Very busy"}
        or hour in {7, 8, 9, 16, 17, 18, 19, 20}
    )


def row_network_critical(row: pd.Series) -> bool:
    try:
        hour = int(row.get("CRSDepHour", -1))
    except (TypeError, ValueError):
        hour = -1
    origin = str(row.get("Origin", ""))
    dest = str(row.get("Dest", ""))
    return is_network_critical(origin, dest, hour, "Normal")


def airport_priority(probability: float, threshold: float, network_critical: bool) -> str:
    tier = risk_tier(probability, threshold)
    if tier == "High" or (tier == "Elevated" and network_critical):
        return "High"
    if tier in {"Elevated", "Watch"} or network_critical:
        return "Medium"
    return "Normal"


def passenger_focus(probability: float, threshold: float) -> str:
    tier = risk_tier(probability, threshold)
    if tier == "High":
        return "Critical Passengers"
    if tier in {"Elevated", "Watch"}:
        return "At-Risk Standard Passengers"
    return "Flexible Passengers"


def airline_review_level(probability: float, threshold: float, network_critical: bool) -> str:
    tier = risk_tier(probability, threshold)
    if tier == "High" or (tier == "Elevated" and network_critical):
        return "Early operational review"
    if tier in {"Elevated", "Watch"}:
        return "Monitor and prepare"
    return "Standard monitoring"


def airport_actions(priority: str, network_critical: bool) -> list[str]:
    if priority == "High":
        actions = [
            "Prioritise this flight by delay risk in the departure bank.",
            "Pre-position gates, ramp teams, and staffing for faster recovery.",
            "Use the risk score for passenger-flow planning and queue pressure.",
        ]
        if network_critical:
            actions.append("Monitor this route as network-critical because disruption may spread to later flights.")
        return actions
    if priority == "Medium":
        return [
            "Place the flight on an airport watchlist instead of treating it as routine.",
            "Check gate, ramp, and staffing availability before the delay escalates.",
            "Monitor nearby peak departure banks and connecting passenger flow.",
        ]
    return [
        "Keep standard airport monitoring active.",
        "Do not pull scarce gate, ramp, or staffing resources away from higher-risk flights.",
        "Continue automated passenger-flow monitoring.",
    ]


def airline_actions(review_level: str, focus_segment: str) -> list[str]:
    if review_level == "Early operational review":
        return [
            "Identify the flight for early operational review.",
            "Check crew legality, aircraft rotation, and recovery options.",
            f"Protect {focus_segment.lower()} before disruption escalates.",
            "Prepare rebooking and communication options before the delay is confirmed.",
        ]
    if review_level == "Monitor and prepare":
        return [
            "Monitor the flight for delay escalation.",
            "Check backup aircraft, crew, and connection exposure if resources allow.",
            f"Prepare automated communication for {focus_segment.lower()}.",
        ]
    return [
        "Keep the flight in standard operations monitoring.",
        "Use automated updates unless the risk score increases.",
        "Reserve manual intervention for higher-risk flights.",
    ]


def action_list_html(title: str, actions: list[str], priority: str | None = None) -> str:
    priority_class = {
        "High": "priority-high",
        "Medium": "priority-medium",
        "Normal": "priority-normal",
    }.get(priority or "", "")
    priority_html = f"<p class='{priority_class}'>{priority} priority</p>" if priority else ""
    items = "".join(f"<li>{action}</li>" for action in actions)
    return f"""
    <div class="decision-card">
        <h4>{title}</h4>
        {priority_html}
        <ul>{items}</ul>
    </div>
    """


def airport_display_name(airport_code: str, airports: pd.DataFrame) -> str:
    matches = airports[airports["iata_code"] == airport_code]
    if matches.empty:
        catalog = load_airport_catalog()
        matches = catalog[catalog["iata_code"] == airport_code]
    if matches.empty:
        fallback = AIRPORT_FALLBACK.get(airport_code)
        return fallback[0] if fallback else airport_code
    return str(matches.iloc[0]["name"])


def passenger_risk_name(probability: float, threshold: float) -> str:
    if probability >= max(0.45, threshold + 0.2):
        return "Very High"
    if probability >= threshold:
        return "High"
    if probability >= max(0.12, threshold * 0.65):
        return "Moderate"
    return "Low"


def weather_passenger_message(weather_name: str, weather: dict[str, float]) -> str:
    if weather_name in {"Storm", "Snow"}:
        return "Weather may slow ground operations, boarding, or departure sequencing."
    if weather_name in {"Rain", "Windy", "Fog"}:
        return "Weather could add some airport handling or departure pressure."
    if weather["wind_gusts_10m"] >= 45:
        return "Wind gusts are elevated, so keep an eye on airport updates."
    return "Weather looks manageable for passengers, but flight status can still change."


def passenger_reasons(
    probability: float,
    threshold: float,
    weather_name: str,
    weather: dict[str, float],
    traffic_name: str,
    recent_delay_name: str,
    departure_time: time,
    network_critical: bool,
    flight_date: date,
) -> list[str]:
    reasons = []
    if traffic_name in {"Busy", "Very busy"}:
        reasons.append(f"{departure_time:%H:%M} is estimated as a {traffic_name.lower()} airport period.")
    if weather_name in {"Rain", "Storm", "Snow", "Fog", "Windy"}:
        reasons.append(f"Departure weather is {weather_name.lower()}, with {weather['wind_speed_10m']:.0f} km/h wind.")
    if recent_delay_name in {"Some", "Many"}:
        reasons.append(f"Recent delay pressure is estimated as {recent_delay_name.lower()}.")
    if network_critical:
        reasons.append("This route or departure time may be sensitive to network disruption.")
    if holiday_flags(flight_date)["IS_NEAR_HOLIDAY"]:
        reasons.append("The date is close to a busy travel period.")

    if not reasons:
        reasons.append("No major weather, timing, or airport-pressure warning stands out.")
    if probability >= threshold and len(reasons) < 3:
        reasons.append("The model still finds enough combined pressure to recommend closer monitoring.")
    return reasons[:3]


def recommended_airport_arrival_minutes(
    probability: float,
    threshold: float,
    traffic_name: str,
    checked_bags: bool,
    needs_extra_time: bool,
) -> int:
    minutes = 120
    if probability >= max(0.45, threshold + 0.2):
        minutes += 45
    elif probability >= threshold:
        minutes += 30
    elif probability >= max(0.12, threshold * 0.65):
        minutes += 15
    if traffic_name == "Very busy":
        minutes += 30
    elif traffic_name == "Busy":
        minutes += 15
    if checked_bags:
        minutes += 20
    if needs_extra_time:
        minutes += 20
    return minutes


def airport_arrival_text(flight_date: date, departure_time: time, minutes_before: int) -> str:
    departure_dt = datetime.combine(flight_date, departure_time)
    arrival_dt = departure_dt - timedelta(minutes=minutes_before)
    return f"{arrival_dt:%H:%M} ({minutes_before // 60}h {minutes_before % 60:02d}m before departure)"


def leave_home_text(flight_date: date, departure_time: time, minutes_before: int, travel_minutes: int) -> str:
    departure_dt = datetime.combine(flight_date, departure_time)
    leave_dt = departure_dt - timedelta(minutes=minutes_before + travel_minutes)
    return f"{leave_dt:%H:%M}"


def estimate_destination_arrival_text(
    flight_date: date,
    departure_time: time,
    distance_miles: float,
    lookup_details: dict[str, str],
) -> tuple[str, str]:
    if lookup_details.get("arrival_time"):
        return lookup_details["arrival_time"], "Scheduled arrival from flight lookup."

    duration_minutes = int(max(45, min(720, (distance_miles / 500.0) * 60 + 35)))
    arrival_dt = datetime.combine(flight_date, departure_time) + timedelta(minutes=duration_minutes)
    return arrival_dt.strftime("%H:%M"), f"Estimated from route distance, about {duration_minutes // 60}h {duration_minutes % 60:02d}m flight time."


def connection_risk(
    has_connection: bool,
    connection_minutes: int,
    probability: float,
    threshold: float,
) -> tuple[str, str]:
    if not has_connection:
        return "Not applicable", "No connection entered."
    if connection_minutes < 45:
        return "Very High", "This is a tight connection even without a delay."
    if probability >= threshold and connection_minutes < 90:
        return "High", "A moderate delay could put this connection at risk."
    if probability >= max(0.12, threshold * 0.65) and connection_minutes < 75:
        return "Moderate", "Keep the airline app open and review backup options."
    return "Low", "Your connection buffer looks reasonable for this risk level."


def passenger_action_plan(
    passenger_risk: str,
    arrival_text: str,
    leave_text: str,
    connection_label: str,
    has_connection: bool,
) -> list[str]:
    actions = [f"Leave for the airport by {leave_text} and be at the airport by {arrival_text}."]
    if passenger_risk in {"High", "Very High"}:
        actions.append("Turn on airline notifications and check the flight again before you leave.")
        actions.append("Do not plan a tight airport arrival.")
    elif passenger_risk == "Moderate":
        actions.append("Keep notifications on and check the flight before leaving home.")
    else:
        actions.append("Normal timing should be okay, but keep airline notifications on.")
    if has_connection and connection_label in {"High", "Very High"}:
        actions.append("Because your connection may be tight, check backup options in the airline app.")
    return actions


def airport_map_rows(origin: str, dest: str, airports: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for code, label, color in [(origin, "Takeoff", [20, 184, 166]), (dest, "Landing", [249, 115, 22])]:
        lat, lon = airport_location(code, airports)
        rows.append(
            {
                "code": code,
                "label": label,
                "lat": lat,
                "lon": lon,
                "name": airport_display_name(code, airports),
                "color": color,
                "text": f"{label}\n{code}",
            }
        )
    points = pd.DataFrame(rows)
    line = pd.DataFrame(
        [
            {
                "from": origin,
                "to": dest,
                "source": [float(points.iloc[0]["lon"]), float(points.iloc[0]["lat"])],
                "target": [float(points.iloc[1]["lon"]), float(points.iloc[1]["lat"])],
                "color": [37, 99, 235],
            }
        ]
    )
    return points, line


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


if not rebuild_split_model_artifact(MODEL_PATH):
    st.error(
        f"Model artifact not found: {MODEL_PATH}. "
        f"Upload {MODEL_PATH.name}.part-001 and {MODEL_PATH.name}.part-002 to {MODEL_PATH.parent}."
    )
    st.stop()

artifact = load_artifact()
metadata = load_metadata()
airports = load_airports()
airport_labels = airport_label_map(airports)
airport_options = list(airport_labels)
default_threshold = float(artifact.get("threshold", metadata.get("best_threshold", 0.21)))
mode_settings = threshold_modes(default_threshold)

st.title("Passenger Flight Delay Assistant")
st.caption("Enter your flight details. The app checks weather and airport context, then gives passenger-friendly travel advice.")

left, right = st.columns([0.95, 1.05], gap="large")

with left:
    st.subheader("Find Your Flight")
    input_mode = st.radio(
        "Input method",
        ["Route and time", "Flight lookup"],
        horizontal=True,
    )

    with st.form("flight_form"):
        lookup_identifier = ""
        lookup_date = date.today()
        lookup_details: dict[str, str] = {}
        origin_index = list(airport_labels.values()).index("JFK") if "JFK" in airport_labels.values() else 0
        dest_index = list(airport_labels.values()).index("LAX") if "LAX" in airport_labels.values() else min(1, len(airport_options) - 1)

        if input_mode == "Route and time":
            route_cols = st.columns(2)
            origin_label = route_cols[0].selectbox("From", airport_options, index=origin_index)
            dest_label = route_cols[1].selectbox("To", airport_options, index=dest_index)

            schedule_cols = st.columns(2)
            selected_date = schedule_cols[0].date_input("Flight date", value=date.today())
            selected_time = schedule_cols[1].time_input("Departure time", value=time(17, 30))

            airline_choice = st.selectbox(
                "Airline, if known",
                [AIRLINE_AUTO_LABEL] + list(AIRLINES),
                index=0,
            )
            origin = airport_labels[origin_label]
            dest = airport_labels[dest_label]
        else:
            lookup_identifier = st.text_input(
                "Flight number or tail number",
                placeholder="Example: AA100, JL740, or N123AA",
            ).strip().upper()
            lookup_date = st.date_input("Date shown in prediction", value=date.today())
            selected_date = lookup_date
            selected_time = time(12, 0)
            origin = "JFK"
            dest = "LAX"
            airline_choice = AIRLINE_AUTO_LABEL

        st.subheader("Your Trip")
        trip_cols = st.columns(2)
        checked_bags = trip_cols[0].checkbox("I have checked bags")
        needs_extra_time = trip_cols[1].checkbox("I may need extra time")
        travel_minutes = st.number_input(
            "Travel time to airport in minutes",
            min_value=0,
            max_value=240,
            value=45,
            step=5,
        )
        has_connection = st.checkbox("I have a connecting flight")
        connection_minutes = 60
        if has_connection:
            connection_minutes = st.number_input(
                "Connection time in minutes",
                min_value=15,
                max_value=360,
                value=60,
                step=5,
            )

        submit_label = "Check my flight" if input_mode == "Route and time" else "Look up flight and check risk"
        submitted = st.form_submit_button(submit_label, type="primary")

        with st.expander("Advanced prediction setting"):
            threshold_mode = st.radio(
                "Threshold mode",
                THRESHOLD_MODE_ORDER,
                index=THRESHOLD_MODE_ORDER.index("Balanced"),
                horizontal=True,
            )
            active_mode = mode_settings[threshold_mode]
            active_threshold = float(active_mode["threshold"])
            st.caption(
                f"{active_mode['description']} "
                f"Validation precision: {percent_text(active_mode['precision'])}; "
                f"recall: {percent_text(active_mode['recall'])}."
            )

    with st.expander("Test examples"):
        st.write("Flight numbers: AA100, DL123, UA200, JL740, VA726, JQ401, S75309, VA500, UA7411, S75227")
        st.write("Tail numbers: N189DN, JA835J")
        st.caption("Flight numbers are more reliable. Tail-number lookup is best-effort because live aircraft registration data is often missing.")

if input_mode == "Flight lookup" and submitted:
    try:
        lookup_details = fetch_flight_by_identifier(lookup_identifier)
        if lookup_details.get("origin"):
            origin = lookup_details["origin"].upper()
        if lookup_details.get("dest"):
            dest = lookup_details["dest"].upper()
        if lookup_details.get("departure_time"):
            hour, minute = lookup_details["departure_time"].split(":")[:2]
            selected_time = time(int(hour), int(minute))
    except FlightLookupError as exc:
        lookup_details = {}
        with left:
            st.error(f"Flight lookup is unavailable: {exc}")

lookup_airline = lookup_details.get("airline", "") if input_mode == "Flight lookup" else ""
airline_code, airline_source_label = resolve_airline(airline_choice, origin, dest, lookup_airline)
airline_label = airline_name_from_code(airline_code)

manual_weather_name = "Cloudy"
weather_error = ""
weather_details: dict[str, object] | None = None
try:
    latitude, longitude = airport_location(origin, airports)
    weather_details = fetch_open_meteo_weather(origin, latitude, longitude, selected_date, int(selected_time.hour))
    weather_values = weather_details["values"]
    weather_name = str(weather_details["condition"])
    weather_source_label = f"{weather_details['source']} at {weather_details['timestamp']}"
except WeatherFetchError as exc:
    weather_error = str(exc)
    weather_values = WEATHER_PRESETS[manual_weather_name]
    weather_name = manual_weather_name
    weather_source_label = f"Automatic fallback: {manual_weather_name}"

context = estimate_operational_context(origin, dest, selected_date, selected_time, weather_values)
traffic_name = str(context["traffic_name"])
recent_delay_name = str(context["recent_delay_name"])

with left:
    if input_mode == "Flight lookup":
        if lookup_details:
            tail_note = f" | tail {lookup_details['tail_number']}" if lookup_details.get("tail_number") else ""
            st.success(
                f"Loaded {lookup_details.get('flight_number') or lookup_identifier}: "
                f"{origin} to {dest}, {selected_time:%H:%M}, {airline_label}{tail_note}."
            )
        elif not AVIATIONSTACK_API_KEY:
            st.info("Flight lookup is optional. Set AVIATIONSTACK_API_KEY to enable Aviationstack lookup.")
        st.caption("Flight-number lookup is direct. Tail-number lookup is best-effort because Aviationstack often omits aircraft registration data.")
    if weather_error:
        st.warning(f"Weather API fallback used. {weather_error}")

if input_mode == "Flight lookup" and not lookup_details:
    with right:
        st.subheader("Delay Probability")
        st.info("Enter a flight number or tail number to load the route, departure time, airline, weather, and context automatically.")
    st.stop()

feature_row = build_feature_row(
    selected_date,
    selected_time,
    airline_code,
    origin,
    dest,
    weather_values,
    traffic_name,
    recent_delay_name,
    airports,
)

if origin == dest:
    right.warning("Choose different origin and destination airports.")
elif submitted or "last_probability" not in st.session_state:
    scored = score_frame(artifact, feature_row, active_threshold)
    st.session_state.last_probability = float(scored["delay_probability"].iloc[0])
    st.session_state.last_prediction = int(scored["predicted_delay"].iloc[0])
    st.session_state.last_row = feature_row

probability = float(st.session_state.get("last_probability", 0.0))
label = risk_label(probability, active_threshold)
classification = "Delay likely" if probability >= active_threshold else "Delay not likely"
network_critical = is_network_critical(origin, dest, int(selected_time.hour), traffic_name)
priority = airport_priority(probability, active_threshold, network_critical)
focus_segment = passenger_focus(probability, active_threshold)
review_level = airline_review_level(probability, active_threshold, network_critical)
passenger_risk = passenger_risk_name(probability, active_threshold)
arrival_minutes = recommended_airport_arrival_minutes(
    probability,
    active_threshold,
    traffic_name,
    bool(checked_bags),
    bool(needs_extra_time),
)
arrival_text = airport_arrival_text(selected_date, selected_time, arrival_minutes)
leave_text = leave_home_text(selected_date, selected_time, arrival_minutes, int(travel_minutes))
distance_miles = float(feature_row["Distance"].iloc[0])
destination_arrival_text, destination_arrival_source = estimate_destination_arrival_text(
    selected_date,
    selected_time,
    distance_miles,
    lookup_details,
)
connection_label, connection_message = connection_risk(
    bool(has_connection),
    int(connection_minutes),
    probability,
    active_threshold,
)
reason_list = passenger_reasons(
    probability,
    active_threshold,
    weather_name,
    weather_values,
    traffic_name,
    recent_delay_name,
    selected_time,
    network_critical,
    selected_date,
)
action_plan = passenger_action_plan(passenger_risk, arrival_text, leave_text, connection_label, bool(has_connection))
weather_message = weather_passenger_message(weather_name, weather_values)

with right:
    st.subheader("Your Delay Risk")
    st.markdown(
        f"""
        <div class="risk-card">
            <div class="risk-percent">{probability:.0%}</div>
            <span class="risk-label">{passenger_risk} risk</span>
            <div class="action-box">{action_plan[0]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(1.0, probability))

    result_cols = st.columns(4)
    result_cols[0].metric("Delay chance", f"{probability:.0%}")
    result_cols[1].metric("Leave by", leave_text)
    result_cols[2].metric("Be at airport by", arrival_text.split(" ")[0])
    result_cols[3].metric("Land around", destination_arrival_text)
    st.caption(f"{origin} to {dest} | {selected_date:%b %d, %Y} at {selected_time:%H:%M}")

    st.subheader("Automatically Loaded Information")
    info_cols = st.columns(4)
    info_cols[0].markdown(
        f"""
        <div class="context-card">
            <div class="label">Weather</div>
            <div class="value">{weather_name}</div>
            <div class="detail">{weather_values['temperature_2m']:.1f} C, {weather_values['precipitation']:.1f} mm rain/snow<br>{weather_message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    info_cols[1].markdown(
        f"""
        <div class="context-card">
            <div class="label">Airport Load</div>
            <div class="value">{traffic_name}</div>
            <div class="detail">{origin} is estimated from route and departure time.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    info_cols[2].markdown(
        f"""
        <div class="context-card">
            <div class="label">Delay Pressure</div>
            <div class="value">{recent_delay_name}</div>
            <div class="detail">Estimated from weather, airport load, and date.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    info_cols[3].markdown(
        f"""
        <div class="context-card">
            <div class="label">Route</div>
            <div class="value">{origin} to {dest}</div>
            <div class="detail">{float(feature_row['Distance'].iloc[0]):,.0f} mi<br>{'Network-critical' if network_critical else 'Standard network route'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary = pd.DataFrame(
        [
            {"Item": "Airline", "Value": f"{airline_label} ({airline_code})"},
            {"Item": "Airline source", "Value": airline_source_label},
            {"Item": "Departure", "Value": f"{selected_date:%b %d, %Y} at {selected_time:%H:%M}"},
            {"Item": "Route", "Value": f"{origin} to {dest}"},
            {"Item": "Distance", "Value": f"{float(feature_row['Distance'].iloc[0]):,.0f} mi"},
            {"Item": "Checked bags", "Value": "Yes" if checked_bags else "No"},
            {"Item": "Extra time needed", "Value": "Yes" if needs_extra_time else "No"},
            {"Item": "Travel time to airport", "Value": f"{int(travel_minutes)} minutes"},
            {"Item": "Leave for airport by", "Value": leave_text},
            {"Item": "Connection", "Value": f"{connection_label} - {connection_message}"},
            {"Item": "Recommended airport arrival", "Value": arrival_text},
            {"Item": "Destination arrival", "Value": f"{destination_arrival_text} - {destination_arrival_source}"},
            {"Item": "Threshold mode", "Value": threshold_mode},
            {"Item": "Network-critical", "Value": "Yes" if network_critical else "No"},
            {"Item": "Flight lookup", "Value": lookup_details.get("source", "Not used")},
            {"Item": "Weather", "Value": weather_name},
            {"Item": "Weather source", "Value": weather_source_label},
            {
                "Item": "Weather inputs",
                "Value": (
                    f"{weather_values['temperature_2m']:.1f} C, "
                    f"{weather_values['precipitation']:.1f} mm precip, "
                    f"{weather_values['wind_speed_10m']:.0f} km/h wind"
                ),
            },
            {"Item": "Airport traffic", "Value": f"{traffic_name} - {context['traffic_source']}"},
            {"Item": "Recent delays", "Value": f"{recent_delay_name} - {context['recent_delay_source']}"},
        ]
    )
st.subheader("Your Travel Plan")
plan_tab, connection_tab, map_tab, details_tab = st.tabs(["What To Do", "Connection", "Map", "Details"])

with plan_tab:
    plan_cols = st.columns([1, 1], gap="large")
    with plan_cols[0]:
        st.markdown(
            action_list_html(
                "What To Do Now",
                action_plan,
            ),
            unsafe_allow_html=True,
        )
    with plan_cols[1]:
        st.markdown(
            action_list_html(
                "Why This Risk Level",
                reason_list,
            ),
            unsafe_allow_html=True,
        )
    st.info(f"Weather at {origin}: {weather_name}. {weather_message}")
    st.success(f"Expected arrival at {dest}: {destination_arrival_text}. {destination_arrival_source}")

with connection_tab:
    conn_cols = st.columns(3)
    conn_cols[0].metric("Connection risk", connection_label)
    conn_cols[1].metric("Connection buffer", f"{int(connection_minutes)} min" if has_connection else "None")
    conn_cols[2].metric("Delay chance", f"{probability:.0%}")
    st.write(connection_message)
    if has_connection and connection_label in {"High", "Very High"}:
        st.warning("Consider checking backup flights or asking the airline about rebooking options before departure.")
    elif has_connection:
        st.success("Your connection buffer looks acceptable for the current risk estimate.")
    else:
        st.info("Tick 'I have a connecting flight' in the form if you want connection advice.")

with map_tab:
    try:
        map_points, route_line = airport_map_rows(origin, dest, airports)
        midpoint_lat = float(map_points["lat"].mean())
        midpoint_lon = float(map_points["lon"].mean())
        if distance_miles >= 3200:
            route_zoom = 1.6
        elif distance_miles >= 1400:
            route_zoom = 2.4
        else:
            route_zoom = 3.6
        st.markdown(
            f"""
            <div class="journey-strip">
                <div class="journey-step">
                    <div class="kicker">Before you go</div>
                    <div class="headline">{leave_text}</div>
                    <div class="note">Leave for the airport</div>
                </div>
                <div class="journey-step">
                    <div class="kicker">Takeoff</div>
                    <div class="headline">{origin} at {selected_time:%H:%M}</div>
                    <div class="note">{airport_display_name(origin, airports)}</div>
                </div>
                <div class="journey-step">
                    <div class="kicker">Landing</div>
                    <div class="headline">{dest} around {destination_arrival_text}</div>
                    <div class="note">{airport_display_name(dest, airports)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="map-note">Flight path: about {distance_miles:,.0f} miles. Hover over each airport marker for details.</p>',
            unsafe_allow_html=True,
        )
        st.pydeck_chart(
            pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=midpoint_lat,
                    longitude=midpoint_lon,
                    zoom=route_zoom,
                    pitch=28,
                ),
                layers=[
                    pdk.Layer(
                        "ArcLayer",
                        data=route_line,
                        get_source_position="source",
                        get_target_position="target",
                        get_source_color=[20, 184, 166],
                        get_target_color=[249, 115, 22],
                        get_width=5,
                    ),
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=map_points,
                        get_position="[lon, lat]",
                        get_fill_color="color",
                        get_line_color=[255, 255, 255],
                        get_line_width=4,
                        get_radius=90000,
                        pickable=True,
                    ),
                    pdk.Layer(
                        "TextLayer",
                        data=map_points,
                        get_position="[lon, lat]",
                        get_text="text",
                        get_size=16,
                        get_color=[15, 23, 42],
                        get_angle=0,
                        get_text_anchor='"middle"',
                        get_alignment_baseline='"bottom"',
                    ),
                ],
                tooltip={"text": "{label}: {code}\n{name}"},
            )
        )
        st.dataframe(
            map_points[["label", "code", "name", "lat", "lon"]],
            hide_index=True,
            width="stretch",
        )
    except WeatherFetchError as exc:
        st.warning(f"Map unavailable: {exc}")

with details_tab:
    st.dataframe(summary, hide_index=True, width="stretch")
    st.download_button(
        "Download travel summary",
        data=csv_bytes(summary),
        file_name="passenger_flight_delay_summary.csv",
        mime="text/csv",
    )
