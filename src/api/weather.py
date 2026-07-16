"""src.api.weather — weather data retrieval with 3-tier fallback.

Provides :func:`fetch_weather`, the main entry point used by the agent
graph, plus the internal helper functions for each provider tier:

    Tier 1: Google Weather API  (``_fetch_google_weather``)
    Tier 2: OpenWeatherMap API  (``_fetch_openweathermap``)
    Tier 3: Static fallback      (``_build_fallback_weather``)
"""

import requests
from datetime import datetime, timedelta

from src.api.base import logger, API_TIMEOUT, _get_session
from src.config import GOOGLE_MAPS_API_KEY, OPENWEATHER_API_KEY

# ---------------------------------------------------------------------------
# Weather Condition Mapping
# ---------------------------------------------------------------------------

# OpenWeatherMap condition main-field -> simplified English description.
# Retained for backward compatibility when OpenWeatherMap is used as fallback.
# Source: https://openweathermap.org/weather-conditions
_WEATHER_CONDITION_MAP = {
    # OpenWeatherMap condition mapping (retained for compatibility)
    "Clear": "Sunny",
    "Clouds": "Cloudy",
    "Rain": "Rainy",
    "Drizzle": "Rainy",
    "Thunderstorm": "Stormy",
    "Snow": "Snowy",
    "Mist": "Foggy",
    "Fog": "Foggy",
    "Haze": "Hazy",
}

# Google Weather API alertType -> severity mapping.
# Each entry maps a Google Weather alert type string to:
#   severity (str): "minor" | "moderate" | "severe" | "extreme"
#   alert_level (int): 1 (low) to 5 (critical)
# Reference: Google Weather API v1 currentConditions.weatherAlerts[].alertType.type
_GOOGLE_WEATHER_ALERT_MAP = {
    "WIND": {"severity": "moderate", "alert_level": 2},
    "SEVERE_WIND": {"severity": "severe", "alert_level": 4},
    "HURRICANE_WIND": {"severity": "extreme", "alert_level": 5},
    "TORNADO": {"severity": "extreme", "alert_level": 5},
    "SEVERE_THUNDERSTORM": {"severity": "severe", "alert_level": 4},
    "THUNDERSTORM": {"severity": "moderate", "alert_level": 2},
    "HEAVY_RAIN": {"severity": "severe", "alert_level": 3},
    "RAIN": {"severity": "minor", "alert_level": 1},
    "FLOOD": {"severity": "severe", "alert_level": 4},
    "SEVERE_FLOOD": {"severity": "extreme", "alert_level": 5},
    "HEAVY_SNOW": {"severity": "severe", "alert_level": 3},
    "SNOW": {"severity": "minor", "alert_level": 1},
    "BLIZZARD": {"severity": "extreme", "alert_level": 5},
    "ICE_STORM": {"severity": "severe", "alert_level": 4},
    "FOG": {"severity": "minor", "alert_level": 1},
    "DENSE_FOG": {"severity": "moderate", "alert_level": 2},
    "EXTREME_HEAT": {"severity": "severe", "alert_level": 4},
    "HEAT": {"severity": "moderate", "alert_level": 2},
    "EXTREME_COLD": {"severity": "severe", "alert_level": 4},
    "COLD": {"severity": "minor", "alert_level": 1},
    "TSUNAMI": {"severity": "extreme", "alert_level": 5},
    "EARTHQUAKE": {"severity": "extreme", "alert_level": 5},
    "VOLCANO": {"severity": "extreme", "alert_level": 5},
}

# OpenWeatherMap weather condition code -> alert info mapping.
# Code ranges follow the OWM standard:
#   2xx = Thunderstorm, 3xx = Drizzle, 5xx = Rain, 6xx = Snow, 7xx = Atmosphere
# Reference: https://openweathermap.org/weather-conditions
# Each entry maps an integer code to:
#   severity (str): "minor" | "moderate" | "severe" | "extreme"
#   alert_level (int): 1 (low) to 5 (critical)
#   alert_type (str): human-readable alert category
_OWM_ALERT_MAP = {
    # Thunderstorm 2xx
    200: {"severity": "moderate", "alert_level": 2, "alert_type": "thunderstorm"},
    201: {"severity": "moderate", "alert_level": 3, "alert_type": "thunderstorm"},
    202: {"severity": "severe", "alert_level": 4, "alert_type": "severe_thunderstorm"},
    210: {"severity": "minor", "alert_level": 1, "alert_type": "thunderstorm"},
    211: {"severity": "moderate", "alert_level": 2, "alert_type": "thunderstorm"},
    212: {"severity": "severe", "alert_level": 4, "alert_type": "severe_thunderstorm"},
    221: {"severity": "severe", "alert_level": 4, "alert_type": "severe_thunderstorm"},
    230: {"severity": "moderate", "alert_level": 3, "alert_type": "thunderstorm_rain"},
    231: {"severity": "moderate", "alert_level": 3, "alert_type": "thunderstorm_rain"},
    232: {"severity": "severe", "alert_level": 4, "alert_type": "thunderstorm_rain"},
    # Drizzle 3xx
    300: {"severity": "minor", "alert_level": 1, "alert_type": "drizzle"},
    301: {"severity": "minor", "alert_level": 1, "alert_type": "drizzle"},
    302: {"severity": "minor", "alert_level": 1, "alert_type": "drizzle"},
    310: {"severity": "minor", "alert_level": 1, "alert_type": "drizzle"},
    311: {"severity": "minor", "alert_level": 1, "alert_type": "drizzle"},
    312: {"severity": "moderate", "alert_level": 2, "alert_type": "drizzle"},
    313: {"severity": "moderate", "alert_level": 2, "alert_type": "drizzle"},
    314: {"severity": "moderate", "alert_level": 2, "alert_type": "drizzle"},
    321: {"severity": "moderate", "alert_level": 2, "alert_type": "drizzle"},
    270: {"severity": "minor", "alert_level": 1, "alert_type": "drizzle"},
    # Rain 5xx
    500: {"severity": "minor", "alert_level": 1, "alert_type": "rain"},
    501: {"severity": "minor", "alert_level": 1, "alert_type": "rain"},
    502: {"severity": "moderate", "alert_level": 2, "alert_type": "heavy_rain"},
    503: {"severity": "severe", "alert_level": 3, "alert_type": "heavy_rain"},
    504: {"severity": "severe", "alert_level": 4, "alert_type": "extreme_rain"},
    511: {"severity": "moderate", "alert_level": 2, "alert_type": "freezing_rain"},
    520: {"severity": "minor", "alert_level": 1, "alert_type": "rain_shower"},
    521: {"severity": "moderate", "alert_level": 2, "alert_type": "rain_shower"},
    522: {"severity": "moderate", "alert_level": 3, "alert_type": "heavy_rain_shower"},
    531: {"severity": "moderate", "alert_level": 2, "alert_type": "ragged_rain"},
    # Snow 6xx
    600: {"severity": "minor", "alert_level": 1, "alert_type": "snow"},
    601: {"severity": "minor", "alert_level": 1, "alert_type": "snow"},
    602: {"severity": "moderate", "alert_level": 3, "alert_type": "heavy_snow"},
    611: {"severity": "moderate", "alert_level": 2, "alert_type": "sleet"},
    612: {"severity": "moderate", "alert_level": 2, "alert_type": "sleet"},
    613: {"severity": "moderate", "alert_level": 2, "alert_type": "sleet"},
    615: {"severity": "moderate", "alert_level": 2, "alert_type": "rain_snow"},
    616: {"severity": "moderate", "alert_level": 2, "alert_type": "rain_snow"},
    620: {"severity": "moderate", "alert_level": 2, "alert_type": "snow_shower"},
    621: {"severity": "moderate", "alert_level": 2, "alert_type": "snow_shower"},
    622: {"severity": "severe", "alert_level": 3, "alert_type": "heavy_snow_shower"},
    # Atmosphere 7xx
    701: {"severity": "minor", "alert_level": 1, "alert_type": "mist"},
    711: {"severity": "minor", "alert_level": 1, "alert_type": "smoke"},
    721: {"severity": "minor", "alert_level": 1, "alert_type": "haze"},
    731: {"severity": "moderate", "alert_level": 2, "alert_type": "dust"},
    741: {"severity": "moderate", "alert_level": 2, "alert_type": "fog"},
    751: {"severity": "moderate", "alert_level": 2, "alert_type": "sand"},
    761: {"severity": "moderate", "alert_level": 2, "alert_type": "dust"},
    762: {"severity": "severe", "alert_level": 3, "alert_type": "volcanic_ash"},
    771: {"severity": "severe", "alert_level": 4, "alert_type": "squalls"},
    781: {"severity": "extreme", "alert_level": 5, "alert_type": "tornado"},
}

# Google Weather API weatherCondition.type -> simplified English description.
# Maps the enum values returned by Google Weather v1 currentConditions and
# forecast/days endpoints to short human-readable strings.
_GOOGLE_WEATHER_CONDITION_MAP = {
    "CLEAR": "Sunny",
    "MOSTLY_CLEAR": "Sunny",
    "PARTLY_CLOUDY": "Cloudy",
    "MOSTLY_CLOUDY": "Cloudy",
    "CLOUDY": "Cloudy",
    "OVERCAST": "Cloudy",
    "RAIN": "Rainy",
    "RAIN_SHOWERS": "Rainy",
    "LIGHT_RAIN": "Rainy",
    "HEAVY_RAIN": "Rainy",
    "DRIZZLE": "Rainy",
    "THUNDERSTORM": "Stormy",
    "SNOW": "Snowy",
    "SNOW_SHOWERS": "Snowy",
    "LIGHT_SNOW": "Snowy",
    "HEAVY_SNOW": "Snowy",
    "FLURRIES": "Snowy",
    "FOG": "Foggy",
    "HAZE": "Hazy",
    "WIND": "Windy",
    "SLEET": "Rainy",
    "FREEZING_RAIN": "Rainy",
    "ICE": "Snowy",
}


# ---------------------------------------------------------------------------
# 1. fetch_weather - Weather Data Retrieval (multi-tier)
# ---------------------------------------------------------------------------
#
# Fallback chain:
#   Tier 0: Open-Meteo Historical (for trips >14 days out, uses last year's data)
#   Tier 1: Google Weather API  (primary, most accurate, up to 14-day forecast)
#   Tier 2: OpenWeatherMap API  (secondary, wide coverage, up to 5 days)
#   Tier 3: Seasonal estimate   (_build_fallback_weather)
# ---------------------------------------------------------------------------


# WMO Weather interpretation codes mapping (used by Open-Meteo)
_WMO_CODE_MAP = {
    0: "Clear Sky", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
    45: "Foggy", 48: "Freezing Fog",
    51: "Light Drizzle", 53: "Moderate Drizzle", 55: "Dense Drizzle",
    61: "Slight Rain", 63: "Moderate Rain", 65: "Heavy Rain",
    71: "Slight Snow", 73: "Moderate Snow", 75: "Heavy Snow",
    80: "Slight Rain Showers", 81: "Moderate Rain Showers", 82: "Violent Rain Showers",
    85: "Slight Snow Showers", 86: "Heavy Snow Showers",
    95: "Thunderstorm", 96: "Thunderstorm with Hail", 99: "Thunderstorm with Heavy Hail",
}


def _fetch_historical_weather_open_meteo(destination: str, duration_days: int, start_date: str) -> dict | None:
    """Fetch last year's same-period historical weather via Open-Meteo Archive API.

    Free API, no key required. Uses last year's actual weather data as a reference
    for trips that are beyond the 14-day forecast horizon.

    API Endpoint: GET https://archive-api.open-meteo.com/v1/archive
    Parameters:
        latitude, longitude: Destination coordinates
        start_date, end_date: Date range in YYYY-MM-DD (last year's same period)
        daily: temperature_2m_max, temperature_2m_min, weathercode
        timezone: auto
    Response: JSON with daily arrays of max/min temp and weather codes

    Args:
        destination (str): Destination city name.
        duration_days (int): Number of travel days.
        start_date (str): Trip start date (YYYY-MM-DD).

    Returns:
        dict | None: Weather data dict with is_forecast=True and historical note,
            or None if the API call fails.
    """
    try:
        # First, geocode the destination using Open-Meteo's free geocoding
        # API: https://geocoding-api.open-meteo.com/v1/search?name=Beijing&count=1
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_resp = _get_session().get(geo_url, params={"name": destination, "count": 1}, timeout=5)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        results = geo_data.get("results", [])
        if not results:
            logger.warning(f"Open-Meteo geocoding failed for '{destination}': no results")
            return None

        lat = results[0]["latitude"]
        lon = results[0]["longitude"]

        # Calculate last year's same period
        trip_start = datetime.strptime(start_date, "%Y-%m-%d")
        hist_start = trip_start.replace(year=trip_start.year - 1)
        hist_end = hist_start + timedelta(days=duration_days - 1)

        # Fetch historical daily weather
        # API: https://archive-api.open-meteo.com/v1/archive
        archive_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": hist_start.strftime("%Y-%m-%d"),
            "end_date": hist_end.strftime("%Y-%m-%d"),
            "daily": "temperature_2m_max,temperature_2m_min,weathercode",
            "timezone": "auto",
        }
        resp = _get_session().get(archive_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])
        codes = daily.get("weathercode", [])

        if not dates:
            return None

        # Build forecast_days using actual trip dates but last year's weather
        forecast_days = []
        for i in range(min(duration_days, len(dates))):
            trip_day = trip_start + timedelta(days=i)
            wmo_code = codes[i] if i < len(codes) else 0
            condition = _WMO_CODE_MAP.get(wmo_code, "Partly Cloudy")
            forecast_days.append({
                "date": trip_day.strftime("%Y-%m-%d"),
                "condition": condition,
                "temp_high": round(max_temps[i]) if i < len(max_temps) and max_temps[i] is not None else 25,
                "temp_low": round(min_temps[i]) if i < len(min_temps) and min_temps[i] is not None else 15,
            })

        # Use average of the period for current condition summary
        valid_highs = [t for t in max_temps if t is not None]
        valid_lows = [t for t in min_temps if t is not None]
        avg_high = round(sum(valid_highs) / len(valid_highs)) if valid_highs else 25
        avg_low = round(sum(valid_lows) / len(valid_lows)) if valid_lows else 15

        logger.info(
            f"Open-Meteo historical weather for '{destination}': "
            f"{hist_start.strftime('%Y-%m-%d')} to {hist_end.strftime('%Y-%m-%d')} "
            f"(avg {avg_low}-{avg_high}°C)"
        )

        return {
            "condition": forecast_days[0]["condition"] if forecast_days else "Partly Cloudy",
            "temp_c": (avg_high + avg_low) // 2,
            "humidity": 60,  # Open-Meteo free tier doesn't include humidity in archive
            "wind_speed_kmh": 12,
            "forecast_days": forecast_days,
            "weather_alerts": [],
            "is_forecast": True,
            "note": (f"Based on actual weather data from {hist_start.strftime('%Y-%m-%d')} to "
                     f"{hist_end.strftime('%Y-%m-%d')} (last year same period). "
                     f"Expect similar conditions."),
        }

    except Exception as e:
        logger.warning(f"Open-Meteo historical weather failed for '{destination}': {e}")
        return None
def _build_fallback_weather(duration_days: int = 3, start_date: str = None, destination: str = "") -> dict:
    """Generate fallback weather data when all weather APIs fail or trip is too far out.

    For trips >14 days in the future, provides seasonal climate estimates based
    on the destination's hemisphere and month. For near-term trips, uses generic
    mild-climate data.

    Args:
        duration_days (int): Number of travel days to generate forecast for.
        start_date (str | None): Trip start date in YYYY-MM-DD format.
        destination (str): Destination name, used to estimate hemisphere.

    Returns:
        dict: Weather data dict with is_forecast=True and note field.
    """
    base_date = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime.now()
    month = base_date.month

    # Seasonal climate estimates by month (Northern Hemisphere baseline)
    # Format: (avg_high, avg_low, dominant_condition, humidity)
    _SEASONAL_CLIMATE = {
        1:  (5, -2, "Cold & Dry", 55),
        2:  (7, 0, "Cold & Dry", 55),
        3:  (13, 5, "Mild & Breezy", 60),
        4:  (18, 9, "Mild & Pleasant", 60),
        5:  (24, 14, "Warm & Sunny", 55),
        6:  (28, 19, "Hot & Humid", 65),
        7:  (31, 23, "Hot & Humid", 70),
        8:  (30, 22, "Hot & Humid", 70),
        9:  (26, 17, "Warm & Pleasant", 60),
        10: (20, 12, "Cool & Comfortable", 55),
        11: (13, 5, "Cool & Dry", 55),
        12: (7, 0, "Cold & Dry", 55),
    }

    # Simple heuristic: southern hemisphere destinations get inverted seasons
    _SOUTHERN_HINTS = ["sydney", "melbourne", "auckland", "cape town", "buenos aires",
                       "santiago", "lima", "sao paulo", "rio", "perth", "brisbane"]
    is_southern = any(hint in destination.lower() for hint in _SOUTHERN_HINTS)
    if is_southern:
        month = ((month + 5) % 12) + 1  # Flip 6 months

    # Tropical destination heuristic
    _TROPICAL_HINTS = ["bangkok", "bali", "singapore", "kuala lumpur", "manila",
                       "phuket", "hanoi", "ho chi minh", "jakarta", "cancun",
                       "hawaii", "maldives", "pattaya"]
    is_tropical = any(hint in destination.lower() for hint in _TROPICAL_HINTS)

    if is_tropical:
        avg_high, avg_low = 32, 25
        condition = "Hot & Humid" if month in (5, 6, 7, 8, 9, 10) else "Warm & Sunny"
        humidity = 75
    else:
        avg_high, avg_low, condition, humidity = _SEASONAL_CLIMATE[month]

    # Generate daily forecast with small variations
    conditions_cycle = ["Sunny", condition, "Cloudy", "Sunny", condition, "Rainy", "Sunny"]
    forecast_days = []
    for i in range(duration_days):
        day_date = base_date + timedelta(days=i)
        # Add small daily variation
        high = avg_high + (i % 3) - 1
        low = avg_low + (i % 3) - 1
        cond = conditions_cycle[i % len(conditions_cycle)]
        forecast_days.append({
            "date": day_date.strftime("%Y-%m-%d"),
            "condition": cond,
            "temp_high": high,
            "temp_low": low
        })

    days_until = (base_date - datetime.now()).days
    if days_until > 14:
        note = (f"Based on historical climate data for {destination} in "
                f"{base_date.strftime('%B')}. Actual weather may vary.")
    else:
        note = "Fallback estimate — real-time forecast unavailable."

    return {
        "condition": condition,
        "temp_c": (avg_high + avg_low) // 2,
        "humidity": humidity,
        "wind_speed_kmh": 12,
        "forecast_days": forecast_days,
        "weather_alerts": [],
        "is_forecast": True,
        "note": note,
    }


def _fetch_google_weather(destination: str, api_key: str, duration_days: int,
                          start_date: str = None) -> dict:
    """Fetch weather via Google Weather API (primary provider).

    Performs a 3-step process:
      1. Geocode the destination to lat/lng via Google Geocoding API.
      2. Fetch current conditions via Google Weather currentConditions:lookup.
      3. Fetch daily forecast via Google Weather forecast/days:lookup,
         optionally anchored to the trip start date via ``forecastStartDate``.

    Args:
        destination (str): Destination city name (e.g. "Wuhan", "Tokyo").
        api_key (str): Google Maps API key with Weather API enabled.
        duration_days (int): Number of forecast days to retrieve.
        start_date (str | None): Trip start date in YYYY-MM-DD format.
            When provided and in the future, the forecast is requested for
            that date range instead of starting from today.

    Returns:
        dict: Unified weather data dict with keys:
            condition (str), temp_c (int), humidity (int),
            wind_speed_kmh (float), forecast_days (list[dict]),
            weather_alerts (list[dict]),
            is_forecast (bool): True when start_date is in the future.

    Raises:
        ValueError: If geocoding cannot resolve the destination.
        requests.HTTPError: On HTTP failure from any of the three endpoints.
        Exception: On any other API or parsing failure.
    """
    # Step 1: Geocode destination to lat/lng coordinates
    # Google Geocoding API
    # API 名称: Google Maps Geocoding API
    # Endpoint: https://maps.googleapis.com/maps/api/geocode/json
    # 请求参数: address (目的地名称), key (API密钥)
    # Response format:
    # {
    #     "results": [
    #         {
    #             "formatted_address": "string",
    #             "geometry": {
    #                 "location": {"lat": float, "lng": float}
    #             },
    #             "address_components": [
    #                 {"long_name": "string", "short_name": "string", "types": ["string"]}
    #             ]
    #         }
    #     ],
    #     "status": "OK" | "ZERO_RESULTS" | "OVER_DAILY_LIMIT" | ...
    # }
    geo_url = "https://maps.googleapis.com/maps/api/geocode/json"
    geo_params = {"address": destination, "key": api_key}
    geo_resp = _get_session().get(geo_url, params=geo_params, timeout=API_TIMEOUT)
    geo_resp.raise_for_status()
    geo_data = geo_resp.json()

    if geo_data.get("status") != "OK" or not geo_data.get("results"):
        raise ValueError(f"Google Geocoding could not resolve '{destination}'")

    location = geo_data["results"][0]["geometry"]["location"]
    lat, lon = location["lat"], location["lng"]
    logger.debug(f"Google Geocoding succeeded: {destination} -> ({lat}, {lon})")

    # Step 2: Fetch current weather conditions
    # Google Weather API - Current Conditions
    # API 名称: Google Weather API v1
    # Endpoint: https://weather.googleapis.com/v1/currentConditions:lookup
    # 请求参数: key (API密钥), location.latitude (纬度), location.longitude (经度)
    # Response format:
    # {
    #     "weatherCondition": {"type": "CLEAR" | "CLOUDY" | "RAIN" | ...},
    #     "temperature": {"degrees": float},
    #     "relativeHumidity": float (0-100),
    #     "wind": {"speed": {"value": float}},
    #     "weatherAlerts": [
    #         {
    #             "alertType": {"type": "WIND" | "RAIN" | "FLOOD" | ...},
    #             "headline": "string"
    #         }
    #     ]
    # }
    current_url = "https://weather.googleapis.com/v1/currentConditions:lookup"
    current_params = {"key": api_key, "location.latitude": lat, "location.longitude": lon}
    current_resp = _get_session().get(current_url, params=current_params, timeout=API_TIMEOUT)
    current_resp.raise_for_status()
    current = current_resp.json()

    # Step 3: Fetch daily weather forecast
    # Google Weather API - Daily Forecast
    # API 名称: Google Weather API v1 Forecast
    # Endpoint: https://weather.googleapis.com/v1/forecast/days:lookup
    # 请求参数: key, location.latitude, location.longitude, days (预报天数)
    # Response format:
    # {
    #     "forecastDays": [
    #         {
    #             "displayDateTime": {"date": "YYYY-MM-DD"},
    #             "weatherCondition": {"type": "CLEAR" | "CLOUDY" | ...},
    #             "temperature": {
    #                 "high": {"degrees": float},
    #                 "low": {"degrees": float}
    #             },
    #             "interval": {"startTime": "ISO8601 string"}
    #         }
    #     ]
    # }
    # Determine whether the trip is in the future; used to flag forecast accuracy
    today_str = datetime.now().strftime("%Y-%m-%d")
    is_future_trip = bool(start_date) and start_date > today_str

    forecast_url = "https://weather.googleapis.com/v1/forecast/days:lookup"
    forecast_params = {
        "key": api_key,
        "location.latitude": lat,
        "location.longitude": lon,
        "days": duration_days,
    }
    # When the trip starts in the future, anchor the forecast to the trip start
    # date so Google Weather returns the relevant period rather than from today.
    # Google Weather API supports up to 14-day forecasts via forecastStartDate.
    if is_future_trip:
        forecast_params["forecastStartDate"] = start_date
        logger.info(
            f"Google Weather: requesting forecast anchored to trip start "
            f"date {start_date} (future trip)"
        )
    forecast_resp = _get_session().get(forecast_url, params=forecast_params, timeout=API_TIMEOUT)
    forecast_resp.raise_for_status()
    forecast_data = forecast_resp.json()

    # Parse current weather
    condition_type = current.get("weatherCondition", {}).get("type", "CLOUDY")
    condition = _GOOGLE_WEATHER_CONDITION_MAP.get(condition_type, condition_type.title())
    temp_c = round(current.get("temperature", {}).get("degrees", 25))
    humidity = current.get("relativeHumidity", 60)
    wind_speed = current.get("wind", {}).get("speed", {})
    wind_speed_kmh = round(wind_speed.get("value", 10), 1)

    # Parse daily forecast
    forecast_days = []
    for item in forecast_data.get("forecastDays", [])[:duration_days]:
        display_dt = item.get("displayDateTime", {})
        day_date = display_dt.get("date", "")
        if not day_date:
            day_date = item.get("interval", {}).get("startTime", "")[:10]
        fc_type = item.get("weatherCondition", {}).get("type", "CLOUDY")
        fc_condition = _GOOGLE_WEATHER_CONDITION_MAP.get(fc_type, fc_type.title())
        fc_high = round(item.get("temperature", {}).get("high", {}).get("degrees", 28))
        fc_low = round(item.get("temperature", {}).get("low", {}).get("degrees", 20))
        forecast_days.append({
            "date": day_date,
            "condition": fc_condition,
            "temp_high": fc_high,
            "temp_low": fc_low
        })

    # Parse weather alerts from Google Weather
    weather_alerts = []
    for alert in current.get("weatherAlerts", []):
        alert_type = alert.get("alertType", {}).get("type", "UNKNOWN")
        alert_info = _GOOGLE_WEATHER_ALERT_MAP.get(alert_type, {"severity": "minor", "alert_level": 1})
        weather_alerts.append({
            "alert_type": alert_type.lower(),
            "severity": alert_info["severity"],
            "alert_level": alert_info["alert_level"],
            "headline": alert.get("headline", f"{alert_type} alert")
        })

    result = {
        "condition": condition,
        "temp_c": temp_c,
        "humidity": humidity,
        "wind_speed_kmh": wind_speed_kmh,
        "forecast_days": forecast_days,
        "weather_alerts": weather_alerts,
        "is_forecast": is_future_trip,
    }
    if is_future_trip:
        logger.info(
            f"Google Weather forecast for {destination} starting {start_date}: "
            f"{condition}, {temp_c}°C (future trip — accuracy may decrease "
            f"for dates beyond 7 days)"
        )
    else:
        logger.info(f"Google Weather data retrieved: {destination} - {condition}, {temp_c}°C")
    return result


def _fetch_openweathermap(destination: str, api_key: str, duration_days: int,
                          start_date: str = None) -> dict:
    """Fetch weather via OpenWeatherMap API (secondary/fallback provider).

    Uses three OpenWeatherMap endpoints:
      1. Geocoding API to resolve city name to lat/lng.
      2. Current Weather API for current conditions.
      3. 5-day/3-hour Forecast API aggregated into daily summaries.
         When ``start_date`` is in the future, only forecast entries on or
         after that date are included.  OWM only supports a 5-day window so
         trips beyond that horizon fall back to partial data.

    Args:
        destination (str): Destination city name (e.g. "Wuhan", "Paris").
        api_key (str): OpenWeatherMap API key.
        duration_days (int): Number of forecast days to retrieve (max 5).
        start_date (str | None): Trip start date in YYYY-MM-DD format.
            When provided and in the future, forecasts are filtered to start
            from this date.

    Returns:
        dict: Unified weather data dict with keys:
            condition (str), temp_c (int), humidity (int),
            wind_speed_kmh (float), forecast_days (list[dict]),
            weather_alerts (list[dict]),
            is_forecast (bool): True when start_date is in the future.

    Raises:
        ValueError: If geocoding cannot resolve the destination.
        requests.HTTPError: On HTTP failure from any endpoint.
        Exception: On any other API or parsing failure.
    """
    # Step 1: Geocode city name to lat/lng
    # OpenWeatherMap Geocoding API
    # API 名称: OpenWeatherMap Geocoding API
    # Endpoint: http://api.openweathermap.org/geo/1.0/direct
    # 请求参数: q (城市名), limit (结果数量), appid (API密钥)
    # Response format:
    # [
    #     {
    #         "name": "string (local name)",
    #         "lat": float,
    #         "lon": float,
    #         "country": "string (country code)"
    #     }
    # ]
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={destination}&limit=1&appid={api_key}"
    geo_resp = _get_session().get(geo_url, timeout=API_TIMEOUT)
    geo_resp.raise_for_status()
    geo_data = geo_resp.json()

    if not geo_data:
        raise ValueError(f"OpenWeatherMap Geocoding could not resolve '{destination}'")

    lat, lon = geo_data[0]["lat"], geo_data[0]["lon"]
    logger.debug(f"OpenWeatherMap Geocoding succeeded: {destination} -> ({lat}, {lon})")

    # Step 2: Fetch current weather conditions
    # OpenWeatherMap Current Weather API
    # API 名称: OpenWeatherMap Current Weather Data
    # Endpoint: https://api.openweathermap.org/data/2.5/weather
    # 请求参数: lat, lon, appid, units (metric)
    # Response format:
    # {
    #     "main": {
    #         "temp": float (Celsius),
    #         "humidity": int (0-100)
    #     },
    #     "weather": [
    #         {
    #             "id": int (condition code, e.g. 800=Clear),
    #             "main": "string (e.g. Clear, Clouds, Rain)",
    #             "description": "string"
    #         }
    #     ],
    #     "wind": {"speed": float (m/s)},
    #     "dt": int (unix timestamp)
    # }
    current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    current_resp = _get_session().get(current_url, timeout=API_TIMEOUT)
    current_resp.raise_for_status()
    current = current_resp.json()

    # Step 3: Fetch 5-day/3-hour forecast (max 40 data points)
    # OpenWeatherMap 5-day/3-hour Forecast API
    # API 名称: OpenWeatherMap 5 Day / 3 Hour Forecast
    # Endpoint: https://api.openweathermap.org/data/2.5/forecast
    # 请求参数: lat, lon, appid, units (metric), cnt (number of 3-hour slots)
    # Response format:
    # {
    #     "list": [
    #         {
    #             "dt": int (unix timestamp),
    #             "main": {"temp": float (Celsius)},
    #             "weather": [
    #                 {"id": int, "main": "string", "description": "string"}
    #             ]
    #         }
    #     ]
    # }
    # Each entry represents a 3-hour window; 8 entries per day.
    cnt = min(duration_days * 8, 40)
    forecast_url = (
        f"https://api.openweathermap.org/data/2.5/forecast"
        f"?lat={lat}&lon={lon}&appid={api_key}&units=metric&cnt={cnt}"
    )
    forecast_resp = _get_session().get(forecast_url, timeout=API_TIMEOUT)
    forecast_resp.raise_for_status()
    forecast_data = forecast_resp.json()

    # Parse current weather
    main = current.get("main", {})
    weather_desc = current.get("weather", [{}])[0]
    condition_main = weather_desc.get("main", "Clouds")
    condition = _WEATHER_CONDITION_MAP.get(condition_main, condition_main)
    temp_c = round(main.get("temp", 25))
    humidity = main.get("humidity", 60)
    wind_speed_ms = current.get("wind", {}).get("speed", 3.0)
    wind_speed_kmh = round(wind_speed_ms * 3.6, 1)  # m/s -> km/h

    # Aggregate 3-hour forecasts into daily summaries
    daily_data = {}  # date_str -> list of forecast entries
    for entry in forecast_data.get("list", []):
        dt_str = datetime.utcfromtimestamp(entry["dt"]).strftime("%Y-%m-%d")
        if dt_str not in daily_data:
            daily_data[dt_str] = []
        daily_data[dt_str].append(entry)

    # Determine whether the trip is in the future
    today_str = datetime.now().strftime("%Y-%m-%d")
    is_future_trip = bool(start_date) and start_date > today_str

    forecast_days = []
    for date_str in sorted(daily_data.keys()):
        # When the trip is in the future, skip forecast entries before start_date
        if is_future_trip and date_str < start_date:
            continue
        if len(forecast_days) >= duration_days:
            break
        entries = daily_data[date_str]
        temps = [e.get("main", {}).get("temp", 25) for e in entries]
        temp_high = round(max(temps))
        temp_low = round(min(temps))
        # Use the most common condition (mode) from the day's entries
        conditions = [e.get("weather", [{}])[0].get("main", "Clouds") for e in entries]
        dominant_condition = max(set(conditions), key=conditions.count)
        fc_condition = _WEATHER_CONDITION_MAP.get(dominant_condition, dominant_condition)
        forecast_days.append({
            "date": date_str,
            "condition": fc_condition,
            "temp_high": temp_high,
            "temp_low": temp_low
        })

    # Extract weather alerts from forecast data
    weather_alerts = []
    seen_alert_types = set()
    all_entries = forecast_data.get("list", [])
    # Also include current weather in alert check
    all_entries = [{"weather": current.get("weather", []), "dt": current.get("dt", 0)}] + all_entries
    for entry in all_entries:
        for w in entry.get("weather", []):
            code = w.get("id", 800)
            alert_info = _OWM_ALERT_MAP.get(code)
            if alert_info:
                alert_type = alert_info["alert_type"]
                if alert_type not in seen_alert_types:
                    seen_alert_types.add(alert_type)
                    weather_alerts.append({
                        "alert_type": alert_type,
                        "severity": alert_info["severity"],
                        "alert_level": alert_info["alert_level"],
                        "headline": w.get("description", f"{alert_type} warning")
                    })

    result = {
        "condition": condition,
        "temp_c": temp_c,
        "humidity": humidity,
        "wind_speed_kmh": wind_speed_kmh,
        "forecast_days": forecast_days,
        "weather_alerts": weather_alerts,
        "is_forecast": is_future_trip,
    }
    if is_future_trip:
        logger.info(
            f"OpenWeatherMap forecast for {destination} starting {start_date}: "
            f"{condition}, {temp_c}°C (future trip — OWM supports max 5-day "
            f"forecast; partial data if trip exceeds that window)"
        )
    else:
        logger.info(f"OpenWeatherMap data retrieved: {destination} - {condition}, {temp_c}°C")
    return result


def fetch_weather(destination: str, duration_days: int = 3, start_date: str = None) -> dict:
    """Fetch destination weather with a 3-tier fallback chain.

    Attempts weather retrieval in the following order:
      1. Google Weather API   - most accurate, requires Google Maps API key
      2. OpenWeatherMap API   - wide global coverage, requires OWM API key
      3. Static fallback      - hardcoded mild-climate data, always succeeds

    Each tier is tried only if the previous one fails or its API key is missing.

    When ``start_date`` is in the future, all tiers attempt to return a forecast
    anchored to that date rather than to today.  A warning is logged and
    ``is_forecast: True`` is set on the result when the trip date exceeds the
    reliable forecast horizon (Google Weather ~14 days, OWM ~5 days).

    Args:
        destination (str): Destination name (e.g. "Wuhan", "Tokyo", "Paris").
        duration_days (int): Number of travel days for daily forecast.
            Defaults to 3.
        start_date (str | None): Trip start date in YYYY-MM-DD format.
            When provided and in the future, the forecast is anchored to this
            date instead of today.  When ``None``, the current date is used.

    Returns:
        dict: Weather data dict guaranteed to succeed. Contains:
            condition (str): Current weather condition (e.g. "Sunny", "Cloudy").
            temp_c (int): Current temperature in Celsius.
            humidity (int): Relative humidity percentage (0-100).
            wind_speed_kmh (float): Wind speed in km/h.
            forecast_days (list[dict]): Daily forecasts with date, condition,
                temp_high, temp_low.
            weather_alerts (list[dict]): Active weather alerts with
                alert_type, severity, alert_level, headline.
            is_forecast (bool): ``True`` when start_date is in the future,
                indicating the data is a forecast rather than current conditions.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    is_future_trip = bool(start_date) and start_date > today_str

    if is_future_trip:
        # Calculate how many days until the trip starts
        days_until_trip = (datetime.strptime(start_date, "%Y-%m-%d") - datetime.now()).days
        if days_until_trip > 14:
            logger.info(
                f"Trip to {destination} starts in {days_until_trip} days "
                f"({start_date}). Beyond 14-day forecast range — fetching "
                f"last year's same-period historical weather."
            )
            # Tier 0: Try Open-Meteo historical weather (last year same period)
            hist_result = _fetch_historical_weather_open_meteo(destination, duration_days, start_date)
            if hist_result:
                return hist_result
            # Fallback to seasonal estimate if historical API fails
            logger.info(f"Historical weather unavailable, using seasonal estimate")
            return _build_fallback_weather(duration_days, start_date=start_date, destination=destination)
        elif days_until_trip > 5:
            logger.info(
                f"Trip to {destination} starts in {days_until_trip} days "
                f"({start_date}). OpenWeatherMap (5-day limit) may not cover "
                f"the full trip; Google Weather (14-day) will be used if available."
            )

    # Tier 1: Google Weather API (supports up to 14-day forecast via forecastStartDate)
    if GOOGLE_MAPS_API_KEY:
        try:
            return _fetch_google_weather(destination, GOOGLE_MAPS_API_KEY, duration_days,
                                        start_date=start_date)
        except Exception as e:
            logger.warning(f"Google Weather failed: {e}, trying OpenWeatherMap...")
    else:
        logger.info("Google Maps API Key is empty, skipping Google Weather")

    # Tier 2: OpenWeatherMap API (5-day forecast limit)
    if OPENWEATHER_API_KEY:
        try:
            return _fetch_openweathermap(destination, OPENWEATHER_API_KEY, duration_days,
                                        start_date=start_date)
        except Exception as e:
            logger.warning(f"OpenWeatherMap failed: {e}, using fallback weather data")
    else:
        logger.info("OpenWeatherMap API Key is empty, skipping OpenWeatherMap")

    # Tier 3: Static fallback (uses start_date to anchor generated dates)
    logger.info(f"All weather APIs failed for '{destination}', using seasonal climate estimate")
    return _build_fallback_weather(duration_days, start_date=start_date, destination=destination)
