"""src.api.geocoding — geocoding and coordinate utilities.

Provides:
    - :func:`geocode_place` — resolve a place name to GPS coordinates.
    - :func:`backfill_missing_coordinates` — fill missing lat/lng for POIs.
    - :func:`_haversine_distance` — great-circle distance between two points.
"""

import requests

from src.api.base import logger, _get_session


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two GPS points.

    Uses the Haversine formula to compute the shortest distance over the
    Earth's surface between two latitude/longitude coordinate pairs.

    Args:
        lat1 (float): Latitude of point 1 in degrees.
        lon1 (float): Longitude of point 1 in degrees.
        lat2 (float): Latitude of point 2 in degrees.
        lon2 (float): Longitude of point 2 in degrees.

    Returns:
        float: Distance in kilometres (km).
    """
    import math
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def geocode_place(place_name: str, destination: str, api_key: str) -> tuple:
    """Geocode a place name to GPS coordinates using Google Geocoding API.

    Args:
        place_name (str): Name of the place to geocode (e.g. "Eiffel Tower").
        destination (str): Destination city for scoping the search
            (e.g. "Paris").
        api_key (str): Google Maps API key.

    Returns:
        tuple: (lat, lng) as floats, or (None, None) if not found or on failure.

    Raises:
        No exceptions raised; errors are caught and (None, None) is returned.
    """
    try:
        # Google Maps Geocoding API
        # API 名称: Google Maps Geocoding API
        # Endpoint: https://maps.googleapis.com/maps/api/geocode/json
        # 请求参数: address ("{place_name}, {destination}"), key (API密钥)
        # Response format:
        # {
        #     "results": [
        #         {
        #             "geometry": {"location": {"lat": float, "lng": float}},
        #             "formatted_address": "string"
        #         }
        #     ],
        #     "status": "OK" | "ZERO_RESULTS" | ...
        # }
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "address": f"{place_name}, {destination}",
            "key": api_key
        }
        resp = _get_session().get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") == "OK" and data.get("results"):
            location = data["results"][0]["geometry"]["location"]
            return location["lat"], location["lng"]
    except Exception as e:
        logger.debug(f"Geocoding failed for '{place_name}': {e}")
    return None, None


def backfill_missing_coordinates(pois: list[dict], destination: str, api_key: str) -> list[dict]:
    """Backfill missing lat/lng coordinates for POIs using Google Geocoding API.

    Iterates through the POI list and geocodes any POI that is missing
    lat or lng coordinates. Modifies POIs in-place and returns the same list.
    Only calls the API for POIs that actually need coordinates to conserve
    API quota.

    Args:
        pois (list[dict]): List of POI dicts (must have 'name' field).
        destination (str): Destination city for scoping geocode queries.
        api_key (str): Google Maps API key.

    Returns:
        list[dict]: The same POI list with coordinates filled in where possible.
            POIs that could not be geocoded remain unchanged.

    Raises:
        No exceptions raised; all errors are caught and logged.
    """
    if not api_key:
        logger.info("Google Maps API Key is empty, skipping coordinate backfill")
        return pois

    missing_coords = [p for p in pois if not p.get("lat") or not p.get("lng")]
    if not missing_coords:
        return pois

    logger.info(f"Backfilling coordinates for {len(missing_coords)} POIs missing lat/lng...")
    filled_count = 0
    for poi in missing_coords:
        name = poi.get("name", "")
        if not name:
            continue
        lat, lng = geocode_place(name, destination, api_key)
        if lat is not None and lng is not None:
            poi["lat"] = lat
            poi["lng"] = lng
            filled_count += 1
            logger.debug(f"Backfilled coordinates for '{name}': ({lat}, {lng})")

    logger.info(f"Coordinate backfill complete: {filled_count}/{len(missing_coords)} POIs filled")
    return pois
