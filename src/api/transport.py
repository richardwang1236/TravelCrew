"""src.api.transport — inter-POI transit time matrix via Google Distance Matrix.

Provides:
    - :func:`fetch_transport_matrix` — main entry point (supports per-day batching).
    - :func:`_fetch_single_batch_matrix` — single batch request (≤10 POIs).
    - :func:`generate_fallback_transport_matrix` — heuristic fallback matrix.
"""

import requests
from typing import Optional

from src.api.base import logger, API_TIMEOUT, GOOGLE_MAPS_API_KEY, _get_session


def _fetch_single_batch_matrix(pois: list[dict], key: str) -> dict[str, dict[str, int]]:
    """Fetch transit matrix for a single batch of POIs (≤10).

    Calls Google Distance Matrix API for the given POI batch. Falls back to
    generate_fallback_transport_matrix() when the API returns an error.

    Args:
        pois (list[dict]): POI list (≤10 items). Each dict must have 'name'
            (str); 'lat' and 'lng' (float) are required.
        key (str): Google Maps API key.

    Returns:
        dict[str, dict[str, int]]: Nested dict representing the transit time
            matrix: {origin_name: {dest_name: time_in_minutes}}.
            Self-pairs (origin == dest) are omitted.
    """
    try:
        # Build origins and destinations strings (batch request)
        locations = [f"{poi['lat']},{poi['lng']}" for poi in pois]
        origins = "|".join(locations)
        destinations = "|".join(locations)

        # Google Distance Matrix API
        # API 名称: Google Maps Distance Matrix API
        # Endpoint: https://maps.googleapis.com/maps/api/distancematrix/json
        # 请求参数:
        #   origins: "lat1,lng1|lat2,lng2|..." (多个起点, 用 | 分隔)
        #   destinations: "lat1,lng1|lat2,lng2|..." (多个终点, 用 | 分隔)
        #   mode: "transit" (公共交通模式)
        #   key: str (API密钥)
        #   language: "en"
        # Response format:
        # {
        #     "status": "OK" | "INVALID_REQUEST" | ...,
        #     "rows": [
        #         {
        #             "elements": [
        #                 {
        #                     "status": "OK" | "NOT_FOUND" | ...,
        #                     "duration": {"value": int (seconds), "text": "string"}
        #                 }
        #             ]
        #         }
        #     ]
        # }
        url = "https://maps.googleapis.com/maps/api/distancematrix/json"
        params = {
            "origins": origins,
            "destinations": destinations,
            "mode": "transit",
            "key": key,
            "language": "en"
        }
        resp = _get_session().get(url, params=params, timeout=API_TIMEOUT * 2)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "OK":
            logger.warning(f"Distance Matrix API returned status: {data.get('status')}, using fallback for batch")
            return generate_fallback_transport_matrix(pois)

        # Parse result matrix
        matrix = {}
        rows = data.get("rows", [])
        for i, row in enumerate(rows):
            origin_name = pois[i]["name"]
            matrix[origin_name] = {}
            for j, element in enumerate(row.get("elements", [])):
                if i == j:
                    continue  # Skip self
                dest_name = pois[j]["name"]
                if element.get("status") == "OK":
                    # Duration is in seconds; convert to minutes
                    duration_min = round(element["duration"]["value"] / 60)
                    matrix[origin_name][dest_name] = duration_min
                else:
                    # Route unreachable; use estimated value
                    matrix[origin_name][dest_name] = 30

        logger.info(f"Transport matrix batch retrieved successfully: {len(pois)} POIs")
        return matrix

    except Exception as e:
        logger.error(f"Failed to retrieve transport matrix batch: {e}, using fallback")
        return generate_fallback_transport_matrix(pois)


def fetch_transport_matrix(pois: list[dict], api_key: Optional[str] = None, daily_groups: Optional[list[list[dict]]] = None) -> dict[str, dict[str, int]]:
    """Fetch inter-POI public transit time matrix.

    When daily_groups is provided, computes per-group matrices and merges them.
    Otherwise, auto-batches to stay under the API's 100-element limit.
    Cross-group transit times use the fallback heuristic.

    Calls Google Distance Matrix API when all POIs have coordinates and an
    API key is available. Falls back to generate_fallback_transport_matrix()
    when API is unavailable or any POI is missing coordinates.

    Args:
        pois (list[dict]): POI list. Each dict must have 'name' (str);
            'lat' and 'lng' (float) are preferred for accurate results.
        api_key (Optional[str]): Google Maps API key. If None, uses the
            globally configured GOOGLE_MAPS_API_KEY.
        daily_groups (Optional[list[list[dict]]]): POIs grouped by day. When
            provided, each group is requested separately to stay under the
            API's 100-element limit (10 origins × 10 destinations).

    Returns:
        dict[str, dict[str, int]]: Nested dict representing the transit time
            matrix: {origin_name: {dest_name: time_in_minutes}}.
            Self-pairs (origin == dest) are omitted.
    """
    key = api_key or GOOGLE_MAPS_API_KEY
    if not key:
        logger.info("Google Maps API Key is empty, using fallback transport matrix")
        return generate_fallback_transport_matrix(pois)

    # Check whether all POIs have coordinates
    has_coords = all(poi.get("lat") and poi.get("lng") for poi in pois)
    if not has_coords:
        logger.info("Some POIs are missing coordinates, using fallback transport matrix")
        return generate_fallback_transport_matrix(pois)

    # Determine groups
    if daily_groups:
        groups = daily_groups
    else:
        # Auto-batch: max 10 POIs per group (10×10=100 elements)
        max_per_batch = 10
        groups = [pois[i:i+max_per_batch] for i in range(0, len(pois), max_per_batch)]

    # Fetch matrix for each group and merge
    merged_matrix = {}
    for group in groups:
        if len(group) <= 1:
            continue
        # Filter out POIs missing coordinates within each group
        valid_group = [poi for poi in group if poi.get("lat") and poi.get("lng")]
        if len(valid_group) <= 1:
            continue
        group_matrix = _fetch_single_batch_matrix(valid_group, key)
        for origin, dests in group_matrix.items():
            if origin not in merged_matrix:
                merged_matrix[origin] = {}
            merged_matrix[origin].update(dests)

    # For cross-group pairs, use fallback heuristic (estimate 30 min)
    all_names = [poi["name"] for poi in pois]
    for origin in all_names:
        if origin not in merged_matrix:
            merged_matrix[origin] = {}
        for dest in all_names:
            if origin == dest:
                continue
            if dest not in merged_matrix[origin]:
                merged_matrix[origin][dest] = 30  # Cross-group default

    logger.info(f"Transport matrix retrieved successfully: {len(pois)} POIs")
    return merged_matrix


# ---------------------------------------------------------------------------
# 7. generate_fallback_transport_matrix - Fallback Transport Matrix
# ---------------------------------------------------------------------------
#
# Heuristic-based estimation used when the Google Distance Matrix API is
# unavailable. Assumes POIs are ordered by proximity and estimates travel
# time based on position gap in the list.
# ---------------------------------------------------------------------------

def generate_fallback_transport_matrix(pois: list[dict]) -> dict[str, dict[str, int]]:
    """Generate an estimated transport time matrix based on POI list order.

    Uses a heuristic where adjacent attractions are ~15 minutes apart,
    and travel time increases by 5 minutes per intermediate POI, capped
    at 45 minutes maximum.

    Args:
        pois (list[dict]): POI list (must include 'name' field).
            Order is assumed to reflect approximate geographic proximity.

    Returns:
        dict[str, dict[str, int]]: Estimated transit time matrix:
            {origin_name: {dest_name: time_in_minutes}}.
            Self-pairs (i == j) are omitted.
    """
    matrix = {}
    n = len(pois)

    for i in range(n):
        origin = pois[i]["name"]
        matrix[origin] = {}
        for j in range(n):
            if i == j:
                continue
            dest = pois[j]["name"]
            # Travel time increases with gap: base 15 min + 5 min per intermediate POI
            gap = abs(i - j)
            travel_time = min(15 + (gap - 1) * 5, 45)  # Max 45 minutes
            matrix[origin][dest] = travel_time

    return matrix
