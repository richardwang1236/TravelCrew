"""src.api.attractions — POI (Point of Interest) data retrieval via Google Places.

Provides:
    - :func:`fetch_attractions` — search nearby attractions for a destination.
    - :func:`search_specific_place` — search a specific place by name.
    - :func:`get_destination_cost_multiplier` — regional cost multiplier lookup.
    - :func:`_fetch_place_detail` — fetch Google Place Details.
    - :func:`_convert_place_to_poi` — convert Google Place data to POI format.

Constants:
    FALLBACK_POIS, _PLACE_TYPE_MAP, _PRICE_LEVEL_TO_COST,
    _TYPE_DEFAULT_COST_USD, _TYPE_TO_VISIT_TIME
"""

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from src.api.base import logger, API_TIMEOUT, GOOGLE_MAPS_API_KEY, _get_session
from src.api.geocoding import _haversine_distance
from src.api.ai_search import fetch_must_visit_places
from src.config import (
    PLACES_SEARCH_RADIUS,
    _DESTINATION_COST_MULTIPLIERS,
)


# ---------------------------------------------------------------------------
# Fallback Constant Data
# Used when all external API calls fail. Provides static placeholder values
# so downstream processing (synthesizer, budget calc) can continue gracefully.
# ---------------------------------------------------------------------------

# Hardcoded fallback POI data for Wuhan (武汉), used when Google Places API
# is unavailable or the API key is missing. Each entry follows the standard
# POI schema produced by _convert_place_to_poi().
# Fields: name, type, cost (USD), rating, review_count, opening_hours,
#         avg_visit_time_min, tags, description, distance_km
FALLBACK_POIS = [
    {
        "name": "黄鹤楼",
        "type": "cultural",
        "cost": 10,
        "rating": 4.6,
        "review_count": 28500,
        "opening_hours": "08:00-18:00",
        "avg_visit_time_min": 90,
        "tags": ["landmark", "history", "poetry"],
        "description": "江南三大名楼之首，千年诗词文化的象征，俯瞰长江壮丽景色",
        "distance_km": 1.0
    },
    {
        "name": "东湖风景区",
        "type": "outdoor",
        "cost": 0,
        "rating": 4.7,
        "review_count": 22000,
        "opening_hours": "06:00-22:00",
        "avg_visit_time_min": 150,
        "tags": ["lake", "nature", "cycling"],
        "description": "中国最大的城中湖，绿道骑行、湖光山色，四季皆宜",
        "distance_km": 8.0
    },
    {
        "name": "户部巷",
        "type": "dining",
        "cost": 15,
        "rating": 4.3,
        "review_count": 18700,
        "opening_hours": "06:00-23:00",
        "avg_visit_time_min": 60,
        "tags": ["street_food", "breakfast", "local_cuisine"],
        "description": "武汉早餐一条街，热干面、豆皮、三鲜豆皮等地道小吃汇聚",
        "distance_km": 1.2
    },
    {
        "name": "武汉大学",
        "type": "outdoor",
        "cost": 0,
        "rating": 4.8,
        "review_count": 15600,
        "opening_hours": "08:00-17:00",
        "avg_visit_time_min": 90,
        "tags": ["cherry_blossom", "campus", "architecture"],
        "description": "中国最美大学之一，春季樱花大道闻名全国，民国建筑群独具魅力",
        "distance_km": 5.0
    },
    {
        "name": "江汉路步行街",
        "type": "shopping",
        "cost": 20,
        "rating": 4.4,
        "review_count": 12300,
        "opening_hours": "09:00-22:00",
        "avg_visit_time_min": 90,
        "tags": ["shopping", "architecture", "nightlife"],
        "description": "百年商业街区，欧式建筑与现代商业交融，夜景璀璨",
        "distance_km": 3.0
    },
    {
        "name": "归元寺",
        "type": "cultural",
        "cost": 3,
        "rating": 4.5,
        "review_count": 9800,
        "opening_hours": "08:00-17:00",
        "avg_visit_time_min": 60,
        "tags": ["temple", "buddhism", "meditation"],
        "description": "武汉四大丛林之一，五百罗汉栩栩如生，闹市中的清净之地",
        "distance_km": 4.5
    },
    {
        "name": "晴川阁",
        "type": "cultural",
        "cost": 0,
        "rating": 4.4,
        "review_count": 7600,
        "opening_hours": "08:30-17:00",
        "avg_visit_time_min": 45,
        "tags": ["history", "architecture", "riverside"],
        "description": "与黄鹤楼隔江相望，古典楼阁与大禹神话交织的文化胜地",
        "distance_km": 2.0
    },
    {
        "name": "武昌起义纪念馆",
        "type": "indoor",
        "cost": 0,
        "rating": 4.5,
        "review_count": 11200,
        "opening_hours": "09:00-17:00",
        "avg_visit_time_min": 60,
        "tags": ["museum", "history", "revolution"],
        "description": "辛亥革命第一枪打响之地，红楼建筑本身即为珍贵历史文物",
        "distance_km": 1.5
    },
    {
        "name": "吉庆街美食街",
        "type": "dining",
        "cost": 20,
        "rating": 4.2,
        "review_count": 8900,
        "opening_hours": "11:00-02:00",
        "avg_visit_time_min": 60,
        "tags": ["nightlife", "local_food", "atmosphere"],
        "description": "武汉夜生活地标，大排档与民间艺人共存，烟火气十足",
        "distance_km": 3.5
    },
    {
        "name": "长江大桥",
        "type": "outdoor",
        "cost": 0,
        "rating": 4.6,
        "review_count": 19500,
        "opening_hours": "24h",
        "avg_visit_time_min": 45,
        "tags": ["landmark", "bridge", "sunset"],
        "description": "万里长江第一桥，步行横跨长江感受工程奇迹，日落时分尤为壮观",
        "distance_km": 1.0
    }
]

# Google Places API "types" array element -> internal POI category mapping.
# Used by _convert_place_to_poi() to classify each place into one of:
#   "dining", "cultural", "outdoor", "indoor", "shopping"
# Default (unmatched) type is "outdoor".
_PLACE_TYPE_MAP = {
    "restaurant": "dining",
    "cafe": "dining",
    "food": "dining",
    "museum": "indoor",
    "art_gallery": "indoor",
    "temple": "cultural",
    "church": "cultural",
    "hindu_temple": "cultural",
    "mosque": "cultural",
    "synagogue": "cultural",
    "park": "outdoor",
    "natural_feature": "outdoor",
    "shopping_mall": "shopping",
    "store": "shopping",
    "tourist_attraction": "outdoor",
}

# Google Places price_level (0-4) -> estimated base cost in USD.
# 0 = Free, 1 = Inexpensive, 2 = Moderate, 3 = Expensive, 4 = Very Expensive
# None (missing) defaults to $10.
# This base cost is further adjusted by _DESTINATION_COST_MULTIPLIERS.
_PRICE_LEVEL_TO_COST = {0: 0, 1: 5, 2: 15, 3: 30, 4: 50, None: 10}

# Fallback default cost (USD) inferred from Google Places types when price_level
# is missing. Lookup uses the first matching type in the place's types list.
_TYPE_DEFAULT_COST_USD = {
    # Free or very cheap
    "park": 0,
    "natural_feature": 0,
    "neighborhood": 0,
    "sublocality": 0,
    "locality": 0,
    "street_address": 0,
    "route": 0,
    "transit_station": 0,
    "bus_station": 0,
    "train_station": 0,
    "point_of_interest": 5,

    # Low cost (public spaces, places of worship)
    "church": 0,
    "mosque": 0,
    "hindu_temple": 0,
    "synagogue": 0,
    "place_of_worship": 0,
    "cemetery": 0,
    "city_hall": 0,
    "library": 0,
    "local_government_office": 0,

    # Medium cost (museums, galleries, zoos)
    "museum": 8,
    "art_gallery": 6,
    "zoo": 12,
    "aquarium": 15,
    "amusement_park": 25,
    "stadium": 15,
    "tourist_attraction": 8,

    # Dining (use price_level if available, else moderate default)
    "restaurant": 12,
    "cafe": 5,
    "bakery": 4,
    "bar": 8,
    "meal_takeaway": 6,
    "food": 8,

    # Shopping
    "shopping_mall": 0,
    "store": 0,
    "clothing_store": 0,
    "department_store": 0,

    # Entertainment
    "movie_theater": 10,
    "night_club": 15,
    "spa": 20,
    "gym": 10,
}


def get_destination_cost_multiplier(destination: str) -> float:
    """Get the regional cost multiplier for a given destination.

    Looks up the destination in _DESTINATION_COST_MULTIPLIERS using exact
    match first, then partial (substring) match. Returns 1.0 (US baseline)
    if the destination is not found.

    Args:
        destination (str): Destination city or region name (case-insensitive).

    Returns:
        float: Cost multiplier relative to US baseline (e.g. 0.7 for Wuhan,
               1.6 for London). 1.0 if unknown.
    """
    dest_lower = destination.lower().strip()
    # Try exact match first
    if dest_lower in _DESTINATION_COST_MULTIPLIERS:
        return _DESTINATION_COST_MULTIPLIERS[dest_lower]
    # Try partial match
    for key, val in _DESTINATION_COST_MULTIPLIERS.items():
        if key in dest_lower or dest_lower in key:
            return val
    return 1.0

# POI internal category -> estimated average visit duration in minutes.
# Used by _convert_place_to_poi() to set avg_visit_time_min when the API
# does not provide visit duration data.
_TYPE_TO_VISIT_TIME = {
    "shopping": 90,
    "dining": 45,
    "cultural": 60,
    "outdoor": 75,
    "indoor": 90,
}


# ---------------------------------------------------------------------------
# 3. fetch_attractions - Attraction Data Retrieval (Google Places)
# ---------------------------------------------------------------------------
#
# 3-step pipeline:
#   1. Geocode destination to get center lat/lng
#   2. Search nearby places by type (tourist_attraction, restaurant, etc.)
#   3. Fetch place details and convert to standard POI format
# Falls back to FALLBACK_POIS on API failure.
# ---------------------------------------------------------------------------

def fetch_attractions(destination: str, api_key: Optional[str] = None, limit: int = 12, language: str = "en", interests: str = "") -> list[dict]:
    """Fetch destination POI (Point of Interest) data via a two-stage pipeline.

    Stage 1 — AI-guided discovery (SerpApi Google AI Mode):
        Queries "what are the must-visit places in {destination}?" and
        extracts a list of specific place names from the AI response.

    Stage 2 — Google Places enrichment:
        For each name returned by Stage 1, calls Google Places Text Search
        to get precise coordinates, ratings, price levels, website URLs,
        and other structured data.

    If Stage 1 fails or returns too few results, falls back to the legacy
    Nearby Search approach (searching by type around the center point).

    Falls back to FALLBACK_POIS if all API calls fail.

    Args:
        destination (str): Destination city name (e.g. "Wuhan", "Tokyo").
        api_key (Optional[str]): Google Maps API key. If None, uses the
            globally configured GOOGLE_MAPS_API_KEY.
        limit (int): Maximum number of POIs to return. Defaults to 12.
        language (str): Language code for results (``"en"`` or ``"zh-CN"``).
        interests (str): Optional user interests string (e.g. "history, food")
            passed to the AI Mode query to refine results.

    Returns:
        list[dict]: List of POI dicts, each containing:
            name, type, cost, rating, review_count, opening_hours,
            avg_visit_time_min, tags, description, distance_km,
            lat, lng, website, maps_url, place_id.

    Raises:
        No exceptions raised; all errors are caught and fallback POIs returned.
    """
    key = api_key or GOOGLE_MAPS_API_KEY
    if not key:
        logger.info("Google Maps API Key is empty, using fallback POI data")
        return FALLBACK_POIS[:limit]

    # ── Stage 1: AI-guided discovery + Geocoding in parallel ──
    # These are independent I/O calls — running them together saves ~0.5-3s.
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_ai = pool.submit(fetch_must_visit_places, destination, language=language, interests=interests, count=limit)
        future_geo = pool.submit(_geocode_destination, destination, key)
        ai_places = future_ai.result()
        lat, lng = future_geo.result()

    if lat == 0.0 and lng == 0.0:
        logger.warning(f"Google Geocoding failed for '{destination}', using fallback")
        if not ai_places:
            return FALLBACK_POIS[:limit]

    pois = []
    seen_place_ids = set()
    seen_names = set()

    # ── Stage 2a: Enrich AI-discovered places via Google Places Text Search ──
    # Each place search is an independent API call — parallelize for ~5-10x speedup.
    if ai_places:
        logger.info(f"Enriching {len(ai_places)} AI-discovered places via Google Places (parallel)...")
        max_workers = min(len(ai_places), 8)  # Cap at 8 concurrent to avoid rate limits
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_text_search_place, name, destination, key, language, lat, lng): name
                for name in ai_places[:limit * 2]
            }
            for future in as_completed(futures):
                if len(pois) >= limit:
                    break
                poi = future.result()
                if poi:
                    pid = poi.get("place_id", poi["name"])
                    if pid not in seen_place_ids and poi["name"] not in seen_names:
                        seen_place_ids.add(pid)
                        seen_names.add(poi["name"])
                        pois.append(poi)

        logger.info(f"AI-guided search yielded {len(pois)} POIs for '{destination}'")

    # ── Stage 2b: Fallback — Nearby Search by type if AI yielded too few ──
    if len(pois) < limit:
        needed = limit - len(pois)
        logger.info(f"AI yielded {len(pois)} POIs, need {needed} more via Nearby Search fallback...")
        fallback_pois = _nearby_search_fallback(
            destination, key, needed * 2, language, lat, lng,
            seen_place_ids, seen_names
        )
        pois.extend(fallback_pois)

    if pois:
        logger.info(f"Attraction data retrieved successfully: {destination}, {len(pois)} POIs")
        return pois[:limit]

    # Ultimate fallback
    logger.warning(f"All search methods failed for '{destination}', using static fallback POIs")
    return FALLBACK_POIS[:limit]


def _geocode_destination(destination: str, key: str) -> tuple[float, float]:
    """Geocode a destination name to (lat, lng) coordinates.

    Returns (0.0, 0.0) on failure.
    """
    try:
        geo_url = "https://maps.googleapis.com/maps/api/geocode/json"
        geo_params = {"address": destination, "key": key}
        geo_resp = _get_session().get(geo_url, params=geo_params, timeout=API_TIMEOUT)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        if geo_data.get("status") == "OK" and geo_data.get("results"):
            loc = geo_data["results"][0]["geometry"]["location"]
            return loc["lat"], loc["lng"]
    except Exception as e:
        logger.debug(f"Geocoding failed for '{destination}': {e}")
    return 0.0, 0.0


def _text_search_place(place_name: str, destination: str, key: str, language: str, center_lat: float, center_lng: float) -> Optional[dict]:
    """Search for a specific place via Google Places Text Search and convert to POI.

    Args:
        place_name (str): Name of the place to search.
        destination (str): Destination city for scoping.
        key (str): Google Maps API key.
        language (str): Language code for results.
        center_lat (float): Center latitude for distance calculation.
        center_lng (float): Center longitude for distance calculation.

    Returns:
        Optional[dict]: Standard POI dict or None if not found.
    """
    try:
        text_search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        text_params = {
            "query": f"{place_name} in {destination}",
            "key": key,
            "language": language,
        }
        resp = _get_session().get(text_search_url, params=text_params, timeout=API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            logger.debug(f"No Google Places results for '{place_name} in {destination}'")
            return None

        place = results[0]
        place_id = place.get("place_id")
        if not place_id:
            return None

        # Fetch detailed info including website URL
        detail = _fetch_place_detail(place_id, key, language=language)

        return _convert_place_to_poi(place, detail, center_lat, center_lng, destination)
    except Exception as e:
        logger.debug(f"Text search failed for '{place_name}': {e}")
        return None


def _nearby_search_fallback(destination: str, key: str, limit: int, language: str, lat: float, lng: float, seen_place_ids: set, seen_names: set) -> list[dict]:
    """Fallback: search nearby places by type (legacy approach).

    Used when AI-guided search yields too few results. Searches multiple
    place types in parallel via ThreadPoolExecutor.
    """
    pois = []
    if lat == 0.0 and lng == 0.0:
        return pois

    search_types = ["tourist_attraction", "restaurant", "cafe", "museum", "park", "temple"]
    raw_places = []

    # Parallelize the 6 Nearby Search API calls (one per place type).
    # Each is independent — this saves ~5 sequential API round-trips.
    def _search_one_type(place_type):
        results = []
        nearby_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        nearby_params = {
            "location": f"{lat},{lng}",
            "radius": PLACES_SEARCH_RADIUS,
            "type": place_type,
            "key": key,
            "language": language,
        }
        try:
            resp = _get_session().get(nearby_url, params=nearby_params, timeout=API_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            for place in data.get("results", [])[:6]:
                place_id = place.get("place_id")
                name = place.get("name", "")
                if place_id and name:
                    results.append(place)
        except Exception as e:
            logger.debug(f"Nearby search failed for type '{place_type}': {e}")
        return results

    with ThreadPoolExecutor(max_workers=min(len(search_types), 4)) as pool:
        futures = {pool.submit(_search_one_type, t): t for t in search_types}
        for future in as_completed(futures):
            if len(raw_places) >= limit:
                continue
            for place in future.result():
                place_id = place.get("place_id")
                name = place.get("name", "")
                if place_id not in seen_place_ids and name not in seen_names:
                    seen_place_ids.add(place_id)
                    seen_names.add(name)
                    raw_places.append(place)
                    if len(raw_places) >= limit:
                        break

    # Fetch place details in parallel for all collected raw places
    if raw_places:
        with ThreadPoolExecutor(max_workers=min(len(raw_places), 6)) as pool:
            detail_futures = {
                pool.submit(_fetch_place_detail, p.get("place_id"), key, language=language): p
                for p in raw_places[:limit]
            }
            for future in as_completed(detail_futures):
                if len(pois) >= limit:
                    break
                place = detail_futures[future]
                detail = future.result()
                poi = _convert_place_to_poi(place, detail, lat, lng, destination)
                if poi:
                    pois.append(poi)

    return pois


def _fetch_place_detail(place_id: str, api_key: str, language: str = "en") -> dict:
    """Fetch detailed information for a single Google Place.

    Args:
        place_id (str): Google Place ID string.
        api_key (str): Google Maps API key.

    Returns:
        dict: Place detail result dict containing:
            rating (float), opening_hours (dict), price_level (int),
            user_ratings_total (int), reviews (list).
            Returns empty dict on failure.

    Raises:
        No exceptions raised; errors are caught and logged.
    """
    try:
        # Google Place Details API
        # API 名称: Google Places API - Place Details
        # Endpoint: https://maps.googleapis.com/maps/api/place/details/json
        # 请求参数:
        #   place_id: str (Place ID)
        #   fields: "rating,opening_hours,price_level,user_ratings_total,reviews"
        #   key: str (API密钥)
        #   language: "en"
        # Response format:
        # {
        #     "result": {
        #         "rating": float (0-5),
        #         "user_ratings_total": int,
        #         "price_level": int (0-4),
        #         "opening_hours": {
        #             "weekday_text": ["Monday: 9:00 AM – 5:00 PM", ...]
        #         },
        #         "reviews": [
        #             {"text": "string", "rating": int, "author_name": "string"}
        #         ]
        #     },
        #     "status": "OK" | "INVALID_REQUEST" | ...
        # }
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {
            "place_id": place_id,
            "fields": "rating,opening_hours,price_level,user_ratings_total,reviews,website,url,formatted_address,international_phone_number",
            "key": api_key,
            "language": language
        }
        resp = _get_session().get(url, params=params, timeout=API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result", {})
    except Exception as e:
        logger.debug(f"Failed to fetch place details (place_id={place_id}): {e}")
        return {}


def _convert_place_to_poi(place: dict, detail: dict, center_lat: float, center_lng: float, destination: str = "") -> Optional[dict]:
    """Convert Google Places API data to the internal standard POI format.

    Merges data from the Nearby Search result (place) and the Place Details
    result (detail) into a single POI dict with regional cost adjustment.

    Input field mapping (Google Places -> POI):
        place["name"]               -> poi["name"]
        place["types"]              -> poi["type"]     (via _PLACE_TYPE_MAP)
        detail/place["price_level"] -> poi["cost"]     (via _PRICE_LEVEL_TO_COST * regional multiplier)
        detail/place["rating"]      -> poi["rating"]
        detail/place["user_ratings_total"] -> poi["review_count"]
        detail["opening_hours"]["weekday_text"][0] -> poi["opening_hours"]
        poi["type"]                 -> poi["avg_visit_time_min"] (via _TYPE_TO_VISIT_TIME)
        place["types"]              -> poi["tags"]     (filtered, max 3)
        place["vicinity"]           -> poi["description"]
        haversine(place_loc, center) -> poi["distance_km"]
        place["geometry"]["location"] -> poi["lat"], poi["lng"]

    Args:
        place (dict): Google Places Nearby Search result entry.
        detail (dict): Google Place Details result (from _fetch_place_detail).
        center_lat (float): Destination center latitude for distance calculation.
        center_lng (float): Destination center longitude for distance calculation.
        destination (str): Destination name for regional cost multiplier.
            Defaults to "" (no adjustment, multiplier = 1.0).

    Returns:
        Optional[dict]: Standard POI dict with fields:
            name, type, cost, rating, review_count, opening_hours,
            avg_visit_time_min, tags, description, distance_km, lat, lng.
            Returns None if conversion fails.

    Raises:
        No exceptions raised; errors are caught and None is returned.
    """
    try:
        name = place.get("name", "Unknown Place")

        # Type mapping
        place_types = place.get("types", [])
        poi_type = "outdoor"  # default
        for t in place_types:
            if t in _PLACE_TYPE_MAP:
                poi_type = _PLACE_TYPE_MAP[t]
                break

        # price_level -> cost (adjusted by regional cost multiplier)
        # When price_level is missing (common for Chinese POIs), infer a more
        # reasonable base cost from the place's Google types instead of always
        # falling back to $10.
        price_level = detail.get("price_level", place.get("price_level"))

        if price_level is not None:
            # Google provided price_level — use standard mapping
            base_cost = _PRICE_LEVEL_TO_COST.get(price_level, 10)
        else:
            # No price_level — infer from place types
            place_types = detail.get("types", place.get("types", []))
            base_cost = 10  # ultimate fallback
            for t in place_types:
                if t in _TYPE_DEFAULT_COST_USD:
                    base_cost = _TYPE_DEFAULT_COST_USD[t]
                    break

        cost = round(base_cost * get_destination_cost_multiplier(destination), 1)

        # Rating
        rating = detail.get("rating", place.get("rating", 4.0))
        review_count = detail.get("user_ratings_total", place.get("user_ratings_total", 0))

        # Opening hours
        opening_hours = "09:00-18:00"  # default
        hours_info = detail.get("opening_hours", {})
        if hours_info and hours_info.get("weekday_text"):
            # Use the first day's hours as representative
            first_day = hours_info["weekday_text"][0]
            # Try to extract the time portion
            if ":" in first_day:
                time_part = first_day.split(":", 1)[1].strip() if ":" in first_day else first_day
                opening_hours = time_part
        elif place.get("opening_hours", {}).get("open_now") is not None:
            opening_hours = "Open" if place["opening_hours"]["open_now"] else "Closed"

        # Average visit time
        avg_visit_time_min = _TYPE_TO_VISIT_TIME.get(poi_type, 60)

        # Calculate distance (simple straight-line estimate)
        place_loc = place.get("geometry", {}).get("location", {})
        place_lat = place_loc.get("lat", center_lat)
        place_lng = place_loc.get("lng", center_lng)
        distance_km = _haversine_distance(center_lat, center_lng, place_lat, place_lng)

        # Tags
        tags = [t for t in place_types if t not in ("point_of_interest", "establishment")][:3]

        # Description
        description = place.get("vicinity", "") or name

        # Website URL from Google Place Details (official website if available)
        website = detail.get("website", "")

        # OpenStreetMap URL for embedding / linking (no VPN required, globally accessible).
        # Prefer the Google-provided URL; fall back to an OpenStreetMap link.
        maps_url = detail.get("url", "")
        if not maps_url and place_lat and place_lng:
            maps_url = f"https://www.openstreetmap.org/?mlat={place_lat}&mlon={place_lng}#map=15/{place_lat}/{place_lng}"

        # Formatted address (more detailed than vicinity)
        formatted_address = detail.get("formatted_address", "")
        if formatted_address:
            description = formatted_address

        # Place ID (useful for downstream lookups)
        place_id = place.get("place_id", "")

        return {
            "name": name,
            "type": poi_type,
            "cost": cost,
            "rating": round(rating, 1),
            "review_count": review_count,
            "opening_hours": opening_hours,
            "avg_visit_time_min": avg_visit_time_min,
            "tags": tags,
            "description": description,
            "distance_km": round(distance_km, 1),
            "lat": place_lat,
            "lng": place_lng,
            "website": website,
            "maps_url": maps_url,
            "place_id": place_id,
        }
    except Exception as e:
        logger.debug(f"POI conversion failed: {e}")
        return None


# ---------------------------------------------------------------------------
# 3b. search_specific_place - Dynamic Place Search by Name
# ---------------------------------------------------------------------------
#
# Searches for a specific place (e.g. user-requested POI) using Google
# Places Text Search API. Returns a standard POI dict compatible with
# the fetch_attractions output format.
# ---------------------------------------------------------------------------

def search_specific_place(place_name: str, destination: str, api_key: str, language: str = "en") -> Optional[dict]:
    """Search for a specific place by name using Google Places Text Search API.

    Performs a 4-step pipeline:
      1. Text Search to find the place by name.
      2. Place Details to fetch enriched info (rating, hours, price).
      3. Geocode destination for center coordinates (distance calculation).
      4. Convert to standard POI format via _convert_place_to_poi.

    Args:
        place_name (str): Name of the place to search (e.g. "Disneyland").
        destination (str): Destination city for scoping the search
            (e.g. "Shanghai").
        api_key (str): Google Maps API key.

    Returns:
        Optional[dict]: Standard POI dict (same schema as fetch_attractions
            output), or None if not found or on any failure.

    Raises:
        No exceptions raised; all errors are caught and None is returned.
    """
    if not api_key:
        logger.info("Google Maps API Key is empty, cannot search for place")
        return None

    try:
        # Step 1: Text Search to find the place
        # Google Places Text Search API
        # API 名称: Google Places API - Text Search
        # Endpoint: https://maps.googleapis.com/maps/api/place/textsearch/json
        # 请求参数:
        #   query: "{place_name} in {destination}" (搜索关键词)
        #   key: str (API密钥)
        #   language: "en"
        # Response format:
        # {
        #     "results": [
        #         {
        #             "name": "string",
        #             "place_id": "string",
        #             "formatted_address": "string",
        #             "geometry": {"location": {"lat": float, "lng": float}},
        #             "rating": float,
        #             "types": ["string", ...]
        #         }
        #     ],
        #     "status": "OK" | "ZERO_RESULTS" | ...
        # }
        text_search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        text_params = {
            "query": f"{place_name} in {destination}",
            "key": api_key,
            "language": language,
        }
        resp = _get_session().get(text_search_url, params=text_params, timeout=API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            logger.warning(f"No Google Places results for '{place_name} in {destination}'")
            return None

        place = results[0]
        place_id = place.get("place_id")
        if not place_id:
            logger.warning(f"First result for '{place_name}' has no place_id")
            return None

        # Step 2: Place Details for enriched info
        detail = _fetch_place_detail(place_id, api_key, language=language)

        # Step 3: Geocode destination for center coordinates (distance calc)
        geo_url = "https://maps.googleapis.com/maps/api/geocode/json"
        geo_params = {"address": destination, "key": api_key}
        geo_resp = _get_session().get(geo_url, params=geo_params, timeout=API_TIMEOUT)
        center_lat, center_lng = 0.0, 0.0
        if geo_resp.ok:
            geo_data = geo_resp.json()
            if geo_data.get("results"):
                loc = geo_data["results"][0]["geometry"]["location"]
                center_lat, center_lng = loc["lat"], loc["lng"]

        # Step 4: Convert to standard POI format (reuse existing helper)
        poi = _convert_place_to_poi(place, detail, center_lat, center_lng, destination)
        if poi:
            logger.info(f"Successfully found place: '{poi['name']}' for query '{place_name}'")
        return poi

    except Exception as e:
        logger.warning(f"search_specific_place failed for '{place_name}' in '{destination}': {e}")
        return None
