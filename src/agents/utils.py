"""Utility functions and shared constants for agent nodes.

This module collects non-node helper functions used across multiple agent
nodes: progress message building, currency detection, exchange rate fetching,
POI classification, daily structure enforcement, and the shared LLM client
singleton.

Imported by individual node modules (intent_parser, information, etc.) to
avoid code duplication.
"""

import ast
import json
import logging
import re
import time

import requests

from src.api.base import _get_session

from src.state import TravelState
from src.llm import DeepSeekChatClient
from src.config import (
    EXCHANGE_RATES,
    _TRANSPORT_COST_PER_MIN,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_TIMEOUT,
    LLM_MAX_RETRIES,
)

logger = logging.getLogger(__name__)


def _safe_json_parse(response: str, context: str = "LLM") -> dict | list:
    """Robustly parse LLM JSON/dict responses with progressive fallbacks.

    LLMs (even with ``json_format=True``) occasionally return malformed JSON:
    - Python-style single-quoted keys (``{'key': 'value'}``)
    - Markdown code fences (`````json ... `````)
    - Trailing commas, unescaped line breaks in strings

    This function tries multiple strategies in order, logging each failure:

    1. Direct ``json.loads`` (standard case).
    2. Strip markdown code fences, then ``json.loads``.
    3. ``ast.literal_eval`` for Python-dict-style responses.
    4. Regex-based key repair (``'key':`` → ``"key":``), then ``json.loads``.

    Args:
        response: Raw LLM text response.
        context: Label for log messages (e.g. "Recommendation", "IntentParser").

    Returns:
        Parsed dict or list.

    Raises:
        ValueError: If all parsing strategies fail, with the original response
            truncated to 300 chars for debugging.
    """
    strategies_tried = []

    # --- Strategy 1: Direct JSON parse ---------------------------------
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        strategies_tried.append(f"json.loads: {e}")

    # --- Strategy 2: Strip markdown code fences ------------------------
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            strategies_tried.append(f"markdown fence + json.loads: {e}")

    # --- Strategy 3: ast.literal_eval (Python dict style) ---------------
    try:
        cleaned = response.strip()
        # Replace JSON boolean/null literals with Python equivalents
        cleaned = re.sub(r'\btrue\b', 'True', cleaned)
        cleaned = re.sub(r'\bfalse\b', 'False', cleaned)
        cleaned = re.sub(r'\bnull\b', 'None', cleaned)
        return ast.literal_eval(cleaned)
    except (ValueError, SyntaxError) as e:
        strategies_tried.append(f"ast.literal_eval: {e}")

    # --- Strategy 4: Regex-based key repair ----------------------------
    # Pattern: single-quoted key followed by colon — replace quotes.
    # This handles the most common LLM failure: {'key': val} instead of {"key": val}
    try:
        # Replace single-quoted keys:  'key':  →  "key":
        # Carefully match: start of line or { or , followed by whitespace then 'key'
        repaired = re.sub(
            r"""(\{|\,)\s*'([^']+)'\s*:""",
            r'\1"\2":',
            response
        )
        # Also handle the first key: 'key': → "key":
        repaired = re.sub(r"^\s*'([^']+)'\s*:", r'"\1":', repaired)
        return json.loads(repaired)
    except (json.JSONDecodeError, Exception) as e:
        strategies_tried.append(f"key-repair + json.loads: {e}")

    # --- All strategies exhausted --------------------------------------
    truncated = response[:300] + ("…" if len(response) > 300 else "")
    logger.error(
        f"[{context}] Failed to parse response after {len(strategies_tried)} strategies:\n"
        + "\n".join(f"  • {s}" for s in strategies_tried)
        + f"\n  Response (truncated): {truncated}"
    )
    raise ValueError(
        f"{context}: unable to parse LLM response as JSON after "
        f"{len(strategies_tried)} attempts. "
        f"First error: {strategies_tried[0] if strategies_tried else 'unknown'}"
    )


def _get_transport_cost_per_min(destination: str) -> float:
    """Get per-minute transit cost (USD) for a given destination city.

    Performs a case-insensitive lookup against the ``_TRANSPORT_COST_PER_MIN``
    table.  If an exact match fails, falls back to substring matching so that
    e.g. "New York City" still matches the "new york" entry.

    Args:
        destination: City or region name (case-insensitive).

    Returns:
        Estimated transit cost in USD per minute.
    """
    dest_lower = destination.lower().strip()
    # Exact match first
    if dest_lower in _TRANSPORT_COST_PER_MIN:
        return _TRANSPORT_COST_PER_MIN[dest_lower]
    # Substring fallback: check if any known key is contained in the query
    for key, val in _TRANSPORT_COST_PER_MIN.items():
        if key in dest_lower or dest_lower in key:
            return val
    # Ultimate fallback for unknown destinations
    return _TRANSPORT_COST_PER_MIN["default"]


# ── Live exchange rate with in-memory cache ──────────────────────
# Cache structure: { "CURRENCY_CODE": (rate_float, timestamp_epoch) }
_exchange_rate_cache = {}
_CACHE_TTL = 3600*24  # Cache time-to-live in seconds (24 hour)


def _fetch_live_exchange_rate(currency: str) -> float:
    """Fetch the live USD→currency exchange rate with caching and fallback.

    Strategy:
        1. Return cached rate if it is still within the TTL window.
        2. Call the free ``exchangerate-api.com`` endpoint for fresh data.
        3. Fall back to the hardcoded ``EXCHANGE_RATES`` dict if the API
           is unreachable or returns no data for the requested currency.

    API: ExchangeRate-API (https://api.exchangerate-api.com/v4/latest/USD)
        Expected JSON response::

            {
                "result": "success",
                "rates": { "CNY": 7.25, "EUR": 0.92, ... },
                ...
            }

    Args:
        currency: ISO 4217 currency code (e.g. "CNY", "EUR").

    Returns:
        Exchange rate as ``1 USD = rate * currency``.
    """
    cache_key = currency.upper()
    now = time.time()

    # Step 1: Return cached rate if still fresh (within TTL)
    if cache_key in _exchange_rate_cache:
        cached_rate, cached_time = _exchange_rate_cache[cache_key]
        if now - cached_time < _CACHE_TTL:
            return cached_rate

    # Step 2: Try the free ExchangeRate-API (no key required)
    try:
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        resp = _get_session().get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            rate = data.get("rates", {}).get(cache_key)
            if rate:
                _exchange_rate_cache[cache_key] = (rate, now)
                return rate
    except Exception:
        pass  # Network error or timeout — fall through to hardcoded rates

    # Step 3: Fallback to hardcoded rates
    return EXCHANGE_RATES.get(cache_key, 1.0)


# Singleton LLM client instance shared across all nodes.
# Parameters (API key, base URL, model name, timeout, retries) are loaded
# from config.py to keep secrets out of source code.
llm_client = DeepSeekChatClient(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model_name=DEEPSEEK_MODEL,
    timeout=LLM_TIMEOUT,
    max_retries=LLM_MAX_RETRIES
)

def _progress(state: TravelState, zh_msg: str, en_msg: str) -> dict:
    """Build a bilingual progress message for frontend display.

    Returns a dict with both Chinese and English messages so the frontend
    can pick the appropriate language based on current UI language setting.

    Args:
        state: Current ``TravelState`` dict (kept for API compatibility).
        zh_msg: Chinese message text.
        en_msg: English message text.

    Returns:
        A dict with keys 'zh' and 'en' containing the respective messages.
    """
    return {"zh": zh_msg, "en": en_msg}


# Google Places API type strings that indicate food/dining establishments.
# Used to classify POIs into dining vs attraction categories during
# daily structure enforcement and critic audits.
_FOOD_TYPES = {
    "restaurant", "cafe", "bakery", "meal_takeaway", "meal_delivery",
    "food", "bar", "night_club", "dining", "street_food",
}

# Google Places API type strings that indicate hotel/accommodation POIs.
# Excluded from attraction classification (a hotel is neither food nor attraction).
_HOTEL_TYPES = {"lodging", "hotel", "hostel"}


def _is_food_poi(poi: dict) -> bool:
    """Check if a POI represents a food/dining establishment.

    Args:
        poi: POI dict with a ``"type"`` field (case-insensitive).

    Returns:
        True if the POI type is in the ``_FOOD_TYPES`` set.
    """
    poi_type = poi.get("type", "").lower()
    return poi_type in _FOOD_TYPES


def _is_attraction_poi(poi: dict) -> bool:
    """Check if a POI is an attraction (neither food nor hotel).

    By default, any POI that is not classified as food or hotel is treated
    as an attraction. This "default-to-attraction" approach handles the
    wide variety of Google Places types (museum, park, temple, etc.).

    Args:
        poi: POI dict with a ``"type"`` field (case-insensitive).

    Returns:
        True if the POI is neither a food nor a hotel type.
    """
    poi_type = poi.get("type", "").lower()
    return poi_type not in _FOOD_TYPES and poi_type not in _HOTEL_TYPES


def _find_nearest_hotel(day_attractions: list, hotels: list) -> dict:
    """Find the hotel closest to the day's activity centroid.

    Calculates the average coordinate (centroid) of all attractions with valid
    lat/lng, then returns the hotel with the smallest Euclidean distance.
    Falls back to the first hotel when coordinates are unavailable.

    Args:
        day_attractions: List of POI dicts for the current day (attractions + dining).
        hotels: List of hotel dicts with optional ``lat``/``lng`` fields.

    Returns:
        The best-matching hotel dict, or ``{}`` if no hotels are available.
    """
    if not hotels or not day_attractions:
        return hotels[0] if hotels else {}

    # Calculate centroid of the day's activities (mean lat/lng).
    # Only POIs with valid coordinates contribute to the centroid.
    lats = [a.get("lat", 0) for a in day_attractions if a.get("lat")]
    lngs = [a.get("lng", 0) for a in day_attractions if a.get("lng")]

    if not lats or not lngs:
        # No coordinates available — cannot compute distance; return first hotel
        return hotels[0]

    centroid_lat = sum(lats) / len(lats)
    centroid_lng = sum(lngs) / len(lngs)

    # Find nearest hotel using simple Euclidean distance on lat/lng.
    # This is a rough approximation (ignores Earth curvature) but sufficient
    # for intra-city hotel selection where distances are small.
    best_hotel = hotels[0]
    best_dist = float('inf')
    for hotel in hotels:
        h_lat = hotel.get("lat", 0)
        h_lng = hotel.get("lng", 0)
        if h_lat and h_lng:
            dist = ((h_lat - centroid_lat) ** 2 + (h_lng - centroid_lng) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_hotel = hotel

    return best_hotel


def enforce_daily_structure(
    daily_plans: list,
    all_pois: list,
    hotels: list,
    duration: int,
    min_attractions: int = 3,
    max_attractions: int = 5,
) -> list:
    """Ensure each day has the configured attraction range + 2 dining + 1 hotel.

    This is a rule-based fix-up step that runs after the LLM recommendation.
    It trims over-populated days, supplements under-populated days from the
    full POI pool, and assigns a hotel to each day based on proximity.

    Key behaviors:
      - Days with > ``max_attractions`` attractions are trimmed to the top
        ``max_attractions`` by rating.
      - Days with < ``min_attractions`` attractions are supplemented from the
        cheapest available attraction POIs (to stay within budget).
      - Days with <2 dining options are supplemented from the cheapest food POIs.
      - Duplicate POI names across days are prevented via a tracking set.
      - If the LLM returned fewer days than ``duration``, empty day stubs are
        appended.

    Args:
        daily_plans: List of day-plan dicts from the LLM (may be incomplete).
        all_pois: Full POI pool from ``raw_knowledge["pois"]`` used for gap-filling.
        hotels: List of hotel dicts for hotel assignment.
        duration: Expected number of travel days.
        min_attractions: Minimum attractions per day (default 3).
        max_attractions: Maximum attractions per day (default 5).

    Returns:
        The modified ``daily_plans`` list with exactly ``duration`` entries,
        each containing ``attractions`` (within range), ``dining`` (2), and ``hotel`` (1).
    """
    if not all_pois:
        logger.warning("enforce_daily_structure: all_pois is empty, cannot supplement")

    # Separate POIs by type for filling gaps
    food_pois = [p for p in all_pois if _is_food_poi(p)]
    attraction_pois = [p for p in all_pois if _is_attraction_poi(p)]
    # Sort by cost ascending — prefer cheaper POIs to stay within budget
    food_pois.sort(key=lambda p: p.get("cost", 0))
    attraction_pois.sort(key=lambda p: p.get("cost", 0))

    logger.info(
        f"enforce_daily_structure: POI pool has {len(attraction_pois)} attractions, "
        f"{len(food_pois)} dining candidates"
    )

    # Ensure we have exactly `duration` days
    if not daily_plans or not isinstance(daily_plans, list):
        daily_plans = []
    while len(daily_plans) < duration:
        daily_plans.append({"day": len(daily_plans) + 1, "attractions": [], "dining": [], "hotel": None})
    daily_plans = daily_plans[:duration]  # trim extra days

    # Track names already used to avoid duplicates when filling
    used_attraction_names = set()
    used_dining_names = set()
    for day in daily_plans:
        for a in day.get("attractions", []):
            used_attraction_names.add(a.get("name", ""))
        for d in day.get("dining", []):
            used_dining_names.add(d.get("name", ""))

    for idx, day in enumerate(daily_plans):
        day.setdefault("day", idx + 1)
        day.setdefault("attractions", [])
        day.setdefault("dining", [])
        day.setdefault("hotel", None)

        # --- Attractions: enforce configured range ---
        attrs = day["attractions"]
        if len(attrs) > max_attractions:
            # Keep top by rating (up to max_attractions)
            attrs.sort(key=lambda p: p.get("rating", 0), reverse=True)
            day["attractions"] = attrs[:max_attractions]
        sup_attr = 0
        while len(day["attractions"]) < min_attractions:
            # Fill from attraction pool
            candidate = None
            for p in attraction_pois:
                if p.get("name", "") not in used_attraction_names:
                    candidate = p
                    break
            if candidate is None:
                break  # no more candidates available
            day["attractions"].append(candidate)
            used_attraction_names.add(candidate.get("name", ""))
            sup_attr += 1

        # --- Dining: enforce exactly 2 ---
        dinings = day["dining"]
        sup_din = 0
        while len(dinings) < 2:
            candidate = None
            for p in food_pois:
                if p.get("name", "") not in used_dining_names:
                    candidate = p
                    break
            if candidate is None:
                break  # no more candidates available
            day["dining"].append(candidate)
            used_dining_names.add(candidate.get("name", ""))
            sup_din += 1

        # --- Hotel: pick the one nearest to today's attractions ---
        if not day.get("hotel") and hotels:
            day_attractions = day.get("attractions", []) + day.get("dining", [])
            day["hotel"] = _find_nearest_hotel(day_attractions, hotels)

        if sup_attr > 0 or sup_din > 0:
            logger.info(
                f"Day {day.get('day', idx + 1)}: supplemented {sup_attr} attractions, "
                f"{sup_din} dining from POI pool"
            )

    return daily_plans


def _detect_currency(query: str) -> str:
    """Detect the user's preferred currency from the query text.

    Scans for keyword/symbol indicators in both English and Chinese:
      - "rmb", "人民币", "元" → CNY
      - "€", "euro", "欧元" → EUR
      - "£", "pound", "英镑" → GBP
      - "日元", "yen" → JPY
      - Default → USD

    Args:
        query: Raw user query string.

    Returns:
        ISO 4217 currency code (e.g. "CNY", "EUR", "USD").
    """
    query_lower = query.lower()
    if "rmb" in query_lower or "人民币" in query or "元" in query:
        return "CNY"
    elif "€" in query or "euro" in query_lower or "欧元" in query:
        return "EUR"
    elif "£" in query or "pound" in query_lower or "英镑" in query:
        return "GBP"
    elif "日元" in query or "yen" in query_lower:
        return "JPY"
    return "USD"  # Default to USD if no currency indicator found


def _detect_is_chinese(query: str) -> bool:
    """Detect if the user query contains Chinese characters.

    Checks each character against the CJK Unified Ideographs Unicode range
    (U+4E00 to U+9FFF). Used to determine the output language for the
    final report.

    Args:
        query: Raw user query string.

    Returns:
        True if at least one Chinese character is found.
    """
    for ch in query:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False
