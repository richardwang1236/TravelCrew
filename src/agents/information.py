"""Information node — fetches real-time travel data from external APIs.

This module implements the data hub of the pipeline. All data here comes
from real APIs (no LLM generation), ensuring factual accuracy. Fetches
weather, attractions, hotels, and wikivoyage data concurrently using a
thread pool.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

from src.state import TravelState
from src.api import (
    fetch_weather,
    fetch_attractions,
    fetch_hotels,
    fetch_wikivoyage_info,
    backfill_missing_coordinates,
    search_specific_place,
)
from src.config import GOOGLE_MAPS_API_KEY
from src.agents.utils import _progress

logger = logging.getLogger(__name__)


def information_node(state: TravelState) -> dict[str, Any]:
    """Fetch real-time travel data from external APIs in parallel.

    Acts as the data hub of the pipeline. All data here comes from real APIs
    (no LLM generation), ensuring factual accuracy. Fetches four data streams
    concurrently using a thread pool:

    1. **Weather** – multi-day forecast via ``fetch_weather``.
    2. **Attractions (POIs)** – Google Places API via ``fetch_attractions``.
    3. **Hotels** – accommodation listings via ``fetch_hotels``.
    4. **Wikivoyage** – destination travel knowledge via ``fetch_wikivoyage_info``.

    After parallel fetch, backfills any POIs missing GPS coordinates using
    the Google Geocoding API.

    Note:
        Image fetching and transport matrix computation are intentionally
        deferred to ``routing_and_strategy_node`` to avoid wasting API calls
        on POIs that may be discarded during the replan loop.

    Args:
        state: Current ``TravelState`` dict. Requires:
            - ``state["intent"]["destination"]`` (str)
            - ``state["intent"]["duration_days"]`` (int)
            - ``state["intent"]["budget"]`` (float, USD) or ``budget_usd``

    Returns:
        A partial state dict with key ``"raw_knowledge"`` containing:
        - ``weather`` (dict): Weather forecast data.
        - ``pois`` (list[dict]): Point-of-interest records.
        - ``hotels`` (list[dict]): Hotel listings.
        - ``wikivoyage`` (dict): Destination knowledge from Wikivoyage.
    """
    dest = state["intent"].get("destination", "Unknown")
    duration = state["intent"].get("duration_days", 3)

    # Use check_in / check_out dates computed by IntentParser (from user-specified
    # start_date or today as default)
    check_in = state["intent"].get("check_in", datetime.now().strftime("%Y-%m-%d"))
    check_out = state["intent"].get("check_out", (datetime.now() + timedelta(days=duration)).strftime("%Y-%m-%d"))

    # The trip start date, passed to fetch_weather so that it retrieves the
    # forecast for the travel period rather than for today.
    start_date = check_in

    # ── Parallel API calls ──────────────────────────────────────
    # Initialize result containers; will be populated by the thread pool.
    weather = None
    pois = []
    hotels = []
    wikivoyage_context = {}

    # Scale POI count with trip duration; floor at 25 to ensure a rich
    # candidate pool even for short trips.
    poi_count = max(25, duration * 8)

    # Derive per-night hotel budget ceiling from total budget / duration.
    # This guides the hotel search to return options within the user's means.
    budget_usd = state["intent"].get("budget_usd", 0)
    hotel_budget_max = int(budget_usd / max(duration, 1)) if budget_usd else None

    # Detect user language from the original query so that Google Places
    # and SerpApi return POI/hotel names in the user's preferred language.
    query = state.get("query", "")
    is_chinese = bool(re.search(r'[\u4e00-\u9fff]', query))
    api_language = "zh-CN" if is_chinese else "en"
    api_language2 = "zh-cn" if is_chinese else "en"
    logger.info(f"Detected API language: {api_language} (is_chinese={is_chinese})")

    logger.info(f"Fetching data in parallel for {dest} ({duration} days, hotel budget/night: ${hotel_budget_max})...")

    # Collect sanitized progress messages for frontend display.
    progress_msgs = [
        _progress(state, f"🔍 正在获取 {dest} 的旅行数据...", f"🔍 Fetching travel data for {dest}..."),
    ]

    # ── Build user preference context for API search queries ──────
    # Instead of passing only the raw query, construct a rich, structured
    # description that incorporates parsed intent fields (theme, interests,
    # dietary_preferences, must_avoid, pacing, physical_level) so that
    # the SerpApi AI-mode and hotel search return results matching the
    # user's actual needs (e.g. Western restaurants if the user wants them,
    # indoor activities if the user dislikes sun exposure).
    intent = state.get("intent", {})
    prefs = state.get("user_preferences", {})
    query = state.get("query", "")

    # Build a preference summary string for the AI search prompt
    pref_parts = []
    theme = intent.get("theme", "")
    if theme and theme.lower() not in ("general", "mixed"):
        pref_parts.append(f"trip theme: {theme}")
    interests = prefs.get("interests", [])
    if interests:
        pref_parts.append(f"interests/hobbies: {', '.join(interests)}")
    dietary = prefs.get("dietary_preferences", [])
    if dietary:
        pref_parts.append(f"dietary/cuisine preferences: {', '.join(dietary)}")
    must_avoid = prefs.get("must_avoid", [])
    if must_avoid:
        pref_parts.append(f"MUST AVOID: {', '.join(must_avoid)}")
    pacing = prefs.get("pacing", "")
    if pacing:
        pref_parts.append(f"preferred pacing: {pacing}")
    physical = prefs.get("physical_level", "")
    if physical:
        pref_parts.append(f"physical level: {physical}")
    # Include the original query for any nuance the parser may have missed
    if query:
        pref_parts.append(f"original request: {query}")

    search_context = "; ".join(pref_parts) if pref_parts else query
    logger.info(f"Built search context ({len(search_context)} chars): {search_context[:150]}...")

    # Launch 4 API calls concurrently; max_workers=4 matches the number of tasks
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {
            executor.submit(fetch_weather, dest, duration, start_date): "weather",
            executor.submit(fetch_wikivoyage_info, dest): "wikivoyage",
            executor.submit(fetch_attractions, dest, GOOGLE_MAPS_API_KEY, poi_count, api_language, search_context): "attractions",
            executor.submit(fetch_hotels, dest, check_in, check_out, adults=2, budget_max=hotel_budget_max, currency="USD", language=api_language2): "hotels",
        }

        # Collect results as they complete (order-independent)
        for future in as_completed(future_map):
            task_name = future_map[future]
            try:
                result = future.result()
                if task_name == "weather":
                    weather = result
                    logger.info(f"Weather data fetched for {dest}.")
                    progress_msgs.append(_progress(state, "☀️ 天气数据获取完成", "☀️ Weather data fetched"))
                elif task_name == "wikivoyage":
                    wikivoyage_context = result
                    logger.info("Wikivoyage context fetched.")
                    progress_msgs.append(_progress(state, "📖 目的地攻略获取完成", "📖 Destination guide fetched"))
                elif task_name == "attractions":
                    pois = result
                    logger.info(f"Fetched {len(pois)} attractions in {dest}.")
                    progress_msgs.append(_progress(state, f"📍 已找到 {len(pois)} 个景点", f"📍 Found {len(pois)} attractions"))
                elif task_name == "hotels":
                    hotels = result
                    logger.info(f"Fetched {len(hotels)} hotels in {dest}.")
                    progress_msgs.append(_progress(state, f"🏨 已找到 {len(hotels)} 家酒店", f"🏨 Found {len(hotels)} hotels"))
            except Exception as e:
                logger.warning(f"Parallel task '{task_name}' failed: {e}")
                # Graceful fallbacks: assign empty defaults so downstream
                # nodes can still function with partial data.
                if task_name == "hotels":
                    hotels = []
                elif task_name == "wikivoyage":
                    wikivoyage_context = {}

    # Fallback if weather fetch failed entirely — downstream nodes expect
    # at least a skeleton weather dict with "condition" and "daily" keys.
    if weather is None:
        weather = {"condition": "Unknown", "daily": []}

    # Backfill missing POI coordinates via Google Geocoding API.
    # Some fallback POIs (e.g. from secondary data sources) may lack lat/lng;
    # this ensures all POIs have valid coordinates for distance calculations.
    backfill_missing_coordinates(pois, dest, GOOGLE_MAPS_API_KEY)

    # ── Pre-search must_visit places ──────────────────────────────
    # The user may have explicitly named places they want to visit (e.g.,
    # "I must see the Eiffel Tower"). These are non-negotiable, so we
    # proactively search for them via Google Places and inject them into
    # the POI pool if not already present.
    must_visit = prefs.get("must_visit", [])
    if must_visit:
        existing_names = {p.get("name", "").lower() for p in pois}
        newly_added = 0
        for place_name in must_visit:
            if not isinstance(place_name, str) or not place_name.strip():
                continue
            if place_name.strip().lower() in existing_names:
                continue  # Already in the pool
            try:
                poi = search_specific_place(
                    place_name.strip(), dest, GOOGLE_MAPS_API_KEY,
                    language=api_language
                )
                if poi:
                    pois.append(poi)
                    existing_names.add(poi.get("name", "").lower())
                    newly_added += 1
                    logger.info(f"Pre-searched must_visit place '{place_name}' → added '{poi.get('name')}'")
            except Exception as e:
                logger.warning(f"Failed to search must_visit place '{place_name}': {e}")
        if newly_added > 0:
            progress_msgs.append(
                _progress(state,
                    f"📍 已预搜索 {newly_added} 个必去地点",
                    f"📍 Pre-searched {newly_added} must-visit place(s)")
            )

    # Package all fetched data into the raw_knowledge state slot
    return {
        "raw_knowledge": {
            "weather": weather,
            "pois": pois,
            "hotels": hotels,
            "wikivoyage": wikivoyage_context,
        },
        "progress_logs": progress_msgs,
    }
