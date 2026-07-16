"""Routing & Strategy node — transport matrix, coordinate backfill, and field restoration.

This node runs after the user has reviewed and confirmed the recommendations.
It performs coordinate backfill, transport matrix computation, and restores
website/maps_url fields for confirmed POIs. Image fetching and static map
generation are deferred to the Synthesizer node to avoid wasted API calls
during replan.
"""

import logging
from typing import Any

from src.state import TravelState
from src.api import fetch_transport_matrix, backfill_missing_coordinates
from src.config import GOOGLE_MAPS_API_KEY
from src.agents.utils import _progress

logger = logging.getLogger(__name__)


def routing_and_strategy_node(state: TravelState) -> dict[str, Any]:
    """Compute transport matrix and restore POI fields for confirmed recommendations.

    This node runs after the user has reviewed and confirmed the recommendations.
    It performs three tasks:

    1. **Coordinate backfill**: Ensures all recommended POIs have valid GPS
       coordinates via the Google Geocoding API.
    2. **Transport matrix**: Computes a pairwise transit-time matrix between all
       POIs using the Google Distance Matrix API.
    3. **Field restoration**: Restores website/maps_url/coordinates from the
       original POI data into daily_itinerary items.

    These operations were intentionally deferred from ``information_node`` to
    avoid wasting API calls on POIs that might be discarded during replan.

    API: Google Distance Matrix API (via ``fetch_transport_matrix``)
        Expected return: ``dict[str, dict[str, int]]``
        Example::

            {
                "Tokyo Tower": { "Senso-ji": 25, "Shibuya": 18 },
                "Senso-ji":    { "Tokyo Tower": 25, "Shibuya": 30 }
            }

    Args:
        state: Current ``TravelState`` dict. Requires:
            - ``recommended_pois`` (list[dict])
            - ``intent["destination"]`` (str)
            - ``daily_itinerary`` (list[dict])

    Returns:
        A partial state dict with keys:
        - ``routing_metrics`` (dict): Cost, hours, and route order.
        - ``transport_matrix`` (dict): Pairwise transit times.
        - ``daily_itinerary`` (list[dict]): Updated with restored website/maps_url fields.
    """
    # Compute transport matrix now (after user confirmed recommendations)
    pois = state["recommended_pois"]
    destination = state["intent"].get("destination", "")

    # Backfill missing POI coordinates via Geocoding API before computing
    # the transport matrix (which requires valid lat/lng for all POIs)
    backfill_missing_coordinates(pois, destination, GOOGLE_MAPS_API_KEY)

    # Build a name→coords lookup from recommended_pois (which have been backfilled)
    coords_map = {}
    for poi in pois:
        if poi.get("lat") and poi.get("lng"):
            coords_map[poi["name"]] = (poi["lat"], poi["lng"])

    logger.info("Computing inter-attraction transit times...")

    # Extract daily POI groups from daily_itinerary if available, so that the
    # transport matrix is requested per-day to avoid exceeding the Google
    # Distance Matrix API 100-element limit (MAX_ELEMENTS_EXCEEDED).
    daily_itinerary = state.get("daily_itinerary", [])
    daily_groups = None
    if daily_itinerary:
        daily_groups = []
        for day in daily_itinerary:
            day_pois = []
            day_pois.extend(day.get("attractions", []))
            day_pois.extend(day.get("dining", []))
            # Backfill coordinates from recommended_pois into daily_itinerary POIs
            for item in day_pois:
                if not item.get("lat") and item.get("name") in coords_map:
                    item["lat"], item["lng"] = coords_map[item["name"]]
            daily_groups.append(day_pois)

    transport = fetch_transport_matrix(pois, GOOGLE_MAPS_API_KEY, daily_groups=daily_groups)

    # Calculate total POI cost + a fixed logistics overhead ($50) for
    # miscellaneous transit expenses not captured by the matrix.
    total_cost = sum(p.get("cost", 0) for p in pois)
    total_cost += 50  # Base transit/logistics overhead

    # Build routing metrics for the critic and synthesizer
    metrics = {
        "total_cost": total_cost,
        "estimated_hours": len(pois) * 2.5,  # Rough estimate: 2.5h per POI
        "route_order": [p["name"] for p in pois]
    }

    daily_itinerary = state.get("daily_itinerary", [])

    # Build a lookup map from recommended_pois for website & maps_url fields.
    # The recommendation LLM may have dropped these fields, so we restore them.
    poi_lookup = {}
    for poi in pois:
        name = poi.get("name", "")
        if name:
            poi_lookup[name] = poi

    # ── Inject website/maps_url from original POI data into daily_itinerary ──
    # The recommendation LLM may have dropped these fields, so we restore them.
    for day in daily_itinerary:
        for item in day.get("attractions", []) + day.get("dining", []):
            name = item.get("name", "")
            lookup_name = item.get("name_original", name)
            orig_poi = poi_lookup.get(lookup_name) or poi_lookup.get(name)
            if orig_poi:
                if not item.get("website") and orig_poi.get("website"):
                    item["website"] = orig_poi["website"]
                if not item.get("maps_url") and orig_poi.get("maps_url"):
                    item["maps_url"] = orig_poi["maps_url"]
                if not item.get("lat") and orig_poi.get("lat"):
                    item["lat"] = orig_poi["lat"]
                    item["lng"] = orig_poi["lng"]
        hotel = day.get("hotel", {})
        if isinstance(hotel, dict):
            hotel_name = hotel.get("name", "")
            orig_hotel_poi = poi_lookup.get(hotel_name)
            if orig_hotel_poi:
                if not hotel.get("website") and orig_hotel_poi.get("website"):
                    hotel["website"] = orig_hotel_poi["website"]
                if not hotel.get("maps_url") and orig_hotel_poi.get("maps_url"):
                    hotel["maps_url"] = orig_hotel_poi["maps_url"]

    progress_msgs = [
        _progress(state, "🗺️ 正在优化路线规划...", "🗺️ Optimizing route planning..."),
        _progress(state, "🗺️ 路线规划完成", "🗺️ Route planning complete"),
    ]

    return {"routing_metrics": metrics, "transport_matrix": transport, "daily_itinerary": daily_itinerary, "progress_logs": progress_msgs}
