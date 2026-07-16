"""ReplanAgent — tool-use agent for autonomous itinerary revision.

Instead of a single LLM call generating a replan instruction, this agent
runs an iterative tool-use loop. It can inspect the current plan, search for
alternative POIs, check constraints, directly modify the itinerary, and
update preferences — all autonomously — until it is satisfied the user's
feedback has been fully addressed.

Architecture::

    User Feedback + Current State
           │
           ▼
    ┌─────────────────────┐
    │   ReplanAgent.run() │
    │                     │
    │  ┌── LLM ◄──────┐   │
    │  │   │          │   │
    │  │   ▼          │   │
    │  │ Tool Call    │   │
    │  │   │          │   │
    │  │   ▼          │   │
    │  │ Execute Tool │   │
    │  │   │          │   │
    │  └──►Result ────┘   │
    │      (loop until    │
    │       finalize)     │
    └─────────────────────┘
           │
           ▼
    Modified itinerary + prefs + new POIs
"""

import copy
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src.state import TravelState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_plan",
            "description": (
                "View the current daily itinerary: all attractions, dining, "
                "and hotels for each day, plus budget and preference info."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_poi_pool",
            "description": (
                "Browse the available POI pool — all attractions, restaurants, "
                "and hotels that can be used in the itinerary. Use this to "
                "find alternatives when the user wants to swap or add places."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["attractions", "restaurants", "hotels", "all"],
                        "description": "Which category to list (default: all).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_place",
            "description": (
                "Search Google Places for a specific place name near the "
                "destination. Use when the user mentions a place not in the "
                "current POI pool, or when you need more options."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Place name to search for (e.g. 'Eiffel Tower').",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotel",
            "description": (
                "Search for hotels via SerpApi Google Hotels engine. Use when "
                "you need more hotel options with specific budget constraints. "
                "Returns hotels with price, rating, amenities, and GPS coordinates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min_price": {
                        "type": "integer",
                        "description": (
                            "Minimum price per night (in USD). "
                            "Use to filter out cheap/low-quality hotels. "
                            "Leave empty for no lower bound."
                        ),
                    },
                    "max_price": {
                        "type": "integer",
                        "description": (
                            "Maximum price per night (in USD). "
                            "Use to respect the user's hotel budget. "
                            "Leave empty for no upper bound."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_weather",
            "description": (
                "View the weather forecast for the trip period: current "
                "conditions, daily forecast (temp high/low, condition), "
                "and any severe weather alerts. Use when the user asks "
                "about weather or when weather might affect outdoor vs "
                "indoor activity planning."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_destination_info",
            "description": (
                "Get local travel knowledge about the destination from "
                "Wikivoyage: best seasons, transport tips, food & dining "
                "recommendations, safety advice, local customs & etiquette, "
                "and highlights. Use when the user asks about local culture, "
                "food scene, safety concerns, or general destination tips."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "enum": [
                            "all", "best_seasons", "transport_tips",
                            "highlights", "food_tips", "safety_tips",
                            "local_customs",
                        ],
                        "description": (
                            "Specific topic to query. 'all' returns everything. "
                            "Use specific topics for targeted questions."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "discover_places",
            "description": (
                "Discover new attraction and restaurant options in the "
                "destination via AI-powered search (SerpApi AI Mode). "
                "Returns a curated list of must-visit places. Use when "
                "the user wants more options, different recommendations, "
                "or you need alternatives beyond the current POI pool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "interests": {
                        "type": "string",
                        "description": (
                            "Optional interest keywords to guide discovery "
                            "(e.g. 'history, nature, food'). Leave empty "
                            "for general recommendations."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_transit",
            "description": (
                "Check estimated public transit time (in minutes) between "
                "two POIs using the pre-computed distance matrix. Use when "
                "the user asks about travel time, whether two places are "
                "close enough for the same day, or route feasibility."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "from_poi": {
                        "type": "string",
                        "description": "Origin POI name.",
                    },
                    "to_poi": {
                        "type": "string",
                        "description": "Destination POI name.",
                    },
                },
                "required": ["from_poi", "to_poi"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_plan",
            "description": (
                "Modify the daily itinerary. Supported actions:\n"
                "- add_attraction: Add an attraction to a day. Needs day, poi_name.\n"
                "- remove_attraction: Remove an attraction. Needs day, poi_name.\n"
                "- swap_attraction: Replace one attraction with another. Needs day, old_poi_name, poi_name.\n"
                "- add_dining: Add a dining option. Needs day, poi_name.\n"
                "- remove_dining: Remove a dining option. Needs day, poi_name.\n"
                "- swap_dining: Replace a dining option. Needs day, old_poi_name, poi_name.\n"
                "- change_hotel: Change hotel. day=0 sets SAME hotel for ALL days; day=N sets hotel for that day only. Needs day, poi_name.\n"
                "- reorder: Reorder attractions in a day. Needs day, order (list of names).\n\n"
                "All poi_name values MUST come from the POI pool — never invent names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {
                        "type": "integer",
                        "description": "Day number (1-indexed). For change_hotel, use day=0 to set the same hotel for ALL days.",
                    },
                    "action": {
                        "type": "string",
                        "enum": [
                            "add_attraction", "remove_attraction",
                            "swap_attraction", "add_dining",
                            "remove_dining", "swap_dining",
                            "change_hotel", "reorder",
                        ],
                        "description": "Type of modification to apply.",
                    },
                    "poi_name": {
                        "type": "string",
                        "description": "POI name to add or the NEW name for a swap.",
                    },
                    "old_poi_name": {
                        "type": "string",
                        "description": "Existing POI name to remove or replace (for swap/remove).",
                    },
                    "order": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New order of attraction names (for reorder action).",
                    },
                },
                "required": ["day", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_constraints",
            "description": (
                "Run constraint checks on the current itinerary: per-day "
                "attraction count, dining count, hotel presence, budget "
                "adherence, and must-include coverage."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_preferences",
            "description": (
                "Update user preference fields. Only include keys that actually "
                "changed. Supported keys: interests, dietary_preferences, "
                "must_avoid, pacing, physical_level, override_weather_rule, "
                "override_budget_rule, min_attractions_per_day, "
                "max_attractions_per_day."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "updates": {
                        "type": "object",
                        "description": "Dict of preference key → new value.",
                    }
                },
                "required": ["updates"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize",
            "description": (
                "Signal that you are satisfied with the itinerary and all "
                "user feedback has been addressed. Provide a brief summary "
                "of what was changed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of all changes made.",
                    }
                },
                "required": ["summary"],
            },
        },
    },
]

# Category types for constraint checking (used by modify_plan and check_constraints)
_DINING_TYPES = {"restaurant", "cafe", "street_food", "dining", "food", "bakery", "bar", "meal_takeaway", "meal_delivery"}

# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """You are an expert travel itinerary revision agent. Your job is to modify
a travel plan to satisfy the user's feedback.

SCOPE — ONLY FIX STATED ISSUES:
Your task is a TARGETED BUGFIX, not a full redesign. You MUST ONLY address
the specific issues listed in the feedback. Do NOT search for or add anything
unrelated. Examples of UNNECESSARY actions you MUST avoid:
  - Feedback says "swap in western restaurants" → do NOT search for nightlife, bars, shopping
  - Feedback says "replace outdoor with indoor" → do NOT search for new hotels or dining
  - Feedback says "attractions too far apart" → do NOT discover new places, just regroup
Every extra search wastes iterations you need for actual modifications.

CRITICAL — BUDGET YOUR ITERATIONS:
You have a limited number of tool-call rounds. Follow this efficient workflow:

1. **Inspect** (1 call): Call get_current_plan to understand the current state.
   Then immediately identify WHICH specific issues from the feedback need fixing.

2. **Search** (ONLY for stated issues, 2-3 calls MAX):
   - Dietary mismatch → 1-2 search_place for the exact cuisine mentioned
     (e.g. "western restaurant Beijing", "Italian restaurant Beijing").
   - Sun/outdoor avoidance → 1-2 search_place for indoor alternatives
     (e.g. "museum Beijing", "indoor gallery Beijing").
   - Distance issues → check_transit, no new searches needed.
   - Stop searching once you have 2-3 candidates. Do NOT keep looking for "better" options.

3. **Modify** (3-5 calls): Use modify_plan to apply changes.
   - swap_attraction / swap_dining handle BOTH remove AND add in ONE call.
     Do NOT use separate remove+add calls — that wastes iterations.
   - For dining swaps: old_poi_name = current dining POI, poi_name = new restaurant.

4. **Verify** (1 call): Call check_constraints to confirm everything passes.

5. **Finalize** (1 call): Call finalize when ALL stated issues are addressed.

RULES:
- Every POI name you use MUST exist in the POI pool or be found via search_place.
- For hotel changes: use search_hotel to find hotels in the user's price range, then use modify_plan change_hotel.
  IMPORTANT — Hotel consistency: Prefer ONE hotel for the ENTIRE trip. Use day=0 with change_hotel
  to apply the same hotel to all days. Only set different hotels per day if the user explicitly requests it.
- Attraction count per day should match the user's pacing preference (check user_preferences.pacing):
  relaxed/悠闲 → 2-4 attractions/day, moderate/适中 → 3-5, intensive/紧凑 → 4-7.
  The user may also specify explicit min_attractions_per_day / max_attractions_per_day values.
  Each day must have 2 dining options and 1 hotel.
- Category separation: Attractions MUST NOT contain dining/restaurant POIs.
  Dining MUST NOT contain hotels or attraction POIs.
  POIs with type \"restaurant\"/\"cafe\"/\"food\" etc. belong in dining, NOT attractions.
- Keep must_visit / must_include places — they are NON-NEGOTIABLE.
- Respect the user's budget — don't add very expensive places unless the user asks.
- **[BUDGET CEILING RULE]** After calling check_constraints, IF it reports budget exceeded ("Budget severely exceeded" or "Budget slightly over"), you MUST NOT increase total cost further. Only swap to CHEAPER or EQUAL-cost alternatives. Do NOT upgrade hotels, dining, or attractions when over budget.
- When weather is bad (rain, storm, extreme heat/cold), suggest indoor alternatives.
- Use check_transit before adding far-away places to the same day — keep transit under 45 min when possible.
- **[BUDGET REBALANCING]** If you reduce costs significantly (e.g. cheaper hotel), use the savings to UPGRADE
  dining and attractions. Don't leave large budget surplus idle — actively reallocate it. After making changes,
  call check_constraints to see if there's unused budget, then upgrade accordingly.
- **[IMPLICIT PREFERENCES]** The original user query (shown in get_current_plan) often contains implicit
  preferences NOT captured in structured fields. Read it carefully and respect signals like: family/kids
  (choose kid-friendly POIs), couple/romantic (scenic/romantic spots), elderly (easy walking, no steep climbs),
  solo (social/budget-friendly), photography (scenic viewpoints), foodie (diverse local cuisine), etc.
  If the feedback mentions an implicit preference the plan ignores, fix it.
"""


# ---------------------------------------------------------------------------
# Tool executor — bridges LLM tool calls to actual state mutations
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Executes tool calls against an in-memory working copy of the state.

    Maintains a mutable working copy of ``daily_itinerary``, ``user_preferences``,
    and a ``new_pois`` accumulator. All modifications are applied to the working
    copy; the caller retrieves the final state after the agent finishes.
    """

    def __init__(
        self,
        state_values: dict[str, Any],
        api_language: str = "en",
        llm_client: Any = None,
    ):
        # Deep-copy mutable structures so the agent works on a sandbox
        self.daily_itinerary = copy.deepcopy(
            state_values.get("daily_itinerary", [])
        )
        self.user_preferences = copy.deepcopy(
            state_values.get("user_preferences", {})
        )
        self.raw_knowledge = copy.deepcopy(
            state_values.get("raw_knowledge", {})
        )
        self.transport_matrix = copy.deepcopy(
            state_values.get("transport_matrix", {})
        )
        self.intent = state_values.get("intent", {})
        self.query = state_values.get("query", "")
        self.must_include_places = list(
            state_values.get("must_include_places", [])
        )
        self.api_language = api_language
        self.llm_client = llm_client

        # Accumulator for newly discovered POIs
        self.new_pois: list[dict] = []

        # Build lookup maps for fast access
        self._poi_by_name: dict[str, dict] = {}
        self._hotel_by_name: dict[str, dict] = {}
        all_pois = self.raw_knowledge.get("pois", [])
        all_hotels = self.raw_knowledge.get("hotels", [])
        for p in all_pois:
            self._poi_by_name[p.get("name", "")] = p
        for h in all_hotels:
            self._hotel_by_name[h.get("name", "")] = h

        # Also index must_visit from initial intent
        self.must_visit_names = {
            n.lower() for n in self.user_preferences.get("must_visit", [])
        }
        self.must_visit_names.update(
            n.lower() for n in self.must_include_places
        )

        # ── Precompute geographic clusters for context ────────────
        # This gives the LLM awareness of which POIs are in the same area,
        # preventing it from mixing distant POIs into the same day.
        self._cluster_summary = self._build_cluster_summary()

    def _build_cluster_summary(self) -> str:
        """Compute geographic clusters from POI pool and return a text summary."""
        try:
            from src.agents.recommendation import (
                _cluster_pois_by_proximity,
                _compute_dynamic_cluster_threshold,
            )
            pois = list(self._poi_by_name.values())
            if len(pois) < 3:
                return ""
            threshold = _compute_dynamic_cluster_threshold(pois)
            # Deep copy to avoid mutating original POI dicts
            import copy as _copy
            pois_copy = _copy.deepcopy(pois)
            clusters = _cluster_pois_by_proximity(pois_copy, threshold_km=threshold)
            valid = [c for c in clusters if len(c) >= 2]
            if not valid:
                return ""
            lines = []
            for cid, cluster in enumerate(valid):
                names = [p.get("name", "?") for p in cluster]
                lines.append(f"  Area {cid}: {', '.join(names)}")
            return (
                f"Geographic clusters (threshold={threshold}km):\n"
                + "\n".join(lines)
                + "\n  ⚠️ Keep same-day attractions within the SAME area."
            )
        except Exception:
            return ""

    # ── Tool implementations ──────────────────────────────────────

    def get_current_plan(self) -> str:
        """Return a human-readable summary of the current itinerary."""
        if not self.daily_itinerary:
            return "(No itinerary yet)"

        lines = []
        for day in self.daily_itinerary:
            dn = day.get("day", "?")
            lines.append(f"--- Day {dn} ---")
            lines.append(f"  Theme: {day.get('day_theme', 'N/A')}")
            attrs = day.get("attractions", [])
            for i, a in enumerate(attrs):
                lines.append(
                    f"  Attraction {i+1}: {a.get('name','?')} "
                    f"(type={a.get('type','?')}, cost=${a.get('cost',0)}, "
                    f"rating={a.get('rating','?')})"
                )
            dines = day.get("dining", [])
            for i, d in enumerate(dines):
                lines.append(
                    f"  Dining {i+1}: {d.get('name','?')} "
                    f"(type={d.get('type','?')}, cost=${d.get('cost',0)})"
                )
            hotel = day.get("hotel")
            if isinstance(hotel, dict) and hotel.get("name"):
                lines.append(
                    f"  Hotel: {hotel['name']} "
                    f"(${hotel.get('price_per_night',0)}/night, "
                    f"rating={hotel.get('rating','?')})"
                )
            else:
                lines.append("  Hotel: (none)")
        lines.append(f"\nPreferences: {json.dumps(self.user_preferences, ensure_ascii=False)}")
        lines.append(f"Must-include places: {list(self.must_visit_names)}")
        lines.append(f"Budget: ${self.intent.get('budget_usd', 'N/A')} (original: "
                     f"{self.intent.get('budget_original_amount','?')} "
                     f"{self.intent.get('budget_original_currency','USD')})")
        if self.query:
            lines.append(f"\nOriginal user query: {self.query}")
        if self._cluster_summary:
            lines.append(f"\n{self._cluster_summary}")
        return "\n".join(lines)

    def get_poi_pool(self, category: str = "all") -> str:
        """Return available POIs filtered by category."""
        all_pois = self._poi_by_name
        all_hotels = self._hotel_by_name

        # Determine which POIs are already used
        used_names = set()
        for day in self.daily_itinerary:
            for a in day.get("attractions", []):
                used_names.add(a.get("name", ""))
            for d in day.get("dining", []):
                used_names.add(d.get("name", ""))

        lines = []
        if category in ("attractions", "all"):
            lines.append(f"=== Available Attractions ({len(all_pois)} total) ===")
            for name, poi in sorted(all_pois.items()):
                marker = " [USED]" if name in used_names else ""
                lines.append(
                    f"  • {name}{marker} — type={poi.get('type','?')}, "
                    f"cost=${poi.get('cost',0)}, rating={poi.get('rating','?')}, "
                    f"tags={poi.get('tags',[])}"
                )
        if category in ("restaurants", "all"):
            # Restaurants are mixed in the POI pool; filter by type
            restaurants = {
                n: p for n, p in all_pois.items()
                if p.get("type") in ("restaurant", "cafe", "street_food", "dining", "food")
            }
            if restaurants:
                lines.append(f"\n=== Available Restaurants ({len(restaurants)}) ===")
                for name, poi in sorted(restaurants.items()):
                    marker = " [USED]" if name in used_names else ""
                    lines.append(
                        f"  • {name}{marker} — cost=${poi.get('cost',0)}, "
                        f"rating={poi.get('rating','?')}"
                    )
        if category in ("hotels", "all"):
            lines.append(f"\n=== Available Hotels ({len(all_hotels)}) ===")
            for name, h in sorted(all_hotels.items()):
                lines.append(
                    f"  • {name} — ${h.get('price_per_night',0)}/night, "
                    f"rating={h.get('rating','?')}"
                )
        return "\n".join(lines)

    def search_place(self, name: str) -> str:
        """Search for a specific place via Google Places API."""
        destination = self.intent.get("destination", "")
        if not destination:
            return "Error: No destination set — cannot search."

        try:
            from src.api import search_specific_place
            from src.config import GOOGLE_MAPS_API_KEY

            poi = search_specific_place(
                name, destination, GOOGLE_MAPS_API_KEY,
                language=self.api_language,
            )
            if poi:
                poi_name = poi.get("name", name)
                self._poi_by_name[poi_name] = poi
                self.new_pois.append(poi)
                return (
                    f"Found: {poi_name} — type={poi.get('type','?')}, "
                    f"cost=${poi.get('cost',0)}, rating={poi.get('rating','?')}, "
                    f"tags={poi.get('tags',[])}"
                )
            else:
                return f"No results found for '{name}'. Try a different search term."
        except Exception as e:
            logger.warning(f"search_place('{name}') failed: {e}")
            return f"Search failed: {str(e)}"

    def search_hotel(self, min_price: int = 0, max_price: int = 0) -> str:
        """Search for hotels via SerpApi with optional price range filter."""
        destination = self.intent.get("destination", "")
        if not destination:
            return "Error: No destination set — cannot search hotels."

        check_in = self.intent.get("check_in", "")
        check_out = self.intent.get("check_out", "")
        adults = self.intent.get("adults", 2)
        if not check_in or not check_out:
            return "Error: Missing check-in/check-out dates in intent."

        budget_min = min_price if min_price and min_price > 0 else None
        budget_max = max_price if max_price and max_price > 0 else None

        try:
            from src.api.hotels import fetch_hotels

            hotels = fetch_hotels(
                destination=destination,
                check_in=check_in,
                check_out=check_out,
                adults=adults,
                budget_min=budget_min,
                budget_max=budget_max,
                currency="USD",
                language=self.api_language,
            )

            if not hotels:
                price_hint = ""
                if budget_min and budget_max:
                    price_hint = f" in range ${budget_min}-${budget_max}/night"
                elif budget_max:
                    price_hint = f" under ${budget_max}/night"
                return f"No hotels found{price_hint}. Try widening the price range."

            # Register found hotels in the lookup map and accumulator
            new_count = 0
            lines = [f"=== Hotel Search Results ({len(hotels)} found) ==="]
            for h in hotels:
                hname = h.get("name", "")
                if hname and hname not in self._hotel_by_name:
                    self._hotel_by_name[hname] = h
                    self.new_pois.append(h)
                    new_count += 1
                lines.append(
                    f"  • {hname} — ${h.get('price_per_night',0)}/night, "
                    f"rating={h.get('rating','?')}★ ({h.get('reviews',0)} reviews), "
                    f"amenities: {', '.join(h.get('amenities',[])[:3])}"
                )

            price_range = ""
            if budget_min and budget_max:
                price_range = f" (price range: ${budget_min}-${budget_max}/night)"
            elif budget_max:
                price_range = f" (max ${budget_max}/night)"
            elif budget_min:
                price_range = f" (min ${budget_min}/night)"

            return (
                f"Found {len(hotels)} hotels{price_range}. "
                f"{new_count} new hotels added to the pool.\n"
                + "\n".join(lines)
            )
        except Exception as e:
            logger.warning(f"search_hotel failed: {e}")
            return f"Hotel search failed: {str(e)}"

    def check_weather(self) -> str:
        """Show weather forecast for the trip period."""
        weather = self.raw_knowledge.get("weather", {})
        if not weather:
            return "No weather data available for this trip."

        lines = [
            f"Current: {weather.get('condition', 'N/A')}, "
            f"{weather.get('temp_c', '?')}°C, "
            f"humidity {weather.get('humidity', '?')}%, "
            f"wind {weather.get('wind_speed_kmh', '?')} km/h"
        ]

        forecast = weather.get("forecast_days", [])
        if forecast:
            lines.append("\nDaily Forecast:")
            for f in forecast[:7]:
                lines.append(
                    f"  {f.get('date', '?')}: {f.get('condition', '?')}, "
                    f"{f.get('temp_low', '?')}-{f.get('temp_high', '?')}°C"
                )

        alerts = weather.get("weather_alerts", [])
        if alerts:
            lines.append("\n⚠️ WEATHER ALERTS:")
            for a in alerts:
                lines.append(
                    f"  • {a.get('alert_type', '?')} "
                    f"(severity: {a.get('severity', '?')}, "
                    f"level: {a.get('alert_level', '?')}/5): "
                    f"{a.get('headline', '')}"
                )

        if weather.get("note"):
            lines.append(f"\nNote: {weather['note']}")
        if weather.get("is_forecast"):
            lines.append("(This is a forecast — accuracy decreases for dates beyond 7 days)")

        return "\n".join(lines)

    def get_destination_info(self, topic: str = "all") -> str:
        """Read Wikivoyage travel knowledge for the destination."""
        wiki = self.raw_knowledge.get("wikivoyage", {})
        if not wiki:
            return "No destination knowledge available. Try discover_places instead."

        all_topics = {
            "best_seasons": "Best Time to Visit",
            "transport_tips": "Getting Around / Transport",
            "highlights": "Highlights & Must-See",
            "food_tips": "Food & Dining",
            "safety_tips": "Safety Advice",
            "local_customs": "Local Customs & Etiquette",
        }

        dest = self.intent.get("destination", "this destination")

        if topic == "all":
            lines = [f"=== Destination Knowledge: {dest} ==="]
            has_any = False
            for key, label in all_topics.items():
                val = wiki.get(key, "")
                if val:
                    has_any = True
                    lines.append(f"\n--- {label} ---")
                    lines.append(val[:500])
            if not has_any:
                return f"No destination knowledge available for {dest}."
            return "\n".join(lines)
        else:
            # Allow both enum values and direct key names
            key_map = {v.lower().replace(" ", "_"): k for k, v in all_topics.items()}
            key_map.update({k: k for k in all_topics})
            matched = key_map.get(topic.lower(), topic)
            val = wiki.get(matched, "")
            if val:
                label = all_topics.get(matched, matched)
                return f"--- {label} ({dest}) ---\n{val[:500]}"
            return (
                f"No info for topic '{topic}'. "
                f"Available: {list(all_topics.keys())}"
            )

    def discover_places(self, interests: str = "") -> str:
        """Discover new places via SerpApi AI Mode."""
        destination = self.intent.get("destination", "")
        if not destination:
            return "Error: No destination set — cannot discover places."

        try:
            from src.api.ai_search import fetch_must_visit_places
            from src.api.attractions import search_specific_place
            from src.config import GOOGLE_MAPS_API_KEY

            place_names = fetch_must_visit_places(
                destination,
                language=self.api_language,
                interests=interests or "",
                count=12,
            )

            if not place_names:
                return (
                    f"No new places discovered for '{destination}'. "
                    f"Try different interest keywords or use search_place "
                    f"for specific places."
                )

            new_count = 0
            lines = [f"=== Discovered Places in {destination} ==="]
            for name in place_names[:12]:
                already_known = name in self._poi_by_name
                marker = "" if already_known else " [NEW]"
                lines.append(f"  • {name}{marker}")

                # Try to enrich new places with Google Places data
                if not already_known and GOOGLE_MAPS_API_KEY:
                    try:
                        poi = search_specific_place(
                            name, destination, GOOGLE_MAPS_API_KEY,
                            language=self.api_language,
                        )
                        if poi:
                            self._poi_by_name[poi.get("name", name)] = poi
                            self.new_pois.append(poi)
                            new_count += 1
                    except Exception:
                        pass  # Enrichment is best-effort

            summary = (
                f"Discovered {len(place_names)} places "
                f"({new_count} new, {len(place_names) - new_count} already known)."
            )
            return summary + "\n" + "\n".join(lines)
        except Exception as e:
            logger.warning(f"discover_places failed: {e}")
            return f"Place discovery failed: {str(e)}"

    def check_transit(self, from_poi: str, to_poi: str) -> str:
        """Check transit time between two POIs."""
        if not self.transport_matrix:
            return "No transit data available. The transport matrix has not been computed yet."
        if not from_poi or not to_poi:
            return "Error: both from_poi and to_poi are required."

        # Try exact match first
        from_times = self.transport_matrix.get(from_poi, {})
        minutes = from_times.get(to_poi)

        # Case-insensitive fallback
        if not minutes:
            for origin, dests in self.transport_matrix.items():
                if origin.lower() == from_poi.lower():
                    for dest, mins in dests.items():
                        if dest.lower() == to_poi.lower():
                            minutes = mins
                            from_poi = origin
                            to_poi = dest
                            break
                    break

        if minutes:
            return (
                f"Transit from '{from_poi}' to '{to_poi}': "
                f"~{minutes} minutes by public transit."
            )

        # List available origins that partially match
        partial_matches = [
            o for o in self.transport_matrix
            if from_poi.lower() in o.lower()
        ]
        if partial_matches:
            hint = f"Did you mean: {', '.join(partial_matches[:5])}?"
            return (
                f"No transit data for '{from_poi}' → '{to_poi}'. "
                f"{hint}"
            )
        return (
            f"No transit data for '{from_poi}' → '{to_poi}'. "
            f"Available origins: {list(self.transport_matrix.keys())[:10]}"
        )

    def modify_plan(
        self,
        day: int,
        action: str,
        poi_name: str = "",
        old_poi_name: str = "",
        order: list[str] | None = None,
    ) -> str:
        """Apply a modification to the working itinerary."""
        # Special case: change_hotel with day=0 applies to ALL days
        if action == "change_hotel" and day == 0:
            if not poi_name:
                return "Error: poi_name required for change_hotel."
            hotel = self._hotel_by_name.get(poi_name)
            if not hotel:
                for hname, hdata in self._hotel_by_name.items():
                    if poi_name.lower() in hname.lower():
                        hotel = hdata
                        break
            if not hotel:
                return (
                    f"Error: Hotel '{poi_name}' not found. "
                    f"Available: {list(self._hotel_by_name.keys())[:10]}"
                )
            for d in self.daily_itinerary:
                d["hotel"] = hotel
            return f"Changed hotel for ALL {len(self.daily_itinerary)} days to '{hotel.get('name', poi_name)}'."

        # Find the target day (0-indexed internally)
        day_idx = day - 1
        if day_idx < 0 or day_idx >= len(self.daily_itinerary):
            return f"Error: Day {day} does not exist (have {len(self.daily_itinerary)} days)."

        day_plan = self.daily_itinerary[day_idx]

        if action == "add_attraction":
            if not poi_name:
                return "Error: poi_name required for add_attraction."
            poi = self._find_poi(poi_name)
            if not poi:
                return f"Error: '{poi_name}' not found in POI pool. Use get_poi_pool or search_place first."
            ptype = (poi.get("type") or "").lower()
            if ptype in _DINING_TYPES:
                return (
                    f"Error: '{poi_name}' has type '{ptype}' — it's a dining POI, not an attraction. "
                    f"Use add_dining instead. Dining types: {sorted(_DINING_TYPES)}"
                )
            day_plan.setdefault("attractions", []).append(poi)
            return f"Added attraction '{poi_name}' to Day {day}."

        elif action == "remove_attraction":
            if not poi_name:
                # Try old_poi_name as fallback
                poi_name = old_poi_name
            if not poi_name:
                return "Error: poi_name or old_poi_name required for remove_attraction."
            attrs = day_plan.get("attractions", [])
            before = len(attrs)
            day_plan["attractions"] = [
                a for a in attrs
                if a.get("name", "").lower() != poi_name.lower()
            ]
            removed = before - len(day_plan["attractions"])
            if removed == 0:
                return f"Warning: '{poi_name}' not found in Day {day} attractions."
            return f"Removed '{poi_name}' from Day {day} attractions."

        elif action == "swap_attraction":
            old = old_poi_name or ""
            new = poi_name or ""
            if not old or not new:
                return "Error: Both old_poi_name and poi_name required for swap_attraction."
            # Remove old
            attrs = day_plan.get("attractions", [])
            found = False
            for i, a in enumerate(attrs):
                if a.get("name", "").lower() == old.lower():
                    found = True
                    break
            if not found:
                return f"Error: '{old}' not found in Day {day} attractions. Available: {[a.get('name') for a in attrs]}"
            day_plan["attractions"] = [
                a for a in attrs if a.get("name", "").lower() != old.lower()
            ]
            # Add new
            new_poi = self._find_poi(new)
            if not new_poi:
                return f"Error: '{new}' not found in POI pool."
            ptype = (new_poi.get("type") or "").lower()
            if ptype in _DINING_TYPES:
                return (
                    f"Error: '{new}' has type '{ptype}' — it's a dining POI, not an attraction. "
                    f"Use swap_dining instead."
                )
            day_plan["attractions"].append(new_poi)
            return f"Swapped '{old}' → '{new}' on Day {day}."

        elif action == "add_dining":
            if not poi_name:
                return "Error: poi_name required for add_dining."
            poi = self._find_poi(poi_name)
            if not poi:
                return f"Error: '{poi_name}' not found in POI pool."
            if poi_name in self._hotel_by_name:
                return (
                    f"Error: '{poi_name}' is a hotel, not dining. "
                    f"Use change_hotel instead."
                )
            ptype = (poi.get("type") or "").lower()
            if ptype and ptype not in _DINING_TYPES:
                return (
                    f"Error: '{poi_name}' has type '{ptype}' — not a dining POI. "
                    f"Use add_attraction instead. Dining types: {sorted(_DINING_TYPES)}"
                )
            day_plan.setdefault("dining", []).append(poi)
            return f"Added dining '{poi_name}' to Day {day}."

        elif action == "remove_dining":
            if not poi_name:
                poi_name = old_poi_name
            if not poi_name:
                return "Error: poi_name required for remove_dining."
            dines = day_plan.get("dining", [])
            before = len(dines)
            day_plan["dining"] = [
                d for d in dines
                if d.get("name", "").lower() != poi_name.lower()
            ]
            removed = before - len(day_plan["dining"])
            if removed == 0:
                return f"Warning: '{poi_name}' not found in Day {day} dining."
            return f"Removed dining '{poi_name}' from Day {day}."

        elif action == "swap_dining":
            old = old_poi_name or ""
            new = poi_name or ""
            if not old or not new:
                return "Error: Both old_poi_name and poi_name required for swap_dining."
            dines = day_plan.get("dining", [])
            found = any(d.get("name", "").lower() == old.lower() for d in dines)
            if not found:
                return f"Error: '{old}' not found in Day {day} dining."
            day_plan["dining"] = [
                d for d in dines if d.get("name", "").lower() != old.lower()
            ]
            new_poi = self._find_poi(new)
            if not new_poi:
                return f"Error: '{new}' not found in POI pool."
            ptype = (new_poi.get("type") or "").lower()
            if ptype and ptype not in _DINING_TYPES:
                return (
                    f"Error: '{new}' has type '{ptype}' — not a dining POI. "
                    f"Use swap_attraction instead. Dining types: {sorted(_DINING_TYPES)}"
                )
            day_plan["dining"].append(new_poi)
            return f"Swapped dining '{old}' → '{new}' on Day {day}."

        elif action == "change_hotel":
            if not poi_name:
                return "Error: poi_name required for change_hotel."
            hotel = self._hotel_by_name.get(poi_name)
            if not hotel:
                # Try fuzzy match
                for hname, hdata in self._hotel_by_name.items():
                    if poi_name.lower() in hname.lower():
                        hotel = hdata
                        break
            if not hotel:
                return (
                    f"Error: Hotel '{poi_name}' not found. "
                    f"Available: {list(self._hotel_by_name.keys())[:10]}"
                )
            day_plan["hotel"] = hotel
            return f"Changed Day {day} hotel to '{hotel.get('name', poi_name)}'."

        elif action == "reorder":
            if not order:
                return "Error: 'order' list required for reorder action."
            attrs = day_plan.get("attractions", [])
            name_to_attr = {a.get("name", ""): a for a in attrs}
            new_attrs = []
            for name in order:
                matched = name_to_attr.get(name)
                if not matched:
                    # Try case-insensitive
                    for aname, a in name_to_attr.items():
                        if aname.lower() == name.lower():
                            matched = a
                            break
                if matched:
                    new_attrs.append(matched)
                else:
                    return f"Error: '{name}' not found in Day {day} attractions."
            day_plan["attractions"] = new_attrs
            return f"Reordered Day {day} attractions: {order}."

        return f"Unknown action: {action}"

    def check_constraints(self) -> str:
        """Run lightweight constraint checks on the working itinerary."""
        issues = []
        budget = self.intent.get("budget_usd", float("inf"))
        duration = len(self.daily_itinerary) or self.intent.get("duration_days", 1)

        # Determine attraction range based on pacing preference
        pacing = self.user_preferences.get("pacing", "适中")
        if pacing in ("悠闲", "relaxed"):
            default_min, default_max = 2, 4
        elif pacing in ("紧凑", "intensive"):
            default_min, default_max = 4, 7
        else:  # moderate / 适中 / default
            default_min, default_max = 3, 5

        min_attr = self.user_preferences.get("min_attractions_per_day", default_min)
        max_attr = self.user_preferences.get("max_attractions_per_day", default_max)

        total_cost = 0
        for i, day in enumerate(self.daily_itinerary):
            dn = day.get("day", i + 1)
            attrs = day.get("attractions", [])
            dines = day.get("dining", [])
            hotel = day.get("hotel", {})

            n_attr = len(attrs)
            if n_attr < min_attr:
                issues.append(f"Day {dn}: only {n_attr} attractions (min {min_attr})")
            if n_attr > max_attr:
                issues.append(f"Day {dn}: {n_attr} attractions (max {max_attr})")

            if len(dines) < 2:
                issues.append(f"Day {dn}: only {len(dines)} dining options (need 2)")

            if not isinstance(hotel, dict) or not hotel.get("name"):
                issues.append(f"Day {dn}: no hotel assigned")

            # Category type checks: attractions must not be dining, dining must not be hotels/attractions
            for a in attrs:
                ptype = (a.get("type") or "").lower()
                if ptype in _DINING_TYPES:
                    issues.append(
                        f"Day {dn}: '{a.get('name')}' (type={ptype}) is a dining POI "
                        f"in attractions — remove it and add via add_dining"
                    )
            for d in dines:
                dname = d.get("name", "")
                if dname in self._hotel_by_name:
                    issues.append(
                        f"Day {dn}: '{dname}' is a hotel in dining "
                        f"— remove it and use change_hotel"
                    )
                else:
                    ptype = (d.get("type") or "").lower()
                    if ptype and ptype not in _DINING_TYPES:
                        issues.append(
                            f"Day {dn}: '{dname}' (type={ptype}) is not a dining POI "
                            f"in dining — remove it and add via add_attraction"
                        )

            for a in attrs:
                total_cost += a.get("cost", 0) or 0
            for d in dines:
                total_cost += d.get("cost", 0) or 0
            if isinstance(hotel, dict):
                total_cost += hotel.get("price_per_night", 0) or 0

        # Check must_visit coverage
        all_poi_names = set()
        for day in self.daily_itinerary:
            for a in day.get("attractions", []):
                all_poi_names.add(a.get("name", "").lower())
            for d in day.get("dining", []):
                all_poi_names.add(d.get("name", "").lower())
        missing_must = [
            n for n in self.must_visit_names
            if n not in all_poi_names and not any(n in pn for pn in all_poi_names)
        ]
        if missing_must:
            issues.append(f"MISSING must-visit places: {missing_must}")

        # ── Budget check (including surplus detection for rebalancing) ──
        if budget < float("inf"):
            if total_cost > budget * 1.5:
                issues.append(
                    f"Budget severely exceeded: ${total_cost:.0f} vs ${budget:.0f} budget"
                )
            elif total_cost > budget:
                issues.append(
                    f"Budget slightly over: ${total_cost:.0f} vs ${budget:.0f} (within tolerance)"
                )

            # Budget surplus: warn if >15% unspent — agent should reallocate to upgrades
            remaining = budget - total_cost
            surplus_pct = (remaining / budget) * 100 if budget > 0 else 0
            if surplus_pct > 25:
                issues.append(
                    f"📊 Budget SURPLUS: ${remaining:.0f} ({surplus_pct:.0f}%) unspent. "
                    f"Consider UPGRADING dining to nicer restaurants or adding premium attractions."
                )
            elif surplus_pct > 15:
                issues.append(
                    f"💡 Budget: ${remaining:.0f} ({surplus_pct:.0f}%) remaining. Could upgrade some dining."
                )

        # Hotel consistency check — prefer one hotel for the entire trip
        hotels_used = set()
        for day in self.daily_itinerary:
            hotel = day.get("hotel", {})
            if isinstance(hotel, dict) and hotel.get("name"):
                hotels_used.add(hotel["name"])
        if len(hotels_used) > 1:
            issues.append(
                f"Multiple hotels across trip: {', '.join(sorted(hotels_used))}. "
                f"Consider using ONE hotel for the entire trip (use change_hotel with day=0)."
            )

        if not issues:
            return (
                f"✅ All constraints passed. "
                f"Total estimated cost: ${total_cost:.0f} / ${budget:.0f}. "
                f"Days: {duration}, attractions: {min_attr}-{max_attr}/day."
            )
        return "ISSUES FOUND:\n" + "\n".join(f"  ❌ {i}" for i in issues)

    def update_preferences(self, updates: dict) -> str:
        """Merge preference updates into the working preferences."""
        if not isinstance(updates, dict):
            return "Error: 'updates' must be a dict."
        allowed = {
            "interests", "dietary_preferences", "must_avoid",
            "pacing", "physical_level", "override_weather_rule",
            "override_budget_rule", "min_attractions_per_day",
            "max_attractions_per_day",
        }
        applied = []
        for key, val in updates.items():
            if key not in allowed:
                continue
            if isinstance(val, list) and key in self.user_preferences and isinstance(self.user_preferences.get(key), list):
                existing = self.user_preferences.get(key, [])
                self.user_preferences[key] = list(dict.fromkeys(existing + val))
            else:
                self.user_preferences[key] = val
            applied.append(key)
        if applied:
            return f"Updated preferences: {', '.join(applied)}."
        return "No valid preference keys to update."

    # ── Helpers ───────────────────────────────────────────────────

    def _find_poi(self, name: str) -> dict | None:
        """Find a POI by exact or case-insensitive name match."""
        # Exact match
        if name in self._poi_by_name:
            return self._poi_by_name[name]
        # Case-insensitive
        name_lower = name.lower()
        for pname, poi in self._poi_by_name.items():
            if pname.lower() == name_lower:
                return poi
        # Substring match (last resort, for "Eiffel" matching "Eiffel Tower")
        for pname, poi in self._poi_by_name.items():
            if name_lower in pname.lower():
                return poi
        return None


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_replan_agent(
    feedback_text: str,
    state_values: dict[str, Any],
    llm_client: Any,
    api_language: str = "en",
    max_iterations: int = 20,
) -> dict:
    """Run the autonomous replan agent.

    The agent iteratively calls tools to inspect and modify the itinerary
    until it calls ``finalize`` or the iteration limit is reached.

    Args:
        feedback_text: The user's raw feedback text.
        state_values: Current graph state snapshot (``current_state.values``).
        llm_client: An LLM client instance with a ``chat_with_tools`` method.
        api_language: Language code for Google Places API (e.g. ``"zh-CN"``).
        max_iterations: Safety cap on tool-call iterations.

    Returns:
        dict with keys:
        - ``daily_itinerary``: The modified day-by-day plan.
        - ``user_preferences``: Updated preferences.
        - ``new_pois``: Newly discovered POIs to inject into the pool.
        - ``summary``: Human-readable summary of changes.
        - ``iterations``: Number of tool-call rounds used.
    """
    executor = ToolExecutor(
        state_values, api_language=api_language, llm_client=llm_client
    )

    # Build initial context for the agent
    plan_brief = executor.get_current_plan()
    user_prompt = (
        f"=== CURRENT ITINERARY ===\n{plan_brief}\n\n"
        f"=== USER FEEDBACK ===\n{feedback_text}\n\n"
        f"Please modify the itinerary to address ALL of the user's feedback. "
        f"Use the available tools to inspect, search, and modify the plan. "
        f"Call finalize when done."
    )

    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    final_summary = ""
    iteration = 0

    # Track tool usage to detect search-loops and inject warnings
    _search_tools = {"search_place", "discover_places", "get_poi_pool",
                     "search_hotel", "get_destination_info", "check_weather"}
    _modify_tools = {"modify_plan", "update_preferences"}
    search_call_count = 0
    has_modified = False
    _WARN_AFTER_SEARCHES = 5  # Inject warning after this many search calls without any modify
    _URGENT_AT_ITERATION = max_iterations - 3  # Inject urgency when close to limit
    _FORCE_FINALIZE_AT = max_iterations - 1  # Force finalize at this iteration

    logger.info(f"ReplanAgent starting (max {max_iterations} iterations)")

    while iteration < max_iterations:
        iteration += 1
        logger.debug(f"ReplanAgent iteration {iteration}")

        # ── Force finalize when near absolute limit ───────────────
        if iteration >= _FORCE_FINALIZE_AT:
            logger.warning(
                f"ReplanAgent at iteration {iteration}: forcing finalize "
                f"to prevent timeout."
            )
            changes_made = "Modifications applied" if has_modified else "No modifications made"
            final_summary = f"Auto-finalized at iteration {iteration}. {changes_made}."
            break

        # ── Inject search-loop warning if stuck ──────────────────
        if search_call_count >= _WARN_AFTER_SEARCHES and not has_modified:
            warning = (
                f"⚠️ You have made {search_call_count} search/discover calls "
                f"but ZERO modifications. STOP SEARCHING. You already have "
                f"enough candidates. Use modify_plan NOW to swap in the "
                f"alternatives you found, then call finalize."
            )
            messages.append({"role": "system", "content": warning})
            logger.warning(f"Injecting search-loop warning at iteration {iteration}")
            search_call_count = 0  # Reset to avoid repeated warnings

        # ── Inject urgency warning near limit ─────────────────────
        if iteration >= _URGENT_AT_ITERATION:
            if not has_modified:
                urgent = (
                    f"⏰ URGENT: Only {max_iterations - iteration} iterations remain. "
                    f"You MUST use modify_plan to apply changes RIGHT NOW, then "
                    f"call finalize. Do NOT search for anything else."
                )
            else:
                urgent = (
                    f"⏰ URGENT: Only {max_iterations - iteration} iterations remain. "
                    f"You have already made modifications. Call finalize NOW with a "
                    f"brief summary. Do NOT call check_constraints or search again."
                )
            messages.append({"role": "system", "content": urgent})
            logger.warning(f"Injecting urgency warning at iteration {iteration}")

        try:
            response = llm_client.chat_with_tools(
                messages=messages,
                tools=AGENT_TOOLS,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=4096,
            )
        except Exception as e:
            logger.error(f"ReplanAgent LLM call failed at iteration {iteration}: {e}")
            break

        # Append the assistant message to history
        assistant_msg: dict = {"role": response["role"], "content": response.get("content")}
        tool_calls = response.get("tool_calls", [])

        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            # ── Parse all tool calls first ─────────────────────────
            parsed_calls: list[tuple[dict, str, dict]] = []
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}
                parsed_calls.append((tc, func_name, func_args))

            # ── Track tool categories for loop detection ──────────
            for _, func_name, _ in parsed_calls:
                if func_name in _search_tools:
                    search_call_count += 1
                elif func_name in _modify_tools:
                    has_modified = True

            # ── Parallelize read-only tools, serialize mutations ──
            # Read-only tools (search_place, discover_places, etc.) make
            # independent API calls — execute them concurrently to cut
            # per-iteration latency by 2-3x. Mutations (modify_plan) must
            # be sequential to avoid state corruption.
            _READONLY_TOOLS = {"search_place", "discover_places", "get_poi_pool",
                               "check_weather", "get_destination_info", "check_transit",
                               "get_current_plan", "search_hotel"}
            _MUTATION_TOOLS = {"modify_plan", "update_preferences"}

            results_map: dict[str, str] = {}

            # Execute read-only calls in parallel
            readonly = [(tc, fn, fa) for tc, fn, fa in parsed_calls
                        if fn in _READONLY_TOOLS]
            if readonly:
                with ThreadPoolExecutor(max_workers=min(6, len(readonly))) as pool:
                    futures = {
                        pool.submit(_dispatch_tool, executor, fn, fa): tc["id"]
                        for tc, fn, fa in readonly
                    }
                    for future in as_completed(futures):
                        tc_id = futures[future]
                        try:
                            results_map[tc_id] = str(future.result())
                        except Exception as e:
                            results_map[tc_id] = f"Error: {str(e)}"

            # Execute mutation calls sequentially
            for tc, func_name, func_args in parsed_calls:
                if func_name in _MUTATION_TOOLS:
                    results_map[tc["id"]] = str(
                        _dispatch_tool(executor, func_name, func_args)
                    )
                elif func_name == "finalize":
                    # finalize is special — execute inline to break early
                    results_map[tc["id"]] = str(
                        _dispatch_tool(executor, func_name, func_args)
                    )
                    final_summary = func_args.get("summary", "Plan updated.")

            # ── Append results in original order ───────────────────
            for tc, func_name, _func_args in parsed_calls:
                result = results_map.get(tc["id"], "(no result)")
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                }
                messages.append(tool_msg)

                logger.debug(
                    f"  Tool: {func_name} → {result[:200]}"
                )

            # ── Break if finalize was called ───────────────────────
            if any(fn == "finalize" for _, fn, _ in parsed_calls):
                logger.info(
                    f"ReplanAgent finalized after {iteration} iterations: "
                    f"{final_summary}"
                )
                break
        else:
            # No tool call — model responded with text only.
            # Treat as final response if content is present.
            messages.append(assistant_msg)
            if response.get("content"):
                final_summary = response["content"]
                logger.info("ReplanAgent completed with text response (no tool calls)")
            break

    else:
        # Max iterations reached without finalize
        logger.warning(
            f"ReplanAgent hit max iterations ({max_iterations}) without finalize. "
            "Returning current plan state."
        )
        final_summary = f"Auto-completed after {max_iterations} iterations."

    return {
        "daily_itinerary": executor.daily_itinerary,
        "user_preferences": executor.user_preferences,
        "new_pois": executor.new_pois,
        "summary": final_summary,
        "iterations": iteration,
    }


def _dispatch_tool(executor: ToolExecutor, func_name: str, args: dict) -> str:
    """Route a tool call to the correct executor method."""
    try:
        if func_name == "get_current_plan":
            return executor.get_current_plan()
        elif func_name == "get_poi_pool":
            return executor.get_poi_pool(category=args.get("category", "all"))
        elif func_name == "search_place":
            return executor.search_place(name=args.get("name", ""))
        elif func_name == "search_hotel":
            return executor.search_hotel(
                min_price=args.get("min_price", 0) or 0,
                max_price=args.get("max_price", 0) or 0,
            )
        elif func_name == "check_weather":
            return executor.check_weather()
        elif func_name == "get_destination_info":
            return executor.get_destination_info(
                topic=args.get("topic", "all")
            )
        elif func_name == "discover_places":
            return executor.discover_places(
                interests=args.get("interests", "")
            )
        elif func_name == "check_transit":
            return executor.check_transit(
                from_poi=args.get("from_poi", ""),
                to_poi=args.get("to_poi", ""),
            )
        elif func_name == "modify_plan":
            return executor.modify_plan(
                day=args.get("day", 0),
                action=args.get("action", ""),
                poi_name=args.get("poi_name", ""),
                old_poi_name=args.get("old_poi_name", ""),
                order=args.get("order"),
            )
        elif func_name == "check_constraints":
            return executor.check_constraints()
        elif func_name == "update_preferences":
            return executor.update_preferences(updates=args.get("updates", {}))
        elif func_name == "finalize":
            return f"Plan finalized: {args.get('summary', 'Done.')}"
        else:
            return f"Unknown tool: {func_name}"
    except Exception as e:
        logger.error(f"Tool execution error ({func_name}): {e}", exc_info=True)
        return f"Error executing {func_name}: {str(e)}"
