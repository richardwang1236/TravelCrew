"""State definitions for the travel planning agent.

This module defines the central ``TravelState`` TypedDict that serves as the
single source of truth shared across all nodes in the LangGraph workflow.

Every node reads from and writes to this state object.  The state is persisted
by LangGraph's checkpointer (``MemorySaver``) so that execution can be paused
(interrupted) and resumed transparently.

Typical data flow::

    query → IntentParser → intent, user_preferences
    intent → Information → raw_knowledge, hotels, transport_matrix, wikivoyage_context
    raw_knowledge → Recommendation → recommended_pois, daily_itinerary
    daily_itinerary → UserReview → display_currency, currency_symbol, exchange_rate, is_chinese
    recommended_pois → Routing → routing_metrics, daily_itinerary
    routing_metrics → Critic → audit_findings, rejected_plans, replan_count
    daily_itinerary + routing_metrics → Synthesizer → final_itinerary

Usage::

    from state import TravelState
    # TravelState is passed as the schema to StateGraph in graph.py.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator


class TravelState(TypedDict):
    """Central state schema shared by every node in the travel-planning graph.

    Fields are grouped into logical layers that mirror the processing pipeline.
    LangGraph merges partial state updates returned by each node into this
    unified structure automatically.

    Layers:
        - **Input & Intent**: user query and parsed travel intent.
        - **Data Hub**: raw API data, transport matrix, preferences, hotels.
        - **Decision**: curated POI recommendations (overwritten each cycle).
        - **Interaction & Feedback**: user free-text feedback for HITL.
        - **Audit & Reflection**: critic findings, rejected plans, replan count.
        - **Review Display**: pre-computed fields for the CLI review screen.
        - **Output**: daily itinerary, routing metrics, final Markdown report.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Input & Intent Layer
    # Populated by: user input (query) and IntentParser node (intent)
    # ──────────────────────────────────────────────────────────────────────

    # The original natural-language travel query supplied by the user.
    # Source: initial_input["query"] in main.py.
    # Example: "我要去上海玩3天, 预算 5000 人民币。"
    query: str

    # Structured travel intent parsed by the IntentParser via LLM.
    # Typical keys: destination, duration_days, budget, travel_style,
    #               group_type, currency, etc.
    # Source: IntentParser node output.
    intent: Dict[str, Any]

    # ──────────────────────────────────────────────────────────────────────
    # Data Hub Layer
    # Populated by: Information node (API calls to Google Places, weather, etc.)
    # ──────────────────────────────────────────────────────────────────────

    # Raw API data bucket.  Typically contains sub-keys like:
    #   "pois"    – list of POI dicts from Google Places,
    #   "weather" – weather forecast data,
    #   "events"  – local events if available.
    # Source: Information node.
    raw_knowledge: Dict[str, Any]

    # Pairwise transit-time matrix between POIs (in minutes).
    # Structure: {poi_name_a: {poi_name_b: transit_minutes, ...}, ...}.
    # Source: Information node (Google Distance Matrix API or fallback estimation).
    transport_matrix: Dict[str, Dict[str, int]]

    # Refined user preferences extracted by the IntentParser from the query.
    # Keys may include: preferred_themes, dietary_restrictions, pace,
    #                   must_visit, disliked_categories, etc.
    # Source: IntentParser node.
    user_preferences: Dict[str, Any]

    # Destination-level knowledge fetched from the Wikivoyage API.
    # Contains cultural tips, local customs, safety advice, etc.
    # Source: Information node.
    wikivoyage_context: Dict[str, Any]

    # Hotel search results for each day of the trip.
    # Each dict typically has: name, price_per_night, rating, location, etc.
    # Source: Information node (Google Places / hotel API).
    hotels: List[Dict[str, Any]]

    # ──────────────────────────────────────────────────────────────────────
    # Decision Layer
    # Populated by: Recommendation node
    # ──────────────────────────────────────────────────────────────────────

    # The curated list of POIs selected for the current recommendation cycle.
    # Overwritten (not appended) on each recommendation pass so that the
    # Critic always evaluates the latest candidate set.
    # Each dict contains: name, category, cost, coordinates, description, etc.
    # Source: Recommendation node.
    recommended_pois: List[Dict[str, Any]]

    # ──────────────────────────────────────────────────────────────────────
    # Interaction & Feedback Layer
    # Populated by: Human-in-the-Loop (user input in main.py)
    # ──────────────────────────────────────────────────────────────────────

    # Free-text feedback from the user during the review phase.
    # ``None`` on the first pass; populated when the user provides modification
    # instructions via the CLI prompt.  Used by Recommendation node to adjust
    # the next round of suggestions.
    # Source: main.py feedback loop (travel_app.update_state).
    user_feedback: Optional[str]

    # ──────────────────────────────────────────────────────────────────────
    # Audit & Reflection Layer
    # Populated by: Critic node
    # ──────────────────────────────────────────────────────────────────────

    # List of audit findings (warnings / errors) from the *current* Critic pass.
    # Cleared at the start of each Critic invocation.  If non-empty, the
    # ``critic_router`` in graph.py may route back to Recommendation for a replan.
    # Source: Critic node.
    audit_findings: List[str]

    # Accumulator of all previously rejected plan POI-name lists.
    # Uses ``Annotated[List[List[str]], operator.add]`` so that each new
    # rejected plan is **appended** rather than overwritten across replan
    # cycles.  This lets the Recommendation node avoid repeating bad plans.
    # Source: Critic node (appended each round).
    rejected_plans: Annotated[List[List[str]], operator.add]

    # Sanitized, user-facing progress messages accumulated across all nodes.
    # Uses ``Annotated[List[dict], operator.add]`` so each node's messages are
    # appended (not overwritten) to provide a running log of system activity.
    # Each message is a dict with 'zh' and 'en' keys so the frontend can pick
    # the appropriate language based on the current UI language setting.
    # Messages exclude sensitive data (API keys, internal URLs, raw budget
    # figures) and are safe to display directly in the frontend.
    # Source: All nodes (optional; nodes that don't produce logs simply omit this key).
    progress_logs: Annotated[List[dict], operator.add]

    # Number of replan attempts so far.  Incremented by the Critic when it
    # decides to reject the current plan.  Compared against
    # ``config.MAX_REPLAN_ATTEMPTS`` to decide whether to force-approve.
    # Source: Critic node.
    replan_count: int

    # Flag indicating the user has reviewed and approved the ReplanAgent's
    # changes.  When True, the Critic's ``critic_router`` skips the audit
    # and force-approves to prevent overwriting the agent's work.
    # Source: AutoReplan node / feedback endpoint via update_state.
    replan_user_approved: bool

    # ──────────────────────────────────────────────────────────────────────
    # Review Display Layer
    # Populated by: UserReview node (pre-computed for print_user_review in main.py)
    # ──────────────────────────────────────────────────────────────────────

    # ISO 4217 currency code for display (e.g. "CNY", "USD", "EUR").
    # Derived from the user's budget currency or locale.
    display_currency: str

    # Unicode symbol for the display currency (e.g. "¥", "$", "€").
    currency_symbol: str

    # Exchange rate from USD to ``display_currency``.
    # Used to convert internally-stored USD costs for display.
    exchange_rate: float

    # ``True`` when the original user query was detected as Chinese.
    # Controls UI label language in print_user_review (main.py).
    is_chinese: bool

    # ──────────────────────────────────────────────────────────────────────
    # Output Layer
    # Populated by: Routing (daily_itinerary, routing_metrics),
    #               Synthesizer (final_itinerary)
    # ──────────────────────────────────────────────────────────────────────

    # Structured daily plan produced by the Routing / Recommendation node.
    # Each element: {day: int, attractions: [3-5 POI dicts],
    #                dining: [2 POI dicts], hotel: {name, price_per_night, rating, ...}}.
    # Source: Routing node (or Recommendation fallback).
    daily_itinerary: List[Dict[str, Any]]

    # Quantitative routing metrics: total distance, estimated travel time,
    # total cost, and other hard KPIs computed by the Routing node.
    routing_metrics: Dict[str, Any]

    # The final Markdown itinerary report generated by the Synthesizer.
    # This is the end-user-facing deliverable saved to ``output/``.
    # Source: Synthesizer node.
    final_itinerary: str
