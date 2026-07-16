"""Recommendation node — LLM-driven daily itinerary generation.

This module implements the core recommendation engine. It operates in two
phases: rule-based pre-filtering of POIs, followed by LLM-driven daily-
structured recommendation with anti-hallucination safeguards.
"""

import json
import logging
from typing import Any

from src.state import TravelState
from src.agents.utils import _progress, llm_client, enforce_daily_structure, _safe_json_parse
from src.api.geocoding import _haversine_distance

logger = logging.getLogger(__name__)

# ── Geographic clustering ──────────────────────────────────────
_MIN_CLUSTER_DISTANCE_KM = 3.0   # Minimum threshold (dense cities)
_MAX_CLUSTER_DYNAMIC_KM = 15.0  # Maximum threshold (sprawling areas like Sanya)


def _compute_dynamic_cluster_threshold(pois: list) -> float:
    """Compute a clustering threshold based on POI spatial distribution.

    Uses the median pairwise haversine distance among all POIs with valid
    coordinates, then takes 40% of that as the cluster radius.  The result
    is clamped between ``_MIN_CLUSTER_DISTANCE_KM`` and
    ``_MAX_CLUSTER_DYNAMIC_KM``.

    Dense cities (Beijing, Shanghai) → ~3-5 km
    Sprawling resort cities (Sanya) → ~8-15 km
    """
    coords = [
        (p.get("lat"), p.get("lng"))
        for p in pois
        if p.get("lat") and p.get("lng")
    ]
    if len(coords) < 2:
        return _MIN_CLUSTER_DISTANCE_KM

    # Sample pairwise distances (cap at 200 pairs for performance)
    import random as _rng
    pairs = []
    n = len(coords)
    max_pairs = 200
    attempts = 0
    while len(pairs) < max_pairs and attempts < max_pairs * 3:
        i, j = _rng.randint(0, n - 1), _rng.randint(0, n - 1)
        if i != j:
            d = _haversine_distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            pairs.append(d)
        attempts += 1

    if not pairs:
        return _MIN_CLUSTER_DISTANCE_KM

    pairs.sort()
    median_dist = pairs[len(pairs) // 2]
    threshold = median_dist * 0.40
    return max(_MIN_CLUSTER_DISTANCE_KM, min(_MAX_CLUSTER_DYNAMIC_KM, round(threshold, 1)))


def _cluster_pois_by_proximity(pois: list, threshold_km: float = 0) -> list:
    """Group POIs into geographic clusters using greedy proximity.

    Each POI with lat/lng coordinates is assigned to a cluster of nearby POIs
    (within ``threshold_km`` of any existing member).  If ``threshold_km`` is
    0 or omitted, defaults to ``_MIN_CLUSTER_DISTANCE_KM``.  POIs without
    coordinates are assigned cluster_id = -1.

    Returns the cluster list (list of list of POIs).  Each POI is mutated
    in-place with a ``cluster_id`` key.
    """
    if threshold_km <= 0:
        threshold_km = _MIN_CLUSTER_DISTANCE_KM
    clusters: list = []
    for poi in pois:
        lat, lng = poi.get("lat"), poi.get("lng")
        if not lat or not lng:
            poi["cluster_id"] = -1
            continue
        found = False
        for cid, cluster in enumerate(clusters):
            for member in cluster:
                mlat, mlng = member.get("lat"), member.get("lng")
                if not mlat or not mlng:
                    continue
                if _haversine_distance(lat, lng, mlat, mlng) <= threshold_km:
                    cluster.append(poi)
                    poi["cluster_id"] = cid
                    found = True
                    break
            if found:
                break
        if not found:
            clusters.append([poi])
            poi["cluster_id"] = len(clusters) - 1
    return clusters


# ── Time-slot conflict detection ───────────────────────────────
_MAX_DAY_MINUTES = 840  # 14 hours (08:00 – 22:00) = max daily activity window


def _validate_time_slots(daily_plans: list) -> int:
    """Validate time assignments across all days and fix obvious overlaps.

    For each day, parses ``start_time`` strings into minutes-since-midnight,
    checks that no two activities overlap, and that the total schedule fits
    within a reasonable daily window.  Simple overlaps are resolved by
    pushing the later activity forward.

    Returns the number of time-slot fixes applied.
    """
    fixes = 0
    for day in daily_plans:
        items = day.get("attractions", []) + day.get("dining", [])
        if not items:
            continue

        # Parse each item's start/end time in minutes since midnight
        parsed = []
        for item in items:
            start_str = str(item.get("start_time", "")).strip()
            if not start_str:
                continue
            try:
                parts = start_str.replace("：", ":").split(":")
                start_min = int(parts[0]) * 60 + int(parts[1])
            except (ValueError, IndexError):
                continue
            duration = item.get("avg_visit_time_min", 60)
            transit = item.get("transit_time_min", 0)
            end_min = start_min + duration + transit
            parsed.append((start_min, end_min, item))

        if len(parsed) < 2:
            continue

        # Sort by start time and resolve overlaps
        parsed.sort(key=lambda x: x[0])
        for i in range(len(parsed) - 1):
            _, cur_end, _ = parsed[i]
            next_start, next_end, next_item = parsed[i + 1]
            if cur_end > next_start:
                # Overlap detected: push the later item to right after the earlier one
                new_start = cur_end
                new_end = new_start + (next_end - next_start)
                next_item["start_time"] = f"{new_start // 60:02d}:{new_start % 60:02d}"
                parsed[i + 1] = (new_start, new_end, next_item)
                fixes += 1
                logger.debug(
                    f"Time fix: moved '{next_item.get('name', '?')}' "
                    f"from {parsed[i][0] // 60:02d}:{parsed[i][0] % 60:02d} "
                    f"to {new_start // 60:02d}:{new_start % 60:02d} (overlap with previous)"
                )

        # Warn if total schedule exceeds max daily window
        if parsed:
            first_start = parsed[0][0]
            last_end = parsed[-1][1]
            total_min = last_end - first_start
            if total_min > _MAX_DAY_MINUTES:
                logger.warning(
                    f"Day {day.get('day', '?')}: total schedule {total_min // 60}h{total_min % 60}m "
                    f"exceeds {_MAX_DAY_MINUTES // 60}h window — itinerary may be over-packed"
                )

    if fixes > 0:
        logger.info(f"Time-slot validation: fixed {fixes} overlapping activities")
    return fixes


def recommendation_node(state: TravelState) -> dict[str, Any]:
    """Generate daily-structured itinerary recommendations via LLM.

    This is the core recommendation engine. It operates in two phases:

    **Phase 1 – Rule-based pre-filtering** (fast, deterministic):
      - Exclude outdoor POIs on rainy days.
      - Exclude POIs whose tags match the user's ``must_avoid`` list.
      - Exclude POIs whose cost exceeds 60% of the daily budget.
      - Safety net: if filtering leaves <5 candidates, revert to the full POI list.

    **Phase 2 – LLM daily-structured recommendation**:
      - Prompts the LLM with a multi-dimensional scoring rubric
        (theme match 40%, budget fit 30%, rating 20%, time efficiency 10%).
      - Expects the LLM to return a ``daily_plans`` JSON array.
      - Injects context: user intent, preferences, weather, Wikivoyage knowledge,
        hotel list, previous audit findings, rejected plans, and user feedback.

    **Post-processing**:
      - Anti-hallucination filter: removes LLM-fabricated POIs not in the real pool.
      - ``enforce_daily_structure``: ensures each day has the configured attraction range, 2 dining,
        and 1 hotel.

    API: LLM (DeepSeek Chat via ``llm_client.chat``)
        Prompt mode: ``json_format=True``, ``temperature=0.5``.
        Expected JSON response::

            {
                "daily_plans": [
                    {
                        "day": 1,
                        "attractions": [
                            {"name": "...", "type": "...", "cost": 0, ...}
                        ],
                        "dining": [
                            {"name": "...", "type": "restaurant", "cost": 0, ...}
                        ],
                        "hotel": {"name": "...", "price_per_night": 0, ...}
                    }
                ]
            }

    Args:
        state: Current ``TravelState`` dict. Requires:
            - ``intent``, ``raw_knowledge`` (weather, pois, hotels, wikivoyage),
            - ``user_preferences``, and optionally ``audit_findings``,
              ``rejected_plans``, ``user_feedback``.

    Returns:
        A partial state dict with keys:
        - ``recommended_pois`` (list[dict]): Flat list of all recommended POIs.
        - ``daily_itinerary`` (list[dict]): Day-by-day structured plans.
    """
    intent = state["intent"]
    weather = state["raw_knowledge"]["weather"]
    pois = state["raw_knowledge"]["pois"]
    preferences = state.get("user_preferences", {})
    hotels = state.get("raw_knowledge", {}).get("hotels", [])
    wikivoyage = state.get("raw_knowledge", {}).get("wikivoyage", {})

    # ═══════════════════════════════════════════
    # Phase 1: Rule-based pre-filtering
    # Fast elimination of clearly unsuitable POIs before sending to the LLM.
    # This reduces token count and improves recommendation quality.
    #
    # IMPORTANT: All three rules can be overridden by user_feedback.
    # Check for explicit overrides before applying each rule.
    # ═══════════════════════════════════════════
    # Compute per-day budget ceiling: total budget / number of days.
    # Used by Rule 3 to reject excessively expensive individual POIs.
    budget_per_day = intent.get("budget_usd", 200) / max(intent.get("duration_days", 1), 1)

    # ── Detect user feedback overrides ──────────────────────────
    user_feedback = state.get("user_feedback", "")
    must_include_places = state.get("must_include_places", [])
    # Also merge must_visit from initial intent (parsed from original query)
    init_must_visit = preferences.get("must_visit", [])
    must_include_names = {n.lower() for n in must_include_places}
    must_include_names.update(n.lower() for n in init_must_visit if isinstance(n, str))

    # Check for rule-bypass signals in user feedback text
    feedback_lower = user_feedback.lower() if user_feedback else ""
    override_weather_rule = (
        "weather" in feedback_lower and ("doesn't matter" in feedback_lower or "don't care" in feedback_lower)
    ) or preferences.get("override_weather_rule", False)
    override_budget_rule = (
        "budget" in feedback_lower and ("flexible" in feedback_lower or "don't care" in feedback_lower or "not a concern" in feedback_lower)
    ) or ("money" in feedback_lower and "not a concern" in feedback_lower) or preferences.get("override_budget_rule", False)

    logger.info(
        f"Rule overrides: weather={override_weather_rule}, budget={override_budget_rule}, "
        f"must_include={len(must_include_names)} place(s)"
    )

    filtered_pois = []

    for poi in pois:
        # Never exclude must-include places, regardless of rules
        poi_name_lower = poi.get("name", "").lower()
        if must_include_names and poi_name_lower in must_include_names:
            filtered_pois.append(poi)
            continue

        # Rule 1: Exclude outdoor attractions on rainy days
        # SKIP this rule if user explicitly says weather doesn't matter
        if not override_weather_rule:
            if weather.get("condition") == "Rainy" and poi.get("type") == "outdoor":
                continue

        # Rule 2: Exclude types the user explicitly wants to avoid
        must_avoid = preferences.get("must_avoid", [])
        if any(avoid_tag in poi.get("tags", []) for avoid_tag in must_avoid):
            continue

        # Rule 3: Single-item cost should not exceed 60% of daily budget
        # SKIP this rule if user says budget is flexible
        if not override_budget_rule:
            if poi.get("cost", 0) > budget_per_day * 0.6:
                continue

        filtered_pois.append(poi)

    # Safety net: keep at least 5 candidates (relax filtering if too aggressive)
    if len(filtered_pois) < 5:
        filtered_pois = pois  # Revert to full list; let LLM decide

    # ── Geographic clustering: group POIs by proximity ──────────
    # Compute clusters so the LLM can select same-day attractions that are
    # geographically close to each other (avoid cross-city daily plans).
    # Threshold is dynamic: dense cities → ~3-5km, sprawling resorts → ~8-15km
    cluster_threshold_km = _compute_dynamic_cluster_threshold(filtered_pois)
    geo_clusters = _cluster_pois_by_proximity(filtered_pois, threshold_km=cluster_threshold_km)
    valid_clusters = [c for c in geo_clusters if len(c) >= 2]
    cluster_summary_lines = []
    for cid, cluster in enumerate(valid_clusters):
        names = [p.get("name", "?") for p in cluster]
        cluster_summary_lines.append(f"Cluster {cid}: {', '.join(names)}")
    cluster_summary = "\n".join(cluster_summary_lines) if cluster_summary_lines else "(insufficient coordinate data — ignore cluster constraint)"
    logger.info(
        f"Geographic clustering: {len(valid_clusters)} valid clusters "
        f"from {len(filtered_pois)} POIs (threshold={cluster_threshold_km}km)"
    )

    # ═══════════════════════════════════════════
    # Phase 2: LLM daily-structured recommendation
    # Ask the LLM to select and organize POIs into a day-by-day plan.
    # ═══════════════════════════════════════════
    duration = intent.get("duration_days", 3)

    # ── Extract flexible attraction count range ─────────────────
    # Derive default range from pacing preference; user can override via
    # min_attractions_per_day / max_attractions_per_day preferences.
    pacing = preferences.get("pacing", "适中")
    if pacing in ("悠闲", "relaxed"):
        default_min, default_max = 2, 4
    elif pacing in ("紧凑", "intensive"):
        default_min, default_max = 4, 7
    else:  # moderate / 适中
        default_min, default_max = 3, 5
    min_attr = preferences.get("min_attractions_per_day", default_min)
    max_attr = preferences.get("max_attractions_per_day", default_max)
    # Recommended count for LLM = mid-point of range per day
    recommended_per_day = (min_attr + max_attr) // 2
    recommended_count = duration * (recommended_per_day + 2)  # +2 for dining

    logger.info(f"Attraction range: {min_attr}-{max_attr} per day, LLM target ~{recommended_count}")

    # ── Build dynamic preference-driven hard constraints ────────
    # These hints inject user preferences directly into the system prompt
    # so the LLM considers them during POI selection — reducing issues the
    # Critic must flag later.  Covers all structured preference dimensions
    # plus implicit signals from the original query.
    pref_hints = ""
    query_lower = state.get("query", "").lower()
    must_avoid = preferences.get("must_avoid", [])
    avoid_text = " ".join(must_avoid).lower() + " " + query_lower

    # ── Dietary ─────────────────────────────────────────────────
    dietary_prefs = preferences.get("dietary_preferences", [])
    if dietary_prefs:
        pref_hints += (
            f"[DIETARY HARD CONSTRAINT] User dietary preference: {', '.join(dietary_prefs)}. "
            f"ONLY select dining POIs whose name, tags, or description match this cuisine. "
            f"Skip restaurants that don't match — do NOT recommend local food if user wants a different cuisine.\n\n"
        )

    # ── Sun / Heat avoidance ────────────────────────────────────
    if any(kw in avoid_text for kw in ["sun", "outdoor", "heat", "sunburn", "晒", "热", "户外", "暴晒"]):
        pref_hints += (
            "[SUN/HEAT AVOIDANCE HARD CONSTRAINT] User wants to avoid sun/heat/outdoor exposure. "
            "PREFER indoor attractions (museum, gallery, shopping mall, temple, cultural site). "
            "AVOID outdoor attractions (park, nature, hiking, lake, beach). "
            "At least 70% of daily attractions must be indoor or semi-indoor (cultural, shopping).\n\n"
        )

    # ── Proximity ───────────────────────────────────────────────
    if any(kw in avoid_text for kw in ["距离近", "close", "nearby", "near each", "walking distance"]):
        pref_hints += (
            "[PROXIMITY HARD CONSTRAINT] User wants attractions close together. "
            "Group attractions within 3-5 km of each other per day. "
            "Do NOT scatter attractions across the city — cluster them geographically.\n\n"
        )

    # ── Physical level ──────────────────────────────────────────
    physical_level = preferences.get("physical_level", "moderate")
    if physical_level == "low":
        pref_hints += (
            "[PHYSICAL LEVEL CONSTRAINT] User has low physical fitness. "
            "AVOID attractions requiring long walks (>1km), steep climbs, or extended standing. "
            "PREFER attractions with seating, short walking distances, elevator/escalator access. "
            "Include rest breaks between major attractions. Use taxi/ride-share for transit > 500m.\n\n"
        )
    elif physical_level == "high":
        pref_hints += (
            "[PHYSICAL LEVEL HINT] User has high fitness — can include hiking, long walks, "
            "active/adventure attractions that others might find tiring.\n\n"
        )

    # ── Group type ──────────────────────────────────────────────
    group_type = state.get("intent", {}).get("group_type", "").lower()
    query_check = query_lower + " " + " ".join(preferences.get("interests", []))
    is_family = group_type in ("family", "family with kids") or any(
        kw in query_check for kw in ["family", "kid", "child", "baby", "家庭", "孩子", "亲子", "小孩"]
    )
    is_couple = group_type in ("couple", "honeymoon") or any(
        kw in query_check for kw in ["couple", "honeymoon", "romantic", "情侣", "蜜月", "浪漫"]
    )
    is_solo = group_type == "solo" or any(
        kw in query_check for kw in ["solo", "alone", "独自", "一个人"]
    )
    is_elderly = any(kw in query_check for kw in ["elderly", "senior", "老人", "长辈", "父母"])
    if is_family:
        pref_hints += (
            "[FAMILY/KID-FRIENDLY CONSTRAINT] User is traveling with family/children. "
            "PREFER attractions with kid-friendly activities, playgrounds, interactive exhibits. "
            "AVOID adult-only venues, bars, late-night activities. "
            "Choose restaurants with kids' menus or casual atmosphere.\n\n"
        )
    if is_couple:
        pref_hints += (
            "[COUPLE/ROMANTIC HINT] User is a couple — prefer romantic spots, "
            "scenic viewpoints, nice restaurants with ambiance.\n\n"
        )
    if is_solo:
        pref_hints += (
            "[SOLO TRAVELER HINT] User is traveling alone — prefer social hostels, "
            "walking tours, easy dining options for solo diners.\n\n"
        )
    if is_elderly:
        pref_hints += (
            "[ELDERLY ACCESSIBILITY CONSTRAINT] Traveling with elderly — ensure "
            "minimal walking, elevator/escalator access, frequent rest stops, "
            "and attractions suitable for seniors.\n\n"
        )

    # ── Crowd / Atmosphere ──────────────────────────────────────
    wants_quiet = any(kw in query_check for kw in [
        "quiet", "peaceful", "relaxing", "not crowded", "less touristy",
        "安静", "人少", "小众", "避开人群", "不想挤"
    ])
    wants_lively = any(kw in query_check for kw in [
        "lively", "bustling", "popular", "hot", "trending",
        "热闹", "网红", "热门", "火爆"
    ])
    if wants_quiet:
        pref_hints += (
            "[ATMOSPHERE CONSTRAINT] User prefers quiet, less-crowded places. "
            "Prioritize hidden gems, off-peak timing, and lesser-known spots over tourist hotspots. "
            "Suggest early-morning or late-afternoon visit times to avoid crowds.\n\n"
        )
    elif wants_lively:
        pref_hints += (
            "[ATMOSPHERE HINT] User enjoys lively/popular spots — include famous "
            "landmarks, trending restaurants, and bustling areas.\n\n"
        )

    # ── Photography / Scenic ────────────────────────────────────
    wants_photos = any(kw in query_check for kw in [
        "photo", "instagram", "scenic", "view", "picturesque",
        "拍照", "打卡", "出片", "风景", "网红点", "摄影"
    ])
    if wants_photos:
        pref_hints += (
            "[PHOTOGRAPHY CONSTRAINT] User wants photogenic/Instagram-worthy spots. "
            "Prioritize attractions with scenic views, iconic architecture, colorful streets, "
            "rooftop views, and beautiful natural scenery. Each day should have at least 1-2 highly photogenic spots.\n\n"
        )

    # ── Shopping ────────────────────────────────────────────────
    wants_shopping = any(kw in query_check for kw in [
        "shopping", "mall", "market", "boutique", "买", "购物", "逛街", "商场", "市场"
    ])
    if wants_shopping:
        pref_hints += (
            "[SHOPPING HINT] User wants shopping — include markets, shopping streets, "
            "malls, or boutique areas in the itinerary.\n\n"
        )

    # ── Nightlife ───────────────────────────────────────────────
    wants_nightlife = any(kw in query_check for kw in [
        "nightlife", "bar", "club", "night market", "evening",
        "夜生活", "酒吧", "夜市", "晚上", "夜景"
    ])
    if wants_nightlife:
        pref_hints += (
            "[NIGHTLIFE HINT] User wants evening/night activities — include night markets, "
            "rooftop bars, evening shows, or night-view spots. Arrange at least one evening activity per day.\n\n"
        )

    # ── Local / Authentic experience ────────────────────────────
    wants_local = any(kw in query_check for kw in [
        "local", "authentic", "real", "hidden", "off the beaten",
        "本地", "地道", "真正", "小众", "深度"
    ])
    if wants_local:
        pref_hints += (
            "[LOCAL EXPERIENCE CONSTRAINT] User wants authentic/local experiences, not tourist traps. "
            "Prioritize neighborhood gems, family-run restaurants, local markets, "
            "and cultural immersion over commercial tourist attractions.\n\n"
        )

    # ── Interests-driven activity type guidance ─────────────────
    interests = preferences.get("interests", [])
    if interests:
        interest_hint = f"[INTEREST MATCHING] User interests: {', '.join(interests)}. "
        interest_hint += (
            "At least 60% of attractions should directly relate to these interests. "
            "For each attraction, explain how it connects to the user's interests in the 'why' field.\n\n"
        )
        pref_hints += interest_hint

    # ── Language detection for report output ───────────────────
    import re as _re_lang
    query = state.get("query", "")
    is_cn = bool(_re_lang.search(r'[\u4e00-\u9fff]', query))
    user_lang = "Chinese" if is_cn else "English"
    # Localized example fields for the JSON template
    if is_cn:
        ex_theme = "皇城历史线"
        ex_avoid = "王府井主街（游客陷阱，价高质低）"
    else:
        ex_theme = "Imperial History Route"
        ex_avoid = "Wangfujing main street (tourist trap, overpriced)"

    # ── Budget data for cost constraint ────────────────────────
    budget_usd = intent.get("budget_usd", 0)
    budget_per_day_usd = budget_usd / max(duration, 1) if budget_usd else 0
    budget_currency = intent.get("budget_currency", "USD")

    system_prompt = (
        f"You are an elite local travel recommender with deep insider knowledge. "
        f"[LANGUAGE RULE — CRITICAL] Write ALL content in {user_lang}. "
        f"All descriptions, tips, day_theme, avoid, why fields MUST be in {user_lang}. "
        f"Do NOT write any content in another language. "
        f"Only POI/hotel names may remain in their original language.\n\n"
        f"[BUDGET HARD CONSTRAINT — CRITICAL]\n"
        f"Total budget: {budget_usd:.0f} USD (= approx {intent.get('budget_amount', budget_usd):.0f} {budget_currency}).\n"
        f"Per-day budget ceiling: {budget_per_day_usd:.0f} USD.\n"
        f"The SUM of all POI costs + hotel prices across the entire trip MUST NOT exceed {budget_usd:.0f} USD.\n"
        f"Select affordable POIs and hotels that fit within this limit. "
        f"If the budget is tight, prefer free/cheap attractions and budget-friendly restaurants.\n\n"
        f"[ANTI-HALLUCINATION FOR COST/RATING — CRITICAL]\n"
        f"For each POI, use the EXACT 'cost' and 'rating' values from the candidate POI data below. "
        f"Do NOT invent, modify, or estimate costs/ratings. "
        f"Copy the numbers verbatim from the input data into your JSON output.\n\n"
        f"Select the top {recommended_count} POIs from the candidate list "
        f"based on a multi-dimensional scoring system for a {duration}-day trip:\n"
        "- Theme Match (40%): How well does the POI match the user's theme and interests?\n"
        "- Budget Fit (25%): Is the cost reasonable within the budget?\n"
        "- User Rating (15%): Consider the POI's rating and review count.\n"
        "- Time Efficiency (10%): Is the visit time reasonable for the trip duration?\n"
        "- Crowd Avoidance (10%): Prefer less crowded alternatives during peak seasons.\n\n"
        f"You MUST organize your recommendations into a daily plan structure.\n"
        f"For a {duration}-day trip, create exactly {duration} days of activities.\n\n"
        "[CRITICAL - NO DUPLICATE POIs]\n"
        "Each attraction and dining spot MUST appear ONLY ONCE across the entire trip. "
        "Do NOT recommend the same place on multiple days. Every day must have unique, "
        "different attractions and restaurants. If the destination doesn't have enough "
        "unique POIs, reduce the number per day rather than repeating.\n\n"
        "[LOCAL INSIDER KNOWLEDGE]\n"
        "- For DINING: you MUST recommend specific signature dishes (2-3 dish names per restaurant). "
        "Prefer time-honored local restaurants that locals frequent, NOT tourist traps.\n"
        "- For ATTRACTIONS: include a 'why' field explaining WHY this POI is chosen "
        "(e.g., 'Less crowded than Badaling, gentler slope, cable car available').\n"
        "- For each day: include a 'day_tips' array with 1-3 practical tips "
        "(e.g., booking reminders, what to bring, crowd avoidance timing).\n"
        "- include an 'avoid' array listing 1-2 tourist traps or crowded spots to skip that day.\n\n"
        "Each day MUST include:\n"
        f"- {min_attr} to {max_attr} attractions (sightseeing, cultural, outdoor, shopping spots)\n"
        "- 2 dining recommendations that are NEAR the day's attractions (within walking distance or 1-2 metro stops). "
        "Do NOT just list famous restaurants across the city — pick ones that are geographically convenient for that day's route.\n"
        "- 1 hotel per day following REAL traveler habits:\n"
        "  * For trips of 1-3 days: use the SAME hotel for all days (travelers don't switch hotels for short trips).\n"
        "  * For trips of 4-6 days: use at most 2 different hotels (switch only if activities move to a significantly different area).\n"
        "  * For trips of 7+ days: use at most 3 different hotels, switching only when the itinerary moves to a new district/area.\n"
        "  * When switching hotels, stay at least 2 consecutive nights at each hotel.\n"
        "  * Choose hotels with good access to that period's attractions.\n\n"
        "[CATEGORY SEPARATION RULES — CRITICAL]\n"
        "- The 'attractions' list must ONLY contain sightseeing, cultural, outdoor, or shopping POIs.\n"
        "  Do NOT put restaurants, cafes, or any food-related POIs in the attractions list.\n"
        "- The 'dining' list must ONLY contain restaurants, cafes, street food stalls.\n"
        "  Do NOT put hotels, attractions, or sightseeing POIs in the dining list.\n"
        "- Hotels must ONLY appear in the 'hotel' field, never in attractions or dining.\n"
        "- Check each POI's 'type' field before assigning it to a section.\n\n"
        "[ROUTE PLANNING RULES — CRITICAL]\n"
        "- Arrange each day's attractions in a geographically logical order (minimize backtracking).\n"
        "- **[GEOGRAPHIC CLUSTER CONSTRAINT]** All attractions for a single day MUST belong to "
        "the SAME geographic cluster (see cluster table below). Do NOT scatter attractions "
        "across different clusters on the same day — this creates unrealistic cross-city schedules.\n"
        "- For each attraction, specify suggested_transport (how to get there from the previous stop): "
        "'walk' (if <15 min), 'metro', 'bus', 'taxi', or 'walk+metro'.\n"
        "- For each attraction, specify avg_visit_time_min (realistic time including queuing).\n"
        "- For each attraction/dining, specify start_time in HH:MM format (24h). "
        "Plan a realistic daily schedule starting from ~08:30-09:00, accounting for transit time between stops.\n"
        "- **[NO TIME OVERLAP]** Adjacent activities MUST NOT overlap. "
        "start_time of next = (start_time of previous + avg_visit_time_min + transit_time_min). "
        "Add at least 15 min buffer between activities for walking, restroom, ticket purchase.\n"
        "- **[MAXIMUM 8 ACTIVITIES PER DAY]** The total schedule (first start to last end) "
        "should NOT exceed 13 hours (08:00–21:00). Prioritise quality over quantity.\n"
        "- Place dining between attractions at natural meal times: lunch 11:30-13:00, dinner 18:00-19:30.\n"
        "- For dining, specify 'near' field indicating which attraction it's close to.\n"
        "- Include transit_time_min for each item (time to get there from previous stop).\n\n"
        + pref_hints +
        "Return JSON format:\n"
        "{\n"
        '  "daily_plans": [\n'
        "    {\n"
        '      "day": 1,\n'
        f'      "day_theme": "{ex_theme}",\n'
        '      "day_tips": ["提前7天在故宫小程序抢票", "穿舒适平底鞋"],\n'
        f'      "avoid": ["{ex_avoid}"],\n'
        '      "attractions": [\n'
        '        {"name": "...", "type": "...", "cost": ..., "rating": ..., "description": "...", '
        '"why": "reason for choosing this over alternatives", '
        '"start_time": "09:00", "avg_visit_time_min": ..., "transit_time_min": ..., "suggested_transport": "metro/walk/bus/taxi"},\n'
        "        ...\n"
        "      ],\n"
        '      "dining": [\n'
        '        {"name": "...", "type": "restaurant/cafe/street_food", "cost": ..., "rating": ..., "description": "...", '
        '"signature_dishes": ["dish1", "dish2"], '
        '"start_time": "12:00", "near": "nearby attraction name", "transit_time_min": ..., "suggested_transport": "walk"},\n'
        "        ...\n"
        "      ],\n"
        '      "hotel": {"name": "...", "price_per_night": ..., "rating": ..., "description": "..."}\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    # ── Build user prompt with all context sources ──────────────
    user_prompt = f"User intent: {intent}\n"
    user_prompt += f"User preferences: {preferences}\n"
    user_prompt += f"Weather conditions: {weather}\n"
    user_prompt += f"Candidate POIs (pre-filtered to {len(filtered_pois)}): {json.dumps(filtered_pois, ensure_ascii=False)}\n"

    # Inject geographic cluster summary — shows which POIs are near each other
    if cluster_summary_lines:
        user_prompt += (
            f"\n[GEOGRAPHIC CLUSTERS — CRITICAL]\n"
            f"POIs in the same cluster are within {cluster_threshold_km}km of each other. "
            f"Each day's attractions MUST all come from the SAME cluster.\n"
            f"{cluster_summary}\n"
            f"Exception: if a cluster has too few attractions, you may mix 2 adjacent "
            f"clusters ONLY IF at least 3 attractions come from the primary cluster.\n"
        )

    # Inject Wikivoyage destination knowledge to ground the LLM and
    # reduce hallucination (truncated to 2000 chars to fit context window)
    if wikivoyage:
        user_prompt += f"\n\n[DESTINATION KNOWLEDGE from Wikivoyage - use as reference to reduce hallucination]:\n{json.dumps(wikivoyage, ensure_ascii=False)[:2000]}"

    # Inject available hotel list (top 10) so the LLM can assign hotels
    # that vary by day based on proximity to each day's activities
    if hotels:
        user_prompt += f"\n\n[AVAILABLE HOTELS - assign based on hotel proximity rules above]:\n{json.dumps(hotels[:10], ensure_ascii=False)}"

    # Inject previous audit findings (from critic_node) so the LLM can
    # adjust recommendations to address identified issues (replan loop)
    if state.get("audit_findings"):
        user_prompt += f"\n⚠️ Previous recommendation failed audit. Failure reasons: {state['audit_findings']}\nPlease adjust recommendations based on the above issues!\n"

    # Inject blacklisted POI combinations so the LLM avoids recommending
    # the same set that was already rejected by the critic
    if state.get("rejected_plans"):
        user_prompt += f"\n🚫 The following combinations have been rejected. Do NOT recommend them again: {state['rejected_plans']}\n"

    # Inject mandatory user feedback (from human-in-the-loop review)
    if state.get("user_feedback"):
        user_prompt += f"\n👤 User mandatory requirements (must comply): {state['user_feedback']}\n"

    # Inject must_visit / must_include places as hard constraints
    if must_include_names:
        user_prompt += (
            f"\n🔴 [HARD CONSTRAINT - NON-NEGOTIABLE]: The following places MUST appear "
            f"in the itinerary (the user explicitly requested them): "
            f"{', '.join(must_include_names)}\n"
            f"You MUST include each of these places at least once in the daily plan. "
            f"If any of these places are not in the candidate POI list, search for them "
            f"by name in the candidate list using fuzzy matching (partial name match).\n"
        )

    # Send prompt to LLM; temperature=0.5 balances creativity and consistency
    response = llm_client.chat(system_prompt, user_prompt, json_format=True, temperature=0.5)
    parsed = _safe_json_parse(response, context="Recommendation")

    # ── Parse LLM response ──────────────────────────────────────
    # The LLM is expected to return {"daily_plans": [...]}, but we handle
    # alternative shapes (bare list, dict with different key) for robustness.
    daily_plans = []
    if isinstance(parsed, dict) and "daily_plans" in parsed:
        # Standard case: response has the expected "daily_plans" key
        daily_plans = parsed["daily_plans"]
    elif isinstance(parsed, dict):
        # Fallback: try to find any list-valued entry that looks like daily plans
        for value in parsed.values():
            if isinstance(value, list):
                daily_plans = value
                break
    elif isinstance(parsed, list):
        # Fallback: response is a bare list of day plans
        daily_plans = parsed

    # Defensive filter: ensure every day entry is a dict (discard malformed items)
    daily_plans = [d for d in daily_plans if isinstance(d, dict)] if isinstance(daily_plans, list) else []

    # ── Post-process: enforce daily structure ────────────────────
    # Use the raw POI pool (state["raw_knowledge"]["pois"]) for gap-filling,
    # NOT filtered_pois which may be too small after rule-based filtering.
    raw_poi_pool = state.get("raw_knowledge", {}).get("pois", [])
    if not raw_poi_pool:
        raw_poi_pool = filtered_pois  # fallback to filtered list if raw unavailable

    # ── Anti-hallucination filter ─────────────────────────────────
    # LLMs sometimes fabricate POIs not in the candidate list. This filter
    # removes any POI whose name does not appear in the real API data,
    # ensuring every recommendation is backed by actual data.
    known_poi_names = {p.get("name", "") for p in raw_poi_pool}
    # Build lookup maps for anti-hallucination backfill.
    # LLMs often fabricate prices/ratings even for real POIs, so we
    # overwrite numeric fields with authoritative API data after validation.
    hotel_by_name = {h.get("name", ""): h for h in hotels if isinstance(h, dict) and h.get("name")}
    poi_by_name = {p.get("name", ""): p for p in raw_poi_pool if isinstance(p, dict) and p.get("name")}
    # Apply the anti-hallucination filter to each day's attractions and dining
    total_removed = 0
    for day in daily_plans:
        orig_attractions = day.get("attractions", [])
        orig_dining = day.get("dining", [])
        # Keep only POIs whose names exist in the real API data
        day["attractions"] = [a for a in orig_attractions if a.get("name", "") in known_poi_names]
        day["dining"] = [d for d in orig_dining if d.get("name", "") in known_poi_names]

        # Backfill POI cost/rating from real API data to prevent LLM price hallucination
        for poi in day["attractions"] + day["dining"]:
            real_poi = poi_by_name.get(poi.get("name", ""))
            if real_poi:
                for field in ("cost", "rating", "lat", "lng", "type", "tags", "website", "maps_url"):
                    real_val = real_poi.get(field)
                    if real_val is not None:
                        poi[field] = real_val

        # Validate hotel assignment: set to None if hotel name is not in the real list
        hotel = day.get("hotel")
        if isinstance(hotel, dict) and hotel.get("name", "") not in hotel_by_name and hotel_by_name:
            day["hotel"] = None
        elif isinstance(hotel, dict) and hotel.get("name", "") in hotel_by_name:
            # Backfill hotel fields from real API data to prevent LLM price hallucination
            real_hotel = hotel_by_name[hotel["name"]]
            for field in ("price_per_night", "total_price", "rating", "reviews",
                          "amenities", "image_url", "lat", "lng", "currency",
                          "website", "maps_url"):
                real_val = real_hotel.get(field)
                if real_val is not None:
                    hotel[field] = real_val

        removed = (len(orig_attractions) - len(day["attractions"])) + (len(orig_dining) - len(day["dining"]))
        total_removed += removed
    if total_removed > 0:
        logger.info(
            f"Anti-hallucination filter: removed {total_removed} LLM-fabricated POIs "
            f"(not in real POI pool of {len(known_poi_names)} items)"
        )

    # Deduplicate POIs across days: each attraction/dining spot appears only once
    seen_names = set()
    dedup_removed = 0
    for day in daily_plans:
        unique_attractions = []
        for a in day.get("attractions", []):
            name = a.get("name", "")
            if name not in seen_names:
                seen_names.add(name)
                unique_attractions.append(a)
            else:
                dedup_removed += 1
        day["attractions"] = unique_attractions

        unique_dining = []
        for d in day.get("dining", []):
            name = d.get("name", "")
            if name not in seen_names:
                seen_names.add(name)
                unique_dining.append(d)
            else:
                dedup_removed += 1
        day["dining"] = unique_dining
    if dedup_removed > 0:
        logger.info(f"Cross-day deduplication: removed {dedup_removed} duplicate POIs")

    # ── Time-slot conflict detection & repair ───────────────────
    # LLM often generates overlapping schedules or over-packed days.
    # This runs a deterministic validator that pushes later activities
    # forward when overlaps are detected and warns on over-packed days.
    _validate_time_slots(daily_plans)

    # ── Proximity enforcement: ensure dining is near day's attractions ──────
    # If a dining spot has coordinates and is >3km from the centroid of the
    # day's attractions, try to swap it with a closer restaurant from the pool.
    MAX_DINING_DISTANCE_KM = 3.0
    dining_pool = [p for p in raw_poi_pool if p.get("type") in
                   ("restaurant", "cafe", "street_food", "dining", "food")
                   and p.get("lat") and p.get("lng")]
    swaps_made = 0
    for day in daily_plans:
        attractions = day.get("attractions", [])
        # Compute centroid of day's attractions that have coordinates
        attr_coords = [(a["lat"], a["lng"]) for a in attractions
                       if a.get("lat") and a.get("lng")]
        if not attr_coords:
            continue
        centroid_lat = sum(c[0] for c in attr_coords) / len(attr_coords)
        centroid_lng = sum(c[1] for c in attr_coords) / len(attr_coords)

        new_dining = []
        for d in day.get("dining", []):
            d_lat, d_lng = d.get("lat"), d.get("lng")
            if d_lat and d_lng:
                dist = _haversine_distance(centroid_lat, centroid_lng, d_lat, d_lng)
                if dist <= MAX_DINING_DISTANCE_KM:
                    new_dining.append(d)
                    continue
                # Too far — try to find a closer restaurant from pool
                best_replacement = None
                best_dist = MAX_DINING_DISTANCE_KM
                for candidate in dining_pool:
                    if candidate.get("name") in seen_names:
                        continue  # already used
                    c_dist = _haversine_distance(
                        centroid_lat, centroid_lng,
                        candidate["lat"], candidate["lng"]
                    )
                    if c_dist < best_dist:
                        best_dist = c_dist
                        best_replacement = candidate
                if best_replacement:
                    seen_names.add(best_replacement["name"])
                    # Preserve LLM fields structure
                    replacement = {
                        "name": best_replacement["name"],
                        "type": best_replacement.get("type", "restaurant"),
                        "cost": best_replacement.get("cost", d.get("cost", 0)),
                        "rating": best_replacement.get("rating", 4.0),
                        "description": best_replacement.get("description", ""),
                        "lat": best_replacement["lat"],
                        "lng": best_replacement["lng"],
                        "near": attractions[0].get("name", "") if attractions else "",
                        "suggested_transport": "walk",
                        "start_time": d.get("start_time", "12:00"),
                        "transit_time_min": int(best_dist / 0.08),  # ~5km/h walk
                    }
                    new_dining.append(replacement)
                    swaps_made += 1
                    logger.debug(
                        f"Swapped distant dining '{d['name']}' ({dist:.1f}km) "
                        f"with nearby '{best_replacement['name']}' ({best_dist:.1f}km)"
                    )
                else:
                    new_dining.append(d)  # no better option, keep original
            else:
                new_dining.append(d)  # no coords, can't validate
        day["dining"] = new_dining
    if swaps_made > 0:
        logger.info(f"Proximity enforcement: swapped {swaps_made} distant dining spots with nearby alternatives")

    # Run deterministic post-processing to enforce daily structure constraints
    daily_plans = enforce_daily_structure(
            daily_plans, raw_poi_pool, hotels, duration,
            min_attractions=min_attr,
            max_attractions=max_attr,
        )

    # Build a flat list of all recommended POIs for backward compatibility
    # (some downstream nodes use the flat list instead of the daily structure)
    all_recommended = []
    for day in daily_plans:
        all_recommended.extend(day.get("attractions", []))
        all_recommended.extend(day.get("dining", []))

    # Last-resort fallback: if the LLM + post-processing produced nothing,
    # return the top 4 pre-filtered candidates so the pipeline doesn't break
    if not all_recommended:
        logger.warning("Recommendation LLM returned no valid POIs; falling back to top candidates.")
        all_recommended = filtered_pois[:4]

    return {
        "recommended_pois": all_recommended,
        "daily_itinerary": daily_plans,
        "progress_logs": [
            _progress(state, "🎯 正在根据偏好筛选推荐...", "🎯 Filtering recommendations based on preferences..."),
            _progress(state, "📋 行程推荐已生成", "📋 Itinerary recommendations generated"),
        ],
    }
