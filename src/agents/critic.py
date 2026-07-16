"""Critic node — risk and quality audit of the itinerary.

The critic acts as a gatekeeper in the replan loop. If any rule fails,
it produces ``audit_findings`` that trigger the AutoReplan node (ReplanAgent
tool-use loop) to fix the issues (up to ``MAX_REPLAN_ATTEMPTS`` times).
"""

import json
import logging
from typing import Any

from src.state import TravelState
from src.agents.utils import _progress, llm_client, _FOOD_TYPES, _is_food_poi, _safe_json_parse

logger = logging.getLogger(__name__)


def critic_node(state: TravelState) -> dict[str, Any]:
    """Audit the itinerary against quality, budget, safety, and feasibility rules.

    The critic acts as a gatekeeper in the replan loop. If any rule fails,
    it produces ``audit_findings`` that trigger the recommendation node to
    regenerate the itinerary (up to ``max_replan`` times).

    **Audit rules** (in order):
      1. **Budget constraint** – hard-fail if cost > 150% of budget.
      1b. **Hotel budget ratio** – warn if hotel costs > 70% of total budget.
      2. **Weather conflict** – fail if rain is forecast and outdoor POIs are included.
      2b. **Weather alerts** – fail if any alert has severity level >= 3/5.
      3. **Time feasibility** – fail if total visit + transit hours exceed
         ``duration * 10`` hours (10 active hours/day max).
      4. **Theme coverage** – LLM-based semantic match; fail if < 30% of POIs
         match user interests.
      5. **Rating floor** – fail if average POI rating < 3.0/5.0.
      6. **Meal arrangement** – fail if no dining POIs are included.
      7. **Daily structure** – pacing-based attraction range, 2 dining, 1 hotel,
         category separation, hotel consistency.
      8. **Dietary compliance** – fail if dining POIs don't match user's
         dietary preferences (e.g. requested Western cuisine but only local food).
      9. **Sun/heat avoidance** – fail if user wants to avoid sun/outdoor
         exposure but itinerary is outdoor-heavy on sunny/hot days.
      10. **Geographic coherence** – always-on check for same-day attraction spread
         (max 8km) AND back-and-forth (zigzag) route patterns.
     11. **Holistic preference compliance (LLM)** – catch-all LLM audit that
          evaluates the itinerary against ALL preference signals (structured
          fields + original query + feedback) for dimensions NOT covered by
          rules 1-10, such as: activity-type/interest match, physical-level
          appropriateness, group suitability, atmosphere, photography value,
          shopping, nightlife, cultural depth, crowd tolerance, accessibility,
          and any implicit preference hidden in natural language.

    API: LLM (DeepSeek Chat via ``llm_client.chat``) — Rules 4 & 11
        Rule 4 prompt mode: ``json_format=True``, ``temperature=0.0``, ``max_tokens=200``.
        Rule 11 prompt mode: ``json_format=True``, ``temperature=0.0``, ``max_tokens=400``.

            {
                "match_ratio": 0.85,
                "explanation": "Most POIs align with food and culture interests."
            }

    Args:
        state: Current ``TravelState`` dict. Requires:
            - ``intent``, ``routing_metrics``, ``recommended_pois``,
            - ``raw_knowledge`` (weather), ``daily_itinerary``,
            - ``user_preferences``.

    Returns:
        A partial state dict with keys:
        - ``audit_findings`` (list[str]): Empty if all rules pass.
        - ``replan_count`` (int): Incremented only if findings are non-empty.
        - ``rejected_plans`` (list): Blacklisted POI combinations (if findings).
    """
    budget = state["intent"].get("budget_usd", float('inf'))
    duration = state["intent"].get("duration_days", 1)
    cost = state["routing_metrics"]["total_cost"]
    interests = state.get("user_preferences", {}).get("interests", [])
    pois = state["recommended_pois"]

    # Collect all audit violations in this list
    findings = []

    # ═══ Rule 1: Budget constraint with 50% tolerance ═══
    # Hard-fail only if cost exceeds 150% of budget. Minor overages (100–150%)
    # are logged as warnings to prevent infinite replan loops when POI costs
    # are fixed values from real API data.
    if cost > budget * 1.5:
        findings.append(
            f"Budget severely exceeded! Budget: ${budget:.0f}, actual cost: ${cost:.0f}.")
    elif cost > budget:
        logger.warning(
            f"Budget slightly exceeded: ${cost:.0f} vs budget ${budget:.0f} "
            f"(within 50% tolerance, skipping replan)")

    # ═══ Rule 1b: Hotel budget ratio ═══
    # Check if hotel costs dominate the budget (>70%); log a suggestion
    # but don't hard-fail (informational only).
    daily_itinerary = state.get("daily_itinerary", [])
    hotel_total = 0
    if daily_itinerary:
        for day_plan in daily_itinerary:
            hotel = day_plan.get("hotel", {})
            if isinstance(hotel, dict):
                hotel_total += hotel.get("price_per_night") or 0

    budget_usd = state["intent"].get("budget_usd", float('inf'))
    if hotel_total > budget_usd * 0.7:
        findings.append(
            f"Hotel costs (${hotel_total:.0f}) exceed 70% of total budget (${budget_usd:.0f}). "
            f"Search for more budget-friendly hotels via search_hotel and swap them in.")
        logger.info(
            f"[Suggestion] Hotel costs (${hotel_total}) exceed 70% of total budget (${budget_usd}). "
            f"Consider offering more budget-friendly accommodation options.")

    # ═══ Rule 2: Weather-itinerary conflict ═══
    # Detect outdoor activities scheduled during rainy weather
    weather = state["raw_knowledge"]["weather"]["condition"]
    has_outdoor = any(p.get("type") == "outdoor" for p in pois)
    if weather == "Rainy" and has_outdoor:
        findings.append(
            "Rain forecast but itinerary includes outdoor activities. Consider substituting indoor attractions.")

    # ═══ Rule 2b: Weather alert severity check ═══
    # Check for high-severity weather alerts (level >= 3 out of 5)
    # that could pose safety risks for outdoor activities.
    weather_data = state["raw_knowledge"]["weather"]
    weather_alerts = weather_data.get("weather_alerts", [])
    for alert in weather_alerts:
        if alert.get("alert_level", 0) >= 3:
            findings.append(
                f"Weather alert [{alert.get('alert_type', 'unknown')}]: "
                f"{alert.get('headline', 'No details')} "
                f"(severity: {alert.get('severity', 'unknown')}, level: {alert.get('alert_level', 0)}/5). "
                f"Consider adjusting the itinerary to avoid outdoor exposure.")

    # ═══ Rule 3: Time feasibility ═══
    # Estimate total hours needed (visit time + transit) and compare
    # against the maximum feasible hours for the trip duration.
    total_visit_time = sum(p.get("avg_visit_time_min", 60) for p in pois)
    # Simplified transit estimate: 20 min per location change
    transport_time = (len(pois) - 1) * 20
    total_hours = (total_visit_time + transport_time) / 60
    max_hours = duration * 10  # Max 10 active hours per day
    if total_hours > max_hours:
        findings.append(
            f"Itinerary is too packed! Estimated {total_hours:.1f} hours needed, "
            f"but {duration} day(s) can accommodate at most {max_hours} hours.")

    # ═══ Rule 4: Theme coverage (LLM-based semantic matching) ═══
    # Uses an LLM to evaluate how well the POI set matches user interests.
    # This is more flexible than keyword matching (e.g. a temple matches
    # "historical exploration", a market matches "food & culture").
    # Fails if the match ratio is below 30%.
    if interests and pois:
        poi_summaries = [f"{p.get('name')} (type: {p.get('type')}, tags: {p.get('tags', [])})" for p in pois]
        coverage_prompt = (
            f"User interests: {interests}\n"
            f"Recommended POIs:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(poi_summaries)) + "\n\n"
            "Evaluate how well these POIs match the user's interests. "
            "Return a JSON object with exactly two fields:\n"
            '{"match_ratio": <float 0.0-1.0>, "explanation": "<brief reason>"}\n'
            "match_ratio = proportion of POIs that are relevant to at least one user interest. "
            "Be generous: a tourist landmark matches 'popular attractions', a restaurant matches 'cuisine/food'.'"
        )
        try:
            # Ask LLM to score POI-interest relevance; temperature=0.0 for consistency
            coverage_resp = llm_client.chat(
                system_prompt="You are a travel relevance evaluator. Return only valid JSON.",
                user_prompt=coverage_prompt,
                temperature=0.0,
                max_tokens=200,
                json_format=True
            )
            coverage_result = _safe_json_parse(coverage_resp, context="Critic/coverage")
            match_ratio = float(coverage_result.get("match_ratio", 1.0))
            # Threshold: <30% match triggers an audit failure
            if match_ratio < 0.3:
                explanation = coverage_result.get("explanation", "")
                findings.append(
                    f"Low theme coverage ({match_ratio:.0%}): {explanation}")
            logger.debug(f"Theme coverage LLM result: ratio={match_ratio}, explanation={coverage_result.get('explanation')}")
        except Exception as e:
            logger.warning(f"Theme coverage LLM check failed: {e}, skipping rule 4")

    # ═══ Rule 5: Rating floor ═══
    # Ensure the average POI rating meets a minimum quality bar (3.0/5.0)
    ratings = [p.get("rating", 0) for p in pois if p.get("rating")]
    if ratings:
        avg_rating = sum(ratings) / len(ratings)
        if avg_rating < 3.0:
            findings.append(
                f"Average POI rating is too low ({avg_rating:.1f}/5.0). "
                "Consider recommending better-rated attractions.")

    # ═══ Rule 6: Meal arrangement ═══
    # Ensure at least one dining POI is included in the itinerary.
    # A full-day trip without meal recommendations is a poor user experience.
    has_dining = any(
        p.get("type", "").lower() in _FOOD_TYPES for p in pois
    )
    if not has_dining and duration >= 1:
        findings.append(
            "No dining arrangements included in the full-day itinerary. "
            "Add at least one food-related attraction.")

    # ═══ Rule 7: Daily structure coverage ═══
    # Verify that each day meets structural requirements:
    # - Pacing-based attraction range (relaxed: 2-4, moderate: 3-5, intensive: 4-7)
    # - 2 dining options and 1 hotel per day
    # - Category separation (attractions ≠ dining, dining ≠ hotels/attractions)
    # - Hotel consistency across days (prefer one hotel)
    user_prefs = state.get("user_preferences", {})
    pacing = user_prefs.get("pacing", "适中")
    if pacing in ("悠闲", "relaxed"):
        min_attr, max_attr = 2, 4
    elif pacing in ("紧凑", "intensive"):
        min_attr, max_attr = 4, 7
    else:  # moderate / 适中
        min_attr, max_attr = 3, 5
    # Allow user-specified overrides
    min_attr = user_prefs.get("min_attractions_per_day", min_attr)
    max_attr = user_prefs.get("max_attractions_per_day", max_attr)

    # Build hotel name lookup from raw knowledge for cross-category checks
    raw_hotels = state.get("raw_knowledge", {}).get("hotels", [])
    hotel_names = {h.get("name", "") for h in raw_hotels if h.get("name")}

    if daily_itinerary:
        for day_plan in daily_itinerary:
            day_num = day_plan.get("day", "?")
            attractions = day_plan.get("attractions", [])
            dining = day_plan.get("dining", [])
            hotel = day_plan.get("hotel")

            if len(attractions) < min_attr:
                findings.append(f"Day {day_num}: Only {len(attractions)} attractions (need at least {min_attr})")
            if len(attractions) > max_attr:
                findings.append(f"Day {day_num}: Too many attractions ({len(attractions)}, max {max_attr})")
            if len(dining) < 2:
                findings.append(f"Day {day_num}: Only {len(dining)} dining options (need 2)")
            if not hotel:
                findings.append(f"Day {day_num}: No hotel assigned")

            # Category type checks: attractions must not be dining, dining must not be hotels/attractions
            for a in attractions:
                if _is_food_poi(a):
                    findings.append(
                        f"Day {day_num}: '{a.get('name')}' (type={a.get('type')}) is a dining POI in attractions"
                    )
            for d in dining:
                dname = d.get("name", "")
                if dname in hotel_names:
                    findings.append(
                        f"Day {day_num}: '{dname}' is a hotel in dining section"
                    )
                elif not _is_food_poi(d) and d.get("type"):
                    findings.append(
                        f"Day {day_num}: '{dname}' (type={d.get('type')}) is not a dining POI in dining section"
                    )

        # Hotel consistency check — prefer one hotel for the entire trip
        itinerary_hotels = set()
        for day_plan in daily_itinerary:
            h = day_plan.get("hotel", {})
            if isinstance(h, dict) and h.get("name"):
                itinerary_hotels.add(h["name"])
        if len(itinerary_hotels) > 1:
            findings.append(
                f"Multiple hotels across trip: {', '.join(sorted(itinerary_hotels))}. "
                f"Consider using one hotel for the entire trip."
            )
    else:
        findings.append("No daily itinerary structure found")

    # ═══ Rule 8: Dietary preference compliance ═══
    # Check whether the dining POIs match the user's explicit dietary
    # preferences (e.g. "Western cuisine", "Japanese food", "vegetarian").
    # If the user specified a cuisine preference and no dining POI matches,
    # flag it so the replan agent can swap in matching restaurants.
    dietary_prefs = user_prefs.get("dietary_preferences", [])
    if dietary_prefs:
        dietary_lower = " ".join(dietary_prefs).lower()
        # Build keyword mapping for common cuisine types
        cuisine_keywords: dict[str, list[str]] = {
            "western": ["western", "italian", "french", "american", "steak",
                        "pasta", "pizza", "grill", "european", "mexican",
                        "mediterranean", "西餐", "牛排", "披萨", "意面"],
            "japanese": ["japanese", "sushi", "ramen", "izakaya", "日料",
                         "寿司", "拉面", "居酒屋"],
            "chinese": ["chinese", "dim sum", "hunan", "sichuan", "cantonese",
                        "中餐", "川菜", "粤菜", "火锅"],
            "korean": ["korean", "bbq", "bibimbap", "韩餐", "烤肉"],
            "seafood": ["seafood", "fish", "crab", "oyster", "海鲜"],
            "vegetarian": ["vegetarian", "vegan", "plant", "素食"],
        }
        target_cuisine: list[str] = []
        for cuisine, keywords in cuisine_keywords.items():
            if any(kw in dietary_lower for kw in keywords):
                target_cuisine.append(cuisine)

        if target_cuisine:
            all_dining = []
            for day_plan in daily_itinerary:
                all_dining.extend(day_plan.get("dining", []))
            # Check if any dining POI matches the target cuisine
            dining_matched = False
            for d in all_dining:
                d_text = (d.get("name", "") + " " +
                          " ".join(d.get("tags", [])) + " " +
                          d.get("description", "") + " " +
                          (d.get("type") or "")).lower()
                for kw_list in [cuisine_keywords.get(c, []) for c in target_cuisine]:
                    if any(kw in d_text for kw in kw_list):
                        dining_matched = True
                        break
                if dining_matched:
                    break
            if not dining_matched:
                findings.append(
                    f"Dietary mismatch: user prefers {', '.join(target_cuisine)} cuisine "
                    f"but no dining POIs match. Swap in matching restaurants."
                )

    # ═══ Rule 9: Sun/heat avoidance compliance ═══
    # Check if the user wants to avoid outdoor exposure (sun/heat) but
    # the itinerary is outdoor-heavy on sunny/hot days.
    wants_indoor = False
    # Signal 1: must_avoid contains sun/outdoor keywords
    must_avoid = user_prefs.get("must_avoid", [])
    avoid_text = " ".join(must_avoid).lower()
    sun_keywords = ["sun", "outdoor", "heat", "sunburn", "tan", "晒", "热",
                    "户外", "暴晒", "indoor", "室内", "sun exposure"]
    if any(kw in avoid_text for kw in sun_keywords):
        wants_indoor = True
    # Signal 2: query mentions sun avoidance
    query_lower = state.get("query", "").lower()
    if any(kw in query_lower for kw in ["不想被晒", "怕晒", "怕热", "avoid sun",
                                         "don't want sun", "stay indoor"]):
        wants_indoor = True
    # Signal 3: user explicitly prefers indoor
    if any(kw in " ".join(interests).lower() for kw in ["indoor", "室内"]):
        wants_indoor = True

    if wants_indoor:
        weather_condition = state["raw_knowledge"]["weather"].get("condition", "")
        weather_temp = state["raw_knowledge"]["weather"].get("temperature_c", 20)
        is_hot_sunny = weather_condition in ("Sunny", "Clear", "Partly Cloudy") or weather_temp > 28
        if is_hot_sunny:
            all_attrs = []
            for day_plan in daily_itinerary:
                all_attrs.extend(day_plan.get("attractions", []))
            outdoor_count = sum(1 for a in all_attrs if a.get("type") == "outdoor")
            total_attrs = len(all_attrs)
            if total_attrs > 0 and outdoor_count / total_attrs > 0.3:
                findings.append(
                    f"Sun avoidance: {outdoor_count}/{total_attrs} attractions are outdoor "
                    f"({weather_condition}, {weather_temp}°C). User wants to avoid sun/heat. "
                    f"Replace outdoor attractions with indoor alternatives (museum, gallery, mall)."
                )

    # ═══ Rule 10: Geographic coherence — no back-and-forth ═══
    # Always-on check: same-day attractions must be geographically coherent.
    # Detects both "too far apart" AND "back-and-forth" (zigzag) patterns.
    if daily_itinerary:
        from src.api.geocoding import _haversine_distance
        transport_matrix = state.get("transport_matrix", {})
        max_spread_km = 8.0  # Max distance between any two same-day attractions
        backforth_threshold_km = 6.0  # Distance that counts as "far jump"
        for day_plan in daily_itinerary:
            day_num = day_plan.get("day", "?")
            attractions = day_plan.get("attractions", [])
            if len(attractions) < 2:
                continue

            # ── 10a: Max spread check ──
            farthest_km = 0.0
            for i in range(len(attractions)):
                for j in range(i + 1, len(attractions)):
                    n1 = attractions[i].get("name", "")
                    n2 = attractions[j].get("name", "")
                    dist = None
                    if transport_matrix:
                        dist = transport_matrix.get(f"{n1}→{n2}") or transport_matrix.get(f"{n2}→{n1}")
                    if dist is None:
                        lat1 = attractions[i].get("latitude") or attractions[i].get("lat")
                        lng1 = attractions[i].get("longitude") or attractions[i].get("lng")
                        lat2 = attractions[j].get("latitude") or attractions[j].get("lat")
                        lng2 = attractions[j].get("longitude") or attractions[j].get("lng")
                        if lat1 and lng1 and lat2 and lng2:
                            dist = _haversine_distance(lat1, lng1, lat2, lng2)
                    if dist is not None and dist > farthest_km:
                        farthest_km = dist
            if farthest_km > max_spread_km:
                findings.append(
                    f"Day {day_num}: Attractions are too far apart (max distance "
                    f"{farthest_km:.1f}km). Group attractions within {max_spread_km:.0f}km "
                    f"per day to avoid excessive travel."
                )

            # ── 10b: Back-and-forth (zigzag) detection ──
            # If consecutive attractions A→B→C form a zigzag (A far from B,
            # B far from C, but A close to C), flag it.
            if len(attractions) >= 3:
                zigzags = 0
                for k in range(len(attractions) - 2):
                    def _dist(a, b):
                        n_a = a.get("name", "")
                        n_b = b.get("name", "")
                        d = None
                        if transport_matrix:
                            d = transport_matrix.get(f"{n_a}→{n_b}") or transport_matrix.get(f"{n_b}→{n_a}")
                        if d is None:
                            la, oa = a.get("latitude") or a.get("lat"), a.get("longitude") or a.get("lng")
                            lb, ob = b.get("latitude") or b.get("lat"), b.get("longitude") or b.get("lng")
                            if la and oa and lb and ob:
                                d = _haversine_distance(la, oa, lb, ob)
                        return d if d is not None else 0.0

                    d_ab = _dist(attractions[k], attractions[k + 1])
                    d_bc = _dist(attractions[k + 1], attractions[k + 2])
                    d_ac = _dist(attractions[k], attractions[k + 2])
                    # Zigzag: A↔B far, B↔C far, but A↔C close → B is an outlier
                    if d_ab > backforth_threshold_km and d_bc > backforth_threshold_km and d_ac < backforth_threshold_km:
                        zigzags += 1
                if zigzags >= 1:
                    b_name = attractions[1].get("name", "?") if zigzags == 1 else "multiple"
                    findings.append(
                        f"Day {day_num}: Back-and-forth route detected ({zigzags} zigzag(s) near '{b_name}'). "
                        f"Reorder attractions to form a smooth path, or swap out the outlier attraction."
                    )

    # ═══ Rule 11: Holistic preference compliance (LLM-driven catch-all) ═══
    # Instead of hardcoding per-dimension keyword rules for every possible
    # preference a user might express, we use a single LLM call to evaluate
    # the itinerary against ALL preference signals — structured fields AND
    # the original natural-language query (which carries rich implicit
    # preferences like "kid-friendly", "Instagram-worthy", "quiet", etc.).
    #
    # This covers dimensions NOT already checked by Rules 8-10:
    #   - Activity type vs interests (e.g. user wants museums but got parks)
    #   - Physical level appropriateness (e.g. low fitness + long walks)
    #   - Group suitability (e.g. family trip but no kid-friendly spots)
    #   - Atmosphere (e.g. wants quiet but got crowded tourist traps)
    #   - Photography/scenic value
    #   - Shopping preferences
    #   - Nightlife/entertainment
    #   - Cultural/educational depth
    #   - Nature vs urban balance
    #   - Crowd tolerance
    #   - Accessibility / mobility concerns
    #   - Language / communication ease
    #   - Budget sensitivity beyond hard limits
    #   - Any other preference hidden in the query or feedback text
    #
    # The LLM is explicitly told NOT to re-check dietary (Rule 8), sun/heat
    # (Rule 9), or distance proximity (Rule 10) — it should focus on gaps.
    # ───────────────────────────────────────────────────────────────────
    query_text = state.get("query", "")
    feedback_text = state.get("user_feedback", "")

    # Build a rich preference profile from all available signals
    pref_profile_parts = []
    # Structured prefs
    if interests:
        pref_profile_parts.append(f"Interests: {', '.join(interests)}")
    pref_profile_parts.append(f"Pacing: {user_prefs.get('pacing', 'moderate')}")
    pref_profile_parts.append(f"Physical level: {user_prefs.get('physical_level', 'moderate')}")
    dietary = user_prefs.get("dietary_preferences", [])
    if dietary:
        pref_profile_parts.append(f"Dietary: {', '.join(dietary)}")
    must_avoid_list = user_prefs.get("must_avoid", [])
    if must_avoid_list:
        pref_profile_parts.append(f"Must avoid: {', '.join(must_avoid_list)}")
    must_visit_list = user_prefs.get("must_visit", [])
    if must_visit_list:
        pref_profile_parts.append(f"Must visit: {', '.join(must_visit_list)}")
    theme = state["intent"].get("theme", "")
    if theme:
        pref_profile_parts.append(f"Theme: {theme}")
    group_type = state["intent"].get("group_type", "")
    if group_type:
        pref_profile_parts.append(f"Group: {group_type}")
    # Original query (richest signal)
    pref_profile_parts.append(f"Original query: {query_text}")
    # Feedback if any
    if feedback_text:
        pref_profile_parts.append(f"User feedback: {feedback_text}")
    pref_profile = "\n".join(pref_profile_parts)

    # Build concise itinerary summary for the LLM
    itinerary_summary_parts = []
    if daily_itinerary:
        for day_plan in daily_itinerary:
            day_num = day_plan.get("day", "?")
            attrs = [a.get("name", "?") for a in day_plan.get("attractions", [])]
            dines = [d.get("name", "?") for d in day_plan.get("dining", [])]
            hotel = day_plan.get("hotel", {})
            hotel_name = hotel.get("name", "none") if isinstance(hotel, dict) else "none"
            itinerary_summary_parts.append(
                f"Day {day_num}: attractions={attrs}, dining={dines}, hotel={hotel_name}"
            )
    else:
        poi_names = [p.get("name", "?") for p in pois if isinstance(p, dict)]
        itinerary_summary_parts.append(f"Flat POI list: {poi_names}")
    itinerary_summary = "\n".join(itinerary_summary_parts)

    # Only run LLM check if there are meaningful preferences to evaluate
    has_meaningful_prefs = bool(
        interests or dietary or must_avoid_list or must_visit_list
        or group_type or feedback_text or len(query_text) > 0
    )
    if has_meaningful_prefs and (daily_itinerary or pois):
        holistic_prompt = (
            "You are a travel quality auditor. Evaluate whether this itinerary "
            "matches ALL of the user's preferences below.\n\n"
            "=== USER PREFERENCE PROFILE ===\n"
            f"{pref_profile}\n\n"
            "=== CURRENT ITINERARY ===\n"
            f"{itinerary_summary}\n\n"
            "=== AUDIT INSTRUCTIONS ===\n"
            "Check EVERY preference dimension (interests, pacing, physical level, "
            "group suitability, atmosphere, must-visit coverage, etc.) against the "
            "itinerary. DO NOT re-check dietary/cuisine matching or sun/heat avoidance "
            "— those are already verified.\n\n"
            "Focus on gaps like:\n"
            "- Activity types don't match user interests\n"
            "- Physical demands exceed user fitness level\n"
            "- Group type mismatch (e.g. family trip but no kid-friendly spots)\n"
            "- Atmosphere mismatch (e.g. wants quiet but all crowded tourist spots)\n"
            "- Missing must-visit places\n"
            "- Any preference expressed in the query that the itinerary ignores\n"
            "- Back-and-forth routes: same-day attractions that bounce between "
            "distant areas (e.g., area A → area B → area A) waste travel time. "
            "Flag if attractions are not geographically grouped per day.\n\n"
            "Return a JSON object with:\n"
            '  "issues": [list of specific, actionable problem descriptions],\n'
            '  "score": <float 0.0-1.0 overall compliance>\n'
            "If everything is fine, return empty issues list and score 1.0."
        )
        try:
            holistic_resp = llm_client.chat(
                system_prompt="You are a precise travel quality auditor. Return only valid JSON.",
                user_prompt=holistic_prompt,
                temperature=0.0,
                max_tokens=400,
                json_format=True,
            )
            holistic_result = _safe_json_parse(holistic_resp, context="Critic/holistic")
            llm_issues = holistic_result.get("issues", [])
            if isinstance(llm_issues, list):
                for issue in llm_issues:
                    if isinstance(issue, str) and issue.strip():
                        findings.append(f"[Pref] {issue.strip()}")
            holistic_score = float(holistic_result.get("score", 1.0))
            if holistic_score < 0.5:
                logger.warning(
                    f"Holistic preference score low ({holistic_score:.2f}): "
                    f"{len(llm_issues) if isinstance(llm_issues, list) else 0} issues"
                )
        except Exception as e:
            logger.warning(f"Holistic preference LLM check failed: {e}, skipping Rule 11")

    # Prepare state update payload
    update_payload = {
        "audit_findings": findings,
    }

    # Only increment replan_count and blacklist the current plan when
    # audit actually fails. This prevents unnecessary replan cycles
    # when the itinerary is already acceptable.
    if findings:
        # Add the current POI combination to the rejection blacklist
        # so the replan agent avoids suggesting the same set again.
        bad_plan = [p["name"] for p in pois if isinstance(p, dict) and p.get("name")]
        if bad_plan:
            existing_rejected = state.get("rejected_plans", [])
            update_payload["rejected_plans"] = existing_rejected + [bad_plan]

    # Add progress logs based on audit outcome
    if findings:
        update_payload["progress_logs"] = [
            _progress(state, f"🔎 审核完成: 发现 {len(findings)} 个问题，正在重新规划...", f"🔎 Audit complete: {len(findings)} issues found, replanning..."),
        ]
    else:
        update_payload["progress_logs"] = [
            _progress(state, "✓ 行程审核通过", "✓ Itinerary passed quality audit"),
        ]

    return update_payload
