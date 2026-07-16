"""AutoReplan node — Critic-triggered autonomous replan via ReplanAgent.

When the Critic finds issues with the generated itinerary, this node calls
the full ``run_replan_agent`` tool-use loop (not a single LLM call) to
autonomously fix the problems.  After the agent finishes, the modified
itinerary is injected into the graph state and flows through UserReview
→ [INTERRUPT SynthEnrich] so the user can review the changes before
final synthesis.
"""

import logging
import re

from src.agents.replan_agent import run_replan_agent
from src.agents.utils import llm_client as _llm_client

logger = logging.getLogger(__name__)


def auto_replan_node(state: dict) -> dict:
    """Critic-triggered autonomous replan using the ReplanAgent tool-use loop.

    Called when the Critic router returns ``"replan"``.  Reads the current
    itinerary and audit findings, runs the replan agent to fix the issues,
    and returns updated state values.  After this node, the graph flows
    through UserReview → [INTERRUPT SynthEnrich] for HITL review.

    Args:
        state: Current graph state.

    Returns:
        dict: Partial state update with keys:
            - ``daily_itinerary``: Modified day-by-day plan.
            - ``recommended_pois``: Rebuilt flat POI list.
            - ``user_preferences``: Updated preferences.
            - ``user_feedback``: Human-readable summary of changes.
            - ``raw_knowledge``: Updated POI pool (new POIs merged).
            - ``replan_count``: Incremented count.
            - ``replan_user_approved``: ``True`` to prevent Critic from
              overriding the agent's work.
    """
    feedback_text = state.get("user_feedback", "")
    audit = state.get("audit_findings", "")
    if audit:
        feedback_text = f"Critic audit found these issues:\n{audit}\n\n" + feedback_text
    feedback_text = feedback_text or "Improve itinerary quality and ensure all constraints are met."

    # Detect query language for Google Places API locale
    query = state.get("query", "")
    is_chinese = bool(re.search(r"[\u4e00-\u9fff]", query))
    api_language = "zh-CN" if is_chinese else "en"

    current_replan_count = state.get("replan_count", 0)

    logger.info(
        f"AutoReplan starting (replan_count={current_replan_count}, "
        f"audit issues: {str(audit)[:150]})"
    )

    # ── Run autonomous replan agent ──────────────────────────────────
    agent_result = run_replan_agent(
        feedback_text=feedback_text,
        state_values=state,
        llm_client=_llm_client,
        api_language=api_language,
        max_iterations=20,  # Full workflow: inspect(1) + search(≤5) + modify(4-8) + verify(1) + finalize(1)
    )

    modified_itinerary = agent_result["daily_itinerary"]
    updated_prefs = agent_result["user_preferences"]
    new_pois = agent_result["new_pois"]
    agent_summary = agent_result["summary"]
    iterations = agent_result["iterations"]

    logger.info(
        f"AutoReplan completed: {iterations} iterations, "
        f"{len(new_pois)} new POIs, summary: {agent_summary[:100]}"
    )

    # ── Build flat recommended_pois from modified itinerary ──────────
    all_recommended = []
    for day in modified_itinerary:
        all_recommended.extend(day.get("attractions", []))
        all_recommended.extend(day.get("dining", []))

    # ── Merge new POIs into the raw_knowledge pool ───────────────────
    raw_knowledge = dict(state.get("raw_knowledge", {}))
    if new_pois:
        existing_pois = raw_knowledge.get("pois", [])
        existing_names = {p.get("name", "").lower() for p in existing_pois}
        for poi in new_pois:
            if poi.get("name", "").lower() not in existing_names:
                existing_pois.append(poi)
        raw_knowledge["pois"] = existing_pois

    return {
        "daily_itinerary": modified_itinerary,
        "recommended_pois": all_recommended,
        "user_preferences": updated_prefs,
        "user_feedback": agent_summary,
        "raw_knowledge": raw_knowledge,
        "replan_count": current_replan_count + 1,
        "replan_user_approved": True,
    }
