"""LangGraph workflow definition for the travel planning agent.

This module builds and compiles the stateful directed graph that orchestrates
every agent node in the pipeline.  It wires together:

    1. **IntentParser** — parses user intent (destination, dates, budget,
       preferences) from the natural-language query.
    2. **Information** — fetches external data (POIs, weather, images,
       Wikivoyage descriptions) for the parsed destination.
    3. **Recommendation** — scores and ranks attractions/dining/hotels
       against the user's preferences and budget.
    4. **UserReview** — pre-computes display fields (localized labels,
       currency-converted costs) so the CLI can render a rich review
       screen before the graph pauses at the Human-in-the-Loop interrupt.
    5. **Routing** — builds a day-by-day route matrix and selects the
       optimal travel strategy.
    6. **Critic** — audits the plan for theme coverage, budget adherence,
       and quality issues; may trigger a replan.
    7. **Synthesizer** — assembles the final Markdown itinerary report.

Graph flow (happy path)::

    IntentParser → Information → Recommendation → Routing → Critic
        ├─ (approve)        → UserReview → [INTERRUPT] → SynthEnrich → Synthesizer
        ├─ (force_approve)  → UserReview → [INTERRUPT] → SynthEnrich → Synthesizer
        └─ (replan, <3)    → AutoReplan → UserReview → [INTERRUPT] → SynthEnrich
                             → Synthesizer

AutoReplan uses the full ReplanAgent tool-use loop (not a single LLM call)
to autonomously fix critic-identified issues. After the agent finishes, the
revised plan flows through UserReview (pre-computing display data) and pauses
at the SynthEnrich interrupt for Human-in-the-Loop review — the user always
gets a chance to inspect and approve changes.

Conditional edges (Critic router)::

    - "approve"        → UserReview  (no issues found)
    - "force_approve"  → UserReview  (max replan attempts / user-approved)
    - "replan"         → AutoReplan  (issues found, < MAX_REPLAN_ATTEMPTS;
                                       replan agent fixes → UserReview →
                                       [INTERRUPT] → SynthEnrich → Synthesizer)

The graph is compiled with ``interrupt_before=["SynthEnrich"]`` so that the
Human-in-the-Loop pause happens *after* UserReview (which prepares display
data) but *before* the final enrichment/synthesis phase. Once the user
confirms, no more nodes can modify the itinerary.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.state import TravelState
from src.agents import (
    intent_parser_node,
    information_node,
    recommendation_node,
    user_review_node,
    routing_and_strategy_node,
    critic_node,
    synth_enrich_node,
    synthesizer_node,
    auto_replan_node,
)
from src.config import MAX_REPLAN_ATTEMPTS

# ──────────────────────────────────────────────
# 1. Initialize the StateGraph with our typed state schema.
#    Every node receives and returns a TravelState dict.
# ──────────────────────────────────────────────
workflow = StateGraph(TravelState)

# ──────────────────────────────────────────────
# 2. Register agent nodes.
#    Each name maps to a callable defined in src.agents.
# ──────────────────────────────────────────────
workflow.add_node("IntentParser", intent_parser_node)
workflow.add_node("Information", information_node)
workflow.add_node("Recommendation", recommendation_node)
workflow.add_node("UserReview", user_review_node)
workflow.add_node("Routing", routing_and_strategy_node)
workflow.add_node("Critic", critic_node)
workflow.add_node("SynthEnrich", synth_enrich_node)
workflow.add_node("Synthesizer", synthesizer_node)
workflow.add_node("AutoReplan", auto_replan_node)

# ──────────────────────────────────────────────
# 3. Define conditional-edge logic for the Critic node.
#
#    The Critic audits the recommendation against the user's intent.
#    Based on its findings it returns one of three verdicts:
#      - "approve":        No issues → proceed to UserReview (HITL).
#      - "force_approve":  Max replan attempts reached / user already
#                           approved → proceed to UserReview anyway.
#      - "replan":         Issues detected → loop back to Recommendation
#                           (then Routing, Critic again).
# ──────────────────────────────────────────────
def critic_router(state: TravelState):
    """Determine the next node after the Critic audit.

    Args:
        state: The current travel-planning state.

    Returns:
        str: One of ``"approve"``, ``"force_approve"``, or ``"replan"``.
    """
    if not state.get("audit_findings"):
        return "approve"  # No issues; proceed to HITL review
    # If the user already reviewed and approved a replan agent's changes,
    # force-approve — do NOT let the Critic trigger a full regeneration
    # that would discard the replan agent's work.
    if state.get("replan_user_approved"):
        return "force_approve"
    if state["replan_count"] >= MAX_REPLAN_ATTEMPTS:
        return "force_approve"  # Max retries exceeded
    return "replan"  # Issues found; invoke AutoReplan (ReplanAgent tool-use loop)

# ──────────────────────────────────────────────
# 4. Build edges — connect nodes into a directed graph.
# ──────────────────────────────────────────────

# The IntentParser is always the first node to execute.
workflow.set_entry_point("IntentParser")

# Linear pipeline: parse intent → fetch data → rank candidates.
workflow.add_edge("IntentParser", "Information")
workflow.add_edge("Information", "Recommendation")

# Critic-triggered replan: Critic → AutoReplan (ReplanAgent tool-use)
# → UserReview → [INTERRUPT SynthEnrich] → Synthesizer.
# AutoReplan calls the full replan agent to fix issues, then the revised
# plan goes through UserReview for display prep before the HITL pause —
# the user always gets to review changes before final synthesis.
workflow.add_edge("Recommendation", "Routing")
workflow.add_edge("Routing", "Critic")

# Critic conditional branches — see critic_router() above.
workflow.add_conditional_edges(
    "Critic",
    critic_router,
    {
        "approve": "UserReview",        # Clean audit → HITL review
        "force_approve": "UserReview",  # Max retries / user-approved → HITL review
        "replan": "AutoReplan"          # Issues found → ReplanAgent fixes
    }
)

# AutoReplan: after replan agent fixes, route through UserReview so the
# user can review the changes before finalizing.
workflow.add_edge("AutoReplan", "UserReview")

# After UserReview, the graph pauses for Human-in-the-Loop feedback.
# Once the user confirms, enrich data and generate the final report.
workflow.add_edge("UserReview", "SynthEnrich")

# After enrichment, generate the final report.
workflow.add_edge("SynthEnrich", "Synthesizer")

# Synthesizer is the terminal node; after it finishes the graph ends.
workflow.add_edge("Synthesizer", END)

# ──────────────────────────────────────────────
# 5. Compile graph with checkpointer and interrupt configuration.
#
#    - MemorySaver: in-memory checkpoint store that persists state
#      between steps and enables resume after interrupt.
#    - interrupt_before=["SynthEnrich"]: the graph pauses *after*
#      UserReview (which prepares display data) and *before* SynthEnrich
#      /Synthesizer. Once the user confirms, no nodes that modify the
#      itinerary remain.
# ──────────────────────────────────────────────
memory = MemorySaver()
travel_app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["SynthEnrich"]
)
