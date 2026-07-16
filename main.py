"""Entry point and CLI driver for the travel planning agent.

This module wires together the LangGraph workflow (``graph.travel_app``)
with a command-line user interface that supports:

    1. **Session initialization** — generates a unique thread ID so that
       LangGraph's checkpointer can track conversation state across
       interrupt/resume cycles.
    2. **Phase 1: Auto-execution** — streams the graph from the initial
       user query through IntentParser → Information → Recommendation →
       Routing → Critic.  The Critic either approves the plan or triggers
       AutoReplan (ReplanAgent tool-use loop) to fix issues.  In both
       cases, the graph flows through UserReview and pauses at the
       SynthEnrich interrupt for Human-in-the-Loop review.
    3. **Human-in-the-Loop review** — displays a rich, localized itinerary
       review screen and collects user feedback.  Supports *multi-round*
       feedback: each round runs the full ReplanAgent tool-use loop, then
       re-streams through Routing → Critic → UserReview, pausing at
       SynthEnrich again.
    4. **Phase 2: Resume & finalize** — resumes the graph past the
       SynthEnrich interrupt through Synthesizer → END.  No itinerary-
       modifying nodes remain after this point.
    5. **Report output** — retrieves the final Markdown itinerary from the
       graph state and saves it to ``output/<destination>_<timestamp>.md``.

Key functions:
    - :func:`print_user_review` — renders the pre-computed review screen.
    - :func:`main` — orchestrates the entire CLI flow.

Usage::

    python main.py
"""

import os
import re
import uuid
import logging
from datetime import datetime
from src.graph import travel_app

# Configure root logger so that all agent-node and system messages appear
# in the console with timestamps.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def print_user_review(state_values: dict):
    """Render a rich, localized itinerary-review screen for the user.

    Reads pre-computed display fields (set by the UserReview node) from
    the graph state and prints a formatted day-by-day breakdown of
    attractions, dining, and hotels with currency-converted costs.

    The function supports both Chinese and English output, selected by
    the ``is_chinese`` flag in state.

    Args:
        state_values: The current graph state dictionary.  Expected keys:
            - ``is_chinese`` (bool): Whether to render in Chinese.
            - ``currency_symbol`` (str): e.g. ``"$"``, ``"\u00a5"``.
            - ``exchange_rate`` (float): USD → local-currency rate.
            - ``daily_itinerary`` (list[dict]): Day-by-day plans.
            - ``recommended_pois`` (list): Fallback flat POI list.

    Returns:
        str: The localized input prompt to display after the review
        screen, asking the user for feedback.
    """
    # ── Read pre-computed display fields from state ──
    is_chinese = state_values.get("is_chinese", False)
    symbol = state_values.get("currency_symbol", "$")
    rate = state_values.get("exchange_rate", 1.0)
    daily_itinerary = state_values.get("daily_itinerary", [])
    recommended_pois = state_values.get("recommended_pois", [])

    # ── Localized labels ──
    if is_chinese:
        title = "💡 [行程审核] AI 为您推荐了以下行程："
        day_label = "📅 第 {n} 天:"
        attraction_label = "🏛️ 景点"
        dining_label = "🍽️ 美食"
        hotel_label = "🏨 酒店"
        cost_fmt = "预计花费: {cost}"
        hotel_cost_fmt = "{price}/晚, 评分: {rating}★"
        total_fmt = "💰 预计总花费: {total}"
        feedback_prompt = "👉 请输入您的反馈（直接回车表示满意，无需修改）：\n> "
    else:
        title = "💡 [Itinerary Review] AI recommends the following itinerary:"
        day_label = "📅 Day {n}:"
        attraction_label = "🏛️ Attraction"
        dining_label = "🍽️ Dining"
        hotel_label = "🏨 Hotel"
        cost_fmt = "est. cost: {cost}"
        hotel_cost_fmt = "{price}/night, rating: {rating}★"
        total_fmt = "💰 Estimated total: {total}"
        feedback_prompt = "👉 Enter your feedback (press Enter if satisfied, no changes needed):\n> "

    def _fmt_cost(usd_cost):
        """Convert a USD-denominated cost to the user's local currency.

        Applies the exchange rate from state and prepends the currency
        symbol.  ``None`` or missing costs are treated as 0.

        Args:
            usd_cost: Cost value in US dollars (may be None).

        Returns:
            str: Formatted cost string, e.g. ``"¥3,500"``.
        """
        if usd_cost is None:
            usd_cost = 0
        converted = round(usd_cost * rate)
        return f"{symbol}{converted:,}"

    # ── Build output ──
    # Print a visual separator and the review title.
    print("\n" + "=" * 50)
    print(title)

    # Accumulate total cost across all days/items (in USD, converted at the end).
    total_cost = 0

    if daily_itinerary:
        # ── Primary display mode: day-by-day structured itinerary ──
        # Each day contains separate lists for attractions, dining, and hotel.
        for day_plan in daily_itinerary:
            day_num = day_plan.get("day", "?")
            print(f"\n{day_label.format(n=day_num)}")

            # Print attractions for this day with per-item estimated cost.
            for poi in day_plan.get("attractions", []):
                cost_usd = poi.get("cost", 0) or 0
                total_cost += cost_usd
                print(f"  {attraction_label}: {poi.get('name', '?')} ({cost_fmt.format(cost=_fmt_cost(cost_usd))})")

            # Print dining options for this day with per-item estimated cost.
            for poi in day_plan.get("dining", []):
                cost_usd = poi.get("cost", 0) or 0
                total_cost += cost_usd
                print(f"  {dining_label}: {poi.get('name', '?')} ({cost_fmt.format(cost=_fmt_cost(cost_usd))})")

            # Print hotel recommendation for this day (if available).
            # Shows per-night price and star rating from Google Places data.
            hotel = day_plan.get("hotel")
            if isinstance(hotel, dict) and hotel.get("name"):
                price = hotel.get("price_per_night", 0) or 0
                rating = hotel.get("rating", "N/A")
                total_cost += price
                hotel_detail = hotel_cost_fmt.format(price=_fmt_cost(price), rating=rating)
                print(f"  {hotel_label}: {hotel['name']} ({hotel_detail})")
    else:
        # ── Fallback display: flat POI list (no daily structure) ──
        # Used when the Recommendation node returned a simple list without
        # day-by-day grouping (e.g. single-day trips or early-stage results).
        for i, poi in enumerate(recommended_pois):
            if isinstance(poi, dict):
                cost_usd = poi.get("cost", 0) or 0
                total_cost += cost_usd
                print(f"  {i+1}. {poi.get('name', '?')} ({cost_fmt.format(cost=_fmt_cost(cost_usd))})")
            else:
                print(f"  {i+1}. {poi}")

    # ── Print total estimated cost ──
    # Convert the accumulated USD total to the user's local currency.
    total_converted = round(total_cost * rate)
    print(f"\n{total_fmt.format(total=f'{symbol}{total_converted:,}')}")
    print("=" * 50 + "\n")

    return feedback_prompt


def main():
    """Run the full travel-planning CLI interaction loop.

    Orchestrates the entire lifecycle of a single travel-planning session:

        1. **Session init** — generates a UUID-based thread ID.
        2. **Phase 1 (auto-execute)** — streams the graph from the initial
           query through IntentParser → Information → Recommendation →
           Routing → Critic.  The Critic either approves or triggers
           AutoReplan (ReplanAgent) to fix issues.  In both cases the graph
           flows through UserReview and pauses at SynthEnrich for HITL.
        3. **Human-in-the-Loop (multi-round)** — enters a feedback loop:
           a) Renders the itinerary review screen via :func:`print_user_review`.
           b) Collects free-text user feedback.
           c) Runs the autonomous ReplanAgent tool-use loop to revise.
           d) Injects the modified itinerary and re-streams through
              Routing → Critic → UserReview → [interrupt again].
           e) Breaks when user presses Enter (empty input = approval).
        4. **Phase 2 (resume & finalize)** — resumes past SynthEnrich
           through Synthesizer → END.  No itinerary-modifying nodes remain.
        5. **Report output** — retrieves the final Markdown itinerary and
           saves it to ``output/<destination>_<timestamp>.md``.

    This function does not return a value; all output is printed to stdout
    and saved to disk.

    Raises:
        Exception: Any unhandled exception from the graph or file I/O
        will propagate to the caller.
    """
    # ==========================================
    # 1. Initialize session (configure Thread ID)
    # ==========================================
    # thread_id is how LangGraph tracks the current conversation state.
    # A new UUID ensures every planning task starts with a clean slate.
    session_id = str(uuid.uuid4())
    thread_config = {"configurable": {"thread_id": session_id}}

    print(f"🚀 [System] Created new travel planning task, session ID: {session_id}\n")

    # User's initial input — a Chinese-language query specifying:
    #   - Destination: Shanghai (上海)
    #   - Duration: 3 days
    #   - Budget: 5000 RMB
    #   - Preferences: relaxed pace, good food, popular attractions
    # This string is passed to the IntentParser node for intent parsing.
    initial_input = {
        "query": "我明天要去武汉玩3天, 预算 5000 人民币。不喜欢太累，想去吃点好的，要去热门的景点。"
    }

    # ==========================================
    # 2. Phase 1: Auto-execute until interrupt
    # ==========================================
    # The graph runs through IntentParser → Information → Recommendation
    # → Routing → Critic.  If the Critic finds issues, AutoReplan
    # (ReplanAgent) fixes them, then flows through UserReview.  In ALL
    # cases, the graph pauses at SynthEnrich for HITL review.
    print("⏳ [System] Parsing intent, fetching data, and generating initial recommendations...")

    for event in travel_app.stream(initial_input, config=thread_config):
        for node_name, node_state in event.items():
            print(f"✅ [{node_name} node completed]")

            if node_name == "Critic" and node_state.get("audit_findings"):
                print(f"   ⚠️ Audit warning: {node_state['audit_findings']}")
            if node_name == "AutoReplan":
                print(f"   🔧 AutoReplan agent fixed {len(node_state.get('audit_findings', '')) or 'reported'} issue(s)")

    # ==========================================
    # 3. Human-in-the-Loop — supports multi-round feedback
    # ==========================================
    # The graph pauses at SynthEnrich (interrupt_before=["SynthEnrich"])
    # after UserReview has pre-computed display data — regardless of
    # whether the Critic approved or AutoReplan fixed issues.
    #
    # In each loop iteration we:
    #   a) Render the review screen and read user input.
    #   b) If feedback is non-empty: run ReplanAgent, inject result,
    #      and re-stream (pausing at SynthEnrich again).
    #   c) If feedback is empty: user is satisfied → break.

    while True:
        current_state = travel_app.get_state(thread_config)

        # Only proceed if the graph is paused at the SynthEnrich interrupt.
        if not (current_state.next and current_state.next[0] == "SynthEnrich"):
            break  # Not at the interrupt point; exit loop

        # Print rich user review using pre-computed display fields
        feedback_prompt = print_user_review(current_state.values)

        user_feedback = input(feedback_prompt)

        # Non-empty input → user wants changes; process feedback.
        if user_feedback.strip():
            print("\n🔄 Feedback received, running autonomous replan agent...")

            # ── Run autonomous replan agent ─────────────────────────
            # The agent inspects the plan, browses POIs, searches for new
            # places, and directly modifies the itinerary until satisfied.
            from src.agents.replan_agent import run_replan_agent
            from src.agents.utils import llm_client as _llm_client

            is_chinese = bool(re.search(r'[\u4e00-\u9fff]', current_state.values.get("query", "")))
            api_language = "zh-CN" if is_chinese else "en"

            agent_result = run_replan_agent(
                feedback_text=user_feedback,
                state_values=current_state.values,
                llm_client=_llm_client,
                api_language=api_language,
                max_iterations=20,
            )

            modified_itinerary = agent_result["daily_itinerary"]
            updated_prefs = agent_result["user_preferences"]
            new_pois = agent_result["new_pois"]
            agent_summary = agent_result["summary"]
            iterations = agent_result["iterations"]

            logging.info(
                f"ReplanAgent completed: {iterations} iterations, "
                f"{len(new_pois)} new POIs, summary: {agent_summary[:100]}"
            )
            print(f"🤖 Agent completed in {iterations} iteration(s): {agent_summary[:120]}...")

            # ── Build flat recommended_pois from modified itinerary ──
            all_recommended = []
            for day in modified_itinerary:
                all_recommended.extend(day.get("attractions", []))
                all_recommended.extend(day.get("dining", []))

            # ── Merge new POIs into the raw_knowledge pool ──────────
            raw_knowledge = dict(current_state.values.get("raw_knowledge", {}))
            if new_pois:
                existing_pois = raw_knowledge.get("pois", [])
                existing_names = {p.get("name", "").lower() for p in existing_pois}
                deduped_new = [p for p in new_pois if p.get("name", "").lower() not in existing_names]
                if deduped_new:
                    raw_knowledge["pois"] = existing_pois + deduped_new
                    print(f"📍 Agent discovered {len(deduped_new)} new place(s): "
                          f"{', '.join(p.get('name', '?') for p in deduped_new)}")

            # ── Update state directly (bypass Recommendation → Routing) ──
            # Since the agent already produced the final itinerary, we update
            # as if Recommendation just completed. The graph will then flow
            # through Routing (recalculate transport), Critic (force-approves
            # due to replan_user_approved), and UserReview before pausing at
            # the SynthEnrich interrupt for the next round of HITL review.
            # Set replan_user_approved=True so Critic will force-approve and
            # NOT overwrite the replan agent's work with a fresh generation.
            update_values = {
                "daily_itinerary": modified_itinerary,
                "recommended_pois": all_recommended,
                "user_preferences": updated_prefs,
                "user_feedback": agent_summary,
                "raw_knowledge": raw_knowledge,
                "replan_user_approved": True,
            }

            travel_app.update_state(
                config=thread_config,
                values=update_values,
                as_node="Recommendation"
            )
            # Re-stream from the current checkpoint; the graph will flow
            # through Routing → Critic → UserReview and interrupt again
            # before SynthEnrich, allowing another round of HITL feedback.
            for event in travel_app.stream(None, config=thread_config):
                for node_name, node_state in event.items():
                    print(f"✅ [{node_name} node completed]")
        else:
            # Empty input → user is satisfied with the current plan.
            break  # Exit the feedback loop

    # ==========================================
    # 4. Phase 2: Resume from interrupt → generate final report
    # ==========================================
    # After the user approves, resume from SynthEnrich through
    # Synthesizer → END.  No itinerary-modifying nodes remain.
    print("\n🚀 [Confirmed] Generating final itinerary report...")

    while True:
        for event in travel_app.stream(None, config=thread_config):
            for node_name, node_state in event.items():
                print(f"✅ [{node_name} node completed]")

        # After streaming, check if the graph has truly terminated.
        current_state = travel_app.get_state(thread_config)
        if not current_state.next:
            break  # Graph has ended; all nodes complete

    # ==========================================
    # 5. Retrieve and display the final itinerary report
    # ==========================================
    final_state = travel_app.get_state(thread_config)
    final_report = final_state.values.get("final_itinerary", "Generation failed")

    print("\n" + "🎉 "*20)
    print("👇 [Final Custom Itinerary] 👇")
    print(final_report)

    # ==========================================
    # 6. Persist the itinerary as a local Markdown file
    # ==========================================
    # Saves to the ``output/`` directory using the pattern:
    #   output/<destination>_<YYYYMMDD_HHMMSS>.md
    if final_report and final_report != "Generation failed":
        try:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
            os.makedirs(output_dir, exist_ok=True)

            dest = final_state.values.get("intent", {}).get("destination", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"{dest}_{timestamp}.md")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_report)

            print(f"\n✅ Report saved to: {output_path}")
        except Exception as e:
            print(f"\n⚠️ Failed to save report: {e}")

if __name__ == "__main__":
    main()
