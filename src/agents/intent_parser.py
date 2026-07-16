"""IntentParser node — parses user query into structured travel intent.

This module implements the entry point of the travel planning pipeline.
It uses an LLM to extract key travel parameters (destination, duration,
budget, theme, interests, pacing, dietary preferences, physical level,
and activities to avoid) from the raw user query.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from src.state import TravelState
from src.config import EXCHANGE_RATES
from src.agents.utils import _progress, llm_client, _safe_json_parse

logger = logging.getLogger(__name__)


def intent_parser_node(state: TravelState) -> dict[str, Any]:
    """Parse the user's natural-language query into structured travel intent.

    This is the entry point of the pipeline. It uses an LLM to extract key
    travel parameters (destination, duration, budget, theme, interests, pacing,
    dietary preferences, physical level, must-visit places, and activities to
    avoid) from the raw user query stored in ``state["query"]``.

    The LLM is instructed to:
      - Output strictly valid JSON.
      - Translate all values to English (even if the query is in Chinese).
      - Apply sensible defaults for missing fields.

    API: LLM (DeepSeek Chat via ``llm_client.chat``)
        Prompt mode: ``json_format=True`` forces the model to return JSON.
        Expected JSON response::

            {
                "destination": "Tokyo",
                "duration_days": 5,
                "budget_amount": 300000,
                "budget_currency": "JPY",
                "theme": "Food & Culture",
                "interests": ["anime", "ramen", "shrines"],
                "pacing": "moderate",
                "dietary_preferences": ["Japanese cuisine"],
                "physical_level": "moderate",
                "must_avoid": [],
                "must_visit": ["Senso-ji", "Tsukiji Market"]
            }

    Args:
        state: Current ``TravelState`` dict. Must contain ``"query"`` (str).

    Returns:
        A partial state dict with keys:
        - ``intent`` (dict): Parsed travel intent.
        - ``user_preferences`` (dict): Refined user preference subset.
        - ``replan_count`` (int): Initialized to 0.
        - ``audit_findings`` (list): Empty list.
        - ``rejected_plans`` (list): Empty list.
    """
    # System prompt instructs the LLM on exactly which fields to extract,
    # the expected JSON schema, default values, and the English-only rule.
    system_prompt = (
        "You are a travel intent parser. Extract the following fields from the user query:\n"
        "1. destination (str): Travel destination city or region\n"
        "2. duration_days (int): Number of travel days\n"
        "3. budget_amount (int or float): The user's original budget amount as stated in the query "
        "(e.g., 5000 if user says '5000元' or '5000 RMB'). Do NOT convert to USD.\n"
        "4. budget_currency (str): The user's original currency code. Map currency mentions as follows:\n"
        "   - '人民币'/'元'/'RMB' → 'CNY'\n"
        "   - '美元'/'dollars' → 'USD'\n"
        "   - '欧元'/'euro'/'euros' → 'EUR'\n"
        "   - '日元'/'円'/'yen' → 'JPY'\n"
        "   - '英镑'/'pound'/'pounds' → 'GBP'\n"
        "   If the user does not explicitly specify a currency, infer from the destination:\n"
        "   - China cities (e.g., Beijing, Shanghai, Chengdu) → 'CNY'\n"
        "   - Japan cities (e.g., Tokyo, Osaka, Kyoto) → 'JPY'\n"
        "   - European cities (e.g., Paris, Rome, Berlin) → 'EUR'\n"
        "   - Other destinations → 'USD'\n"
        "5. theme (str): Trip theme (e.g. 'Food & Culture', 'Historical Exploration')\n"
        "6. interests (list[str]): Specific interest points (e.g. ['anime', 'ramen', 'shrines'])\n"
        "7. pacing (str): Trip pacing, choose from 'relaxed'/'moderate'/'intensive'\n"
        "8. dietary_preferences (list[str]): Dietary preferences (e.g. ['Japanese cuisine', 'ramen'], empty list if none)\n"
        "9. physical_level (str): Physical fitness level, choose from 'low'/'moderate'/'high'\n"
        "10. must_avoid (list[str]): Types of activities the user explicitly dislikes or wants to avoid (empty list if none)\n"
        "11. must_visit (list[str]): Specific places/attractions the user explicitly says they want to visit "
        "(e.g., 'Disneyland', 'Eiffel Tower', '故宫'). Extract ALL named places the user mentions wanting to go to. "
        "Empty list if none explicitly mentioned. These are NON-NEGOTIABLE and MUST appear in the final plan.\n"
        "12. start_date (str or null): The trip start date in YYYY-MM-DD format. "
        "Extract from user query if mentioned (e.g., 'next Monday', 'July 15', 'National Day'). "
        "If not mentioned, return null.\n"
        "Return strictly valid JSON with all fields. "
        "If a field cannot be inferred from the query, use reasonable defaults: "
        "budget_amount=1000, budget_currency='USD', pacing='moderate', physical_level='moderate', empty lists for unmentioned preferences."
        "IMPORTANT: ALL field values in your JSON output MUST be in English. "
                "Translate any non-English input to English equivalents "
                "(e.g., '北京' → 'Beijing', '美食' → 'cuisine/food', '热门景点' → 'popular attractions'). "
                "Do NOT copy Chinese characters into the output. Every string value must be pure English."
    )

    # Force LLM to output JSON format via the json_format flag
    response = llm_client.chat(
        system_prompt=system_prompt,
        user_prompt=state["query"],
        json_format=True
    )
    logger.debug(f"IntentParser LLM response: {response[:200]}...")

    # Parse the JSON string returned by the LLM into a Python dict
    intent = _safe_json_parse(response, context="IntentParser")

    # ── Budget currency handling ─────────────────────────────────
    # Preserve the original budget amount and currency from the LLM,
    # then convert to USD for internal calculations. Downstream nodes
    # use intent["budget"] (or the backward-compatible budget_usd key)
    # which always holds the USD-equivalent value.
    # Budget: use "or" to guard against LLM returning null (None),
    # since dict.get(x, default) only fires when key is absent, not when value is None.
    budget_amount = intent.get("budget_amount") or 1000
    budget_currency = intent.get("budget_currency") or "USD"

    # Convert original currency to USD using module-level EXCHANGE_RATES.
    # EXCHANGE_RATES stores "1 USD = rate * currency", so USD = amount / rate.
    from_rate = EXCHANGE_RATES.get(budget_currency, 1.0)
    budget_usd = budget_amount / from_rate

    # Store both the unified USD budget and original budget info.
    intent["budget"] = budget_usd             # Internal unified USD value
    intent["budget_usd"] = budget_usd         # Backward compat for downstream nodes
    intent["budget_original_amount"] = budget_amount       # User's original amount
    intent["budget_original_currency"] = budget_currency   # User's original currency

    logger.info(
        f"Budget parsed: {budget_amount} {budget_currency} → {budget_usd:.2f} USD "
        f"(rate: 1 USD = {from_rate} {budget_currency})")

    # Resolve start_date: use user-specified date if extracted, otherwise default to today
    start_date = intent.get("start_date")
    if not start_date:
        start_date = datetime.now().strftime("%Y-%m-%d")
    else:
        # Fix past dates: LLM may output wrong year (e.g. 2024 instead of current year)
        try:
            parsed_start = datetime.strptime(start_date, "%Y-%m-%d")
            if parsed_start.date() < datetime.now().date():
                # Bump to current or next year
                corrected = parsed_start.replace(year=datetime.now().year)
                if corrected.date() < datetime.now().date():
                    corrected = parsed_start.replace(year=datetime.now().year + 1)
                start_date = corrected.strftime("%Y-%m-%d")
                logger.info(f"Corrected past start_date to: {start_date}")
        except ValueError:
            start_date = datetime.now().strftime("%Y-%m-%d")
    intent["start_date"] = start_date

    # Derive check_in / check_out dates for hotel search from start_date + duration
    # Use "or" to guard against LLM returning null (None) for duration_days.
    duration = intent.get("duration_days") or 3
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    intent["check_in"] = start_date
    intent["check_out"] = (start_dt + timedelta(days=duration)).strftime("%Y-%m-%d")

    # Build the partial state update:
    # - "intent" holds the full parsed travel intent.
    # - "user_preferences" is a focused subset used by downstream nodes
    #   (recommendation, critic) to avoid repeatedly extracting common fields.
    # - "replan_count", "audit_findings", "rejected_plans" are initialized
    #   here so the critic/replan loop has a clean starting point.
    dest = intent.get("destination") or "Unknown"
    duration_val = intent.get("duration_days") or 3
    return {
        "intent": intent,
        "user_preferences": {
            "interests": intent.get("interests", []),
            "pacing": intent.get("pacing", "moderate"),
            "dietary_preferences": intent.get("dietary_preferences", []),
            "physical_level": intent.get("physical_level", "moderate"),
            "must_avoid": intent.get("must_avoid", []),
            "must_visit": intent.get("must_visit", [])
        },
        "replan_count": 0,
        "replan_user_approved": False,
        "audit_findings": [],
        "rejected_plans": [],
        "progress_logs": [
            _progress(state, f"✓ 已解析旅行意图: {dest}, {duration_val}天", f"✓ Travel intent parsed: {dest}, {duration_val} days"),
        ],
    }
