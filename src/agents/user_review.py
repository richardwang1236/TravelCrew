"""User Review node — pre-computes display fields for Human-in-the-Loop.

This lightweight node runs before the user sees the recommendations.
It determines the display currency, currency symbol, live exchange rate,
and whether the user prefers Chinese or English output.
"""

import logging
from typing import Any

from src.state import TravelState
from src.config import CURRENCY_SYMBOLS
from src.agents.utils import (
    _progress,
    llm_client,
    _detect_currency,
    _detect_is_chinese,
    _fetch_live_exchange_rate,
)

logger = logging.getLogger(__name__)


def user_review_node(state: TravelState) -> dict[str, Any]:
    """Pre-compute display fields for the Human-in-the-Loop review screen.

    This lightweight node runs before the user sees the recommendations.
    It determines:
      - The display currency (detected from the query).
      - The currency symbol and live exchange rate.
      - Whether the user prefers Chinese or English output.

    These fields are consumed by the synthesizer to format prices and
    choose the report language.

    Args:
        state: Current ``TravelState`` dict. Requires ``"query"`` (str).

    Returns:
        A partial state dict with keys:
        - ``display_currency`` (str): ISO currency code.
        - ``currency_symbol`` (str): Display symbol (e.g. "$").
        - ``exchange_rate`` (float): Live or cached rate.
        - ``is_chinese`` (bool): Whether query is in Chinese.
        - ``user_preferred_language`` (str): "Chinese" or "English".
    """
    query = state.get("query", "")
    # Prefer the currency captured by the IntentParser from the
    # LLM's intent parsing (which can infer from destination); fall back to
    # keyword-based detection from the raw query text.
    currency = state.get("intent", {}).get("budget_original_currency", "")
    if not currency:
        currency = _detect_currency(query)
    symbol = CURRENCY_SYMBOLS.get(currency, "$")
    rate = _fetch_live_exchange_rate(currency)
    is_chinese = _detect_is_chinese(query)

    # ── Translate POI/hotel names if user language doesn't match ──
    # When the user's language is Chinese but POI names are English (from Google API),
    # batch-translate all names so they display correctly during the review screen.
    daily_itinerary = state.get("daily_itinerary", [])
    if is_chinese and daily_itinerary:
        items_to_translate = []  # list of (dict_ref, original_name)

        for day_plan in daily_itinerary:
            # Collect attractions
            for item in day_plan.get("attractions", []):
                if isinstance(item, dict) and item.get("name"):
                    name = item["name"]
                    chinese_chars = sum(1 for c in name if '\u4e00' <= c <= '\u9fff')
                    if chinese_chars < len(name) * 0.3:
                        items_to_translate.append(item)
            # Collect dining
            for item in day_plan.get("dining", []):
                if isinstance(item, dict) and item.get("name"):
                    name = item["name"]
                    chinese_chars = sum(1 for c in name if '\u4e00' <= c <= '\u9fff')
                    if chinese_chars < len(name) * 0.3:
                        items_to_translate.append(item)
            # Collect hotel
            hotel = day_plan.get("hotel")
            if isinstance(hotel, dict) and hotel.get("name"):
                name = hotel["name"]
                chinese_chars = sum(1 for c in name if '\u4e00' <= c <= '\u9fff')
                if chinese_chars < len(name) * 0.3:
                    items_to_translate.append(hotel)

        if items_to_translate:
            # Batch translate all names via LLM (max ~50 names per call)
            names_str = "\n".join(f"- {item['name']}" for item in items_to_translate)
            translate_prompt = (
                "Translate the following place/restaurant/hotel names to Chinese. "
                "Return ONLY the translations, one per line, in the same order. "
                "For well-known landmarks, use their official Chinese names (e.g., Forbidden City=故宫, "
                "Great Wall=长城, Tokyo Tower=东京塔). "
                "For restaurants and hotels, keep brand names recognizable "
                "(e.g., Marriott=万豪, Starbucks=星巴克, McDonald's=麦当劳):\n" + names_str
            )
            try:
                translated = llm_client.chat("You are a professional place-name translator.", translate_prompt)
                translated_names = [line.strip().lstrip("- ") for line in translated.strip().split("\n") if line.strip()]
                for i, item in enumerate(items_to_translate):
                    if i < len(translated_names) and translated_names[i]:
                        item["name_translated"] = translated_names[i]
                        item["name_original"] = item["name"]
                        item["name"] = f"{translated_names[i]}（{item['name']}）"
                logger.info(f"Translated {len(items_to_translate)} POI/hotel names to Chinese for review display")
            except Exception as e:
                logger.warning(f"POI name translation failed in user_review: {e}")

    return {
        "display_currency": currency,
        "currency_symbol": symbol,
        "exchange_rate": rate,
        "is_chinese": is_chinese,
        "user_preferred_language": "Chinese" if is_chinese else "English",
        "daily_itinerary": daily_itinerary,  # Return updated itinerary with translated POI/hotel names
        "progress_logs": [
            _progress(state, "✓ 行程预览已准备完成", "✓ Itinerary preview ready for review"),
        ],
    }
