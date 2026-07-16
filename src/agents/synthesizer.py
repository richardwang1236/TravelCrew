"""Synthesizer node — generates the final Markdown travel report.

This is the last node in the pipeline. It fetches images and static maps
for confirmed POIs (deferred from Routing to avoid wasted API calls during
replan), then assembles all gathered data into a rich context prompt and
asks the LLM to produce a beautifully formatted 9-section Markdown report.
"""

import json
import logging
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from src.state import TravelState
from src.agents.utils import (
    _progress,
    llm_client,
    _detect_currency,
    _fetch_live_exchange_rate,
    _get_transport_cost_per_min,
)

logger = logging.getLogger(__name__)


def synth_enrich_node(state: TravelState) -> dict[str, Any]:
    """Enrich daily itinerary with images, maps, AI descriptions, and translations.

    This node fetches enrichment data (images, static maps, AI descriptions)
    and translates POI names before the final report generation. Splitting
    this from the Synthesizer provides real-time progress to the frontend.

    Args:
        state: Current ``TravelState`` dict.

    Returns:
        - ``daily_itinerary`` (list[dict]): Enriched with image_url, static_map_url,
          website, maps_url, wikipedia_intro.
        - ``progress_logs`` (list[dict]): Sanitized progress messages.
    """
    intent = state["intent"]
    pois = state["recommended_pois"]
    metrics = state["routing_metrics"]
    weather = state["raw_knowledge"]["weather"]
    transport = state.get("transport_matrix", {})
    original_query = state.get("query", "")

    daily_itinerary = state.get("daily_itinerary", [])

    # ── Step 1: Restore website/maps_url from raw_knowledge.pois ──
    # Images and static maps are now fetched here (moved from Routing node)
    # to avoid wasted API calls during Critic-triggered replan cycles.
    raw_pois = state.get("raw_knowledge", {}).get("pois", [])
    website_map: dict[str, str] = {}
    maps_url_map: dict[str, str] = {}
    for poi in raw_pois:
        name = poi.get("name", "")
        if not name:
            continue
        website = poi.get("website", "")
        if website:
            website_map[name] = website
        maps_url = poi.get("maps_url", "")
        if maps_url:
            maps_url_map[name] = maps_url

    # ── Step 1.1: Fetch images for all POIs (moved from Routing node) ──
    # Collect all POI and hotel names from the final daily_itinerary.
    poi_names: list[str] = []
    for day_plan in daily_itinerary:
        for item in day_plan.get("attractions", []) + day_plan.get("dining", []):
            name = item.get("name", "")
            if name:
                poi_names.append(name)
        hotel = day_plan.get("hotel", {})
        if isinstance(hotel, dict) and hotel.get("name"):
            poi_names.append(hotel["name"])

    images: dict[str, str] = {}
    static_maps: dict[str, str] = {}
    if poi_names:
        destination = intent.get("destination", "")
        from src.api import fetch_images
        from src.api.static_map import generate_static_maps
        from src.config import SERPER_API_KEY, GOOGLE_MAPS_API_KEY
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # ── Prepare map coordinates before parallel execution ──────
        poi_data_for_maps = []
        for day in daily_itinerary:
            for item in day.get("attractions", []) + day.get("dining", []):
                lat = item.get("lat")
                lng = item.get("lng")
                name = item.get("name", "")
                if name and lat and lng:
                    try:
                        lat_f, lng_f = float(lat), float(lng)
                        if lat_f != 0.0 or lng_f != 0.0:
                            poi_data_for_maps.append({"name": name, "lat": lat_f, "lng": lng_f})
                    except (ValueError, TypeError):
                        pass
            hotel = day.get("hotel", {})
            if isinstance(hotel, dict):
                lat = hotel.get("lat")
                lng = hotel.get("lng")
                name = hotel.get("name", "")
                if name and lat and lng:
                    try:
                        lat_f, lng_f = float(lat), float(lng)
                        if lat_f != 0.0 or lng_f != 0.0:
                            poi_data_for_maps.append({"name": name, "lat": lat_f, "lng": lng_f})
                    except (ValueError, TypeError):
                        pass

        # ── Run images + maps in parallel to reduce wait time ──────
        logger.info(f"Fetching images + static maps in parallel for {len(poi_names)} POIs...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_images = executor.submit(fetch_images, poi_names, SERPER_API_KEY, destination=destination)
            future_maps = executor.submit(generate_static_maps, poi_data_for_maps, api_key=GOOGLE_MAPS_API_KEY)
            for future in as_completed([future_images, future_maps]):
                try:
                    result = future.result()
                    if future == future_images:
                        images = result
                        logger.info(f"Images fetched: {len(images)}/{len(poi_names)}")
                    else:
                        static_maps = result
                        logger.info(f"Static maps generated: {len(static_maps)}/{len(poi_data_for_maps)}")
                except Exception as e:
                    logger.warning(f"Parallel enrichment task failed: {e}")

    # ── Step 1.3: Inject images, static maps, website, maps_url into daily_itinerary ──
    if daily_itinerary:
        injected_count = 0
        website_injected = 0
        maps_injected = 0
        for day_plan in daily_itinerary:
            for attraction in day_plan.get("attractions", []):
                name = attraction.get("name", "")
                lookup_name = attraction.get("name_original", name)
                if not attraction.get("image_url") and (name in images or lookup_name in images):
                    attraction["image_url"] = images.get(name, images.get(lookup_name, ""))
                    injected_count += 1
                if not attraction.get("website") and (name in website_map or lookup_name in website_map):
                    attraction["website"] = website_map.get(name, website_map.get(lookup_name, ""))
                    website_injected += 1
                if not attraction.get("maps_url") and (name in maps_url_map or lookup_name in maps_url_map):
                    attraction["maps_url"] = maps_url_map.get(name, maps_url_map.get(lookup_name, ""))
                    maps_injected += 1
                if not attraction.get("static_map_url") and (name in static_maps or lookup_name in static_maps):
                    attraction["static_map_url"] = static_maps.get(name, static_maps.get(lookup_name, ""))
            for dining in day_plan.get("dining", []):
                name = dining.get("name", "")
                lookup_name = dining.get("name_original", name)
                if not dining.get("image_url") and (name in images or lookup_name in images):
                    dining["image_url"] = images.get(name, images.get(lookup_name, ""))
                if not dining.get("website") and (name in website_map or lookup_name in website_map):
                    dining["website"] = website_map.get(name, website_map.get(lookup_name, ""))
                    website_injected += 1
                if not dining.get("maps_url") and (name in maps_url_map or lookup_name in maps_url_map):
                    dining["maps_url"] = maps_url_map.get(name, maps_url_map.get(lookup_name, ""))
                    maps_injected += 1
                if not dining.get("static_map_url") and (name in static_maps or lookup_name in static_maps):
                    dining["static_map_url"] = static_maps.get(name, static_maps.get(lookup_name, ""))
            hotel = day_plan.get("hotel")
            if isinstance(hotel, dict):
                name = hotel.get("name", "")
                if not hotel.get("image_url") and name in images:
                    hotel["image_url"] = images[name]
                if not hotel.get("website") and name in website_map:
                    hotel["website"] = website_map[name]
                    website_injected += 1
                if not hotel.get("maps_url") and name in maps_url_map:
                    hotel["maps_url"] = maps_url_map[name]
                    maps_injected += 1
                if not hotel.get("static_map_url") and name in static_maps:
                    hotel["static_map_url"] = static_maps[name]
        logger.info(
            f"Injected {injected_count} images, {len(static_maps)} static maps, "
            f"{website_injected} websites, {maps_injected} maps_url into daily itinerary"
        )

    # ── Step 1.2: Fetch AI descriptions for POIs missing wikipedia_intro ──
    # Wikipedia API may not find articles for every POI (especially
    # translated names, small attractions, restaurants). For any POI
    # without a 'wikipedia_intro', use Google AI Mode (via SerpApi) to
    # fetch a short AI-generated description so EVERY POI has an intro.
    if daily_itinerary:
        missing_names: list[str] = []
        missing_items: list[tuple[dict, str]] = []  # (item_ref, name)
        for day_plan in daily_itinerary:
            for item in day_plan.get("attractions", []) + day_plan.get("dining", []):
                if not isinstance(item, dict):
                    continue
                if not item.get("wikipedia_intro"):
                    name = item.get("name", "")
                    if name:
                        missing_names.append(name)
                        missing_items.append((item, name))

        if missing_items:
            logger.info(f"Fetching AI descriptions for {len(missing_names)} POIs missing Wikipedia data (parallel)...")
            import re as _re
            query = state.get("query", "")
            is_chinese = bool(_re.search(r'[\u4e00-\u9fff]', query))
            ai_lang = "zh" if is_chinese else "en"
            destination = state.get("intent", {}).get("destination", "")
            from src.api import fetch_place_descriptions
            ai_intros = fetch_place_descriptions(
                missing_names, destination=destination, language=ai_lang
            )
            for item, name in missing_items:
                intro = ai_intros.get(name)
                if intro:
                    item["wikipedia_intro"] = intro
            logger.info(f"AI descriptions fetched for {len(ai_intros)}/{len(missing_names)} POIs")

    # ── Step 1.5: Translate POI/hotel names if language mismatch ──
    # If the user's language is Chinese but POI names are English (from API),
    # use LLM to batch-translate all names and store both versions.
    # This is a fallback for names not already translated by user_review_node.
    is_chinese = state.get("is_chinese", False)
    if is_chinese and daily_itinerary:
        items_to_translate = []  # list of dict refs to translate
        for day_plan in daily_itinerary:
            # Collect attractions
            for item in day_plan.get("attractions", []):
                if isinstance(item, dict) and item.get("name"):
                    # Skip if already translated by user_review_node
                    if item.get("name_translated"):
                        continue
                    name = item["name"]
                    chinese_chars = sum(1 for c in name if '\u4e00' <= c <= '\u9fff')
                    if chinese_chars < len(name) * 0.3:
                        items_to_translate.append(item)
            # Collect dining
            for item in day_plan.get("dining", []):
                if isinstance(item, dict) and item.get("name"):
                    if item.get("name_translated"):
                        continue
                    name = item["name"]
                    chinese_chars = sum(1 for c in name if '\u4e00' <= c <= '\u9fff')
                    if chinese_chars < len(name) * 0.3:
                        items_to_translate.append(item)
            # Collect hotel
            hotel = day_plan.get("hotel")
            if isinstance(hotel, dict) and hotel.get("name"):
                if hotel.get("name_translated"):
                    continue
                name = hotel["name"]
                chinese_chars = sum(1 for c in name if '\u4e00' <= c <= '\u9fff')
                if chinese_chars < len(name) * 0.3:
                    items_to_translate.append(hotel)

        if items_to_translate:
            # Batch translate all names via LLM
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
                        # Update name to show both: "中文名 (English Name)"
                        item["name"] = f"{translated_names[i]}（{item['name']}）"
                logger.info(f"Translated {len(items_to_translate)} POI/hotel names to Chinese")
            except Exception as e:
                logger.warning(f"POI name translation failed: {e}")

    # ── Collect progress messages ────────────────────────────────
    progress_msgs = [
        _progress(state, "🖼️ 正在加载景点图片和地图...", "🖼️ Loading POI images and maps..."),
    ]
    if images:
        progress_msgs.append(_progress(state, f"📸 已获取 {len(images)} 张图片", f"📸 Fetched {len(images)} images"))
    if static_maps:
        progress_msgs.append(_progress(state, f"🗺️ 已生成 {len(static_maps)} 张地图", f"🗺️ Generated {len(static_maps)} static maps"))
    progress_msgs.append(_progress(state, "✓ 数据丰富完成，开始生成报告...", "✓ Enrichment complete, generating report..."))

    return {
        "daily_itinerary": daily_itinerary,
        "progress_logs": progress_msgs,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Synthesizer node — generates the final Markdown report via LLM
# ═══════════════════════════════════════════════════════════════════════════


def synthesizer_node(state: TravelState, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
    """Generate the final Markdown travel report using an LLM.

    Reads the enriched daily itinerary from state and assembles all data
    into a comprehensive prompt for the LLM. Budget, hotel data, and
    currency conversions are computed here for accuracy.

    API: LLM (DeepSeek Chat via ``llm_client.chat``)
        Prompt mode: ``temperature=0.5``, ``max_tokens=8000``.

    Args:
        state: Current ``TravelState`` dict with enriched daily_itinerary.

    Returns:
        A partial state dict with key:
        - ``final_itinerary`` (str): The complete Markdown travel report.
    """
    intent = state["intent"]
    pois = state["recommended_pois"]
    metrics = state["routing_metrics"]
    weather = state["raw_knowledge"]["weather"]
    transport = state.get("transport_matrix", {})
    original_query = state.get("query", "")
    daily_itinerary = state.get("daily_itinerary", [])

    # ── Resolve output language ─────────────────────────────────
    user_lang = state.get("user_preferred_language", "")
    if not user_lang:
        user_lang = "Chinese" if state.get("is_chinese", False) else "English"

    # ── Localized UI labels (report link text) ───────────────────
    # The LLM prompt uses these as explicit output format examples;
    # they MUST match the report language so English reports don't
    # show Chinese link labels ("在地图中查看" in an English report).
    is_zh = (user_lang == "Chinese")
    LBL_OFFICIAL_SITE = "官方网站" if is_zh else "Official Website"
    LBL_VIEW_ON_MAP = "在地图中查看" if is_zh else "View on Map"
    LBL_MAP_ALT = "地图" if is_zh else "Map"

    system_prompt = (
        "You are an experienced and eloquent travel expert, skilled at crafting vivid, "
        "informative, and beautifully structured travel itineraries.\n\n"

        "[OUTPUT FORMAT RULES]\n"
        "- Start the report DIRECTLY with the title heading (# ...).\n"
        "- Do NOT include any introductory text, greetings, self-introduction, or conversational filler before the report.\n"
        "- Do NOT say things like \"Sure!\", \"Here's your plan!\", \"As your travel expert...\" etc.\n"
        "- The output must be pure Markdown report content starting with the first heading.\n\n"

        f"[LANGUAGE RULE - CRITICAL]\n"
        f"Write the ENTIRE report in {user_lang}. "
        f"All section titles, descriptions, tips, and narrative text must be in {user_lang}. "
        f"Do NOT mix languages. Do NOT output any content in a language other than {user_lang}.\n\n"

        "[FORMATTING & AESTHETICS]\n"
        "- Use emoji icons as section markers (e.g., 🗺️ for overview, 🏨 for hotels, 🍜 for food, 📍 for attractions)\n"
        "- Use tables for structured data (budget breakdown, daily costs, hotel comparison)\n"
        "- Use horizontal rules (---) between major sections\n"
        "- Use blockquotes (>) for tips and important notes\n"
        "- Use bullet points with consistent indentation\n"
        "- Include a brief highlight summary box at the top (key stats: days, budget, destinations)\n"
        "- For each day's itinerary, use a clear timeline format with time markers (🌅 Morning / ☀️ Afternoon / 🌙 Evening)\n\n"

        "[IMAGE EMBEDDING RULE]\n"
        "If a POI or hotel's data contains a non-null image_url field, embed the image using:\n"
        "![Name](image_url)\n"
        "CRITICAL: Place each image IMMEDIATELY after its corresponding POI/hotel entry — NOT grouped separately.\n"
        "For the Accommodation section, each hotel row in the table should be followed by its image on the next line.\n"
        "Do NOT create a separate 'images' subsection — images must be inline with their respective entries.\n\n"

        "[WEBSITE & MAP RULE]\n"
        "For each attraction/dining POI, check its data for these fields and include them:\n"
        "- If 'website' field is non-empty: add a clickable link line:  🔗 ["
        f"{LBL_OFFICIAL_SITE}](website_url)\n"
        "- If 'maps_url' field is non-empty: add a clickable link:  📍 ["
        f"{LBL_VIEW_ON_MAP}](maps_url)\n"
        "- If lat and lng are available (but no maps_url): generate a link as:\n"
        "  📍 ["
        f"{LBL_VIEW_ON_MAP}](https://www.openstreetmap.org/?mlat={{lat}}&mlon={{lng}}#map=15/{{lat}}/{{lng}})\n"
        "- STATIC MAP IMAGE: If the POI's data contains a 'static_map_url' field (non-empty), embed it as:\n"
        "  !["
        f"{LBL_MAP_ALT}](static_map_url)\n"
        "  This is a locally cached map image — always include it if available. It loads instantly.\n"
        "- Do NOT generate any iframe or embed.html URLs — only use the static_map_url field if available.\n"
        "Place the website link and map image IMMEDIATELY after the POI's description/highlights.\n\n"

        "[WIKIPEDIA INTRO RULE]\n"
        "If a POI's data contains a non-empty 'wikipedia_intro' field, include it as the POI's background description.\n"
        "Format it as a blockquote (>) after the POI name, BEFORE the highlights section.\n"
        "Do NOT fabricate background info if wikipedia_intro is not available — use the 'description' field instead.\n\n"

        "[BUDGET DATA RULE]\n"
        "Use ONLY the pre-computed budget data provided below. Do NOT invent your own numbers. "
        "Use the EXACT currency unit specified in the budget data. "
        "If 'original_budget_amount' and 'original_budget_currency' are present in the budget data, "
        "also display the user's original budget (e.g., 'Original budget: 5000 CNY') in the budget section.\n\n"

        "[DATA SOURCE RULE]\n"
        "Strictly base content on provided data; do NOT fabricate attractions, prices, or image links. "
        "For sections 7-9 where no structured data exists, generate reasonable suggestions based on "
        "destination knowledge, but clearly mark them as recommendations.\n\n"

        "[STRICT ITINERARY FIDELITY RULE — CRITICAL]\n"
        "The daily_itinerary below is the CONFIRMED plan the user has already reviewed and approved. "
        "In Section 4 (Detailed Daily Itinerary), you MUST:\n"
        "- Include ONLY the attractions and dining entries that exist in the daily_itinerary data.\n"
        "- Do NOT add any attractions, restaurants, night markets, viewpoints, or activities that "
        "are NOT in the daily_itinerary.\n"
        "- Do NOT invent 'evening suggestions' like night views, night markets, or bars unless they "
        "are explicitly listed in that day's dining or attractions.\n"
        "- The 'Evening suggestion' at the end of each day should be GENERIC advice only "
        "(e.g., 'rest early for tomorrow', 'take a walk near the hotel') — do NOT name specific "
        "POIs or venues that are not in the daily_itinerary.\n"
        "- In Sections 6 and 8, any suggested alternatives MUST also come from the daily_itinerary "
        "or be clearly labeled as 'optional recommendations not in your confirmed plan'.\n\n"

        "[NO DUPLICATE POI RULE]\n"
        "Each attraction and dining spot must appear ONLY ONCE in the entire report. "
        "Do NOT repeat a POI on multiple days. Do NOT add notes like 'same as Day 1' or "
        "'revisit' — every POI entry must be unique across all days.\n\n"

        "[ANTI-HALLUCINATION RULES]\n"
        "- ONLY include prices/costs that are explicitly provided in the input data (daily_itinerary, budget_data, hotel data).\n"
        "- Do NOT invent or guess specific prices for attractions, transport, food items, or any other services.\n"
        "- If a specific price is not available in the provided data, either omit it or write \"see official website for latest pricing\" (or equivalent in output language).\n"
        "- Do NOT fabricate historical facts, opening hours, or operational details that are not in the provided data.\n"
        "- For transport suggestions (bus routes, ferry, subway), do NOT include specific fare amounts unless provided in the data.\n"
        "- When describing POIs, only use information from the provided POI data fields (name, rating, cost, opening_hours, description).\n\n"

        "Generate a Markdown report with the following 9 sections IN ORDER:\n\n"

        "## Section 1: Trip Overview & Essentials\n"
        "Extract from intent data: trip title (e.g. '{Destination} {Days}-Day ... Trip'), "
        "traveler info (group size, type, budget tier), travel dates (start/end/total days/timezone), "
        "transport plan (inter-city + local), and budget summary table "
        "(transport/accommodation/dining/tickets/shopping/emergency breakdown).\n\n"

        "## Section 2: Pre-Trip Preparation\n"
        "Extract from wikivoyage_context (or supplement with your knowledge if unavailable): "
        "packing checklist (documents/clothing/medicine/chargers), local policies & precautions "
        "(visa/prohibited items/reservation rules), best travel tips (best season/off-peak times/"
        "rainy-day alternatives/photo spots).\n\n"

        "## Section 3: Accommodation Plan\n"
        "Use ONLY the pre-computed hotel data provided below. "
        "For EACH hotel, display as an individual card/block:\n"
        "- Hotel name (translated to output language if original name is in a different language), area, rating, per-night price, recommendation reason\n"
        "- If hotel has image_url, embed the image IMMEDIATELY below that hotel's info (not grouped separately)\n"
        "Do NOT use a single comparison table followed by grouped images. Each hotel should be a self-contained block with its own image.\n"
        "[CRITICAL] Use the EXACT per-night price from the pre-computed hotel data. Do NOT modify, round, or estimate hotel prices.\n\n"

        "## Section 4: Detailed Daily Itinerary (Core Section)\n"
        "For each day from daily_itinerary, generate an HOUR-BY-HOUR schedule:\n"
        "- Day theme title (from day_theme field)\n"
        "- ⚠️ Today's Tips: practical reminders from day_tips (booking deadlines, what to bring, etc.)\n"
        "- 🚫 Avoid Today: list tourist traps to skip from 'avoid' field, with brief reason\n"
        "- Use EXACT start_time from the data to build a precise timeline. Format each entry as:\n"
        "  **HH:MM - HH:MM** | 📍 POI Name\n"
        "  > 🚇 Transport: How to get here from previous stop (transit_time_min minutes)\n"
        "  > ⏱️ Duration: X hours Y minutes\n"
        "  > 💰 Cost: ticket price\n"
        "  > 💡 Why here: from 'why' field — explain what makes it special or better than alternatives\n"
        "  > 📖 Background: from 'wikipedia_intro' field if available — include the Wikipedia intro text as the POI's background description\n"
        "  > ✨ Highlights: what to see/do, tips\n"
        "  > 🔗 ["
        f"{LBL_OFFICIAL_SITE}](website) — include if website field is non-empty\n"
        "  > 📍 ["
        f"{LBL_VIEW_ON_MAP}](maps_url) — include if maps_url is non-empty or use lat/lng to generate OpenStreetMap link\n"
        "  > 🗺️ !["
        f"{LBL_MAP_ALT}](static_map_url) — include if static_map_url is non-empty (locally cached map image)\n"
        "  > 🖼️ [image if available]\n\n"
        "- For dining entries, additionally include:\n"
        "  > 🍽️ Must-order: specific dish names from signature_dishes field\n"
        "  > 📍 Why here: explain proximity to which attraction (from 'near' field)\n"
        "  > 💡 Insider tip: ordering advice, peak hours to avoid, etc.\n\n"
        "- Between activities, show transit segments clearly:\n"
        "  **12:30 - 12:45** | 🚇 Transit: Metro Line X from Station A to Station B (~15 min)\n\n"
        "- End each day with:\n"
        "  > 💰 Daily cost subtotal\n"
        "  > 💡 Evening reminder: generic rest advice only (e.g., 'rest early', 'stretch your legs near the hotel'). "
        "Do NOT name any specific venue, night market, or attraction here.\n"
        "  > 🏨 Return to hotel: transport method & estimated time\n\n"

        "## Section 5: Food & Dining Guide\n"
        "Aggregate ONLY the dining entries that appear in daily_itinerary into a categorized food guide. "
        "Do NOT add restaurants or food items that are not in the daily_itinerary:\n"
        "- 🍖 Signature Main Meals: list the must-try main restaurants with signature_dishes and per-person cost\n"
        "- 🥟 Local Snacks & Street Food: specific snack items from the daily_itinerary dining entries, where to find them, price\n"
        "- ☕ Breakfast/Tea: morning food recommendations from existing dining data only\n"
        "- 💡 Dining Tips: peak hours to avoid, reservation advice, local eating customs\n"
        "Format as a quick-reference table with: Restaurant | Signature Dish | Per-Person Cost | Which Day.\n\n"

        "## Section 6: Supplementary Experiences (Optional)\n"
        "List additional experiences ONLY from the daily_itinerary data (e.g., optional activities "
        "at existing POIs). Do NOT add new attractions or venues not in the confirmed plan. "
        "If no supplementary data exists, briefly describe the type of experiences available "
        "at the destination without naming specific venues.\n\n"

        "## Section 7: Safety & Avoidance Guide\n"
        "Based on 'avoid' fields from each day + general destination knowledge:\n"
        "- 🚫 Tourist Traps to Avoid: specific places/streets and WHY (overpriced, fake, crowded)\n"
        "- ⚠️ Common Scams: destination-specific scams and how to avoid them\n"
        "- 🌡️ Clothing & Weather: temperature range, what to wear each day, rain gear?\n"
        "- 📦 What to Pack: day bag essentials, charging, comfortable shoes, etc.\n"
        "- 🏥 Emergency: useful phone numbers, nearest hospital, embassy if international\n\n"

        "## Section 8: Itinerary Alternatives\n"
        "Provide three variants based on rearranging the EXISTING daily_itinerary POIs: "
        "time-tight condensed version (skip some POIs), low-energy relaxed version (fewer POIs per day), "
        "rainy-day backup route (swap outdoor POIs for indoor ones from the same pool). "
        "Do NOT introduce new attractions or venues not already in the daily_itinerary.\n\n"

        "## Section 9: Trip Summary\n"
        "Summarize: full-trip highlights, one-day quick preview, overall travel advice.\n\n"

        "Writing style: vivid, approachable — like an experienced friend sharing recommendations. "
        "Keep scheduling realistic with transit and meal times accounted for."
    )

    # ── Step 3: Build rich context with accurate budget breakdown ──
    # Pre-compute a per-day cost breakdown from actual POI data so the
    # LLM uses real numbers in the report's budget section (anti-hallucination).
    duration = intent.get("duration_days", 3)

    # Use the original currency captured by the IntentParser (preferred),
    # falling back to query-based detection if the IntentParser didn't set it.
    currency = intent.get("budget_original_currency", "")
    if not currency:
        currency = _detect_currency(original_query)

    # The IntentParser already converted the budget to USD.
    # intent["budget"] holds the USD value; intent["budget_usd"] is backward-compat.
    budget_usd = intent.get("budget", intent.get("budget_usd", 1000))

    # Fetch live exchange rate (falls back to hardcoded rates if API fails)
    rate = _fetch_live_exchange_rate(currency)

    # Progress messages (collected and emitted at node completion)
    progress_msgs = [
        _progress(state, f"💰 计算每日预算明细 ({duration}天)...", f"💰 Computing daily budget ({duration} days)..."),
    ]

    # Round-robin distribute POIs across days as a fallback for budget calc
    # when daily_itinerary is unavailable.
    pois_per_day = [[] for _ in range(duration)]
    for i, poi in enumerate(pois):
        pois_per_day[i % duration].append(poi)

    daily_budget = []
    for day_idx in range(duration):
        # Prefer the structured daily_itinerary; fall back to round-robin distribution
        if daily_itinerary and day_idx < len(daily_itinerary):
            day_plan = daily_itinerary[day_idx]
            day_attractions = day_plan.get("attractions", [])
            day_dining = day_plan.get("dining", [])
            day_hotel = day_plan.get("hotel", {})
            if not isinstance(day_hotel, dict):
                day_hotel = {}
        else:
            day_attractions = [p for p in pois_per_day[day_idx] if p.get("type") not in ("dining", "restaurant", "food")]
            day_dining = [p for p in pois_per_day[day_idx] if p.get("type") in ("dining", "restaurant", "food")]
            day_hotel = {}

        # All costs computed in USD (POI costs from Google Places are in USD).
        # Dining fallback: $25 * 3 meals when no dining POIs are available.
        tickets_usd = sum(p.get("cost", 0) for p in day_attractions)
        dining_usd = sum(p.get("cost", 0) for p in day_dining) if day_dining else 25 * 3

        # Transit cost: use real transport matrix when available, else $8/day baseline.
        # Cost = transit_minutes * destination-specific cost_per_minute (mixed transit).
        # Cap at $25/day to prevent unrealistic accumulation on packed days.
        transit_usd = 8  # Base transit estimate per day in USD
        day_pois = day_attractions + day_dining
        if transport and len(day_pois) > 1:
            day_transit_mins = sum(
                transport.get(day_pois[j].get("name", ""), {}).get(day_pois[j+1].get("name", ""), 20)
                for j in range(len(day_pois)-1)
            )
            # Calculate real transit cost from transport matrix + per-minute rate
            destination = intent.get("destination", "")
            cost_per_min = _get_transport_cost_per_min(destination)
            transit_usd = max(8, min(day_transit_mins * cost_per_min, 25))

        # Hotel cost for this day (in USD); 0 if no hotel assigned
        hotel_usd = day_hotel.get("price_per_night") or 0

        # Accumulate daily costs (all in USD for accurate total)
        daily_budget.append({
            "day": day_idx + 1,
            "tickets_usd": tickets_usd,
            "dining_usd": dining_usd,
            "transit_usd": transit_usd,
            "hotel_usd": hotel_usd,
        })

    # Sum all daily costs in USD first (avoids currency mismatch)
    total_spent_usd = sum(
        d["tickets_usd"] + d["dining_usd"] + d["transit_usd"] + d["hotel_usd"]
        for d in daily_budget
    )
    hotel_total_cost_usd = sum(d["hotel_usd"] for d in daily_budget)

    # Calculate remaining budget in USD (same currency as budget_usd)
    remaining_usd = budget_usd - total_spent_usd

    # Convert daily breakdown to user's preferred currency for display.
    # All values multiplied by the live exchange rate.
    daily_breakdown = []
    for d in daily_budget:
        tickets = round(d["tickets_usd"] * rate, 1)
        dining = round(d["dining_usd"] * rate, 1)
        transit = round(d["transit_usd"] * rate, 1)
        accommodation = round(d["hotel_usd"] * rate, 1)
        daily_breakdown.append({
            "day": d["day"],
            "tickets": tickets,
            "dining": dining,
            "transit": transit,
            "accommodation": accommodation,
            "subtotal": round(tickets + dining + transit + accommodation, 1)
        })

    # Build pre-computed hotel summary with authoritative prices.
    # This prevents the LLM from fabricating hotel prices in Section 3.
    # Deduplicate hotels by name and record which days each hotel covers.
    hotel_summary = []
    seen_hotels = {}
    for day_idx in range(duration):
        if daily_itinerary and day_idx < len(daily_itinerary):
            day_plan = daily_itinerary[day_idx]
            h = day_plan.get("hotel", {})
            if not isinstance(h, dict) or not h.get("name"):
                continue
            hname = h["name"]
            if hname in seen_hotels:
                seen_hotels[hname]["days"].append(day_idx + 1)
            else:
                entry = {
                    "name": hname,
                    "price_per_night_usd": h.get("price_per_night", 0),
                    "price_per_night_display": round((h.get("price_per_night", 0) or 0) * rate, 1),
                    "currency": currency,
                    "rating": h.get("rating"),
                    "image_url": h.get("image_url"),
                    "description": h.get("description"),
                    "amenities": h.get("amenities", []),
                    "lat": h.get("lat"),
                    "lng": h.get("lng"),
                    "days": [day_idx + 1],
                }
                seen_hotels[hname] = entry
                hotel_summary.append(entry)
    hotel_data = {
        "hotels": hotel_summary,
        "total_accommodation_display": round(hotel_total_cost_usd * rate, 1),
        "currency": currency,
        "exchange_rate_note": f"1 USD = {rate} {currency}" if currency != "USD" else "N/A",
    }

    progress_msgs.append(
        _progress(state, "🏨 已汇总酒店数据", f"🏨 Hotel data aggregated ({len(hotel_summary)} hotels)")
    )

    # Package all budget data for the LLM prompt.
    # Display values are in user's currency; exchange rate note explains conversion.
    budget_data = {
        "currency": currency,
        "exchange_rate_note": f"1 USD = {rate} {currency}" if currency != "USD" else "N/A",
        "total_budget": round(budget_usd * rate, 1),
        "original_budget_amount": intent.get("budget_original_amount", round(budget_usd * rate, 1)),
        "original_budget_currency": intent.get("budget_original_currency", currency),
        "daily_breakdown": daily_breakdown,
        "accommodation_total": round(hotel_total_cost_usd * rate, 1),
        "total_estimated_spend": round(total_spent_usd * rate, 1),
        "remaining": round(remaining_usd * rate, 1)
    }

    # ── Step 3.5: Convert POI costs in daily_itinerary from USD → user currency ──
    # The pre-computed budget_data above is already in user currency, but the
    # individual POI cost fields inside daily_itinerary are still in USD (set by
    # _convert_place_to_poi with _PRICE_LEVEL_TO_COST mapping).  If the LLM sees
    # budget_data in CNY and POI costs in USD, it presents the USD number as CNY,
    # producing absurdly low prices (e.g. "7.2 CNY" for a full meal).
    #
    # Convert every cost field in-place so the LLM prompt is self-consistent.
    if daily_itinerary:
        for day in daily_itinerary:
            for item in day.get("attractions", []) + day.get("dining", []):
                if isinstance(item, dict) and isinstance(item.get("cost"), (int, float)):
                    item["cost"] = round(item["cost"] * rate, 1)
            hotel = day.get("hotel")
            if isinstance(hotel, dict) and isinstance(hotel.get("price_per_night"), (int, float)):
                hotel["price_per_night"] = round(hotel["price_per_night"] * rate, 1)

    # ── Step 4: Build the comprehensive user prompt ────────────────
    # Assemble all data sources into a single prompt for the LLM.
    # Each section is labeled with emoji markers for clarity.
    user_prompt = f"📍 User's Original Query:\n\"{original_query}\"\n\n"
    user_prompt += f"📍 Destination Info:\n{json.dumps(intent, ensure_ascii=False, indent=2)}\n\n"
    user_prompt += f"🌤️ Weather Data:\n{json.dumps(weather, ensure_ascii=False, indent=2)}\n\n"

    # Prefer the structured daily itinerary (contains day-by-day plans)
    # over the flat POI list for accurate daily reporting.
    if daily_itinerary:
        # ── Strip internal fields to keep prompt within LLM context window ──
        # For multi-city / long trips (e.g. 10-day Italy), the full daily_itinerary
        # JSON with lat/lng/tags/metadata can exceed 50K characters, blowing past
        # the LLM's effective context and causing empty responses.
        # Keep only the fields the LLM actually needs for report generation.
        _LLM_KEEP_FIELDS = {
            "name", "type", "cost", "rating", "start_time",
            "avg_visit_time_min", "transit_time_min", "suggested_transport",
            "why", "signature_dishes", "near", "day_theme", "day_tips", "avoid",
            "image_url", "website", "maps_url", "static_map_url",
            "wikipedia_intro", "description",
            # Hotel fields
            "price_per_night", "total_price", "amenities", "currency",
            "day",  # top-level per-day
        }
        _DESC_MAX = 200  # chars — truncate long descriptions

        def _strip_poi(item: dict) -> dict:
            out = {}
            for k in _LLM_KEEP_FIELDS:
                if k in item:
                    val = item[k]
                    # Truncate long text fields
                    if k in ("description", "wikipedia_intro") and isinstance(val, str) and len(val) > _DESC_MAX:
                        val = val[:_DESC_MAX] + "…"
                    out[k] = val
            return out

        stripped_itinerary = []
        for day in daily_itinerary:
            d = {"day": day.get("day", 0)}
            if "day_theme" in day:
                d["day_theme"] = day["day_theme"]
            if "day_tips" in day:
                d["day_tips"] = day["day_tips"]
            if "avoid" in day:
                d["avoid"] = day["avoid"]
            d["attractions"] = [_strip_poi(a) for a in day.get("attractions", []) if isinstance(a, dict)]
            d["dining"] = [_strip_poi(a) for a in day.get("dining", []) if isinstance(a, dict)]
            hotel = day.get("hotel")
            d["hotel"] = _strip_poi(hotel) if isinstance(hotel, dict) else (hotel or None)
            stripped_itinerary.append(d)

        itinerary_data = json.dumps(stripped_itinerary, ensure_ascii=False, indent=2)
        if len(itinerary_data) > 30000:
            logger.warning(
                f"Stripped itinerary still {len(itinerary_data)} chars — "
                f"further truncation may be needed for very long trips"
            )
        user_prompt += (
            f"[DAILY ITINERARY - structured daily plan, use this as the PRIMARY data source]\n"
            f"{itinerary_data}\n\n"
            f"Each day contains:\n"
            f"- attractions: 3-5 sightseeing spots\n"
            f"- dining: 2 restaurant/food recommendations\n"
            f"- hotel: 1 accommodation\n\n"
            f"Generate the report following this daily structure exactly.\n\n"
        )
    else:
        # Backward compatibility: fall back to flat POI list if no daily structure
        user_prompt += f"⭐ Recommended POIs (in route order, includes image_url for image embedding):\n{json.dumps(pois, ensure_ascii=False, indent=2)}\n\n"

    # Inject route metrics and pre-computed budget data.
    # The budget data instruction explicitly tells the LLM to use these exact
    # numbers, preventing it from fabricating costs.
    user_prompt += f"🚇 Route & Metrics:\n{json.dumps(metrics, ensure_ascii=False, indent=2)}\n\n"
    user_prompt += f"💰 Pre-Computed Budget Data (USE THESE EXACT NUMBERS in Budget Breakdown section):\n{json.dumps(budget_data, ensure_ascii=False, indent=2)}\n\n"
    user_prompt += f"🏨 Pre-Computed Hotel Data (USE THESE EXACT PRICES in Accommodation Plan section — do NOT invent prices):\n{json.dumps(hotel_data, ensure_ascii=False, indent=2)}\n\n"

    # Include transit time matrix if available (for detailed transport info)
    if transport:
        user_prompt += f"🗺️ Transit Time Matrix (minutes):\n{json.dumps(transport, ensure_ascii=False, indent=2)}\n\n"

    # Inject Wikivoyage travel tips (truncated to 200 chars per category
    # to keep the prompt within the LLM's context window).
    wikivoyage = state.get("raw_knowledge", {}).get("wikivoyage", {})
    if wikivoyage:
        tips_section = "[WIKIVOYAGE TRAVEL TIPS - include these in the report's travel tips section]:\n"
        if wikivoyage.get("transport_tips"):
            tips_section += f"- Transport: {wikivoyage['transport_tips'][:200]}\n"
        if wikivoyage.get("local_customs"):
            tips_section += f"- Local Customs: {wikivoyage['local_customs'][:200]}\n"
        if wikivoyage.get("safety_tips"):
            tips_section += f"- Safety: {wikivoyage['safety_tips'][:200]}\n"
        if wikivoyage.get("best_seasons"):
            tips_section += f"- Best Seasons: {wikivoyage['best_seasons'][:200]}\n"
        if wikivoyage.get("food_tips"):
            tips_section += f"- Food Guide: {wikivoyage['food_tips'][:200]}\n"
        user_prompt += tips_section + "\n"

    # Inject high-severity weather alerts (level >= 3/5) so the LLM
    # highlights them in the safety and important notes sections.
    weather_alerts = weather.get("weather_alerts", [])
    high_severity_alerts = [a for a in weather_alerts if a.get("alert_level", 0) >= 3]
    if high_severity_alerts:
        alert_summaries = [f"[{a.get('alert_type','?')}] {a.get('headline','')} (level {a.get('alert_level',0)}/5)" for a in high_severity_alerts]
        user_prompt += "⚠️ Weather Alerts (MUST be highlighted in Travel Tips and Important Notes):\n"
        for a in alert_summaries:
            user_prompt += f"- {a}\n"
        user_prompt += "\n"
    
    # Inject any remaining audit warnings (from the last critic pass)
    # so the LLM addresses them in the report's notes section.
    if state.get("audit_findings"):
        user_prompt += f"⚠️ Audit Warnings (address these in the Important Notes section):\n{state['audit_findings']}\n\n"
    else:
        user_prompt += "✅ This itinerary has passed all quality checks.\n\n"

    # -- Final LLM call: generate the complete Markdown travel report --
    # temperature=0.5 balances creativity with factual grounding.
    # max_tokens=16000 allows for lengthy multi-day itineraries without truncation.
    logger.info("Calling LLM to generate final report (streaming mode)...")
    progress_msgs.append(
        _progress(state, "📝 正在调用 AI 生成最终报告...", "📝 Calling AI to generate final report...")
    )

    # Try to stream tokens to the frontend in real-time via the SSE context.
    # The push_event helper is thread-safe (uses loop.call_soon_threadsafe).

    # Extract session_id from LangGraph config (thread_id == session_id)
    # so push_event can route events to the correct session's queue.
    session_id = None
    if config:
        session_id = config.get("configurable", {}).get("thread_id")

    try:
        from server.streaming import push_event
        logger.info(f"Streaming push_event available, session_id={session_id}")
    except ImportError as e:
        logger.warning(f"Cannot import push_event: {e}")
        push_event = None

    accumulated = []
    char_count = 0
    chunk_batch = ""  # Buffer to batch small chunks before pushing

    try:
        for chunk in llm_client.chat_stream(
            system_prompt, user_prompt,
            temperature=0.5, max_tokens=32000
        ):
            accumulated.append(chunk)
            char_count += len(chunk)
            chunk_batch += chunk

            # Push to frontend every ~20 chars or on newline for smooth rendering
            if push_event and (len(chunk_batch) >= 20 or '\n' in chunk):
                push_event({
                    "type": "markdown_chunk",
                    "chunk": chunk_batch,
                    "total_chars": char_count,
                }, session_id=session_id)
                if char_count <= 100:
                    logger.info(f"Streaming: pushed {len(chunk_batch)} chars (total {char_count})")
                chunk_batch = ""

        # Flush any remaining buffered text
        if push_event and chunk_batch:
            push_event({
                "type": "markdown_chunk",
                "chunk": chunk_batch,
                "total_chars": char_count,
            }, session_id=session_id)
            logger.info(f"Streaming: flushed final {len(chunk_batch)} chars")

        final_report = "".join(accumulated)
        logger.info(f"Streaming report generation complete: {char_count} chars, {len(accumulated)} chunks")

    except Exception as e:
        logger.warning(f"Streaming LLM call failed ({e}), falling back to synchronous mode")
        final_report = llm_client.chat(
            system_prompt, user_prompt,
            temperature=0.5, max_tokens=32000
        )

    logger.debug(f"Synthesizer LLM response: {final_report[:200]}...")
    progress_msgs.append(
        _progress(state, "✓ 报告生成完成", "✓ Report generation complete")
    )
    return {
        "final_itinerary": final_report,
        "progress_logs": progress_msgs,
    }
