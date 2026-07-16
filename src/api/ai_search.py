"""src.api.ai_search — SerpApi Google AI Mode for must-visit place discovery.

Uses Google's AI Mode (via SerpApi) to ask "what are the must-visit places
in {destination}?" and extract a structured list of attraction names.
The returned names are then used as input for targeted Google Places
Text Search lookups, replacing the previous approach of blindly searching
by type around a coordinate.

Also provides :func:`fetch_place_descriptions` — a fallback for when
Wikipedia has no article for a POI. Queries Google AI Mode for each
place name and returns a short AI-generated description.

API: SerpApi Google AI Mode
    Endpoint: https://serpapi.com/search
    Engine: google_ai_mode+
    Docs: https://serpapi.com/google-ai-mode-api

Public Functions:
    fetch_must_visit_places(destination, language) -> list[str]
    fetch_place_descriptions(place_names, destination, language) -> dict[str, str]
"""

import re
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from src.api.base import logger, API_TIMEOUT, SERPAPI_KEY, _get_session


def fetch_must_visit_places(
    destination: str,
    api_key: Optional[str] = None,
    language: str = "en",
    interests: str = "",
    count: int = 20,
) -> list[str]:
    """Query Google AI Mode for must-visit attractions in a destination.

    Sends a natural-language query like "What are the top must-visit
    attractions and landmarks in {destination}?" to SerpApi's
    ``google_ai_mode`` engine. Parses the AI-generated response to
    extract a clean list of place names.

    Args:
        destination (str): Destination city or region (e.g. "Wuhan", "Tokyo").
        api_key (Optional[str]): SerpApi key. If None, uses the globally
            configured SERPAPI_KEY.
        language (str): Language code for results (``"en"`` or ``"zh-CN"``).
        interests (str): Optional user interests string to refine the query
            (e.g. "history, food, nature").

    Returns:
        list[str]: List of attraction/place names extracted from the AI
        response (e.g. ``["Yellow Crane Tower", "East Lake", ...]``).
        Returns an empty list if the API call fails or no names can be
        extracted.

    Raises:
        No exceptions raised; all errors are caught and logged.
    """
    key = api_key or SERPAPI_KEY
    if not key:
        logger.info("SerpApi Key is empty, skipping AI Mode search")
        return []

    # Build the query based on language and optional interests.
    # Explicitly request a numbered list format so the regex parser can
    # reliably extract place names. Ask for a comprehensive list.
    # User's original query (interests) is integrated into the main question
    # so the AI tailors recommendations to the user's actual needs.
    if language.startswith("zh"):
        if interests:
            query = (
                f"用户想要：{interests}\n\n"
                f"请根据以上需求，列出{destination}最值得去的{count}个景点、地标、博物馆和美食地点。"
                f"推荐应与用户的需求和偏好相关。"
                f"请用编号列表格式回答，每行一个地点名称，格式如：\n"
                f"1. 地点名称\n2. 地点名称\n...\n"
                f"请确保列出至少{count}个不同的具体地点名称。"
            )
        else:
            query = (
                f"请列出{destination}最值得去的{count}个景点、地标、博物馆和美食地点。"
                f"请用编号列表格式回答，每行一个地点名称，格式如：\n"
                f"1. 地点名称\n2. 地点名称\n...\n"
                f"请确保列出至少{count}个不同的具体地点名称。"
            )
    else:
        if interests:
            query = (
                f"User request: {interests}\n\n"
                f"Based on the above request, list the top {count} must-visit attractions, "
                f"landmarks, museums, and famous food spots in {destination}. "
                f"Recommendations should be relevant to the user's needs and preferences. "
                f"Please format your answer as a numbered list, one place per line, like:\n"
                f"1. Place Name\n2. Place Name\n...\n"
                f"Make sure to list at least {count} different specific place names."
            )
        else:
            query = (
                f"List the top {count} must-visit attractions, landmarks, museums, "
                f"and famous food spots in {destination}. "
                f"Please format your answer as a numbered list, one place per line, like:\n"
                f"1. Place Name\n2. Place Name\n...\n"
                f"Make sure to list at least {count} different specific place names."
            )

    try:
        # SerpApi Google AI Mode API
        # API 名称: SerpApi - Google AI Mode Engine
        # Endpoint: https://serpapi.com/search
        # Engine: google_ai_mode
        # Request parameters:
        #   engine: "google_ai_mode" (required)
        #   q: str (search query)
        #   api_key: str (SerpApi key, required)
        #   hl: str (language, e.g. "en" or "zh-CN")
        #   gl: str (country code, e.g. "us" or "cn")
        # Response format:
        # {
        #     "search_metadata": {"status": "Success", "id": "..."},
        #     "ai_overview": {
        #         "text_blocks": [
        #             {"type": "text", "snippet": "..."},
        #             {"type": "list", "snippet": "1. Place Name - description"},
        #             ...
        #         ],
        #         "reconstructed_markdown": "# Title\n\n1. Place Name\n...",
        #         "references": [...]
        #     }
        # }
        params = {
            "engine": "google_ai_mode",
            "q": query,
            "api_key": key,
            "hl": "zh-CN" if language.startswith("zh") else "en",
            "gl": "cn" if language.startswith("zh") else "us",
        }

        logger.info(f"Querying Google AI Mode for must-visit places in '{destination}'...")
        resp = _get_session().get("https://serpapi.com/search", params=params, timeout=API_TIMEOUT * 3)
        resp.raise_for_status()
        data = resp.json()

        # Check search status
        search_status = data.get("search_metadata", {}).get("status", "")
        if search_status != "Success":
            logger.warning(f"Google AI Mode search status: {search_status}")
            # Still try to parse whatever we got

        # Extract text from the AI overview response.
        # The response may contain text_blocks, reconstructed_markdown, or both.
        raw_text = ""

        # Method 1: Try reconstructed_markdown (most reliable)
        ai_overview = data.get("ai_overview", {})
        if isinstance(ai_overview, dict):
            raw_text = ai_overview.get("reconstructed_markdown", "")

            # Method 2: Concatenate text_blocks if markdown is empty
            if not raw_text:
                text_blocks = ai_overview.get("text_blocks", [])
                if isinstance(text_blocks, list):
                    raw_text = "\n".join(
                        block.get("snippet", "") if isinstance(block, dict) else str(block)
                        for block in text_blocks
                    )

        # Method 3: Try top-level text_blocks (some response shapes differ)
        if not raw_text:
            text_blocks = data.get("text_blocks", [])
            if isinstance(text_blocks, list):
                raw_text = "\n".join(
                    block.get("snippet", "") if isinstance(block, dict) else str(block)
                    for block in text_blocks
                )

        if not raw_text:
            logger.warning(f"Google AI Mode returned no text for '{destination}'")
            return []

        logger.debug(f"AI Mode raw text (first 500 chars): {raw_text[:500]}")

        # Parse the AI response text to extract place names.
        place_names = _extract_place_names(raw_text, destination, language)

        # Always run LLM extraction as a supplement (not just as fallback).
        # The LLM can catch place names mentioned inline in paragraphs that
        # regex patterns miss, and can also handle non-standard list formats.
        if len(place_names) < 15:
            logger.info(f"Regex extracted only {len(place_names)} places, running LLM extraction as supplement...")
            llm_names = _llm_extract_place_names(raw_text, destination, language)
            for name in llm_names:
                if name not in place_names:
                    place_names.append(name)

        if place_names:
            logger.info(f"AI Mode extracted {len(place_names)} must-visit places for '{destination}': {place_names[:10]}...")
        else:
            logger.warning(f"Could not extract any place names from AI Mode response for '{destination}'")

        return place_names

    except Exception as e:
        logger.error(f"Google AI Mode search failed for '{destination}': {e}")
        return []


def _extract_place_names(text: str, destination: str, language: str) -> list[str]:
    """Extract attraction/place names from AI Mode response text.

    Parses numbered/bulleted lists and heading patterns to extract
    clean place names. Handles both English and Chinese text.

    Args:
        text (str): Raw AI Mode response text (markdown or plain text).
        destination (str): Destination name (used for context filtering).
        language (str): Language code to guide parsing.

    Returns:
        list[str]: List of extracted place names (deduplicated, cleaned).
    """
    names = []
    seen = set()

    # Strategy 1: Extract from numbered/bulleted list items.
    # Matches patterns like:
    #   "1. **Yellow Crane Tower** - description..."
    #   "2. 黄鹤楼 - description..."
    #   "- **Tokyo Tower**: description..."
    #   "* Senso-ji Temple"
    #   "• Eiffel Tower (iconic landmark...)"
    list_patterns = [
        # Numbered list with bold name: "1. **Place Name**"
        r'(?:^|\n)\d+[\.、)]\s+\*\*([^*]+)\*\*',
        # Numbered list with bold name using __underline__: "1. __Place Name__"
        r'(?:^|\n)\d+[\.、)]\s+__([^_]+)__',
        # Numbered list with plain name + separator: "1. Place Name - desc"
        r'(?:^|\n)\d+[\.、)]\s+([^\-\—:|•·\n]+?)\s*(?:[-\—:|•·]|$)',
        # Numbered list with name in parentheses: "1. Place Name（描述）"
        r'(?:^|\n)\d+[\.、)]\s+([^\n（(]{2,80}?)[\s（(]',
        # Bullet list with bold name: "- **Place Name**"
        r'(?:^|\n)[-*•·▪◦]\s+\*\*([^*]+)\*\*',
        # Bullet list with bold name using __: "- __Place Name__"
        r'(?:^|\n)[-*•·▪◦]\s+__([^_]+)__',
        # Bullet list with plain name + separator: "- Place Name - desc"
        r'(?:^|\n)[-*•·▪◦]\s+([^\-\—:|•·\n]+?)\s*(?:[-\—:|•·]|$)',
        # Heading pattern: "### Place Name"
        r'(?:^|\n)#{1,4}\s+(.+)',
        # Table row pattern: "| Place Name | desc |"
        r'(?:^|\n)\|\s*([^|\n]+?)\s*\|',
    ]

    for pattern in list_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            name = match.strip()
            # Clean up common artifacts
            name = name.strip('*_`').strip()
            # Remove trailing description after dash, em-dash, or en-dash
            name = re.split(r'\s+[-\—–]\s+', name)[0].strip()
            # Remove trailing description after colon (both ASCII and CJK)
            name = re.split(r'[:：]', name)[0].strip()
            # Remove trailing parenthetical descriptions
            name = re.sub(r'[（(].*$', '', name).strip()

            # Filter: must be reasonable length (2-80 chars), not a sentence
            if 2 <= len(name) <= 80 and not name.startswith(('The ', 'A ', 'An ', 'This ', 'These ', 'Some ')):
                # Skip if it looks like a sentence (more than 12 words = likely a sentence)
                word_count = len(name.split())
                if word_count > 12:
                    continue
                if name not in seen:
                    # Skip generic words that are not place names
                    skip_words = {
                        "introduction", "overview", "summary", "tips",
                        "攻略", "介绍", "概述", "总结", "建议",
                        "注意事项", "交通", "住宿", "美食", "购物",
                        "place name", "place", "name",
                    }
                    if name.lower() not in skip_words:
                        seen.add(name)
                        names.append(name)

    # Strategy 2: Always run LLM extraction as supplement.
    # The regex patterns above only catch strict list formats. The AI may also
    # mention places inline within paragraphs, which the LLM can catch.
    # This runs unconditionally — the results are merged with regex results.
    if len(names) < 15:
        llm_names = _llm_extract_place_names(text, destination, language)
        for name in llm_names:
            if name not in seen:
                seen.add(name)
                names.append(name)

    return names


def _llm_extract_place_names(text: str, destination: str, language: str) -> list[str]:
    """Use LLM to extract place names from AI Mode response text as a fallback.

    Args:
        text (str): Raw AI Mode response text.
        destination (str): Destination name for context.
        language (str): Language code.

    Returns:
        list[str]: List of extracted place names.
    """
    try:
        from src.agents.utils import llm_client

        if language.startswith("zh"):
            prompt = (
                f"以下是一段关于「{destination}」旅游景点的AI回答。"
                f"请从中提取所有提到的具体景点、地标、博物馆、餐厅或美食名称。"
                f"尽量提取到15个以上不同的地点名称。"
                f"只提取具体地点名称，不要提取句子或描述性短语。"
                f"返回JSON格式: {{\"places\": [\"名称1\", \"名称2\", ...]}}\n\n"
                f"AI回答：\n{text[:4000]}"
            )
            system = "你是一个景点名称提取器，只返回JSON，提取所有具体地点名称。"
        else:
            prompt = (
                f"The following is an AI-generated response about tourist attractions in {destination}. "
                f"Extract ALL specific place names, landmarks, museums, restaurants, or food spots mentioned. "
                f"Try to extract at least 15 different place names. "
                f"Only extract specific place names, not sentences or descriptive phrases. "
                f'Return JSON format: {{"places": ["Name1", "Name2", ...]}}\n\n'
                f"AI response:\n{text[:4000]}"
            )
            system = "You are a place name extractor. Return only JSON. Extract all specific place names."

        response = llm_client.chat(
            system_prompt=system,
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=800,
            json_format=True,
        )
        result = json.loads(response)
        places = result.get("places", [])
        if isinstance(places, list):
            return [p.strip() for p in places if isinstance(p, str) and p.strip()]
    except Exception as e:
        logger.debug(f"LLM place name extraction failed: {e}")

    return []


def fetch_place_descriptions(
    place_names: list[str],
    destination: str = "",
    api_key: Optional[str] = None,
    language: str = "en",
) -> dict[str, str]:
    """Fetch short AI-generated descriptions for places via Google AI Mode.

    This is a fallback for POIs where Wikipedia has no article. For each
    place name, queries Google AI Mode with "What is {place_name} in
    {destination}?" and extracts the AI overview text as a brief
    description (1-3 sentences).

    Args:
        place_names (list[str]): List of place/POI names needing descriptions.
        destination (str): Destination context for disambiguation
            (e.g. "Paris"). Helps AI Mode return relevant results.
        api_key (Optional[str]): SerpApi key. If None, uses SERPAPI_KEY.
        language (str): Language code (``"en"`` or ``"zh-CN"``).

    Returns:
        dict[str, str]: Mapping of {place_name: description_text}.
            Places without results are omitted. Returns empty dict if
            API key is missing or all queries fail.

    Raises:
        No exceptions raised; all errors are caught and logged.
    """
    key = api_key or SERPAPI_KEY
    if not key:
        logger.info("SerpApi Key is empty, skipping AI Mode description fetch")
        return {}

    # Filter out empty names
    valid_names = [n for n in place_names if n and n.strip()]
    if not valid_names:
        return {}

    # Determine language params once
    hl = "zh-CN" if language.startswith("zh") else "en"
    gl = "cn" if language.startswith("zh") else "us"

    def _fetch_one(name: str) -> tuple[str, str]:
        """Fetch AI description for a single place. Returns (name, description) or (name, "")."""
        if language.startswith("zh"):
            query = f"{name}是什么？有什么特色和亮点？"
            if destination:
                query = f"{destination}的{name}是什么？有什么特色和亮点？请简要介绍。"
        else:
            query = f"What is {name}? What are its highlights and key features?"
            if destination:
                query = f"What is {name} in {destination}? Briefly describe its highlights and key features."

        try:
            params = {
                "engine": "google_ai_mode",
                "q": query,
                "api_key": key,
                "hl": hl,
                "gl": gl,
            }

            resp = _get_session().get("https://serpapi.com/search", params=params, timeout=API_TIMEOUT * 3)
            resp.raise_for_status()
            data = resp.json()

            # Extract text from AI overview
            raw_text = ""
            ai_overview = data.get("ai_overview", {})
            if isinstance(ai_overview, dict):
                raw_text = ai_overview.get("reconstructed_markdown", "")
                if not raw_text:
                    text_blocks = ai_overview.get("text_blocks", [])
                    if isinstance(text_blocks, list):
                        raw_text = "\n".join(
                            block.get("snippet", "") if isinstance(block, dict) else str(block)
                            for block in text_blocks
                        )
            if not raw_text:
                text_blocks = data.get("text_blocks", [])
                if isinstance(text_blocks, list):
                    raw_text = "\n".join(
                        block.get("snippet", "") if isinstance(block, dict) else str(block)
                        for block in text_blocks
                    )

            if not raw_text or len(raw_text.strip()) < 20:
                return (name, "")

            desc = _clean_ai_description(raw_text)
            return (name, desc)

        except Exception as e:
            logger.debug(f"AI Mode description failed for '{name}': {e}")
            return (name, "")

    # Run all requests in parallel (max 5 concurrent to avoid rate limits)
    results: dict[str, str] = {}
    max_workers = min(5, len(valid_names))
    logger.info(f"Fetching AI descriptions for {len(valid_names)} places in parallel (max {max_workers} workers)...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {executor.submit(_fetch_one, name): name for name in valid_names}
        for future in as_completed(future_to_name):
            name, desc = future.result()
            if desc:
                results[name] = desc
                logger.debug(f"AI description found for '{name}': {desc[:60]}...")

    if results:
        logger.info(f"AI Mode descriptions fetched: {len(results)}/{len(valid_names)} places")
    return results


def _clean_ai_description(text: str) -> str:
    """Clean and truncate AI Mode response into a concise description.

    Strips markdown formatting, removes citations, and truncates to
    ~300 characters at a sentence boundary.

    Args:
        text (str): Raw AI Mode response text.

    Returns:
        str: Cleaned, concise description text.
    """
    # Remove markdown headers
    text = re.sub(r'^#{1,4}\s+', '', text, flags=re.MULTILINE)
    # Remove markdown bold/italic markers
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    # Remove citation brackets: [1], [citation needed]
    text = re.sub(r'\[[^\]]*\]', '', text)
    # Remove markdown links, keep text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Truncate to ~350 chars at sentence boundary
    if len(text) > 350:
        cut = text.rfind('. ', 250, 350)
        if cut > 0:
            text = text[:cut + 1]
        else:
            text = text[:300].rsplit(' ', 1)[0] + "..."
    return text
