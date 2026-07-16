"""src.api.images — POI image retrieval via Serper.dev Google Images.

Provides:
    - :func:`fetch_images` — main entry point for image retrieval.
    - :func:`merge_images_to_pois` — attach image URLs to POI data.
    - :func:`_serper_image_search` — Serper.dev Google Images search.
    - :func:`_llm_extract_image_keyword` — LLM-based keyword extraction.
    - :func:`_translate_to_english` — name translation helper.

Two-level fallback for image retrieval:
    Attempt 1: Direct POI name search (supports Chinese/English natively)
    Attempt 2: LLM-extracted concise English keyword search
"""

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from src.api.base import logger, API_TIMEOUT, SERPER_API_KEY, GOOGLE_MAPS_API_KEY, _get_session


def _translate_to_english(name: str) -> str:
    """Translate a non-ASCII place name to English using Google Geocoding as a proxy.

    If the name is already ASCII, returns it unchanged. Otherwise calls
    Google Geocoding API with language=en to get the English name from
    address_components.

    Args:
        name (str): Place name to translate (may contain Chinese or other
            non-ASCII characters).

    Returns:
        str: English name if translation succeeded, otherwise the original name.

    Raises:
        No exceptions raised; errors are caught and original name returned.
    """
    # If already ASCII, no translation needed
    if all(ord(c) < 128 for c in name):
        return name
    # Try Google Geocoding to get English name
    # Google Maps Geocoding API (used as translation proxy)
    # API 名称: Google Maps Geocoding API
    # Endpoint: https://maps.googleapis.com/maps/api/geocode/json
    # 请求参数: address (原始名称), key (API密钥), language ("en" 返回英文)
    # Response format:
    # {
    #     "results": [
    #         {
    #             "address_components": [
    #                 {"long_name": "string (English name)", "types": [...]}
    #             ],
    #             "formatted_address": "string (full address in English)"
    #         }
    #     ]
    # }
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": name, "key": GOOGLE_MAPS_API_KEY, "language": "en"}
        resp = _get_session().get(url, params=params, timeout=API_TIMEOUT)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            # Extract the formatted address or first address component
            components = results[0].get("address_components", [])
            if components:
                return components[0].get("long_name", name)
            return results[0].get("formatted_address", name).split(",")[0]
    except Exception:
        pass
    # Fallback: just use the original name (Unsplash may still find results)
    return name


def _llm_extract_image_keyword(original_name: str, translated_name: str) -> Optional[str]:
    """Use LLM to extract a concise, visually-searchable keyword from a POI name.

    When a POI name is too specific or branded (e.g. '小杨生煎馆吴江路店'),
    this function uses the LLM to extract a generic visual keyword
    (e.g. 'pan-fried bun') suitable for image search.

    Args:
        original_name (str): Original POI name (may be in Chinese or other language).
        translated_name (str): English translation of the POI name.

    Returns:
        Optional[str]: Concise 1-3 word English keyword for image search,
            or None if extraction fails.

    Raises:
        No exceptions raised; errors are caught and None is returned.
    """
    from llm import DeepSeekChatClient
    from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
    try:
        client = DeepSeekChatClient(
            api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, model_name=DEEPSEEK_MODEL
        )
        prompt = (
            f"Given this place name: '{original_name}' (English: '{translated_name}'),\n"
            "extract a single concise English keyword (1-3 words) that visually represents "
            "this place for an image search. Focus on the CORE concept, not the brand or branch.\n"
            "Examples:\n"
            "- '小杨生煎馆吴江路店' -> 'pan-fried bun'\n"
            "- 'Starbucks Reserve Roastery Shanghai' -> 'coffee roastery'\n"
            "- '南京东路步行街' -> 'shopping street'\n"
            "Return ONLY the keyword, nothing else."
        )
        result = client.chat(
            system_prompt="You extract concise image search keywords. Return only the keyword.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=20
        )
        keyword = result.strip().strip('"').strip("'")
        return keyword if keyword else None
    except Exception as e:
        logger.debug(f"LLM keyword extraction failed for '{original_name}': {e}")
        return None


def _serper_image_search(query: str, key: str) -> Optional[str]:
    """Search Google Images via Serper.dev API and return the first image URL.

    Supports multilingual queries (Chinese, English, etc.) natively.
    Prefers full-size 'imageUrl'; falls back to 'thumbnailUrl'.

    Args:
        query (str): Image search query (e.g. "Eiffel Tower", "黄鹤楼").
        key (str): Serper.dev API key.

    Returns:
        Optional[str]: Image URL (full-size or thumbnail), or None if no
            results or on failure.

    Raises:
        requests.HTTPError: On HTTP failure.
    """
    # Serper.dev Google Images Search API
    # API 名称: Serper.dev - Google Images
    # Endpoint: https://google.serper.dev/images
    # Method: POST
    # Headers:
    #   X-API-KEY: str (Serper.dev API key)
    #   Content-Type: "application/json"
    # Body:
    #   q: str (search query, supports Chinese/English)
    # Response format:
    # {
    #     "searchParameters": {"q": "...", "type": "images", ...},
    #     "images": [
    #         {
    #             "title": "string",
    #             "imageUrl": "string (full-size image URL)",
    #             "imageWidth": int,
    #             "imageHeight": int,
    #             "thumbnailUrl": "string (thumbnail URL)",
    #             "thumbnailWidth": int,
    #             "thumbnailHeight": int,
    #             "source": "string",
    #             "domain": "string",
    #             "link": "string (page URL where image was found)",
    #             "googleUrl": "string",
    #             "position": int
    #         }
    #     ]
    # }
    url = "https://google.serper.dev/images"
    headers = {
        "X-API-KEY": key,
        "Content-Type": "application/json",
    }
    payload = {"q": query}
    resp = _get_session().post(url, headers=headers, json=payload, timeout=API_TIMEOUT * 2)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("images", [])
    if results:
        # Prefer 'imageUrl' (full-size), fallback to 'thumbnailUrl'
        return results[0].get("imageUrl") or results[0].get("thumbnailUrl")
    return None


def fetch_images(poi_names: list[str], api_key: Optional[str] = None, destination: str = "") -> dict[str, str]:
    """Fetch attraction image URLs using Serper.dev Google Images API.

    Serper.dev supports multilingual queries natively, so Chinese POI names
    can be searched directly without translation. Uses a two-level fallback:
      1. Direct POI name search with destination context for disambiguation.
      2. LLM-extracted concise English keyword if no results from step 1.

    Args:
        poi_names (list[str]): List of attraction/POI names to fetch images for.
        api_key (Optional[str]): Serper.dev API key. If None, uses the globally
            configured SERPER_API_KEY.
        destination (str): Destination city name for search disambiguation.
            Appended to queries to avoid returning unrelated results
            (e.g. "Angelina Paris" instead of just "Angelina").

    Returns:
        dict[str, str]: Mapping of {attraction_name: image_url}.
            POIs without images are omitted from the result dict.
            Returns empty dict if API key is missing or all searches fail.

    Raises:
        No exceptions raised; all errors are caught and logged per-POI.
    """
    key = api_key or SERPER_API_KEY
    if not key:
        logger.info("Serper.dev API Key is empty, skipping image retrieval")
        return {}

    # Filter out empty names
    valid_names = [n for n in poi_names if n and n.strip()]
    if not valid_names:
        return {}

    def _fetch_one(name: str) -> tuple[str, str]:
        """Fetch image for a single POI. Returns (name, image_url) or (name, "")."""
        try:
            # Build a disambiguated query by appending the destination city.
            if destination:
                search_query = f"{name} {destination}"
            else:
                search_query = name

            # Attempt 1: Direct name + destination search
            image_url = _serper_image_search(search_query, key)
            if image_url:
                return (name, image_url)

            # Attempt 2: LLM-extracted concise English keyword
            translated = _translate_to_english(name)
            llm_query = _llm_extract_image_keyword(name, translated)
            if llm_query:
                if destination:
                    llm_query = f"{llm_query} {destination}"
                image_url = _serper_image_search(llm_query, key)
                if image_url:
                    logger.debug(f"LLM fallback image found for: {name}")
                    return (name, image_url)

            logger.debug(f"No images found after all attempts for: {name}")
            return (name, "")

        except Exception as e:
            logger.debug(f"Failed to fetch image for '{name}': {e}")
            return (name, "")

    # Run all image searches in parallel (max 5 concurrent to avoid rate limits)
    images: dict[str, str] = {}
    max_workers = min(5, len(valid_names))
    logger.info(f"Fetching images for {len(valid_names)} POIs in parallel (max {max_workers} workers)...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {executor.submit(_fetch_one, name): name for name in valid_names}
        for future in as_completed(future_to_name):
            name, image_url = future.result()
            if image_url:
                images[name] = image_url
                logger.debug(f"Image retrieved successfully: {name}")

    logger.info(f"Image retrieval complete: {len(images)}/{len(valid_names)} images")
    return images
