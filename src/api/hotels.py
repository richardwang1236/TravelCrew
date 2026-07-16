"""src.api.hotels — hotel search via SerpApi Google Hotels engine.

Provides :func:`fetch_hotels`, which queries SerpApi's google_hotels engine
to find hotels near a destination with pricing, rating, and amenities.
"""

import requests
from typing import Optional, List

from src.api.base import logger, API_TIMEOUT, SERPAPI_KEY, _get_session


def fetch_hotels(
    destination: str,
    check_in: str,
    check_out: str,
    adults: int = 2,
    budget_max: Optional[int] = None,
    budget_min: Optional[int] = None,
    currency: str = "USD",
    language: str = "en",
) -> List[dict]:
    """Search hotels via SerpApi Google Hotels engine.

    Queries SerpApi's google_hotels engine to find hotels near the given
    destination for the specified dates. Extracts pricing, rating,
    amenities, and GPS coordinates from the response.

    Args:
        destination (str): Destination city or region name (e.g. "Paris").
        check_in (str): Check-in date in YYYY-MM-DD format.
        check_out (str): Check-out date in YYYY-MM-DD format.
        adults (int): Number of adult guests. Defaults to 2.
        budget_max (Optional[int]): Maximum price per night filter in the
            specified currency. None means no upper bound.
        budget_min (Optional[int]): Minimum price per night filter in the
            specified currency. None means no lower bound.
        currency (str): ISO currency code for pricing (e.g. "USD", "EUR").
            Defaults to "USD".

    Returns:
        List[dict]: Up to 10 standardised hotel dicts, each containing:
            {
                "name" (str): Hotel name,
                "price_per_night" (float): Nightly rate in specified currency,
                "total_price" (float): Total stay cost,
                "currency" (str): Currency code,
                "rating" (float): Overall guest rating (0-5),
                "reviews" (int): Number of guest reviews,
                "amenities" (list[str]): Top 5 amenity names,
                "image_url" (str|None): Primary hotel image URL,
                "description" (str|None): Hotel description,
                "check_in_time" (str|None): Check-in time,
                "check_out_time" (str|None): Check-out time,
                "lat" (float|None): GPS latitude,
                "lng" (float|None): GPS longitude
            }
            Returns empty list on failure.

    Raises:
        No exceptions raised; all errors are caught and logged internally.
    """
    if not SERPAPI_KEY:
        logger.info("SerpApi Key is empty, skipping hotel search")
        return []

    # SerpApi Google Hotels Search
    # API 名称: SerpApi - Google Hotels Engine
    # Endpoint: https://serpapi.com/search
    # Engine 类型: google_hotels
    # 请求参数:
    #   engine: "google_hotels"
    #   q: "Hotels in {destination}" (搜索关键词)
    #   check_in_date: YYYY-MM-DD (入住日期)
    #   check_out_date: YYYY-MM-DD (退房日期)
    #   adults: int (成人数量)
    #   currency: str (货币代码, 如 "USD")
    #   gl: "us" (国家代码)
    #   hl: "en" (语言代码)
    #   max_price: int (可选, 每晚最高价格过滤)
    # Response format (SerpApi properties 数组):
    # {
    #     "properties": [
    #         {
    #             "name": "string (hotel name)",
    #             "overall_rating": float (0-5),
    #             "reviews": int,
    #             "description": "string",
    #             "rate_per_night": {
    #                 "extracted_lowest": float,
    #                 "lowest": "string ($xxx)"
    #             },
    #             "total_rate": {
    #                 "extracted_lowest": float,
    #                 "lowest": "string ($xxx)"
    #             },
    #             "amenities": [{"name": "string"}, ...],
    #             "images": ["url1", "url2", ...],
    #             "gps_coordinates": {"latitude": float, "longitude": float},
    #             "check_in_time": "string (e.g. 3:00 PM)",
    #             "check_out_time": "string (e.g. 11:00 AM)"
    #         }
    #     ]
    # }
    params = {
        "engine": "google_hotels",
        "q": f"Hotels in {destination}",
        "check_in_date": check_in,
        "check_out_date": check_out,
        "adults": adults,
        "api_key": SERPAPI_KEY,
        "currency": currency,
        "gl": "us",
        "hl": language,
    }
    # Price range filter — SerpApi supports min_price / max_price for the
    # google_hotels engine, allowing callers to narrow by budget.
    if budget_min is not None:
        params["min_price"] = budget_min
    if budget_max is not None:
        params["max_price"] = budget_max

    try:
        resp = _get_session().get(
            "https://serpapi.com/search",
            params=params,
            timeout=API_TIMEOUT * 2,
        )
        resp.raise_for_status()
        data = resp.json()

        properties = data.get("properties", [])
        hotels: List[dict] = []
        for prop in properties[:20]:
            if len(hotels) >= 10:
                break
            rate_per_night = prop.get("rate_per_night") or {}
            total_rate = prop.get("total_rate") or {}
            images = prop.get("images", [])
            amenities_list = prop.get("amenities", [])

            # Try multiple price field paths — SerpApi response format varies
            price_per_night = (
                rate_per_night.get("extracted_lowest")
                or rate_per_night.get("lowest")
                or prop.get("extracted_price")
                or total_rate.get("extracted_lowest")
                or 0
            )
            total_price = (
                total_rate.get("extracted_lowest")
                or total_rate.get("lowest")
                or prop.get("extracted_total_price")
                or 0
            )

            # Debug log when price is missing — helps diagnose SerpApi schema changes
            if not price_per_night:
                price_debug = {
                    k: v for k, v in prop.items()
                    if "price" in k.lower() or "rate" in k.lower() or "cost" in k.lower()
                }
                logger.warning(
                    f"Hotel '{prop.get('name')}' has no parseable price. "
                    f"Price-related fields: {price_debug}"
                )

            if price_per_night == 0:
                continue  # Skip hotels without price data

            # Extract GPS coordinates for proximity-based hotel selection
            gps = prop.get("gps_coordinates") or {}
            hotel_lat = gps.get("latitude")
            hotel_lng = gps.get("longitude")

            hotels.append({
                "name": prop.get("name"),
                "price_per_night": price_per_night,
                "total_price": total_price,
                "currency": currency,
                "rating": prop.get("overall_rating"),
                "reviews": prop.get("reviews"),
                "amenities": [a.get("name", a) if isinstance(a, dict) else a
                              for a in amenities_list[:5]],
                "image_url": images[0] if images else None,
                "description": prop.get("description"),
                "check_in_time": prop.get("check_in_time"),
                "check_out_time": prop.get("check_out_time"),
                "lat": hotel_lat,
                "lng": hotel_lng,
            })

        logger.info(f"Hotel search complete: {destination}, {len(hotels)} hotels found")
        return hotels

    except Exception as e:
        logger.warning(f"Hotel search failed for '{destination}': {e}")
        return []
