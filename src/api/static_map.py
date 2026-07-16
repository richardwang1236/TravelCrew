"""src.api.static_map — Generate and cache static map images via Google Maps.

Downloads static map images from the Google Static Maps API for given
coordinates, caches them to disk, and serves them as local files. This
replaces the interactive <iframe> embed approach which was slow and
unreliable (especially in regions where map services are slow or blocked).

The backend downloads the map image once during the routing phase; after
that, the frontend loads the cached local image instantly with no network
dependency.

API: Google Static Maps API
    Endpoint: https://maps.googleapis.com/maps/api/staticmap
    Docs: https://developers.google.com/maps/documentation/maps-static

Public Functions:
    get_static_map_url(lat, lng, api_key, zoom, size) -> str (local URL)
    generate_static_maps(pois_data, api_key) -> dict (batch)
"""

import os
import hashlib
import requests
from typing import Optional

from src.api.base import logger, API_TIMEOUT, GOOGLE_MAPS_API_KEY, _get_session

# Directory to cache generated map images.
_MAPS_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "reports", "maps"
)
os.makedirs(_MAPS_CACHE_DIR, exist_ok=True)


def get_static_map_url(
    lat: float,
    lng: float,
    api_key: Optional[str] = None,
    zoom: int = 15,
    width: int = 600,
    height: int = 400,
) -> Optional[str]:
    """Generate a static map image for the given coordinates.

    Downloads a map image from Google Static Maps API with a marker at the
    given coordinates, saves it to the cache directory, and returns the local
    URL path that can be used in <img> tags.

    The image filename is derived from the coordinates, so subsequent calls
    with the same parameters return the cached image without any network
    requests.

    Args:
        lat: Center latitude.
        lng: Center longitude.
        api_key: Google Maps API key. If None, uses the globally
            configured GOOGLE_MAPS_API_KEY.
        zoom: Zoom level (default 15 — shows streets and landmarks).
        width: Image width in pixels (default 600, max 640 for free tier).
        height: Image height in pixels (default 400, max 640 for free tier).

    Returns:
        Local URL path (e.g. "/maps/map_15_48.8606_2.3376_600x400.png") that
        can be used directly in <img src="..."> tags. Returns None if the
        API key is missing or the download fails.
    """
    key = api_key or GOOGLE_MAPS_API_KEY
    if not key:
        logger.info("Google Maps API Key is empty, skipping static map generation")
        return None

    if lat == 0.0 and lng == 0.0:
        return None

    # Clamp dimensions to API max (640x640 for free tier).
    w = min(width, 640)
    h = min(height, 640)

    # Round coordinates to 4 decimal places for cache key (~11m precision).
    lat_r = round(lat, 4)
    lng_r = round(lng, 4)

    # Generate cache filename.
    cache_key = f"{zoom}_{lat_r}_{lng_r}_{w}x{h}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    filename = f"map_{cache_hash}.png"
    filepath = os.path.join(_MAPS_CACHE_DIR, filename)

    # Return cached image if it exists.
    if os.path.exists(filepath):
        logger.debug(f"Static map cache hit: {filename}")
        return f"/maps/{filename}"

    try:
        # Google Static Maps API
        # API 名称: Google Static Maps API
        # Endpoint: https://maps.googleapis.com/maps/api/staticmap
        # Request parameters:
        #   center: "lat,lng"
        #   zoom: int (0-21)
        #   size: "WIDTHxHEIGHT" (max 640x640)
        #   scale: 2 (returns 2x resolution for retina)
        #   markers: "color:red|lat,lng"
        #   key: str (API key)
        #   maptype: "roadmap" (default)
        # Response: PNG image
        params = {
            "center": f"{lat_r},{lng_r}",
            "zoom": str(zoom),
            "size": f"{w}x{h}",
            "scale": "2",  # 2x resolution for crisp display on retina
            "markers": f"color:red|label:📍|{lat_r},{lng_r}",
            "key": key,
            "maptype": "roadmap",
        }

        logger.debug(f"Downloading static map: ({lat_r}, {lng_r}) zoom={zoom}")
        resp = _get_session().get(
            "https://maps.googleapis.com/maps/api/staticmap",
            params=params,
            timeout=API_TIMEOUT,
        )
        resp.raise_for_status()

        # Verify the response is actually an image (not an error JSON).
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type:
            logger.warning(f"Static map API returned non-image: {content_type}")
            return None

        # Save to cache.
        with open(filepath, "wb") as f:
            f.write(resp.content)

        file_size = os.path.getsize(filepath)
        logger.info(f"Generated static map: {filename} ({file_size // 1024}KB)")

        return f"/maps/{filename}"

    except Exception as e:
        logger.error(f"Failed to generate static map for ({lat_r}, {lng_r}): {e}")
        return None


def generate_static_maps(
    pois_data: list[dict],
    api_key: Optional[str] = None,
    zoom: int = 15,
) -> dict[str, str]:
    """Batch-generate static map images for multiple POIs.

    Args:
        pois_data: List of POI dicts with 'name', 'lat', 'lng' fields.
        api_key: Google Maps API key. If None, uses global key.
        zoom: Zoom level for all maps (default 15).

    Returns:
        dict[str, str]: Mapping of {poi_name: local_url}. POIs without
        valid coordinates or if generation fails are omitted.
    """
    results: dict[str, str] = {}
    for poi in pois_data:
        name = poi.get("name", "")
        lat = poi.get("lat")
        lng = poi.get("lng")
        if not name or lat is None or lng is None:
            continue
        try:
            lat_f = float(lat)
            lng_f = float(lng)
            if lat_f == 0.0 and lng_f == 0.0:
                continue
            url = get_static_map_url(lat_f, lng_f, api_key=api_key, zoom=zoom)
            if url:
                results[name] = url
        except (ValueError, TypeError):
            continue

    if results:
        logger.info(f"Generated {len(results)}/{len(pois_data)} static maps")
    return results
