"""src.api — external API client modules.

Re-exports all public API functions so that callers can import from the
package directly::

    from src.api import fetch_weather, fetch_attractions, fetch_hotels
"""

from src.api.weather import fetch_weather
from src.api.attractions import (
    fetch_attractions,
    search_specific_place,
)
from src.api.hotels import fetch_hotels
from src.api.images import fetch_images
from src.api.transport import (
    fetch_transport_matrix,
)
from src.api.geocoding import backfill_missing_coordinates
from src.api.wikivoyage import fetch_wikivoyage_info
from src.api.ai_search import fetch_place_descriptions

__all__ = [
    "fetch_weather",
    "fetch_attractions",
    "search_specific_place",
    "fetch_hotels",
    "fetch_images",
    "fetch_transport_matrix",
    "backfill_missing_coordinates",
    "fetch_wikivoyage_info",
    "fetch_place_descriptions",
]
