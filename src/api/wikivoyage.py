"""src.api.wikivoyage — travel knowledge retrieval from Wikimedia Enterprise API.

Fetches travel tips and local knowledge from the Wikimedia Enterprise
On-demand API (Wikivoyage project). Parses HTML to extract section-based
content (Understand, Get in, See, Eat, Stay safe, Respect, etc.).

Authentication flow:
1. POST username/password to https://auth.enterprise.wikimedia.com/v1/login
2. Receive access_token (24h TTL) + refresh_token (90d TTL)
3. Use Bearer access_token for all API calls
4. Auto-refresh when token expires

Falls back to the free Wikivoyage REST API if Enterprise credentials are
not configured or if the Enterprise API returns an error.
"""

import re
import time
import threading
import requests

from src.api.base import logger, _get_session
from src.config import (
    WIKIVOYAGE_TIMEOUT,
    WIKIMEDIA_ENTERPRISE_USERNAME,
    WIKIMEDIA_ENTERPRISE_PASSWORD,
)

# Simple in-memory cache to avoid repeated requests for the same destination
_wikivoyage_cache: dict[str, dict] = {}

# ─────────────────────────────────────────────────────────────────────────────
# Wikimedia Enterprise Authentication Manager
# ─────────────────────────────────────────────────────────────────────────────

_AUTH_URL = "https://auth.enterprise.wikimedia.com/v1/login"
_TOKEN_REFRESH_URL = "https://auth.enterprise.wikimedia.com/v1/token-refresh"

# Thread-safe token storage
_token_lock = threading.Lock()
_access_token: str | None = None
_refresh_token: str | None = None
_token_expires_at: float = 0  # Unix timestamp when access_token expires


def _login() -> bool:
    """Authenticate with Wikimedia Enterprise and store tokens.

    POST https://auth.enterprise.wikimedia.com/v1/login
    Body: {"username": "...", "password": "..."}
    Response: {"access_token": "...", "refresh_token": "...", "expires_in": 86400}

    Returns:
        bool: True if login succeeded.
    """
    global _access_token, _refresh_token, _token_expires_at

    if not WIKIMEDIA_ENTERPRISE_USERNAME or not WIKIMEDIA_ENTERPRISE_PASSWORD:
        return False

    try:
        resp = _get_session().post(
            _AUTH_URL,
            json={
                "username": WIKIMEDIA_ENTERPRISE_USERNAME.lower(),
                "password": WIKIMEDIA_ENTERPRISE_PASSWORD,
            },
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        _access_token = data.get("access_token")
        _refresh_token = data.get("refresh_token")
        expires_in = data.get("expires_in", 86400)
        # Set expiry 5 minutes early to avoid edge-case failures
        _token_expires_at = time.time() + expires_in - 300

        logger.info("Wikimedia Enterprise: login successful")
        return True

    except Exception as e:
        logger.warning(f"Wikimedia Enterprise login failed: {e}")
        return False


def _refresh_access_token() -> bool:
    """Refresh the access token using the stored refresh token.

    POST https://auth.enterprise.wikimedia.com/v1/token-refresh
    Body: {"refresh_token": "...", "username": "..."}
    Response: {"access_token": "...", "expires_in": 86400}

    Returns:
        bool: True if refresh succeeded.
    """
    global _access_token, _token_expires_at

    if not _refresh_token:
        return False

    try:
        resp = _get_session().post(
            _TOKEN_REFRESH_URL,
            json={
                "refresh_token": _refresh_token,
                "username": WIKIMEDIA_ENTERPRISE_USERNAME.lower(),
            },
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        _access_token = data.get("access_token", _access_token)
        expires_in = data.get("expires_in", 86400)
        _token_expires_at = time.time() + expires_in - 300

        logger.info("Wikimedia Enterprise: token refreshed")
        return True

    except Exception as e:
        logger.warning(f"Wikimedia Enterprise token refresh failed: {e}")
        return False


def _get_access_token() -> str | None:
    """Get a valid access token, logging in or refreshing as needed.

    Thread-safe. Returns None if authentication is unavailable.
    """
    global _access_token

    with _token_lock:
        # Token still valid
        if _access_token and time.time() < _token_expires_at:
            return _access_token

        # Try refresh first (cheaper than full login)
        if _refresh_token and _refresh_access_token():
            return _access_token

        # Full login
        if _login():
            return _access_token

        return None


# ─────────────────────────────────────────────────────────────────────────────
# HTML Section Parsing
# ─────────────────────────────────────────────────────────────────────────────


def _strip_html_tags(html: str) -> str:
    """Remove HTML tags and decode common HTML entities from a string.

    Strips all HTML tags, replaces common HTML entities with their character
    equivalents, and collapses multiple whitespace into single spaces.

    Args:
        html (str): Raw HTML string to clean.

    Returns:
        str: Plain text with no HTML tags or entities.
    """
    text = re.sub(r'<[^>]+>', ' ', html)
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _extract_wikivoyage_section(html: str, section_titles: list[str]) -> str:
    """Extract text content from a Wikivoyage HTML page by section heading.

    Searches for <h2> headings matching any of the given titles (case-insensitive)
    and captures all text content between that heading and the next <h2> heading
    (or end of document). Returns the first match found.

    Args:
        html (str): Full HTML string from Wikivoyage.
        section_titles (list[str]): List of candidate section heading titles.

    Returns:
        str: Extracted plain text (max 500 chars), or empty string if not found.
    """
    for title in section_titles:
        pattern = re.compile(
            r'<h2[^>]*>\s*' + re.escape(title) + r'\s*</h2>(.*?)(?=<h2[^>]*>|$)',
            re.DOTALL | re.IGNORECASE
        )
        match = pattern.search(html)
        if match:
            text = _strip_html_tags(match.group(1))
            return text[:500]
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Data Fetching
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_via_enterprise_api(destination: str) -> str | None:
    """Fetch Wikivoyage article HTML via Wikimedia Enterprise On-demand API.

    API Endpoint: POST https://api.enterprise.wikimedia.com/v2/articles/{name}
    Auth: Bearer token in Authorization header
    Request body: filters for English Wikivoyage (enwikivoyage)
    Response: JSON array with article_body.html field

    Args:
        destination (str): Destination city name in English.

    Returns:
        str | None: Article HTML body, or None if request fails.
    """
    token = _get_access_token()
    if not token:
        return None

    safe_dest = requests.utils.quote(destination.strip().replace(' ', '_').title())
    url = f"https://api.enterprise.wikimedia.com/v2/articles/{safe_dest}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    body = {
        "fields": [
            "name",
            "article_body.html",
            "abstract"
        ],
        "filters": [
            {"field": "in_language.identifier", "value": "en"},
            {"field": "is_part_of.identifier", "value": "enwikivoyage"}
        ],
        "limit": 1
    }

    try:
        resp = _get_session().post(url, json=body, headers=headers, timeout=WIKIVOYAGE_TIMEOUT)

        # If 401, token may have expired mid-flight; retry once after refresh
        if resp.status_code == 401:
            token = _get_access_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"
                resp = _get_session().post(url, json=body, headers=headers, timeout=WIKIVOYAGE_TIMEOUT)

        resp.raise_for_status()
        data = resp.json()

        # Response is a JSON array
        if isinstance(data, list) and len(data) > 0:
            article = data[0]
            html = article.get("article_body", {}).get("html", "")
            if html:
                logger.info(f"Wikimedia Enterprise API: got article for '{destination}'")
                return html
        return None

    except requests.exceptions.HTTPError as e:
        logger.warning(f"Wikimedia Enterprise API HTTP error for '{destination}': {e}")
        return None
    except requests.exceptions.Timeout:
        logger.warning(f"Wikimedia Enterprise API timeout for '{destination}'")
        return None
    except Exception as e:
        logger.warning(f"Wikimedia Enterprise API failed for '{destination}': {e}")
        return None


def _fetch_via_free_api(destination: str) -> str | None:
    """Fallback: fetch Wikivoyage article HTML via free REST API.

    API Endpoint: GET https://en.wikivoyage.org/api/rest_v1/page/html/{destination}
    No auth required. Includes retry logic for rate-limiting.

    Args:
        destination (str): Destination city name in English.

    Returns:
        str | None: Article HTML body, or None if request fails.
    """
    safe_dest = requests.utils.quote(destination.strip())
    url = f"https://en.wikivoyage.org/api/rest_v1/page/html/{safe_dest}"
    headers = {"User-Agent": "TravelPlanner/1.0 (educational project)"}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = _get_session().get(url, headers=headers, timeout=WIKIVOYAGE_TIMEOUT)
            if resp.status_code in (429, 403) and attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                logger.info(f"Wikivoyage rate-limited for '{destination}', retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.HTTPError as e:
            if attempt < max_retries - 1 and "429" in str(e):
                time.sleep(2 ** (attempt + 1))
                continue
            logger.warning(f"Wikivoyage free API HTTP error for '{destination}': {e}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"Wikivoyage free API timeout for '{destination}'")
            return None
        except Exception as e:
            logger.warning(f"Wikivoyage free API failed for '{destination}': {e}")
            return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public Interface
# ─────────────────────────────────────────────────────────────────────────────


def fetch_wikivoyage_info(destination: str) -> dict:
    """Fetch travel knowledge from Wikivoyage.

    Tries the Wikimedia Enterprise On-demand API first (no rate limits).
    Falls back to the free Wikivoyage REST API if Enterprise is unavailable.
    Results are cached in memory to avoid repeated requests.

    Args:
        destination (str): Destination city name in English
            (e.g. "Tokyo", "Paris", "Wuhan").

    Returns:
        dict: Travel knowledge dict with keys:
            best_seasons (str): From "Understand" section (max 500 chars).
            transport_tips (str): From "Get in"/"Get around" sections.
            highlights (str): From "See"/"Do" sections.
            food_tips (str): From "Eat" section.
            safety_tips (str): From "Stay safe" section.
            local_customs (str): From "Respect" section.
            day_trips (str): Always empty string (reserved for future use).
        Returns empty dict {} on any failure.
    """
    # Check cache first
    cache_key = destination.strip().lower()
    if cache_key in _wikivoyage_cache:
        logger.info(f"Wikivoyage cache hit for '{destination}'")
        return _wikivoyage_cache[cache_key]

    # Try Enterprise API first, fall back to free API
    html = _fetch_via_enterprise_api(destination)
    if not html:
        html = _fetch_via_free_api(destination)
    if not html:
        return {}

    try:
        result = {
            "best_seasons": _extract_wikivoyage_section(html, ["Understand"]),
            "transport_tips": _extract_wikivoyage_section(html, ["Get in", "Get around"]),
            "highlights": _extract_wikivoyage_section(html, ["See", "Do"]),
            "food_tips": _extract_wikivoyage_section(html, ["Eat"]),
            "safety_tips": _extract_wikivoyage_section(html, ["Stay safe"]),
            "local_customs": _extract_wikivoyage_section(html, ["Respect"]),
            "day_trips": ""
        }

        filled = sum(1 for v in result.values() if v)
        logger.info(f"Wikivoyage info retrieved for '{destination}': {filled}/{len(result)} sections")

        # Cache the result
        _wikivoyage_cache[cache_key] = result
        return result

    except Exception as e:
        logger.warning(f"Wikivoyage parsing failed for '{destination}': {e}")
        return {}