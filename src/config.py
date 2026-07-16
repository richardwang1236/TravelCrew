"""Centralized configuration for the travel planning agent.

This module loads environment variables (via python-dotenv when available)
and exposes every tunable knob used across the application — LLM settings,
external API keys, search parameters, and business-logic constants.

All values fall back to sensible defaults so the application can start
even when a .env file is missing or incomplete.

Configuration sections:
    1. **LLM Configuration** — DeepSeek API credentials, model selection,
       timeout, and retry settings for the language-model backend.
    2. **Business Logic** — Application-level knobs such as the maximum
       number of replan attempts before force-approving a plan.
    3. **External API Keys** — Credentials for Google Maps, OpenWeather,
       Pixabay, SerpApi, and the legacy Unsplash provider.
    4. **API Call Parameters** — HTTP timeouts, retry counts, and search
       radius/limits for external REST API calls.
    5. **Wikivoyage & Hotel** — Timeout for Wikivoyage content fetches
       and the number of hotel recommendations per travel day.
    6. **Currency & Cost Constants** — Exchange rates, currency symbols,
       transport cost per minute, and destination cost multipliers
       extracted from agents.py and api_client.py.

Usage::

    from config import DEEPSEEK_API_KEY, GOOGLE_MAPS_API_KEY
    # All constants are module-level and can be imported directly.
"""

import os

# Attempt to load .env file for local development; silently skip if
# python-dotenv is not installed (e.g. in production containers that
# inject env vars directly).
#
# load_dotenv() reads key=value pairs from a .env file in the current
# working directory and exports them into os.environ, making them
# available via os.getenv() below.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ──────────────────────────────────────────────
# LLM (Large Language Model) Configuration
# ──────────────────────────────────────────────

# API key used to authenticate with the DeepSeek inference endpoint.
# Default: a project-level key; override via DEEPSEEK_API_KEY env var.
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# Base URL of the DeepSeek-compatible API server.
# Default: official DeepSeek cloud endpoint.
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Model identifier sent in the `model` field of every chat-completion request.
# Default: deepseek-v4-flash for fast, cost-effective responses.
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# Per-request timeout (in seconds) for LLM API calls.
# Set high enough to accommodate long-running reasoning requests.
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300.0"))

# Maximum number of retry attempts when an LLM call fails transiently.
# Uses exponential backoff (2^n seconds) between retries.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# ──────────────────────────────────────────────
# Business Logic Configuration
# ──────────────────────────────────────────────

# Maximum number of times the Critic node can send recommendations back
# for re-ranking before the graph force-approves and moves to synthesis.
# Default: 3 — allows up to 3 re-ranking passes before the plan is accepted
# with warnings.  Used by critic_router() in graph.py.
MAX_REPLAN_ATTEMPTS = int(os.getenv("MAX_REPLAN_ATTEMPTS", "3"))

# ═══════════════════════════════════════════════
# External API Configuration
# ═══════════════════════════════════════════════

# Google Maps / Places API key — used for POI search, geocoding,
# and place-detail lookups.  Required for core functionality.
# Powers: fetch_attractions, fetch_dining, search_specific_place,
#         backfill_missing_coordinates, and routing in api_client.py.
# Obtain from: https://console.cloud.google.com/apis/credentials
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# OpenWeatherMap API key — provides current weather and forecasts
# for the destination city.  Optional; graceful fallback if absent.
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# SerpApi key — powers Google AI Mode search for must-visit place discovery.
# Obtain from: https://serpapi.com/manage-api-key
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

# Serper.dev API key — powers Google Images search for POI image retrieval.
# Serper.dev provides fast, affordable Google search results via API.
# Obtain from: https://serper.dev/
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

# ──────────────────────────────────────────────
# API Call Parameters
# ──────────────────────────────────────────────

# HTTP timeout (seconds) applied to all external REST API calls
# (Google Places, weather, image APIs, etc.).
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "10.0"))

# Number of retry attempts for transient HTTP errors on external APIs.
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "2"))

# Radius (in meters) for Google Places nearby/text search around the
# destination center.  5000 m ≈ 5 km covers most urban attractions.
PLACES_SEARCH_RADIUS = int(os.getenv("PLACES_SEARCH_RADIUS", "5000"))

# Maximum number of POI results returned per Google Places query.
# Higher values give the recommender more candidates but increase
# API cost and response time.
PLACES_RESULT_LIMIT = int(os.getenv("PLACES_RESULT_LIMIT", "30"))

# ──────────────────────────────────────────────
# Wikimedia Enterprise & Hotel Configuration
# ──────────────────────────────────────────────

# Timeout (seconds) for Wikimedia Enterprise API calls that fetch destination
# descriptions and travel tips.
WIKIVOYAGE_TIMEOUT = float(os.getenv("WIKIVOYAGE_TIMEOUT", "10.0"))

# Wikimedia Enterprise On-demand API credentials for fetching Wikivoyage articles
# Login: POST https://auth.enterprise.wikimedia.com/v1/login
# Access tokens expire after 24 hours; refresh tokens expire after 90 days.
WIKIMEDIA_ENTERPRISE_USERNAME = os.getenv("WIKIMEDIA_ENTERPRISE_USERNAME", "")
WIKIMEDIA_ENTERPRISE_PASSWORD = os.getenv("WIKIMEDIA_ENTERPRISE_PASSWORD", "")

# Number of hotel recommendations to retrieve per travel day.
# Default: 1 — the top-rated option per day; increase for more variety.
# Used by the Recommendation node in agents.py when building daily plans.
HOTELS_PER_DAY = int(os.getenv("HOTELS_PER_DAY", "1"))

# ──────────────────────────────────────────────
# Web API Server Configuration
# ──────────────────────────────────────────────

# Host address for the FastAPI server.
API_HOST = os.getenv("API_HOST", "0.0.0.0")

# Port for the FastAPI server.
API_PORT = int(os.getenv("API_PORT", "8000"))

# Session time-to-live in seconds (1 hour default).
# Sessions older than this are cleaned up automatically.
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))

# Maximum number of concurrent planning sessions allowed.
MAX_CONCURRENT_SESSIONS = int(os.getenv("MAX_CONCURRENT_SESSIONS", "10"))

# ═══════════════════════════════════════════════
# Currency & Cost Constants
# (migrated from agents.py and api_client.py)
# ═══════════════════════════════════════════════

# ── Exchange Rates (from agents.py) ───────────────────────────
# Static exchange rates from USD (POI costs are in USD from Google Places API).
# Used as fallback when the live exchange rate API is unreachable.
# Keys: ISO 4217 currency codes; Values: approximate rate per 1 USD.
# For production, replace with a real-time forex API (e.g. Open Exchange Rates).
EXCHANGE_RATES = {
    "USD": 1.0,
    "CNY": 6.8,
    "EUR": 0.87,
    "GBP": 0.75,
    "JPY": 162,
    "CAD": 1.42
}

# ── Currency Symbols (from agents.py) ────────────────────────
# Maps ISO 4217 currency codes to their display symbols.
# Used by the synthesizer to format prices in the final report.
CURRENCY_SYMBOLS = {
    "USD": "$",
    "CNY": "¥",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CAD": "$",
}

# ── Transport Cost Per Minute (from agents.py) ───────────────
# Lookup table for estimating local transit costs per minute of travel.
# Keys are lowercase city names; "default" is used for unknown destinations.
# Values are hand-tuned estimates based on typical taxi/rideshare pricing.
_TRANSPORT_COST_PER_MIN = {
    # Asia — mixed transit (metro/bus/occasional taxi), per-minute USD
    "beijing": 0.08, "shanghai": 0.10, "guangzhou": 0.08, "shenzhen": 0.10,
    "chengdu": 0.07, "hangzhou": 0.08, "wuhan": 0.07, "xian": 0.07,
    "tokyo": 0.15, "osaka": 0.12, "kyoto": 0.12,
    "seoul": 0.12, "busan": 0.10,
    "bangkok": 0.05, "singapore": 0.10, "hong kong": 0.12,
    "taipei": 0.08,
    # Europe — mixed transit
    "paris": 0.12, "london": 0.15, "rome": 0.10, "barcelona": 0.10,
    "amsterdam": 0.12, "berlin": 0.10, "munich": 0.12,
    "prague": 0.08, "budapest": 0.07,
    # Americas
    "new york": 0.12, "los angeles": 0.12, "san francisco": 0.12,
    "chicago": 0.10, "miami": 0.10,
    "toronto": 0.10, "vancouver": 0.10,
    "mexico city": 0.06, "cancun": 0.08,
    # Oceania
    "sydney": 0.12, "melbourne": 0.12,
    # Default
    "default": 0.10,
}

# ── Destination Cost Multipliers (from api_client.py) ────────
# Regional cost multipliers applied on top of price_level-based cost.
# Values represent relative cost of living compared to US baseline (1.0).
# Used by _convert_place_to_poi() to produce more realistic cost estimates.
# Keys are lowercase city/region names; lookup uses exact then partial match.
_DESTINATION_COST_MULTIPLIERS = {
    # Asia
    "beijing": 0.8, "shanghai": 0.9, "guangzhou": 0.8, "shenzhen": 0.9,
    "chengdu": 0.7, "hangzhou": 0.8, "nanjing": 0.8, "wuhan": 0.7,
    "xian": 0.6, "chongqing": 0.6, "suzhou": 0.7, "xiamen": 0.7,
    "tokyo": 1.5, "osaka": 1.3, "kyoto": 1.3,
    "seoul": 1.1, "busan": 1.0,
    "bangkok": 0.5, "phuket": 0.6, "chiang mai": 0.4,
    "singapore": 1.4, "hong kong": 1.3, "taipei": 0.8,
    "hanoi": 0.4, "ho chi minh": 0.4,
    "bali": 0.4, "jakarta": 0.5,
    "mumbai": 0.4, "delhi": 0.4,
    # Europe
    "paris": 1.5, "london": 1.6, "rome": 1.2, "milan": 1.3,
    "barcelona": 1.1, "madrid": 1.1, "amsterdam": 1.4,
    "berlin": 1.1, "munich": 1.3, "vienna": 1.2,
    "prague": 0.8, "budapest": 0.7, "lisbon": 0.9,
    "zurich": 1.8, "geneva": 1.8, "stockholm": 1.4,
    # Americas
    "new york": 1.5, "los angeles": 1.4, "san francisco": 1.5,
    "chicago": 1.2, "miami": 1.3, "las vegas": 1.2,
    "toronto": 1.2, "vancouver": 1.3,
    "mexico city": 0.6, "cancun": 0.8,
    # Oceania
    "sydney": 1.4, "melbourne": 1.3, "auckland": 1.2,
    # Middle East & Africa
    "dubai": 1.3, "istanbul": 0.7, "cairo": 0.4,
}
