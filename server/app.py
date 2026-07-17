import asyncio
import copy
import json
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone

# On macOS with Homebrew, WeasyPrint needs to locate shared libraries
# (pango, cairo, glib) that may not be in the default library search path.
# We prepend the Homebrew lib directory to DYLD_LIBRARY_PATH before importing
# weasyprint so cffi can dlopen them successfully.
for _brew_lib in ("/opt/homebrew/lib", "/usr/local/lib"):
    if os.path.isdir(_brew_lib):
        _cur = os.environ.get("DYLD_LIBRARY_PATH", "")
        if _brew_lib not in _cur.split(":"):
            os.environ["DYLD_LIBRARY_PATH"] = (
                f"{_cur}:{_brew_lib}" if _cur else _brew_lib
            )

import logging
from typing import Any

logger = logging.getLogger(__name__)

import markdown
# NOTE: weasyprint is imported lazily inside the download endpoint to avoid
# crashing the server at startup when system libraries (libpango, libcairo)
# are not installed. See: https://doc.courtbouillon.org/weasyprint/stable/first_steps.html

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.graph import travel_app
from server.schemas import (
    CreateSessionResponse,
    PlanRequest,
    FeedbackRequest,
    SessionStateResponse,
    ReportResponse,
)
from server.session_manager import SessionManager
from server.streaming import stream_graph_execution, sse_event_generator

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
SESSION_TTL = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
MAX_SESSIONS = int(os.getenv("MAX_CONCURRENT_SESSIONS", "100"))

session_manager = SessionManager(ttl_seconds=SESSION_TTL, max_sessions=MAX_SESSIONS)


# ---------------------------------------------------------------------------
# Report persistence — save completed reports as .md and standalone .html
# files so they survive server restarts and can be shared via permanent links.
# ---------------------------------------------------------------------------
REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Session activity log — persisted to log/{session_id}.log for debugging.
# Records: user query, feedback, replan summary, final report path.
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "log"
)
os.makedirs(LOG_DIR, exist_ok=True)


def _session_log(session_id: str, tag: str, content: str):
    """Append a formatted, readable entry to the per-session activity log.

    Format:
        ╔══════════════════════════════════════════════
        ║ 📌 TAG                          2026-07-16 10:38:19
        ╠══════════════════════════════════════════════
        ║ (multi-line content here, indented for readability)
        ╚══════════════════════════════════════════════
    """
    log_path = os.path.join(LOG_DIR, f"{session_id}.log")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "═" * 60
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"╔{sep}\n")
            f.write(f"║ {tag:<45} {ts}\n")
            f.write(f"╠{sep}\n")
            for line in content.splitlines():
                f.write(f"║ {line}\n")
            f.write(f"╚{sep}\n\n")
    except Exception as e:
        logger.warning(f"Failed to write session log for {session_id}: {e}")


def _generate_share_html(markdown_content: str, session_id: str) -> str:
    """Generate a standalone shareable HTML page from markdown report.

    Uses server-side Python markdown rendering — zero CDN dependencies.
    This ensures the share page loads reliably even when CDN connections
    are slow or blocked (common in mainland China).

    Map images are post-processed: /maps/ URLs are wrapped in collapsible
    <details>, and OSM/Google embed URLs are converted to clickable links.
    """
    import re as _re

    # ── Pre-process: tables inside blockquotes ──
    # The LLM sometimes generates summary tables inside blockquotes, e.g.:
    #   > | 项目 | 详情 |
    #   > |---|---|
    # Python's markdown library (tables extension) does not process tables
    # nested inside <blockquote> — the blockquote parser consumes them first.
    # Fix: (1) strip '> ' from table rows, (2) insert a blank line before
    # the first table row to terminate the surrounding blockquote, then
    # (3) collapse blank lines between consecutive table rows.
    markdown_content = _re.sub(
        r'^> (\|.+\|[ \t]*)$', r'\1', markdown_content, flags=_re.MULTILINE
    )
    # Insert blank line before a table that immediately follows a blockquote
    # line — otherwise the markdown parser treats the table as blockquote text.
    markdown_content = _re.sub(
        r'(> [^\n]+\n)(\|)', r'\1\n\2', markdown_content
    )
    markdown_content = _re.sub(
        r'(\|[^\n]*\|)[ \t]*\n\n(\|[^\n]*\|)', r'\1\n\2', markdown_content
    )

    # ── Server-side MD → HTML (Python markdown with GFM table support) ──
    html_body = markdown.markdown(
        markdown_content,
        extensions=['tables', 'fenced_code', 'toc']
    )

    # ── Post-process: preserve line breaks inside blockquote paragraphs ──
    # Python markdown collapses multi-line blockquote content into a single
    # <p>, but HTML treats newlines as whitespace — list items and multi-line
    # summaries run together into one unreadable blob.  Inject <br> so each
    # logical line inside the blockquote paragraph renders on its own line.
    def _fix_blockquote_newlines(m):
        inner = m.group(1).strip('\n')
        inner = _re.sub(r'\n', '<br>\n', inner)
        return '<blockquote>\n<p>' + inner + '</p>\n</blockquote>'

    html_body = _re.sub(
        r'<blockquote>\s*<p>(.*?)</p>\s*</blockquote>',
        _fix_blockquote_newlines,
        html_body,
        flags=_re.DOTALL
    )

    # ── Post-process: image handling (same logic as the frontend) ──
    # 1. /maps/ static map images → collapsible <details>
    html_body = _re.sub(
        r'<img alt="([^"]*)" src="(/maps/[^"]+)"\s*/?>',
        r'<details class="map-collapsible"><summary>🗺️ 点击展开地图</summary>'
        r'<img alt="\1" src="\2" style="width:100%;border-radius:0 0 0.5rem 0.5rem;margin:0;display:block;"></details>',
        html_body
    )
    # 2. Google Static Maps API images → collapsible <details>
    html_body = _re.sub(
        r'<img alt="([^"]*)" src="([^"]*(?:maps\.googleapis\.com|staticmap)[^"]*)"\s*/?>',
        r'<details class="map-collapsible"><summary>🗺️ 点击展开地图</summary>'
        r'<img alt="\1" src="\2" style="width:100%;border-radius:0 0 0.5rem 0.5rem;margin:0;display:block;"></details>',
        html_body
    )
    # 3. OpenStreetMap embed URLs → clickable link
    html_body = _re.sub(
        r'<img alt="([^"]*)" src="([^"]*openstreetmap\.org[^"]*embed\.html[^"]*)"\s*/?>',
        r'<a href="\2" target="_blank" rel="noopener noreferrer" '
        r'style="display:inline-block;padding:0.25rem 0.75rem;background:#f0f9ff;color:#2563eb;'
        r'border:1px solid #bae6fd;border-radius:0.375rem;font-size:0.8125rem;text-decoration:none;">'
        r'🗺️ 点击查看交互地图</a>',
        html_body
    )
    # 4. Google Maps embed URLs → clickable link
    html_body = _re.sub(
        r'<img alt="([^"]*)" src="([^"]*maps\.google\.com[^"]*output=embed[^"]*)"\s*/?>',
        r'<a href="\2" target="_blank" rel="noopener noreferrer" '
        r'style="display:inline-block;padding:0.25rem 0.75rem;background:#f0f9ff;color:#2563eb;'
        r'border:1px solid #bae6fd;border-radius:0.375rem;font-size:0.8125rem;text-decoration:none;">'
        r'🗺️ 点击查看交互地图</a>',
        html_body
    )
    # 5. Remaining images → add poi-image class for uniform sizing
    html_body = _re.sub(
        r'<img alt="([^"]*)" src="(?!\/maps\/)([^"]+)"\s*/?>',
        r'<img alt="\1" src="\2" class="poi-image">',
        html_body
    )

    # ── Self-contained HTML page (no CDN, no JS rendering dependency) ──
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Travel Plan</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                         "Microsoft YaHei", "Noto Sans SC", sans-serif;
            line-height: 1.75; color: #333;
            background: linear-gradient(135deg, #eff6ff 0%, #faf5ff 50%, #eff6ff 100%);
            min-height: 100vh;
        }}
        .container {{ max-width: 800px; margin: 0 auto; padding: 40px 20px; }}
        .card {{
            background: #fff; border-radius: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1);
            border: 1px solid #dbeafe; padding: 32px;
        }}
        .header {{
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 24px; flex-wrap: wrap; gap: 12px;
        }}
        .header h1 {{ font-size: 1.5rem; font-weight: 700; color: #1e3a5f; }}
        .btn-print {{
            font-size: 0.875rem; background: #2563eb; color: #fff;
            padding: 8px 16px; border-radius: 0.5rem; border: none;
            cursor: pointer; font-weight: 500; transition: background 0.2s;
        }}
        .btn-print:hover {{ background: #1d4ed8; }}

        /* ── Prose / Markdown body ── */
        .prose {{ color: #374151; word-wrap: break-word; }}
        .prose h1 {{ font-size: 1.75rem; font-weight: 700; margin: 1.5rem 0 0.75rem;
                     color: #1a365d; border-bottom: 2px solid #3182ce; padding-bottom: 8px; }}
        .prose h2 {{ font-size: 1.375rem; font-weight: 600; margin: 1.25rem 0 0.5rem; color: #2c5282; }}
        .prose h3 {{ font-size: 1.125rem; font-weight: 600; margin: 1rem 0 0.5rem; color: #2d3748; }}
        .prose h4 {{ font-size: 1rem; font-weight: 600; margin: 0.75rem 0 0.25rem; color: #4a5568; }}
        .prose p  {{ margin-bottom: 0.75rem; line-height: 1.8; }}
        .prose ul, .prose ol {{ padding-left: 1.5rem; margin-bottom: 0.75rem; }}
        .prose li {{ margin-bottom: 0.25rem; }}
        .prose strong {{ font-weight: 600; }}
        .prose a {{ color: #2563eb; text-decoration: underline; }}
        .prose img {{ max-width: 100%; border-radius: 0.5rem; margin: 0.5rem 0; }}
        .prose img.poi-image {{ width: 100%; height: 280px; object-fit: cover;
                                 border-radius: 0.5rem; margin: 0.5rem 0; }}
        .prose table {{ border-collapse: collapse; width: 100%; margin: 0.75rem 0; font-size: 0.875rem; }}
        .prose th, .prose td {{ border: 1px solid #e5e7eb; padding: 0.5rem 0.75rem; text-align: left; }}
        .prose th {{ background-color: #f0f9ff; font-weight: 600; }}
        .prose tr:nth-child(even) {{ background-color: #f9fafb; }}
        .prose hr {{ border: none; border-top: 1px solid #e5e7eb; margin: 1.25rem 0; }}
        .prose blockquote {{
            border-left: 4px solid #93c5fd; padding: 0.5rem 1rem;
            color: #4b5563; margin: 0.75rem 0; background: #f0f9ff;
            border-radius: 0 4px 4px 0;
        }}
        .prose code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 0.8125rem; }}
        .prose pre {{
            background: #1e293b; color: #e2e8f0; padding: 1rem;
            border-radius: 0.5rem; overflow-x: auto; margin: 0.75rem 0;
        }}

        /* ── Collapsible maps ── */
        details.map-collapsible {{
            margin: 0.5rem 0; border: 1px solid #e5e7eb;
            border-radius: 0.5rem; overflow: hidden;
        }}
        details.map-collapsible > summary {{
            cursor: pointer; padding: 0.5rem 0.75rem; font-size: 0.8125rem;
            color: #2563eb; background: #f0f9ff; list-style: none;
            display: flex; align-items: center; gap: 0.25rem; user-select: none;
        }}
        details.map-collapsible > summary::-webkit-details-marker {{ display: none; }}
        details.map-collapsible > summary::before {{
            content: '▶'; font-size: 0.7rem; transition: transform 0.2s ease;
        }}
        details.map-collapsible[open] > summary::before {{ transform: rotate(90deg); }}
        details.map-collapsible > img {{
            margin: 0; border-radius: 0 0 0.5rem 0.5rem; display: block; width: 100%;
        }}

        /* ── Print ── */
        @media print {{
            body {{ background: #fff; }}
            .card {{ box-shadow: none; border: none; padding: 0; }}
            .btn-print {{ display: none; }}
            details.map-collapsible {{ display: block; }}
            details.map-collapsible > summary {{ display: none; }}
            details.map-collapsible > img {{ display: block !important; max-width: 100%; height: auto; }}
        }}

        .footer {{ text-align: center; font-size: 0.75rem; color: #9ca3af; margin-top: 24px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="header">
                <h1>🗺️ Travel Plan</h1>
                <button class="btn-print" onclick="window.print()">📥 Download PDF</button>
            </div>
            <div class="prose">
                {html_body}
            </div>
        </div>
        <p class="footer">Generated by AI Travel Planner</p>
    </div>
    <script>
        // Expand collapsible maps before printing, restore after.
        (function() {{
            var _printDetails = [];
            window.addEventListener('beforeprint', function() {{
                _printDetails = [];
                document.querySelectorAll('details.map-collapsible:not([open])').forEach(function(d) {{
                    d.setAttribute('open', '');
                    _printDetails.push(d);
                }});
            }});
            window.addEventListener('afterprint', function() {{
                _printDetails.forEach(function(d) {{ d.removeAttribute('open'); }});
                _printDetails = [];
            }});
        }})();
    </script>
</body>
</html>"""


def _save_report_files(session_id: str, markdown_content: str):
    """Save report as both .md and standalone .html files in REPORTS_DIR.

    This is called when a report finishes generating so that the download and
    share endpoints can serve the files directly from disk.
    """
    # Save markdown file.
    md_path = os.path.join(REPORTS_DIR, f"{session_id}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    # Generate and save standalone HTML.
    html_content = _generate_share_html(markdown_content, session_id)
    html_path = os.path.join(REPORTS_DIR, f"{session_id}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    _session_log(session_id, "📄 REPORT_SAVED",
                 f"MD: {md_path}\nHTML: {html_path}\nSize: {len(markdown_content):,} chars")


# ---------------------------------------------------------------------------
# Lifespan hooks
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Travel Planning API starting...")
    yield
    # Shutdown
    print("🛑 Travel Planning API shutting down...")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Travel Planning Agent API",
    description="Web API for the LangGraph-based travel planning agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "travel-planning-agent"}


# ---------------------------------------------------------------------------
# Session lifecycle endpoints
# ---------------------------------------------------------------------------
@app.post("/api/sessions", response_model=CreateSessionResponse)
async def create_session():
    """Create a new planning session."""
    session_id = await session_manager.create_session()
    return CreateSessionResponse(
        session_id=session_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/api/sessions/{session_id}/plan")
async def start_planning(session_id: str, request: PlanRequest):
    """Submit a travel query and start Phase 1 execution."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["status"] not in ("idle", "reviewing"):
        raise HTTPException(
            status_code=409,
            detail=f"Session is in '{session['status']}' state, cannot start planning",
        )

    # Transition to streaming phase 1.
    await session_manager.update_session(session_id, {"status": "streaming_phase1"})

    # Drain any stale events from a previous run.
    event_queue = session["event_queue"]
    while not event_queue.empty():
        try:
            event_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

    # Kick off graph execution in the background.
    initial_input = {"query": request.query}
    _session_log(session_id, "❓ USER_QUERY", request.query)
    thread_config = session["thread_config"]
    asyncio.create_task(
        stream_graph_execution(travel_app, thread_config, initial_input, event_queue)
    )

    return {
        "session_id": session_id,
        "status": "streaming_phase1",
        "stream_url": f"/api/sessions/{session_id}/stream",
    }


@app.get("/api/sessions/{session_id}/stream")
async def stream_events(session_id: str):
    """SSE endpoint for streaming graph execution events in real-time."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    event_queue = session["event_queue"]

    async def event_stream():
        # Pass a mutable dict so sse_event_generator can signal whether the
        # stream ended due to timeout (vs. a normal terminal event).
        result = {}
        last_event_type = None
        async for event_str in sse_event_generator(event_queue, result=result):
            # Track the last event type for post-stream status handling.
            if event_str.startswith("event: "):
                last_event_type = event_str.split("\n")[0].split("event: ")[1].strip()
            yield event_str

        # If the stream timed out, do NOT update the session status — the
        # background graph task may still be running and will set the
        # appropriate status (completed / reviewing) when it finishes.
        # Updating to 'completed' here would be premature and incorrect.
        if result.get("timed_out"):
            return

        # If the stream ended due to an error, do NOT transition to reviewing —
        # the user should see the error message, not an empty review screen.
        if last_event_type == "error":
            return

        # After the stream ends normally, transition session status.
        # If execution completed (AutoReplan shortcut in Phase 1, or normal
        # Phase 2), the report is ready — go directly to 'completed'.
        # Otherwise, the graph paused at interrupt → 'reviewing'.
        current_session = await session_manager.get_session(session_id)
        if last_event_type == "execution_complete":
            await session_manager.update_session(session_id, {"status": "completed"})
        elif current_session and current_session["status"] == "streaming_phase1":
            await session_manager.update_session(session_id, {"status": "reviewing"})
        elif current_session and current_session["status"] == "streaming_phase2":
            await session_manager.update_session(session_id, {"status": "completed"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/sessions/{session_id}/state", response_model=SessionStateResponse)
async def get_session_state(session_id: str):
    """Get current session state snapshot."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Pull live state from LangGraph checkpoint.
    thread_config = session["thread_config"]
    try:
        state = await asyncio.to_thread(travel_app.get_state, thread_config)
        values = state.values if state else {}
    except Exception:
        values = {}

    # Convert POI/hotel costs from USD to the user's display currency so
    # the numbers match the currency_symbol shown in the frontend.
    raw_itinerary = values.get("daily_itinerary")
    exchange_rate = values.get("exchange_rate") or 1.0
    converted_itinerary = copy.deepcopy(raw_itinerary) if raw_itinerary else None
    if converted_itinerary:
        for day in converted_itinerary:
            for poi in day.get("attractions", []):
                if poi.get("cost") is not None:
                    poi["cost"] = round(poi["cost"] * exchange_rate, 1)
            for poi in day.get("dining", []):
                if poi.get("cost") is not None:
                    poi["cost"] = round(poi["cost"] * exchange_rate, 1)
            hotel = day.get("hotel", {})
            if hotel and hotel.get("price_per_night") is not None:
                hotel["price_per_night"] = round(
                    hotel["price_per_night"] * exchange_rate, 1
                )

    return SessionStateResponse(
        session_id=session_id,
        status=session["status"],
        daily_itinerary=converted_itinerary,
        currency_symbol=values.get("currency_symbol"),
        exchange_rate=values.get("exchange_rate"),
        is_chinese=values.get("is_chinese"),
    )


@app.post("/api/sessions/{session_id}/feedback")
async def submit_feedback(session_id: str, request: FeedbackRequest):
    """Submit user feedback and trigger re-planning."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["status"] != "reviewing":
        raise HTTPException(status_code=409, detail="Session is not in reviewing state")

    thread_config = session["thread_config"]
    event_queue = session["event_queue"]

    # Drain stale events.
    while not event_queue.empty():
        try:
            event_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

    async def _process_feedback():
        """Run the ReplanAgent to autonomously revise the itinerary, then re-stream.

        Instead of a single LLM call that generates an instruction for the
        Recommendation node, we now run an autonomous tool-use agent. The agent:
          1. Inspects the current plan
          2. Browses the POI pool for alternatives
          3. Searches Google Places for user-mentioned locations
          4. Directly modifies the daily itinerary (add/remove/swap/reorder)
          5. Updates user preferences
          6. Checks constraints (budget, structure, must-visit coverage)
          7. Iterates until satisfied, then finalizes

        The agent's output (modified itinerary + prefs) is injected directly
        into the graph state, bypassing the Recommendation node entirely.
        """
        try:
            from src.agents.replan_agent import run_replan_agent
            from src.agents.utils import llm_client
            from src.config import GOOGLE_MAPS_API_KEY
            import re

            # Snapshot current graph state.
            current_state = await asyncio.to_thread(travel_app.get_state, thread_config)
            state_values = current_state.values

            # Detect query language for Google Places API locale.
            query = state_values.get("query", "")
            is_chinese = bool(re.search(r"[\u4e00-\u9fff]", query))
            api_language = "zh-CN" if is_chinese else "en"

            feedback_text = request.feedback
            _session_log(session_id, "💬 USER_FEEDBACK", feedback_text)

            # ── Snapshot itinerary BEFORE replan (for diff logging) ──
            def _itinerary_poi_names(itinerary):
                """Extract per-day POI name sets from a daily itinerary."""
                result = {}
                for day in (itinerary or []):
                    dn = day.get("day", "?")
                    attrs = {a.get("name", "?") for a in day.get("attractions", [])}
                    dines = {d.get("name", "?") for d in day.get("dining", [])}
                    hotel = day.get("hotel", {})
                    h = hotel.get("name", "?") if isinstance(hotel, dict) else "?"
                    result[dn] = {"attractions": attrs, "dining": dines, "hotel": h}
                return result

            before_plan = _itinerary_poi_names(state_values.get("daily_itinerary"))

            logger.info(
                f"ReplanAgent starting for session {session_id} "
                f"(feedback: {feedback_text[:100]}...)"
            )

            # ── Run the autonomous replan agent ──────────────────────
            agent_result = await asyncio.to_thread(
                run_replan_agent,
                feedback_text=feedback_text,
                state_values=state_values,
                llm_client=llm_client,
                api_language=api_language,
                max_iterations=20,
            )

            modified_itinerary = agent_result["daily_itinerary"]
            updated_prefs = agent_result["user_preferences"]
            new_pois = agent_result["new_pois"]
            agent_summary = agent_result["summary"]
            iterations = agent_result["iterations"]

            logger.info(
                f"ReplanAgent completed: {iterations} iterations, "
                f"{len(new_pois)} new POIs, summary: {agent_summary[:100]}"
            )
            _session_log(
                session_id,
                "🔧 REPLAN_RESULT",
                f"Iterations: {iterations}\n"
                f"New POIs discovered: {len(new_pois)}\n"
                f"Summary: {agent_summary}"
            )

            # ── Log itinerary diff (before vs after replan) ──
            after_plan = _itinerary_poi_names(modified_itinerary)
            diff_lines = []
            all_days = sorted(set(list(before_plan.keys()) + list(after_plan.keys())))
            for dn in all_days:
                b = before_plan.get(dn, {"attractions": set(), "dining": set(), "hotel": "?"})
                a = after_plan.get(dn, {"attractions": set(), "dining": set(), "hotel": "?"})
                added_a = a["attractions"] - b["attractions"]
                removed_a = b["attractions"] - a["attractions"]
                added_d = a["dining"] - b["dining"]
                removed_d = b["dining"] - a["dining"]
                hotel_changed = b["hotel"] != a["hotel"]
                if added_a or removed_a or added_d or removed_d or hotel_changed:
                    diff_lines.append(f"Day {dn}:")
                    if removed_a:
                        diff_lines.append(f"  ➖ Attractions: {', '.join(sorted(removed_a))}")
                    if added_a:
                        diff_lines.append(f"  ➕ Attractions: {', '.join(sorted(added_a))}")
                    if removed_d:
                        diff_lines.append(f"  ➖ Dining: {', '.join(sorted(removed_d))}")
                    if added_d:
                        diff_lines.append(f"  ➕ Dining: {', '.join(sorted(added_d))}")
                    if hotel_changed:
                        diff_lines.append(f"  🏨 Hotel: {b['hotel']} → {a['hotel']}")
            if diff_lines:
                _session_log(session_id, "📊 ITINERARY_CHANGES", "\n".join(diff_lines))
            else:
                _session_log(session_id, "📊 ITINERARY_CHANGES", "(no changes)")

            # ── Build flat recommended_pois from the modified itinerary ──
            all_recommended = []
            for day in modified_itinerary:
                all_recommended.extend(day.get("attractions", []))
                all_recommended.extend(day.get("dining", []))

            # ── Merge new POIs into the raw_knowledge pool ──────────
            raw_knowledge = dict(state_values.get("raw_knowledge", {}))
            if new_pois:
                existing_pois = raw_knowledge.get("pois", [])
                existing_names = {p.get("name", "").lower() for p in existing_pois}
                for poi in new_pois:
                    if poi.get("name", "").lower() not in existing_names:
                        existing_pois.append(poi)
                raw_knowledge["pois"] = existing_pois

            # ── Update graph state ───────────────────────────────────
            # We update as if the Recommendation node just completed, so
            # when the graph resumes it flows through Routing (recalculate
            # transport) → Critic (force-approves due to replan_user_approved)
            # → UserReview → [INTERRUPT before SynthEnrich].
            # This bypasses full LLM re-generation since the agent already
            # produced the final itinerary.
            state_update: dict[str, Any] = {
                "daily_itinerary": modified_itinerary,
                "recommended_pois": all_recommended,
                "user_preferences": updated_prefs,
                "user_feedback": agent_summary,
                "raw_knowledge": raw_knowledge,
                "replan_user_approved": True,
            }

            await asyncio.to_thread(
                travel_app.update_state,
                thread_config,
                state_update,
                as_node="Recommendation",
            )

            # Re-stream through Routing → Critic → UserReview → [INTERRUPT]
            await session_manager.update_session(session_id, {"status": "streaming_phase1"})
            await stream_graph_execution(travel_app, thread_config, None, event_queue)

        except Exception as e:
            await event_queue.put({
                "type": "error",
                "error_message": f"Replan agent failed: {str(e)}",
                "recoverable": True,
            })

    asyncio.create_task(_process_feedback())

    return {
        "session_id": session_id,
        "status": "processing_feedback",
        "stream_url": f"/api/sessions/{session_id}/stream",
    }


@app.post("/api/sessions/{session_id}/confirm")
async def confirm_plan(session_id: str):
    """Confirm the current plan and start Phase 2 (SynthEnrich → Synthesizer)."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["status"] not in ("reviewing", "streaming_phase2"):
        raise HTTPException(status_code=409, detail=f"Session is not in reviewing state (current: {session['status']})")

    # If already streaming_phase2, the user likely hit an error and is retrying.
    is_retry = session["status"] == "streaming_phase2"

    # Prevent duplicate background tasks on retry — if a previous
    # _resume_execution is still running, reject the retry request.
    if is_retry and session.get("background_task_running"):
        raise HTTPException(
            status_code=409,
            detail="A background execution task is still running. Please wait for it to complete.",
        )

    thread_config = session["thread_config"]
    event_queue = session["event_queue"]

    # Drain stale events.
    while not event_queue.empty():
        try:
            event_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

    await session_manager.update_session(session_id, {
        "status": "streaming_phase2",
        "background_task_running": True,
    })

    async def _resume_execution():
        """Resume graph from the SynthEnrich interrupt through to completion.

        After user confirmation, only SynthEnrich and Synthesizer remain —
        no nodes that modify the itinerary exist past the interrupt point,
        so the confirmed plan reaches the final report unchanged.
        """
        try:
            while True:
                # Passing None as input resumes from the SynthEnrich checkpoint.
                # Only SynthEnrich → Synthesizer → END remain; no replan possible.
                await stream_graph_execution(travel_app, thread_config, None, event_queue, emit_interrupt=False)

                # Graph should complete on first pass — only SynthEnrich and
                # Synthesizer remain after user confirmation.
                state = await asyncio.to_thread(travel_app.get_state, thread_config)

                if not state.next:
                    # Graph ran to completion — persist the final report.
                    break

                # If the graph paused again (shouldn't happen in the new
                # structure), auto-resume as a safety net.

            # Persist final report.
            state = await asyncio.to_thread(travel_app.get_state, thread_config)
            final_report = state.values.get("final_itinerary", "")
            await session_manager.update_session(session_id, {
                "status": "completed",
                "final_report": final_report,
                "background_task_running": False,
            })
    
            # Save .md and .html files for download / share endpoints.
            if final_report:
                try:
                    _save_report_files(session_id, final_report)
                except Exception as e:
                    print(f"\u26a0\ufe0f Failed to save report files for {session_id}: {e}")
        except Exception as e:
            # Restore session to reviewing so the user can retry confirm.
            await session_manager.update_session(session_id, {
                "status": "reviewing",
                "background_task_running": False,
            })
            await event_queue.put({
                "type": "error",
                "error_message": f"Execution failed: {str(e)}",
                "recoverable": False,
            })

    asyncio.create_task(_resume_execution())

    return {
        "session_id": session_id,
        "status": "streaming_phase2",
        "stream_url": f"/api/sessions/{session_id}/stream",
    }


@app.get("/api/sessions/{session_id}/report", response_model=ReportResponse)
async def get_report(session_id: str):
    """Get the final travel plan report."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["status"] != "completed":
        raise HTTPException(status_code=409, detail="Planning is not yet completed")

    # Prefer live graph state; fall back to the cached copy in the session.
    thread_config = session["thread_config"]
    state = await asyncio.to_thread(travel_app.get_state, thread_config)
    report = state.values.get("final_itinerary", "") if state else ""
    if not report:
        report = session.get("final_report", "")

    # Fallback: if the report files don't exist yet but we have the
    # report content in the session / graph state, persist them now so the
    # download and share endpoints work even for sessions that were active
    # across a server restart or that predate this feature.
    if report:
        html_path = os.path.join(REPORTS_DIR, f"{session_id}.html")
        if not os.path.exists(html_path):
            try:
                _save_report_files(session_id, report)
            except Exception as e:
                print(f"⚠️ Failed to save report files for {session_id}: {e}")

    return ReportResponse(
        session_id=session_id,
        report=report,
        status="completed",
    )


# ---------------------------------------------------------------------------
# Report download & share endpoints
# ---------------------------------------------------------------------------
@app.get("/api/sessions/{session_id}/download")
async def download_report(session_id: str):
    """Download the final travel report as a PDF file."""
    md_path = os.path.join(REPORTS_DIR, f"{session_id}.md")
    if not os.path.exists(md_path):
        raise HTTPException(status_code=404, detail="Report not found")

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'toc', 'nl2br']
    )

    # Wrap in a styled HTML document for PDF rendering
    styled_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
    line-height: 1.8;
    max-width: 800px;
    margin: 0 auto;
    padding: 40px 30px;
    color: #333;
    font-size: 14px;
  }}
  h1 {{ color: #1a365d; border-bottom: 2px solid #3182ce; padding-bottom: 10px; font-size: 24px; }}
  h2 {{ color: #2c5282; margin-top: 30px; font-size: 20px; }}
  h3 {{ color: #2d3748; font-size: 16px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 8px 12px; text-align: left; }}
  th {{ background-color: #ebf8ff; font-weight: 600; }}
  tr:nth-child(even) {{ background-color: #f7fafc; }}
  blockquote {{ border-left: 4px solid #3182ce; margin: 15px 0; padding: 10px 20px; background: #ebf8ff; }}
  img {{ max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0; }}
  code {{ background: #edf2f7; padding: 2px 6px; border-radius: 4px; font-size: 13px; }}
  hr {{ border: none; border-top: 1px solid #e2e8f0; margin: 25px 0; }}
  ul, ol {{ padding-left: 20px; }}
  li {{ margin-bottom: 5px; }}
  a {{ color: #3182ce; text-decoration: none; }}
</style>
</head>
<body>
{html_content}
</body>
</html>"""

    # Import WeasyPrint lazily — raises a clear HTTP 500 if system libs are missing
    try:
        from weasyprint import HTML
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "PDF export is unavailable: WeasyPrint system libraries are not installed. "
                "Please install libpango and libcairo (e.g. `apt install libpango1.0-dev libcairo2-dev` "
                "on Debian/Ubuntu, or `brew install pango cairo` on macOS). "
                f"Original error: {e}"
            ),
        )

    # Generate PDF — skip images that fail to load (403, timeout, etc.)
    import logging as _logging
    _wp_logger = _logging.getLogger('weasyprint')
    _wp_logger.setLevel(_logging.ERROR)

    def _url_fetcher(url, timeout=10, ssl_context=None):
        """Custom URL fetcher that gracefully handles failed image loads."""
        from weasyprint import default_url_fetcher
        try:
            return default_url_fetcher(url, timeout=timeout)
        except Exception:
            # Return a 1x1 transparent PNG placeholder for failed images
            return {
                'string': b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82',
                'mime_type': 'image/png',
            }

    pdf_path = os.path.join(tempfile.gettempdir(), f"travel-plan-{session_id[:8]}.pdf")
    HTML(string=styled_html, url_fetcher=_url_fetcher).write_pdf(pdf_path)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"travel-plan-{session_id[:8]}.pdf",
    )


@app.get("/share/{session_id}")
async def share_report(session_id: str):
    """Serve a permanent shareable standalone HTML page for the report.

    Reads the saved .md file and regenerates the HTML on the fly so the
    page always uses the latest share template (fixing old broken links).
    """
    md_path = os.path.join(REPORTS_DIR, f"{session_id}.md")
    if not os.path.exists(md_path):
        # Fallback: try reading the old .html file directly
        html_path = os.path.join(REPORTS_DIR, f"{session_id}.html")
        if os.path.exists(html_path):
            return FileResponse(html_path, media_type="text/html")
        raise HTTPException(status_code=404, detail="Report not found")

    with open(md_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    html_content = _generate_share_html(markdown_content, session_id)
    return HTMLResponse(content=html_content)


# ---------------------------------------------------------------------------
# Static frontend file serving
# ---------------------------------------------------------------------------
# Ensure correct MIME types for ES modules and CSS — browsers refuse to
# execute <script type="module"> files unless the Content-Type is
# application/javascript or text/javascript.  Python's mimetypes module may
# not include these mappings on every platform, so we register them
# explicitly before mounting StaticFiles.
import mimetypes

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("text/html", ".html")

# Mount the pure-HTML/JS frontend so that visiting http://localhost:8000/
# serves the single-page application directly — no separate Node.js server
# is required.
_frontend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"
)

app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")

# Mount the cached static map images directory so that <img src="/maps/...">
# URLs can serve the locally cached Google Static Maps images.
_maps_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "maps"
)
os.makedirs(_maps_dir, exist_ok=True)
app.mount("/maps", StaticFiles(directory=_maps_dir), name="maps")


@app.get("/")
async def serve_index():
    """Serve the frontend single-page application."""
    return FileResponse(os.path.join(_frontend_dir, "index.html"))
