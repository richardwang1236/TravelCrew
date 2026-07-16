import asyncio
import copy
import json
import threading
import time
from typing import AsyncGenerator, Optional

# ── Streaming context (per-session) ──────────────────────────────────────────
# Each session gets its own entry keyed by session_id, containing its event
# queue and event loop.  This prevents concurrent sessions from overwriting
# each other's streaming context.
#
# IMPORTANT: contextvars do NOT propagate through ThreadPoolExecutor.submit()
# which LangGraph uses internally to run nodes.  We use a session-keyed dict
# protected by a threading.Lock instead.  The session_id is passed explicitly
# to push_event() — nodes obtain it from the LangGraph config (thread_id).
_stream_ctxs: dict[str, dict] = {}
_stream_lock = threading.Lock()


def set_stream_context(session_id: str, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
    """Register the streaming context for a specific session.

    Called from ``stream_graph_execution`` before graph execution begins so
    that nodes (e.g. Synthesizer) can push real-time events via ``push_event``.
    """
    with _stream_lock:
        _stream_ctxs[session_id] = {
            "queue": queue,
            "loop": loop,
        }
    import logging
    _log = logging.getLogger(__name__)
    _log.info(f"[streaming] Context set for session {session_id}: loop={loop}, queue={'ok' if queue else 'missing'}")


def push_event(event: dict, session_id: Optional[str] = None) -> None:
    """Thread-safe helper: push an SSE event from a sync node to the async queue.

    Args:
        event:       The event dict to push onto the session's event queue.
        session_id:  The session this event belongs to.  Nodes obtain this
                     from the LangGraph config (``config['configurable']['thread_id']``).
                     If ``None`` or the session has no registered context,
                     the event is dropped with a warning.
    """
    with _stream_lock:
        ctx = _stream_ctxs.get(session_id, {}).copy() if session_id is not None else {}
    if ctx.get("loop") and ctx.get("queue"):
        ctx["loop"].call_soon_threadsafe(ctx["queue"].put_nowait, event)
    else:
        import logging
        _log = logging.getLogger(__name__)
        # ctx missing is normal when graph runs outside SSE (e.g., evaluation runner)
        _log.debug(
            f"[streaming] push_event: no stream ctx for session_id={session_id} "
            f"(loop={'ok' if ctx.get('loop') else 'missing'}, queue={'ok' if ctx.get('queue') else 'missing'})"
        )


def clear_stream_context(session_id: str) -> None:
    """Remove the streaming context for a session after graph execution ends."""
    with _stream_lock:
        _stream_ctxs.pop(session_id, None)


async def stream_graph_execution(
    travel_app, thread_config: dict, initial_input, event_queue: asyncio.Queue,
    emit_interrupt: bool = True
) -> None:
    """Run travel_app.stream() in a thread and push events to queue in real-time.

    Uses loop.call_soon_threadsafe() to push each graph event into the async
    queue the moment it is produced, so SSE consumers see node-by-node progress
    instead of receiving everything only after the graph finishes.
    """

    # Capture the running event loop BEFORE entering the thread, so the sync
    # callback can schedule work back onto it.
    loop = asyncio.get_running_loop()

    # Extract session_id from thread_config (thread_id == session_id).
    session_id = thread_config.get("configurable", {}).get("thread_id")

    def _run_sync():
        """Synchronous graph execution — runs inside a thread-pool worker.

        Each event emitted by travel_app.stream() is immediately forwarded to
        the asyncio event queue via call_soon_threadsafe, giving the SSE
        endpoint real-time per-node updates.
        """
        # Register the per-session streaming context so that nodes (e.g.
        # Synthesizer) can push streaming events via push_event(session_id=...).
        set_stream_context(session_id, event_queue, loop)

        try:
            for event in travel_app.stream(initial_input, config=thread_config):
                for node_name, node_output in event.items():
                    payload = {
                        "type": "node_completed",
                        "node": node_name,
                        "timestamp": time.time(),
                    }
                    # Thread-safe push: schedule put_nowait on the async loop.
                    loop.call_soon_threadsafe(event_queue.put_nowait, payload)

                    # Send sanitized progress-log messages produced by this node.
                    # Each message is forwarded as a separate ``progress_log`` SSE
                    # event so the frontend can display real-time activity updates.
                    logs = node_output.get("progress_logs", []) if isinstance(node_output, dict) else []
                    for msg in logs:
                        log_payload = {
                            "type": "progress_log",
                            "node": node_name,
                            "message": msg if isinstance(msg, dict) else {"zh": msg, "en": msg},
                            "timestamp": time.time(),
                        }
                        loop.call_soon_threadsafe(event_queue.put_nowait, log_payload)
        finally:
            # Clean up the per-session context so it doesn't linger.
            clear_stream_context(session_id)

    try:
        start = time.time()

        # Execute the blocking graph stream in a worker thread.
        # Real-time events are already being pushed via the callback above.
        await asyncio.to_thread(_run_sync)

        # After the stream finishes, inspect the checkpoint state to decide
        # whether the graph paused at an interrupt or ran to completion.
        state = await asyncio.to_thread(travel_app.get_state, thread_config)

        if state.next:
            if not emit_interrupt:
                # Phase 2 (post-confirm): graph paused at SynthEnrich interrupt
                # but we should NOT notify the frontend — the caller
                # (_resume_execution) will auto-resume. Just return silently.
                return

            # Graph paused at interrupt (before SynthEnrich) — surface display
            # data so the frontend can render the review screen.
            # Deep-copy the itinerary and convert POI/hotel costs from
            # USD to the user's display currency so the numbers match the
            # currency_symbol shown in the frontend review screen.
            raw_itinerary = state.values.get("daily_itinerary", [])
            exchange_rate = state.values.get("exchange_rate", 1.0)

            converted_itinerary = copy.deepcopy(raw_itinerary)
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

            display_data = {
                "daily_itinerary": converted_itinerary,
                "currency_symbol": state.values.get("currency_symbol", ""),
                "exchange_rate": exchange_rate,
                "is_chinese": state.values.get("is_chinese", False),
                "recommended_pois": state.values.get("recommended_pois", []),
            }
            await event_queue.put({
                "type": "interrupt",
                "next_node": state.next[0] if state.next else None,
                "display_data": display_data,
            })
        else:
            # Graph ran to completion (Phase 2: SynthEnrich → Synthesizer).
            await event_queue.put({
                "type": "execution_complete",
                "final_report": state.values.get("final_itinerary", ""),
                "duration_ms": int((time.time() - start) * 1000),
            })

    except Exception as e:
        import traceback
        await event_queue.put({
            "type": "error",
            "error_message": str(e),
            "error_traceback": traceback.format_exc(),
            "recoverable": True,
        })


async def sse_event_generator(
    event_queue: asyncio.Queue, timeout: float = 300,
    *, result: Optional[dict] = None,
) -> AsyncGenerator[str, None]:
    """Generate SSE-formatted event strings from the async queue.

    Sends a heartbeat comment every 30 seconds to keep the connection alive,
    and stops as soon as a terminal event (execution_complete / interrupt /
    error / timeout) is received or the idle timeout elapses (no events
    received for ``timeout`` seconds).

    Args:
        event_queue: The async queue from which events are consumed.
        timeout:     Maximum idle seconds (no events received) before the
                     stream is force-closed.
        result:      Optional mutable dict.  If provided, ``result['timed_out']``
                     is set to ``True`` when the stream ends due to timeout,
                     allowing the caller to distinguish timeout from normal
                     completion.
    """
    last_activity = time.time()
    last_heartbeat = time.time()

    while True:
        try:
            # Idle timeout guard — resets whenever an event is received.
            if time.time() - last_activity > timeout:
                if result is not None:
                    result["timed_out"] = True
                yield (
                    f"event: timeout\n"
                    f"data: {json.dumps({'type': 'timeout', 'message': 'Stream timeout'})}\n\n"
                )
                break

            # Heartbeat keep-alive every 30 seconds.
            if time.time() - last_heartbeat > 30:
                yield ": heartbeat\n\n"
                last_heartbeat = time.time()

            # Non-blocking wait for the next event (1 s poll interval so we
            # can still check the heartbeat / timeout conditions above).
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # Reset idle timer — we received an event.
            last_activity = time.time()

            # Serialise and yield as an SSE frame.
            event_type = event.get("type", "message")
            yield f"event: {event_type}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

            # Terminal events end the stream.
            if event_type in ("execution_complete", "interrupt", "error", "timeout"):
                break

        except Exception as e:
            yield (
                f"event: error\n"
                f"data: {json.dumps({'type': 'error', 'error_message': str(e)})}\n\n"
            )
            break
