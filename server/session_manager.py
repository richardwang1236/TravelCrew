"""Session manager with per-session granular locking.

Uses a segmented locking strategy to avoid contention between unrelated
sessions in high-concurrency scenarios:

- ``_dict_lock`` protects structural dict changes (create / delete / cleanup
  operations that add or remove keys from ``_sessions``).
- ``_session_locks`` stores one ``asyncio.Lock`` per active session_id,
  allowing concurrent ``get_session`` and ``update_session`` calls on
  *different* sessions to proceed without blocking each other.

All operations are O(1) dict lookups, so locking overhead is negligible.
"""

import asyncio
import time
import uuid
from typing import Dict, Optional, Any


class SessionManager:
    """Manage concurrent user sessions with TTL cleanup and per-session locking."""

    def __init__(self, ttl_seconds: int = 3600, max_sessions: int = 100):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._dict_lock = asyncio.Lock()  # protects _sessions key set (add/remove)
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions

    async def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Return (or lazily create) the per-session lock.

        Must be called while NOT holding ``_dict_lock``, because
        acquiring ``_dict_lock`` inside a per-session lock would deadlock.
        """
        lock = self._session_locks.get(session_id)
        if lock is None:
            async with self._dict_lock:
                # Double-check after acquiring dict_lock — another coroutine
                # may have created the lock between our get() and here.
                lock = self._session_locks.get(session_id)
                if lock is None:
                    lock = self._session_locks[session_id] = asyncio.Lock()
        return lock

    async def create_session(self) -> str:
        """Create a new session; returns session_id."""
        session_id = str(uuid.uuid4())
        async with self._dict_lock:
            # Cleanup expired sessions if at capacity
            if len(self._sessions) >= self.max_sessions:
                await self._cleanup_expired()
            self._sessions[session_id] = {
                "thread_config": {"configurable": {"thread_id": session_id}},
                "status": "idle",
                "created_at": time.time(),
                "last_activity": time.time(),
                "event_queue": asyncio.Queue(),
                "state_snapshot": None,
                "final_report": None,
            }
        return session_id

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return session data or None if not found.

        Uses per-session lock — only blocks concurrent operations on the
        *same* session_id, not other sessions.
        """
        lock = await self._get_session_lock(session_id)
        async with lock:
            session = self._sessions.get(session_id)
            if session:
                session["last_activity"] = time.time()
            return session

    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        """Apply partial updates to an existing session.

        Uses per-session lock — only blocks concurrent operations on the
        *same* session_id, not other sessions.
        """
        lock = await self._get_session_lock(session_id)
        async with lock:
            if session_id in self._sessions:
                self._sessions[session_id].update(updates)
                self._sessions[session_id]["last_activity"] = time.time()

    async def delete_session(self, session_id: str) -> None:
        """Remove a session and its lock."""
        async with self._dict_lock:
            self._sessions.pop(session_id, None)
            self._session_locks.pop(session_id, None)

    async def _cleanup_expired(self) -> int:
        """Remove expired sessions from the store.

        Must be called while holding ``_dict_lock``.
        """
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s["last_activity"] > self.ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]
            self._session_locks.pop(sid, None)
        return len(expired)
