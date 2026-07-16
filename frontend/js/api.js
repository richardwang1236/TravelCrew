/**
 * API client — all fetch calls to the FastAPI backend.
 * Uses same-origin (window.location.origin) since the backend
 * serves both the API and the static frontend files.
 */

const API_BASE = window.location.origin;

/** Default fetch timeout in milliseconds. */
const FETCH_TIMEOUT_MS = 30000;

/**
 * Fetch wrapper with timeout support.
 * @param {string} url
 * @param {RequestInit} [options]
 * @param {number} [timeoutMs]
 * @returns {Promise<Response>}
 */
async function fetchWithTimeout(url, options = {}, timeoutMs = FETCH_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    return res;
  } catch (e) {
    if (e.name === 'AbortError') {
      throw new Error('Request timed out. Please check your network and try again.');
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Create a new planning session.
 * @returns {Promise<string>} session_id
 */
export async function createSession() {
  const res = await fetchWithTimeout(`${API_BASE}/api/sessions`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to create session');
  const data = await res.json();
  return data.session_id;
}

/**
 * Submit a travel query and start Phase 1 execution.
 * @param {string} sessionId
 * @param {string} query
 */
export async function startPlanning(sessionId, query) {
  const res = await fetchWithTimeout(`${API_BASE}/api/sessions/${sessionId}/plan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) throw new Error('Failed to start planning');
}

/**
 * Submit user feedback and trigger re-planning.
 * @param {string} sessionId
 * @param {string} feedback
 */
export async function submitFeedback(sessionId, feedback) {
  const res = await fetchWithTimeout(`${API_BASE}/api/sessions/${sessionId}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ feedback }),
  });
  if (!res.ok) throw new Error('Failed to submit feedback');
}

/**
 * Confirm the current plan and start Phase 2 (Routing → Critic → Synthesizer).
 * @param {string} sessionId
 */
export async function confirmPlan(sessionId) {
  const res = await fetchWithTimeout(`${API_BASE}/api/sessions/${sessionId}/confirm`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error('Failed to confirm plan');
}

/**
 * Get the final travel plan report (Markdown string).
 * @param {string} sessionId
 * @returns {Promise<string>} Markdown report
 */
export async function getReport(sessionId) {
  const res = await fetchWithTimeout(`${API_BASE}/api/sessions/${sessionId}/report`);
  if (!res.ok) throw new Error('Failed to get report');
  const data = await res.json();
  return data.report;
}

/**
 * Get the current session state snapshot.
 * @param {string} sessionId
 */
export async function getSessionState(sessionId) {
  const res = await fetchWithTimeout(`${API_BASE}/api/sessions/${sessionId}/state`);
  if (!res.ok) throw new Error('Failed to get session state');
  return res.json();
}

/**
 * Build the SSE stream URL for a given session.
 * @param {string} sessionId
 * @returns {string}
 */
export function getStreamUrl(sessionId) {
  return `${API_BASE}/api/sessions/${sessionId}/stream`;
}
