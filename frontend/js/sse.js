/**
 * SSE client — manages an EventSource connection to the backend
 * streaming endpoint, dispatching typed events to registered callbacks.
 *
 * Event types:
 *   - node_completed    A graph node finished executing
 *   - progress_log      A sanitized real-time progress message from a node
 *   - interrupt         Graph paused for user review
 *   - execution_complete Graph ran to completion (terminal)
 *   - error             An error occurred (terminal)
 *   - audit_warning     A non-fatal quality warning from the Critic
 *   - timeout           Stream timed out (terminal)
 *
 * Terminal events (execution_complete, error, interrupt, timeout)
 * automatically disconnect the EventSource.
 */

import { getStreamUrl } from './api.js?v=20260715b';

// Event types we listen for on the EventSource.
const EVENT_TYPES = [
  'node_completed',
  'progress_log',
  'interrupt',
  'execution_complete',
  'error',
  'audit_warning',
  'markdown_chunk',
  'timeout',
];

// Events that signal the end of the stream.
const TERMINAL_EVENTS = new Set(['execution_complete', 'error', 'interrupt', 'timeout']);

export class SSEClient {
  /**
   * @param {string} sessionId
   */
  constructor(sessionId) {
    this.sessionId = sessionId;
    this.eventSource = null;
    this._callback = null;
    this._onClose = null;
    this.connected = false;
    this._errorCount = 0;
    this._connectTimer = null;
  }

  /**
   * Open the EventSource connection and register listeners for all event types.
   */
  connect() {
    // Close any existing connection first.
    this.disconnect();

    this._errorCount = 0;

    const url = getStreamUrl(this.sessionId);
    const es = new EventSource(url);
    this.eventSource = es;

    // ── Connection timeout: if no successful connection within 15 s, report error ──
    this._connectTimer = setTimeout(() => {
      if (!this.connected && this._callback) {
        console.error('[SSE] connection timeout — no response from server within 15 s');
        this._callback({ type: 'connect_error', error_message: 'Connection timed out. Please try again.' });
        this.disconnect();
      }
    }, 15000);

    es.onopen = () => {
      this.connected = true;
      if (this._connectTimer) {
        clearTimeout(this._connectTimer);
        this._connectTimer = null;
      }
    };

    es.onerror = () => {
      // EventSource fires onerror on normal close too; only treat CLOSED as failure.
      this._errorCount++;
      if (es.readyState === EventSource.CLOSED) {
        console.error('[SSE] connection closed after ' + this._errorCount + ' error(s)');
        this.connected = false;
        if (this._connectTimer) {
          clearTimeout(this._connectTimer);
          this._connectTimer = null;
        }
        // Report connection failure to the app so the user sees an error.
        if (this._callback) {
          this._callback({
            type: 'connect_error',
            error_message: 'Failed to connect to server. Please check your network and try again.',
          });
        }
        this.disconnect();
      } else {
        // Still CONNECTING — EventSource will auto-retry.
        this.connected = false;
      }
    };

    // Register a listener for each known event type.
    for (const type of EVENT_TYPES) {
      es.addEventListener(type, (e) => {
        try {
          const data = JSON.parse(e.data);
          if (this._callback) {
            this._callback(data);
          }
          // Auto-disconnect on terminal events.
          if (TERMINAL_EVENTS.has(type)) {
            this.disconnect();
          }
        } catch (err) {
          console.error('Failed to parse SSE event:', err, e.data);
        }
      });
    }
  }

  /**
   * Close the EventSource connection if open.
   */
  disconnect() {
    if (this._connectTimer) {
      clearTimeout(this._connectTimer);
      this._connectTimer = null;
    }
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
      this.connected = false;
      if (this._onClose) {
        this._onClose();
      }
    }
  }

  /**
   * Register a single callback invoked for every SSE event.
   * @param {(event: object) => void} callback
   * @param {() => void} [onClose]  Optional callback when the connection closes.
   */
  onEvent(callback, onClose) {
    this._callback = callback;
    this._onClose = onClose || null;
  }
}
