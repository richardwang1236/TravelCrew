/**
 * Main application — state machine driven single-page app.
 *
 * Phase transitions:
 *   input             →  streaming_phase1  (user submits query)
 *   streaming_phase1  →  reviewing         (SSE interrupt event —
 *                          Critic approved or AutoReplan fixed, always HITL)
 *   reviewing         →  streaming_phase1  (user submits feedback → re-plan)
 *   reviewing         →  streaming_phase2  (user confirms → finalise)
 *   streaming_phase2  →  completed         (SSE execution_complete event)
 */

import * as api from './api.js?v=20260715b';
import { SSEClient } from './sse.js?v=20260715b';
import * as components from './components.js?v=20260715b';
import { t, toggleLang, getLang, setLang } from './i18n.js?v=20260715b';

console.log('[app.js] modules loaded successfully:', { api, SSEClient, components });

// ---- Application state ---------------------------------------------------

let state = {
  phase: 'input', // input | streaming_phase1 | reviewing | streaming_phase2 | completed
  sessionId: null,
  events: [],
  displayData: null,
  report: null,
  streamingReport: '',   // Accumulated markdown chunks during streaming
  streamingChars: 0,     // Total character count from backend
  loading: false,
  error: null,
};

// ---- Throttled render for streaming chunks --------------------------------
// When many markdown_chunk events arrive rapidly, we coalesce renders using
// requestAnimationFrame.  Once the streaming report is visible, we do
// INCREMENTAL updates (only the report body) instead of full DOM rebuilds.
let _renderScheduled = false;
function scheduleRender() {
  if (_renderScheduled) return;
  _renderScheduled = true;
  requestAnimationFrame(() => {
    _renderScheduled = false;
    updateStreamingReport();
  });
}

/**
 * Incremental update for the streaming report — only touches the report
 * body innerHTML and scroll position, leaving the rest of the DOM intact.
 * Falls back to a full render() if the report container doesn't exist yet.
 */
function updateStreamingReport() {
  // If we're not in streaming phase2, fall back to full render
  if (state.phase !== 'streaming_phase2' || !state.streamingReport) {
    render();
    return;
  }

  const body = document.querySelector('.streaming-report-body');
  const charSpan = document.querySelector('.streaming-chars');

  if (!body) {
    // First chunk: report container doesn't exist yet — do a full render
    // to switch from progress bar to streaming report view.
    render();
    return;
  }

  // Incremental update: only swap the report body HTML
  body.innerHTML = components.markdownToHtml(state.streamingReport || '');

  // Update char count
  if (charSpan) {
    charSpan.textContent = (state.streamingChars || 0) + ' chars';
  }

  // Auto-scroll to bottom
  body.scrollTop = body.scrollHeight;
}

let currentSSE = null;

// Progress timer for real-time elapsed time display
let _progressTimer = null;

// ---- Render --------------------------------------------------------------

function render() {
  console.log('[app] render() called, phase=' + state.phase + ' lang=' + getLang());
  // Update language toggle button text on every render
  const langBtn = document.getElementById('lang-btn');
  if (langBtn) langBtn.textContent = t('langSwitch');

  // Sync <html lang="..."> attribute
  document.documentElement.lang = getLang();

  // Clear progress timer before re-rendering
  if (_progressTimer) {
    clearInterval(_progressTimer);
    _progressTimer = null;
  }

  const app = document.getElementById('app');
  app.innerHTML = ''; // Clear previous view

  const wrapper = document.createElement('main');
  wrapper.className =
    'min-h-screen flex flex-col items-center px-4 pt-6 pb-10 bg-gradient-to-br from-blue-50 via-purple-50 to-blue-50 animate-gradient'

  switch (state.phase) {
    case 'input':
      renderInputView(wrapper);
      break;
    case 'streaming_phase1':
      renderStreamingView(wrapper, 'phase1');
      break;
    case 'reviewing':
      renderReviewingView(wrapper);
      break;
    case 'streaming_phase2':
      renderStreamingView(wrapper, 'phase2');
      break;
    case 'completed':
      renderCompletedView(wrapper);
      break;
  }

  // Error display
  if (state.error) {
    const errP = document.createElement('p');
    errP.className = 'mt-4 text-sm text-red-600';
    errP.textContent = state.error;
    wrapper.appendChild(errP);
  }

  app.appendChild(wrapper);

  // Start progress timer AFTER the wrapper is in the DOM so that
  // document.getElementById('progress-elapsed') can find the element.
  if (state.phase === 'streaming_phase1' ||
      (state.phase === 'streaming_phase2' && !state.streamingReport)) {
    _progressTimer = components.initProgressTimer();
  }
}

// ---- View: input ---------------------------------------------------------

function renderInputView(wrapper) {
  const form = components.renderQueryForm(handleSubmitQuery, state.loading);
  wrapper.appendChild(form);
}

// ---- View: streaming -----------------------------------------------------

function renderStreamingView(wrapper, phase) {
  // When streaming report is active (phase2 with chunks arriving),
  // show ONLY the streaming report — no progress bar to avoid flickering.
  const showStreamingReport = phase === 'phase2' && state.streamingReport;

  if (showStreamingReport) {
    // ── Streaming Report Only ──────────────────────────────────────────
    // Hide progress bar once chunks start arriving; the report IS the progress.
    const streamDiv = document.createElement('div');
    streamDiv.className = 'w-full flex justify-center';
    streamDiv.appendChild(
      components.renderStreamingReport(state.streamingReport, state.streamingChars)
    );
    wrapper.appendChild(streamDiv);
  } else {
    // ── Progress Bar (phase1, or phase2 before chunks arrive) ─────────
    // Phase label
    const titleDiv = document.createElement('div');
    titleDiv.className = 'text-center mb-8';
    const label =
      phase === 'phase1' ? t('planningTitle') : t('finalizingTitle');

    let liveIndicator = '';
    if (currentSSE && currentSSE.connected) {
      liveIndicator = `
        <p class="text-xs text-green-600 mt-2 flex items-center justify-center gap-1">
          <span class="pulse-dot"></span> ${escapeText(t('liveIndicator'))}
        </p>
      `;
    }

    titleDiv.innerHTML = `
      <h1 class="text-2xl sm:text-3xl font-bold text-gray-900">${escapeText(label)}</h1>
      ${liveIndicator}
    `;
    wrapper.appendChild(titleDiv);

    // Progress component
    const progress = components.renderStreamingProgress(state.events, phase);
    wrapper.appendChild(progress);
    // NOTE: timer is started in render() after wrapper is appended to DOM
  }

  // Retry button for Phase 2 timeout/error — lets the user re-trigger
  // confirm without being stuck in the streaming view.
  if (state.error && phase === 'phase2') {
    wrapper.appendChild(
      components.renderRetryButton(() => {
        state.error = null;
        handleConfirm();
      })
    );
  }
}

// ---- View: reviewing -----------------------------------------------------

function renderReviewingView(wrapper) {
  // Back link
  const backDiv = document.createElement('div');
  backDiv.className = 'w-full max-w-3xl mb-6';
  backDiv.innerHTML = `
    <button id="back-home" class="text-sm text-blue-600 hover:underline flex items-center gap-1">
      ${escapeText(t('backHome'))}
    </button>
  `;
  wrapper.appendChild(backDiv);
  backDiv.querySelector('#back-home').addEventListener('click', resetToInput);

  // Phase title
  const titleDiv = document.createElement('div');
  titleDiv.className = 'text-center mb-8';
  titleDiv.innerHTML = `
    <h1 class="text-2xl sm:text-3xl font-bold tracking-tight">
      <span class="gradient-text">${escapeText(t('reviewTitle'))}</span>
    </h1>
    <p class="mt-2 text-sm text-gray-500">${escapeText(t('reviewSubtitle'))}</p>
  `;
  wrapper.appendChild(titleDiv);

  // Content container
  const content = document.createElement('div');
  content.className = 'flex flex-col items-center gap-6 w-full';

  if (state.displayData) {
    content.appendChild(components.renderItineraryReview(state.displayData));
  }

  content.appendChild(
    components.renderFeedbackForm(handleFeedback, handleConfirm, state.loading)
  );

  wrapper.appendChild(content);
}

// ---- View: completed -----------------------------------------------------

function renderCompletedView(wrapper) {
  // Back link
  const backDiv = document.createElement('div');
  backDiv.className = 'w-full max-w-3xl mb-6';
  backDiv.innerHTML = `
    <button id="back-home" class="text-sm text-blue-600 hover:underline flex items-center gap-1">
      ${escapeText(t('backHome'))}
    </button>
  `;
  wrapper.appendChild(backDiv);
  backDiv.querySelector('#back-home').addEventListener('click', resetToInput);

  // Success animation + title
  const titleDiv = document.createElement('div');
  titleDiv.className = 'text-center mb-8';
  titleDiv.innerHTML = `
    <div class="text-5xl mb-2 animate-success">🎉</div>
    <h1 class="text-2xl sm:text-3xl font-bold tracking-tight">
      <span class="gradient-text">${escapeText(t('doneTitle'))}</span>
    </h1>
  `;
  wrapper.appendChild(titleDiv);

  // Report + "plan another trip" button
  const content = document.createElement('div');
  content.className = 'flex flex-col items-center gap-6 w-full';

  if (state.report) {
    content.appendChild(components.renderReport(state.report));
  } else if (state.sessionId) {
    // Fallback: try to fetch report via REST
    content.appendChild(showLoading(t('loadingReport')));
    api.getReport(state.sessionId).then((report) => {
      state.report = report;
      render();
    }).catch(() => {
      state.error = t('failedToLoadReport');
      render();
    });
  }

  // Action buttons (download + share)
  const actionsDiv = document.createElement('div');
  actionsDiv.className = 'flex flex-wrap gap-3 justify-center mt-4';
  actionsDiv.innerHTML = `
    <button id="download-pdf"
       class="inline-flex items-center gap-2 px-4 py-2 btn-gradient text-white text-sm rounded-lg transition disabled:opacity-60 disabled:cursor-wait">
      ${t('downloadReport')}
    </button>
    <button id="copy-share-link" 
       class="inline-flex items-center gap-2 px-4 py-2 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 transition">
      ${t('copyShareLink')}
    </button>
  `;
  content.appendChild(actionsDiv);

  // Download PDF handler — uses browser native print-to-PDF via a popup
  // window. This is more reliable than html2canvas which fails on cross-origin
  // iframes (OpenStreetMap embeds) and produces blank PDFs.
  // Falls back to the backend WeasyPrint endpoint if the popup fails.
  const downloadBtn = actionsDiv.querySelector('#download-pdf');
  downloadBtn.addEventListener('click', () => {
    const reportBody = document.querySelector('.markdown-body');
    if (!reportBody) {
      window.open(`/api/sessions/${state.sessionId}/download`, '_blank');
      return;
    }

    // Open a new window with just the report content and print styles.
    // The user can then "Save as PDF" from the browser's print dialog.
    const printWin = window.open('', '_blank', 'width=800,height=900');
    if (!printWin) {
      // Popup blocked — fall back to backend WeasyPrint
      window.open(`/api/sessions/${state.sessionId}/download`, '_blank');
      return;
    }

    // Collect all stylesheets and styles needed for the report
    const styles = `
      <style>
        @page { margin: 15mm; }
        body {
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans SC", sans-serif;
          line-height: 1.75;
          color: #333;
          max-width: 800px;
          margin: 0 auto;
          padding: 20px;
        }
        h1 { color: #1a365d; border-bottom: 2px solid #3182ce; padding-bottom: 8px; font-size: 22px; }
        h2 { color: #2c5282; margin-top: 24px; font-size: 18px; }
        h3 { color: #2d3748; font-size: 15px; }
        h4 { color: #4a5568; font-size: 14px; }
        table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }
        th, td { border: 1px solid #e2e8f0; padding: 6px 10px; text-align: left; }
        th { background: #ebf8ff; font-weight: 600; }
        tr:nth-child(even) { background: #f7fafc; }
        blockquote {
          border-left: 4px solid #3182ce;
          margin: 12px 0;
          padding: 8px 16px;
          background: #f0f9ff;
          color: #1e40af;
          border-radius: 0 4px 4px 0;
        }
        img { max-width: 100%; height: auto; border-radius: 8px; margin: 8px 0; }
        code { background: #edf2f7; padding: 2px 6px; border-radius: 4px; font-size: 13px; }
        hr { border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }
        ul, ol { padding-left: 20px; }
        li { margin-bottom: 4px; }
        a { color: #3182ce; text-decoration: none; }
        /* Expand all collapsible maps for printing */
        details.map-collapsible { display: block; }
        details.map-collapsible > summary { display: none; }
        /* In print, force-open all map <details> and show images */
        details.map-collapsible > img { display: block !important; max-width: 100%; height: auto; }
      </style>
    `;

    // Clone the report content and expand all collapsed maps for printing.
    const contentClone = reportBody.cloneNode(true);
    contentClone.querySelectorAll('details.map-collapsible').forEach((d) => {
      d.setAttribute('open', '');
    });

    printWin.document.write(`<!DOCTYPE html><html><head><meta charset="utf-8"><title>Travel Plan</title>${styles}</head><body>${contentClone.innerHTML}</body></html>`);
    printWin.document.close();
    // Use setTimeout instead of onload — onload fires during document.close()
    // and setting it afterwards misses the event entirely.
    // 800ms gives images time to start loading before print dialog opens.
    setTimeout(() => {
      try {
        printWin.focus();
        printWin.print();
      } catch (e) {
        console.error('[PDF export] print failed:', e);
      }
    }, 800);
  });

  // Copy share link handler
  actionsDiv.querySelector('#copy-share-link').addEventListener('click', () => {
    const shareUrl = `${window.location.origin}/share/${state.sessionId}`;
    if (navigator.clipboard) {
      navigator.clipboard.writeText(shareUrl).then(() => {
        const btn = actionsDiv.querySelector('#copy-share-link');
        btn.textContent = t('linkCopied');
        setTimeout(() => { btn.innerHTML = t('copyShareLink'); }, 2000);
      });
    } else {
      // Fallback for older browsers
      const textarea = document.createElement('textarea');
      textarea.value = shareUrl;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      const btn = actionsDiv.querySelector('#copy-share-link');
      btn.textContent = t('linkCopied');
      setTimeout(() => { btn.innerHTML = t('copyShareLink'); }, 2000);
    }
  });

  const newTripBtn = document.createElement('button');
  newTripBtn.className =
    'btn-gradient px-6 py-2.5 rounded-xl text-white font-medium text-sm transition';
  newTripBtn.textContent = t('newTrip');
  newTripBtn.addEventListener('click', resetToInput);
  content.appendChild(newTripBtn);

  wrapper.appendChild(content);
}

// ---- Loading helper ------------------------------------------------------

function showLoading(text) {
  const div = document.createElement('div');
  div.className = 'flex items-center gap-2 text-gray-500';
  div.innerHTML = `<span class="spinner" style="width:20px;height:20px;border-color:#dbeafe;border-top-color:#3b82f6;"></span> ${escapeText(text)}`;
  return div;
}

function escapeText(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ---- Event handlers ------------------------------------------------------

/**
 * User submits the initial travel query.
 */
async function handleSubmitQuery(query) {
  // Auto-detect language from query: if no Chinese characters, switch to English
  const hasChinese = /[\u4e00-\u9fff]/.test(query);
  if (!hasChinese && getLang() === 'zh') {
    setLang('en');
    console.log('[app] Auto-switched language to en based on query');
  } else if (hasChinese && getLang() === 'en') {
    setLang('zh');
    console.log('[app] Auto-switched language to zh based on query');
  }

  state.loading = true;
  state.error = null;
  render();
  try {
    state.sessionId = await api.createSession();
    await api.startPlanning(state.sessionId, query);
    console.log(`%c[LOG] QUERY submitted | session=${state.sessionId}`, 'color:#2196F3;font-weight:bold');
    console.log(`%c  "${query}"`, 'color:#2196F3');
    state.phase = 'streaming_phase1';
    state.events = [];
    state.loading = false;
    render();
    connectSSE();
  } catch (e) {
    state.loading = false;
    const rawMsg = e.message || '';
    // Map common browser/infra errors to user-friendly messages.
    if (rawMsg.includes('Failed to fetch') || rawMsg.includes('NetworkError')) {
      state.error = getLang() === 'zh'
        ? '无法连接服务器，请检查网络后重试'
        : 'Cannot connect to server. Please check your network and try again.';
    } else if (rawMsg.includes('timed out') || rawMsg === 'Request timed out. Please check your network and try again.') {
      state.error = getLang() === 'zh'
        ? '请求超时，请检查网络后重试'
        : 'Request timed out. Please check your network and try again.';
    } else {
      state.error = rawMsg || (getLang() === 'zh' ? '启动规划失败' : 'Failed to start planning');
    }
    render();
  }
}

/**
 * Open an SSE connection and drive state transitions based on events.
 */
function connectSSE() {
  // Disconnect any existing connection.
  if (currentSSE) {
    currentSSE.disconnect();
    currentSSE = null;
  }

  currentSSE = new SSEClient(state.sessionId);
  currentSSE.onEvent((event) => {
    state.events.push(event);

    if (event.type === 'interrupt') {
      state.displayData = event.display_data;
      state.phase = 'reviewing';
      console.log(`%c[LOG] PLAN READY for review | session=${state.sessionId}`, 'color:#9C27B0;font-weight:bold');
      render();
    } else if (event.type === 'execution_complete') {
      state.report = event.final_report;
      state.phase = 'completed';
      console.log(`%c[LOG] REPORT COMPLETE | session=${state.sessionId} | chars=${(event.final_report||'').length}`, 'color:#4CAF50;font-weight:bold');
      console.log(`%c  Download: /api/sessions/${state.sessionId}/download`, 'color:#4CAF50');
      console.log(`%c  Share: /share/${state.sessionId}`, 'color:#4CAF50');
      render();
    } else if (event.type === 'markdown_chunk') {
      // Accumulate streaming markdown chunks for live preview
      state.streamingReport += (event.chunk || '');
      state.streamingChars = event.total_chars || state.streamingReport.length;
      // Re-render to show the live report preview (throttled via rAF)
      scheduleRender();
    } else if (event.type === 'timeout') {
      // Stream timed out — show a user-friendly message and allow retry.
      state.error = getLang() === 'zh'
        ? '报告生成超时，请重试'
        : 'Report generation timed out, please retry';
      // If in Phase 1, go back to input so the user can resubmit.
      // If in Phase 2, stay in the current view — the user can retry via confirm.
      if (state.phase === 'streaming_phase1') {
        state.phase = 'input';
      }
      render();
    } else if (event.type === 'error') {
      state.error = event.error_message || 'An error occurred';
      render();
    } else if (event.type === 'connect_error') {
      // SSE connection failed — show error and allow retry.
      state.error = event.error_message || 'Connection failed. Please try again.';
      if (state.phase === 'streaming_phase1') {
        state.phase = 'input';
      }
      render();
    } else {
      // node_completed or audit_warning — just re-render to update progress.
      render();
    }
  });

  currentSSE.connect();
}

/**
 * User submits feedback — triggers re-planning (back to Phase 1).
 */
async function handleFeedback(feedback) {
  state.loading = true;
  state.error = null;
  render();
  try {
    await api.submitFeedback(state.sessionId, feedback);
    console.log(`%c[LOG] FEEDBACK submitted | session=${state.sessionId}`, 'color:#FF9800;font-weight:bold');
    console.log(`%c  "${feedback}"`, 'color:#FF9800');
    state.phase = 'streaming_phase1';
    state.events = [];
    state.loading = false;
    render();
    connectSSE();
  } catch (e) {
    state.loading = false;
    const rawMsg = e.message || '';
    if (rawMsg.includes('Failed to fetch') || rawMsg.includes('NetworkError')) {
      state.error = getLang() === 'zh'
        ? '无法连接服务器，请检查网络后重试'
        : 'Cannot connect to server. Please check your network and try again.';
    } else if (rawMsg.includes('timed out')) {
      state.error = getLang() === 'zh'
        ? '请求超时，请重试'
        : 'Request timed out, please retry.';
    } else {
      state.error = rawMsg || 'Failed to submit feedback';
    }
    render();
  }
}

/**
 * User confirms the plan — triggers Phase 2 (Routing → Critic → Synthesizer).
 */
async function handleConfirm() {
  state.loading = true;
  state.error = null;
  render();
  try {
    await api.confirmPlan(state.sessionId);
    state.phase = 'streaming_phase2';
    state.events = [];
    state.streamingReport = '';
    state.streamingChars = 0;
    state.loading = false;
    render();
    connectSSE();
  } catch (e) {
    state.loading = false;
    const rawMsg = e.message || '';
    if (rawMsg.includes('Failed to fetch') || rawMsg.includes('NetworkError')) {
      state.error = getLang() === 'zh'
        ? '无法连接服务器，请检查网络后重试'
        : 'Cannot connect to server. Please check your network and try again.';
    } else if (rawMsg.includes('timed out')) {
      state.error = getLang() === 'zh'
        ? '请求超时，请重试'
        : 'Request timed out, please retry.';
    } else {
      state.error = rawMsg || 'Failed to confirm plan';
    }
    render();
  }
}

/**
 * Reset to the initial input view.
 */
function resetToInput() {
  if (currentSSE) {
    currentSSE.disconnect();
    currentSSE = null;
  }
  state = {
    phase: 'input',
    sessionId: null,
    events: [],
    displayData: null,
    report: null,
    streamingReport: '',
    streamingChars: 0,
    loading: false,
    error: null,
  };
  render();
}

// ---- Initialize ----------------------------------------------------------

// Wrap the initial render in a try-catch so that any error is surfaced
// both in the console and in the on-page error overlay (set up in index.html).
try {
  // Set up language toggle handler (button lives in index.html, persists across renders)
  const langBtn = document.getElementById('lang-btn');
  if (langBtn) {
    langBtn.addEventListener('click', () => {
      try {
        toggleLang();
        render(); // render() updates the button text and <html lang>
      } catch (e) {
        console.error('[app.js] language toggle failed:', e);
        // Attempt to recover: at least update button text and lang attribute
        const langBtn2 = document.getElementById('lang-btn');
        if (langBtn2) langBtn2.textContent = getLang() === 'zh' ? 'EN' : '中文';
        document.documentElement.lang = getLang();
        // Force full page reload as last resort
        window.location.reload();
      }
    });
    console.log('[app.js] language toggle handler attached');
  }

  console.log('[app.js] initialising render…');
  render();
  console.log('[app.js] render complete');

  // ---- Print support: force-expand all collapsible maps before printing ----
  // <details> elements hide content via shadow DOM, not CSS display.
  // @media print alone cannot open them, so we add/remove the `open` attribute.
  let _printExpandedDetails = [];
  window.addEventListener('beforeprint', () => {
    _printExpandedDetails = [];
    document.querySelectorAll('details.map-collapsible:not([open])').forEach((d) => {
      d.setAttribute('open', '');
      _printExpandedDetails.push(d);
    });
  });
  window.addEventListener('afterprint', () => {
    _printExpandedDetails.forEach((d) => d.removeAttribute('open'));
    _printExpandedDetails = [];
  });
} catch (e) {
  console.error('[app.js] failed during initial render:', e);
  // Dispatch a synthetic error event so the global handler in index.html
  // can display it in the overlay.
  window.dispatchEvent(new ErrorEvent('error', {
    message: 'app.js initial render failed: ' + (e && e.message ? e.message : String(e)),
    filename: '/static/js/app.js',
    lineno: 0,
    colno: 0,
    error: e,
  }));
}
