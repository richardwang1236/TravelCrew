/**
 * UI components — pure functions that return HTML strings.
 * Each function mirrors a React component from the original Next.js app.
 * All user-facing strings are pulled from the i18n module.
 */

import { t, getExamples, getLang } from './i18n.js?v=20260715b';

// ---- Constants -----------------------------------------------------------

const PHASE_NODES = {
  phase1: ['intentparser', 'information', 'recommendation', 'user_review'],
  phase2: ['routing', 'critic', 'synthenrich', 'synthesizer'],
};

/** Look up a node label via i18n. Falls back to the raw node key. */
function nodeLabel(node) {
  const label = t('nodes.' + node);
  return label === 'nodes.' + node ? node : label;
}

// ---- Helper: escape HTML -------------------------------------------------

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text == null ? '' : String(text);
  return div.innerHTML;
}

function formatNumber(n) {
  if (n == null) return '';
  return Number(n).toLocaleString();
}

// ---- QueryForm -----------------------------------------------------------

/**
 * Render the travel query input form.
 * @param {(query: string) => void} onSubmit
 * @param {boolean} loading
 * @returns {HTMLElement}
 */
export function renderQueryForm(onSubmit, loading) {
  const container = document.createElement('div');
  container.className = 'animate-fadeSlideUp';

  container.innerHTML = `
    <div class="text-center mb-4">
      <h1 class="text-4xl sm:text-5xl font-bold tracking-tight">
        <span class="gradient-text">${escapeHtml(t('appTitle'))}</span>
      </h1>
      <p class="mt-3 text-base sm:text-lg text-gray-500 max-w-md mx-auto">
        ${escapeHtml(t('appSubtitle'))}
      </p>
    </div>

    <form class="w-full max-w-2xl bg-white rounded-2xl shadow-lg border border-blue-100 p-8 card-hover">
      <label for="query" class="block text-lg font-semibold text-gray-800 mb-2">
        ${escapeHtml(t('formLabel'))}
      </label>
      <p class="text-sm text-gray-500 mb-4">
        ${escapeHtml(t('formHelper'))}
      </p>

      <div class="mb-4 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-100 text-sm text-gray-600 whitespace-pre-line leading-relaxed">${t('inputGuide')}</div>

      <textarea
        id="query"
        rows="4"
        placeholder="${escapeHtml(t('inputPlaceholder'))}"
        class="focus-glow w-full rounded-xl border border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 px-4 py-3 text-gray-800 resize-none transition outline-none"
      ></textarea>

      <div class="flex flex-wrap gap-2 mt-3 mb-6" id="example-chips"></div>

      <button
        type="submit"
        class="btn-gradient w-full py-3 rounded-xl text-white font-semibold text-base transition disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center justify-center gap-2"
      >
        <span id="submit-text">${escapeHtml(t('startPlanning'))}</span>
      </button>
    </form>

    <div class="mt-8 text-center text-2xl opacity-60 select-none">
      🗺️ &nbsp; 🏖️ &nbsp; 🍜 &nbsp; 🏛️ &nbsp; ✈️
    </div>
    <p class="mt-4 text-xs text-gray-400 text-center">
      ${escapeHtml(t('poweredBy'))}
    </p>
  `;

  // Populate example chips
  const chipsContainer = container.querySelector('#example-chips');
  getExamples().forEach((ex) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className =
      'text-xs bg-blue-50 text-blue-700 rounded-full px-3 py-1 hover:bg-blue-100 transition';
    btn.textContent = ex;
    btn.addEventListener('click', () => {
      container.querySelector('#query').value = ex;
    });
    chipsContainer.appendChild(btn);
  });

  // Form submit
  const form = container.querySelector('form');
  const submitText = container.querySelector('#submit-text');
  const textarea = container.querySelector('#query');

  // Inline validation hint element (shown when user clicks submit on empty input)
  let hintEl = null;
  const showHint = (msg) => {
    if (!hintEl) {
      hintEl = document.createElement('p');
      hintEl.className = 'text-sm text-red-600 mt-2';
      textarea.parentElement.appendChild(hintEl);
    }
    hintEl.textContent = msg;
    hintEl.style.display = msg ? 'block' : 'none';
  };

  const updateLoadingState = () => {
    const disabled = loading;
    form.querySelector('button[type="submit"]').disabled = disabled;
    if (loading) {
      submitText.innerHTML =
        '<span class="spinner" style="width:20px;height:20px;"></span> ' + escapeHtml(t('creatingPlan'));
    } else {
      submitText.textContent = t('startPlanning');
    }
    if (!loading) showHint('');
  };

  textarea.addEventListener('input', () => {
    showHint('');
    updateLoadingState();
  });
  updateLoadingState();

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const trimmed = textarea.value.trim();
    if (loading) return;
    if (!trimmed) {
      showHint(t('emptyInputHint'));
      textarea.focus();
      return;
    }
    showHint('');
    onSubmit(trimmed);
  });

  return container;
}

// ---- StreamingProgress ---------------------------------------------------

/**
 * Render the node execution progress list.
 * @param {Array} events  SSE events array
 * @param {'phase1'|'phase2'} phase
 * @returns {HTMLElement}
 */
export function renderStreamingProgress(events, phase) {
  const container = document.createElement('div');
  container.className = 'animate-fadeSlideUp w-full max-w-lg bg-white rounded-2xl shadow-md border border-blue-100 p-6 card-hover';

  const orderedNodes = PHASE_NODES[phase] || [];
  const totalNodes = orderedNodes.length;

  const completedSet = new Set(
    events
      .filter((e) => e.type === 'node_completed')
      .map((e) => (e.node || '').toLowerCase())
  );

  // Only count completed nodes that belong to the current phase.
  // Replan cycles or cross-phase events may add extra names to completedSet,
  // which would cause "6/4" style over-counts and fake 100%.
  const completedCount = orderedNodes.filter((n) => completedSet.has(n)).length;
  const currentIdx = orderedNodes.findIndex((n) => !completedSet.has(n));

  // Progress percentage: completed nodes / total, with partial bump for current step
  let progressPct = totalNodes > 0 ? Math.round((completedCount / totalNodes) * 100) : 0;
  const isActive = currentIdx >= 0 && currentIdx < totalNodes;
  if (isActive && completedCount < totalNodes) {
    // Only show partial progress after first node completes to avoid fake initial %
    if (completedCount === 0) {
      progressPct = 0;
    } else {
      progressPct = Math.min(95, Math.round(((completedCount + 0.25) / totalNodes) * 100));
    }
  }
  if (completedCount >= totalNodes) {
    progressPct = 100;
  }

  // Elapsed time since first event
  // Use data-start-time attribute for real-time updates via initProgressTimer()
  let startTimestamp = 0;
  const firstEvent = events.find((e) => e.timestamp);
  if (firstEvent && firstEvent.timestamp) {
    startTimestamp = firstEvent.timestamp;
  }
  let elapsedStr = '';
  if (startTimestamp > 0) {
    const elapsed = Math.max(0, Math.round(Date.now() / 1000 - startTimestamp));
    if (elapsed < 60) {
      elapsedStr = elapsed + 's';
    } else {
      const mins = Math.floor(elapsed / 60);
      const secs = elapsed % 60;
      elapsedStr = mins + 'm ' + secs + 's';
    }
  }

  const errorEvents = events.filter((e) => e.type === 'error');

  let html = `
    <h2 class="text-lg font-semibold text-gray-800 mb-3">
      ${escapeHtml(phase === 'phase1' ? t('analyzingTrip') : t('finalizingPlan'))}
    </h2>
  `;

  // ---- Visual Progress Bar ----
  const barActiveClass = isActive ? ' active' : '';
  html += `
    <div class="mb-4">
      <div class="progress-bar-container">
        <div class="progress-bar-fill${barActiveClass}" style="width:${progressPct}%;"></div>
      </div>
      <div class="flex justify-between items-center mt-2">
        <span class="text-xs text-gray-500">
          ${escapeHtml(t('progressStep').replace('{current}', String(completedCount)).replace('{total}', String(totalNodes)))}
        </span>
        <span class="text-xs font-semibold" style="color:#3b82f6;">
          ${escapeHtml(t('progressPercent').replace('{percent}', String(progressPct)))}
        </span>
        ${elapsedStr ? `<span class="text-xs text-gray-400" id="progress-elapsed" data-start-time="${startTimestamp}">${escapeHtml(t('progressElapsed').replace('{time}', elapsedStr))}</span>` : ''}
      </div>
    </div>
  `;

  html += '<ul class="space-y-2">';

  orderedNodes.forEach((node, idx) => {
    const isComplete = completedSet.has(node);
    const isCurrent = idx === currentIdx;
    const label = nodeLabel(node);
    const dimClass = !isComplete && !isCurrent ? 'opacity-30' : 'opacity-100';

    let iconHtml;
    if (isComplete) {
      iconHtml = '<span class="animate-checkmark">✅</span>';
    } else if (isCurrent) {
      iconHtml = '<span class="spinner" style="width:18px;height:18px;border-color:#dbeafe;border-top-color:#3b82f6;"></span>';
    } else {
      iconHtml = '<span class="inline-block w-3 h-3 rounded-full bg-gray-300"></span>';
    }

    let labelClass;
    if (isComplete) {
      labelClass = 'text-gray-600 line-through decoration-gray-400';
    } else if (isCurrent) {
      labelClass = 'text-blue-700 font-medium';
    } else {
      labelClass = 'text-gray-400';
    }

    html += `
      <li class="flex items-center gap-3 transition-opacity duration-500 ${dimClass} animate-fadeSlideUp" style="animation-delay:${idx * 0.1}s;">
        <span class="flex-shrink-0 w-6 h-6 flex items-center justify-center text-base">${iconHtml}</span>
        <span class="text-sm ${labelClass}">${escapeHtml(label)}</span>
      </li>
    `;
  });

  html += '</ul>';

  // Error messages
  if (errorEvents.length > 0) {
    html += errorEvents
      .map((e) => `<p class="mt-4 text-sm text-red-600">${escapeHtml(e.error_message || '')}</p>`)
      .join('');
  }

  // Progress log messages
  const logEvents = events.filter((e) => e.type === 'progress_log');
  if (logEvents.length > 0) {
    html += `
      <div class="mt-4 border-t border-gray-100 pt-3">
        <h3 class="text-xs font-medium text-gray-400 mb-2">${escapeHtml(t('progressTitle'))}</h3>
        <div class="space-y-1 max-h-48 overflow-y-auto text-xs text-gray-600" id="progress-log-container">
    `;
    logEvents.forEach((e) => {
      const msg = typeof e.message === 'object'
        ? (e.message[getLang()] || e.message.en || '')
        : (e.message || '');
      html += `<p class="log-fade-in">${escapeHtml(msg)}</p>`;
    });
    html += `
        </div>
      </div>
    `;
  }

  container.innerHTML = html;
  return container;
}

/**
 * Start a timer that updates the elapsed time display every second.
 * Call this after renderStreamingProgress is added to the DOM.
 * Returns the interval ID (pass to clearInterval to stop).
 */
export function initProgressTimer() {
  const el = document.getElementById('progress-elapsed');
  if (!el) return null;
  const startTime = parseFloat(el.getAttribute('data-start-time') || '0');
  if (!startTime) return null;
  
  const intervalId = setInterval(() => {
    const elNow = document.getElementById('progress-elapsed');
    if (!elNow) {
      clearInterval(intervalId);
      return;
    }
    const elapsed = Math.max(0, Math.round(Date.now() / 1000 - startTime));
    let elapsedStr;
    if (elapsed < 60) {
      elapsedStr = elapsed + 's';
    } else {
      const mins = Math.floor(elapsed / 60);
      const secs = elapsed % 60;
      elapsedStr = mins + 'm ' + secs + 's';
    }
    elNow.textContent = t('progressElapsed').replace('{time}', elapsedStr);
  }, 1000);
  
  return intervalId;
}

// ---- ItineraryReview -----------------------------------------------------

/**
 * Render a POI card (attraction or restaurant).
 * @param {object} poi
 * @param {string} currency
 * @returns {string}
 */
function poiCardHtml(poi, currency) {
  const imgHtml = typeof poi.image_url === 'string' && poi.image_url
    ? `<img src="${escapeHtml(poi.image_url)}" alt="${escapeHtml(poi.name)}" class="w-16 h-16 rounded-lg object-cover flex-shrink-0" />`
    : '';

  const costHtml =
    poi.cost != null
      ? `<span class="text-blue-700 font-medium">${escapeHtml(currency)}${formatNumber(poi.cost)}</span>`
      : '';
  const ratingHtml = poi.rating != null ? `<span>⭐ ${Number(poi.rating).toFixed(1)}</span>` : '';

  // Build action links: website + map link
  const linkParts = [];
  if (poi.website) {
    linkParts.push(
      `<a href="${escapeHtml(poi.website)}" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:text-blue-800 underline">🔗 ${escapeHtml(t('websiteLabel'))}</a>`
    );
  }
  // Use maps_url if available, otherwise generate from lat/lng (OpenStreetMap, no VPN needed)
  let mapUrl = poi.maps_url || '';
  if (!mapUrl && poi.lat && poi.lng) {
    mapUrl = `https://www.openstreetmap.org/?mlat=${poi.lat}&mlon=${poi.lng}#map=15/${poi.lat}/${poi.lng}`;
  }
  if (mapUrl) {
    linkParts.push(
      `<a href="${escapeHtml(mapUrl)}" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:text-blue-800 underline">📍 ${escapeHtml(t('mapLabel'))}</a>`
    );
  }
  const linksHtml = linkParts.length
    ? `<div class="flex items-center gap-3 mt-1 text-xs">${linkParts.join('')}</div>`
    : '';

  // Wikipedia intro or description (truncated)
  const descText = poi.wikipedia_intro || poi.description || '';
  const descHtml = descText
    ? `<p class="text-xs text-gray-500 mt-1 line-clamp-2">${escapeHtml(descText.substring(0, 120))}${descText.length > 120 ? '...' : ''}</p>`
    : '';

  return `
    <div class="card-hover flex gap-3 rounded-xl border border-gray-100 bg-gray-50 p-3">
      ${imgHtml}
      <div class="min-w-0 flex-1">
        <p class="font-medium text-sm text-gray-900 truncate">${escapeHtml(poi.name)}</p>
        ${poi.type ? `<p class="text-xs text-gray-500 truncate">${escapeHtml(poi.type)}</p>` : ''}
        <div class="flex items-center gap-3 mt-1 text-xs text-gray-600">${costHtml}${ratingHtml}</div>
        ${descHtml}
        ${linksHtml}
      </div>
    </div>
  `;
}

/**
 * Render a hotel card.
 * @param {object} hotel
 * @param {string} currency
 * @returns {string}
 */
function hotelCardHtml(hotel, currency) {
  const imgHtml = typeof hotel.image_url === 'string' && hotel.image_url
    ? `<img src="${escapeHtml(hotel.image_url)}" alt="${escapeHtml(hotel.name)}" class="w-16 h-16 rounded-lg object-cover flex-shrink-0" />`
    : '';

  const priceHtml =
    hotel.price_per_night != null
      ? `<span class="text-blue-700 font-medium">${escapeHtml(currency)}${formatNumber(hotel.price_per_night)} ${escapeHtml(t('perNight'))}</span>`
      : '';
  const ratingHtml = hotel.rating != null ? `<span>⭐ ${Number(hotel.rating).toFixed(1)}</span>` : '';

  // Build action links: website + map link
  const linkParts = [];
  if (hotel.website) {
    linkParts.push(
      `<a href="${escapeHtml(hotel.website)}" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:text-blue-800 underline">🔗 ${escapeHtml(t('websiteLabel'))}</a>`
    );
  }
  let mapUrl = hotel.maps_url || '';
  if (!mapUrl && hotel.lat && hotel.lng) {
    mapUrl = `https://www.openstreetmap.org/?mlat=${hotel.lat}&mlon=${hotel.lng}#map=15/${hotel.lat}/${hotel.lng}`;
  }
  if (mapUrl) {
    linkParts.push(
      `<a href="${escapeHtml(mapUrl)}" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:text-blue-800 underline">📍 ${escapeHtml(t('mapLabel'))}</a>`
    );
  }
  const linksHtml = linkParts.length
    ? `<div class="flex items-center gap-3 mt-1 text-xs">${linkParts.join('')}</div>`
    : '';

  return `
    <div class="card-hover flex gap-3 rounded-xl border border-gray-100 bg-gray-50 p-3">
      ${imgHtml}
      <div class="min-w-0 flex-1">
        <p class="font-medium text-sm text-gray-900 truncate">${escapeHtml(hotel.name)}</p>
        <div class="flex items-center gap-3 mt-1 text-xs text-gray-600">${priceHtml}${ratingHtml}</div>
        ${linksHtml}
      </div>
    </div>
  `;
}

/**
 * Render the day-by-day itinerary review.
 * Uses a timeline layout (left vertical line + dots) for each day.
 * @param {object} displayData  DisplayData from the interrupt event
 * @returns {HTMLElement}
 */
export function renderItineraryReview(displayData) {
  const container = document.createElement('div');
  container.className = 'animate-fadeSlideUp w-full max-w-3xl';

  const daily = displayData.daily_itinerary || [];
  const currency = displayData.currency_symbol || '';

  if (!daily.length) {
    container.innerHTML = `
      <div class="w-full max-w-3xl bg-white rounded-2xl shadow-md border border-blue-100 p-8 text-center text-gray-500">
        ${escapeHtml(t('noItinerary'))}
      </div>
    `;
    return container;
  }

  let html = '<div class="timeline-line space-y-6">';

  for (const day of daily) {
    const attractionsHtml =
      day.attractions && day.attractions.length > 0
        ? `<div><h4 class="text-sm font-semibold text-gray-700 mb-2">🏛️ ${escapeHtml(t('attractions'))}</h4>
           <div class="grid gap-3 sm:grid-cols-2">
             ${day.attractions.map((p) => poiCardHtml(p, currency)).join('')}
           </div></div>`
        : '';

    const diningHtml =
      day.dining && day.dining.length > 0
        ? `<div><h4 class="text-sm font-semibold text-gray-700 mb-2">🍽️ ${escapeHtml(t('dining'))}</h4>
           <div class="grid gap-3 sm:grid-cols-2">
             ${day.dining.map((p) => poiCardHtml(p, currency)).join('')}
           </div></div>`
        : '';

    const hotelHtml =
      day.hotel && day.hotel.name
        ? `<div><h4 class="text-sm font-semibold text-gray-700 mb-2">🏨 ${escapeHtml(t('hotel'))}</h4>
           ${hotelCardHtml(day.hotel, currency)}</div>`
        : '';

    html += `
      <div class="relative">
        <span class="timeline-dot"></span>
        <div class="bg-white rounded-2xl shadow-md border border-blue-100 overflow-hidden card-hover">
          <div class="bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-3">
            <h3 class="text-white font-semibold text-base">${escapeHtml(t('dayLabel').replace('{n}', String(day.day)))}</h3>
          </div>
          <div class="p-6 space-y-5">
            ${attractionsHtml}
            ${diningHtml}
            ${hotelHtml}
          </div>
        </div>
      </div>
    `;
  }

  html += '</div>';
  container.innerHTML = html;
  return container;
}

// ---- FeedbackForm --------------------------------------------------------

/**
 * Render the feedback form with "Revise" and "Confirm" buttons.
 * @param {(feedback: string) => void} onFeedback
 * @param {() => void} onConfirm
 * @param {boolean} loading
 * @returns {HTMLElement}
 */
export function renderFeedbackForm(onFeedback, onConfirm, loading) {
  const container = document.createElement('div');
  container.className =
    'animate-fadeSlideUp w-full max-w-3xl bg-white rounded-2xl shadow-md border border-blue-100 p-6 card-hover';

  container.innerHTML = `
    <h3 class="text-base font-semibold text-gray-800 mb-1">
      ${escapeHtml(t('reviewPrompt'))}
    </h3>
    <p class="text-sm text-gray-500 mb-4">
      ${escapeHtml(t('reviewHelper'))}
    </p>

    <textarea
      id="feedback-text"
      rows="3"
      placeholder="${escapeHtml(t('feedbackPlaceholder'))}"
      class="focus-glow w-full rounded-xl border border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 px-4 py-3 text-sm text-gray-800 resize-none transition outline-none"
    ></textarea>

    <div class="flex flex-col sm:flex-row gap-3 mt-4">
      <button
        id="btn-revise"
        class="flex-1 py-2.5 rounded-xl border border-blue-600 text-blue-600 font-medium text-sm hover:bg-blue-50 transition disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
      >
        <span id="revise-text">${escapeHtml(t('submitFeedback'))}</span>
      </button>

      <button
        id="btn-confirm"
        class="btn-gradient flex-1 py-2.5 rounded-xl text-white font-medium text-sm transition disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center justify-center gap-2"
      >
        <span id="confirm-text">${escapeHtml(t('confirmPlan'))}</span>
      </button>
    </div>
  `;

  const textarea = container.querySelector('#feedback-text');
  const reviseBtn = container.querySelector('#btn-revise');
  const confirmBtn = container.querySelector('#btn-confirm');
  const reviseText = container.querySelector('#revise-text');
  const confirmText = container.querySelector('#confirm-text');

  const updateButtons = () => {
    const hasText = textarea.value.trim().length > 0;
    reviseBtn.disabled = loading || !hasText;
    confirmBtn.disabled = loading;

    if (loading) {
      reviseText.innerHTML =
        '<span class="spinner" style="width:16px;height:16px;"></span> ' + escapeHtml(t('submitting'));
      confirmText.innerHTML =
        '<span class="spinner" style="width:16px;height:16px;"></span> ' + escapeHtml(t('confirming'));
    } else {
      reviseText.textContent = t('submitFeedback');
      confirmText.textContent = t('confirmPlan');
    }
  };

  textarea.addEventListener('input', updateButtons);
  updateButtons();

  reviseBtn.addEventListener('click', () => {
    const trimmed = textarea.value.trim();
    if (!trimmed || loading) return;
    onFeedback(trimmed);
  });

  confirmBtn.addEventListener('click', () => {
    if (loading) return;
    onConfirm();
  });

  return container;
}

// ---- ReportDisplay -------------------------------------------------------

/**
 * Convert a limited subset of Markdown to HTML.
 * Supports: # ## ### #### headings, **bold**, *italic*, - / * unordered lists,
 * 1. ordered lists, --- horizontal rules, [link](url), ![alt](url) images,
 * | table | syntax, and paragraph text.
 *
 * @param {string} md  Markdown source
 * @returns {string}   HTML string
 */
export function markdownToHtml(md) {
  // ── Pre-process: tables inside blockquotes ──
  // The LLM sometimes generates summary tables inside blockquotes, e.g.:
  //   > | 项目 | 详情 |
  //   > |---|---|
  // Our line-by-line parser treats blockquote lines as <p> text, so tables
  // inside blockquotes are rendered as literal pipe characters.
  // Fix: strip the '> ' prefix from lines that look like table rows so the
  // table-detection regex picks them up as regular tables.
  md = md.replace(/^> (\|.+\|[ \t]*)$/gm, '$1');

  // ── Pre-process: collapse blank lines between consecutive table rows ──
  // GFM allows blank lines between table rows, but our line-by-line parser
  // needs them to be contiguous.  Collapsing them avoids premature table
  // closure.
  md = md.replace(/(\|[^\n]*\|)[ \t]*\n\n(\|[^\n]*\|)/g, '$1\n$2');

  const lines = md.split('\n');
  let html = '';
  let inUl = false;
  let inOl = false;
  let inTable = false;
  let inBlockquote = false;
  let tableHeader = null;

  const closeLists = () => {
    if (inUl) {
      html += '</ul>';
      inUl = false;
    }
    if (inOl) {
      html += '</ol>';
      inOl = false;
    }
  };

  const closeBlockquote = () => {
    if (inBlockquote) {
      html += '</blockquote>';
      inBlockquote = false;
    }
  };

  const closeTable = () => {
    if (inTable) {
      html += '</tbody></table>';
      inTable = false;
      tableHeader = null;
    }
  };

  const inline = (text) =>
    text
      // Images: ![alt](url)
      // Static map images from /maps/ are now served as regular <img> tags.
      // Legacy OSM/Google Maps embed.html URLs are converted to a clickable link.
      .replace(
        /!\[([^\]]*)\]\(([^)]+)\)/g,
        (match, alt, url) => {
          // Skip URLs that are clearly not valid (JavaScript object coercion artifacts)
          if (!url || url === '[object Object]' || url === 'undefined' || url === 'null') {
            return '<em>' + (alt || 'image') + '</em>';
          }
          // Locally cached static map images (e.g. /maps/map_xxxx.png)
          // Wrapped in <details> for default-collapsed display.
          if (url.startsWith('/maps/')) {
            return '<details class="map-collapsible"><summary>🗺️ ' + escapeHtml(t('expandMapLabel')) + '</summary><img alt="' + (alt || escapeHtml(t('mapLabel'))) + '" src="' + url + '" style="width:100%;border-radius:0 0 0.5rem 0.5rem;margin:0;display:block;" /></details>';
          }
          // External Google Static Maps API URLs - also wrap in collapsible <details>
          if (url.includes('maps.googleapis.com') || (url.includes('staticmap') && url.includes('zoom='))) {
            return '<details class="map-collapsible"><summary>🗺️ ' + escapeHtml(t('expandMapLabel')) + '</summary><img alt="' + (alt || escapeHtml(t('mapLabel'))) + '" src="' + url + '" style="width:100%;border-radius:0 0 0.5rem 0.5rem;margin:0;display:block;" /></details>';
          }
          if (url.includes('openstreetmap.org') && url.includes('embed.html')) {
            return '<a href="' + url + '" target="_blank" rel="noopener noreferrer" style="display:inline-block;padding:0.25rem 0.75rem;background:#f0f9ff;color:#2563eb;border:1px solid #bae6fd;border-radius:0.375rem;font-size:0.8125rem;text-decoration:none;">🗺️ ' + escapeHtml(t('viewInteractiveMapLabel')) + '</a>';
          }
          // Legacy Google Maps embed: convert to a clickable link
          if (url.includes('maps.google.com/maps') && url.includes('output=embed')) {
            return '<a href="' + url + '" target="_blank" rel="noopener noreferrer" style="display:inline-block;padding:0.25rem 0.75rem;background:#f0f9ff;color:#2563eb;border:1px solid #bae6fd;border-radius:0.375rem;font-size:0.8125rem;text-decoration:none;">🗺️ ' + escapeHtml(t('viewInteractiveMapLabel')) + '</a>';
          }
          return '<img alt="' + alt + '" src="' + url + '" class="poi-image" />';
        }
      )
      // Filter out [object Object] from links (defense against object-typed fields)
      .replace(
        /\[([^\]]+)\]\(\[object Object\]\)/g,
        '<em>$1</em>'
      )
      // Links: [text](url)
      .replace(
        /\[([^\]]+)\]\(([^)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
      )
      // Bold: **text**
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      // Italic: *text*
      .replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>')
      // Inline code: `code`
      .replace(/`([^`]+)`/g, '<code>$1</code>');

  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    const trimmed = raw.trimEnd();

    // Table row detection: | col | col |
    if (/^\|.*\|/.test(trimmed)) {
      const cells = trimmed
        .split('|')
        .map((c) => c.trim())
        .filter((c, idx, arr) => idx !== 0 && idx !== arr.length - 1 || c !== '');

      // Check if this is a separator row: |---|---|
      if (/^\|[\s-:]+\|/.test(trimmed) && cells.every((c) => /^[-:\s]+$/.test(c))) {
        // Skip separator; header was the previous row.
        continue;
      }

      if (!inTable) {
        closeLists();
        inTable = true;
        // This row is the header.
        html += '<table><thead><tr>';
        html += cells.map((c) => `<th>${inline(c)}</th>`).join('');
        html += '</tr></thead><tbody>';
        continue;
      }

      // Regular table row
      const rowCells = trimmed
        .split('|')
        .map((c) => c.trim())
        .filter((c, idx, arr) => idx !== 0 && idx !== arr.length - 1 || c !== '');
      html += '<tr>';
      html += rowCells.map((c) => `<td>${inline(c)}</td>`).join('');
      html += '</tr>';
      continue;
    } else if (trimmed !== '') {
      closeTable();
    }

    // Blank line inside a table — skip without adding spacing
    if (inTable && trimmed === '') {
      continue;
    }

    // Close blockquote before non-blockquote content
    if (inBlockquote && !/^>/.test(trimmed)) {
      closeBlockquote();
    }

    // Headings
    const headingMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (headingMatch) {
      closeLists();
      closeBlockquote();
      const level = headingMatch[1].length;
      html += `<h${level}>${inline(headingMatch[2])}</h${level}>`;
      continue;
    }

    // Horizontal rule
    if (/^-{3,}$/.test(trimmed)) {
      closeLists();
      closeBlockquote();
      html += '<hr />';
      continue;
    }

    // Blockquote: > text
    const bqMatch = trimmed.match(/^>\s*(.*)$/);
    if (bqMatch) {
      closeLists();
      if (!inBlockquote) {
        html += '<blockquote>';
        inBlockquote = true;
      }
      if (bqMatch[1].trim()) {
        html += `<p>${inline(bqMatch[1])}</p>`;
      } else {
        html += '<br/>';
      }
      continue;
    }

    // Unordered list
    const ulMatch = trimmed.match(/^[-*]\s+(.+)$/);
    if (ulMatch) {
      closeBlockquote();
      if (inOl) {
        html += '</ol>';
        inOl = false;
      }
      if (!inUl) {
        html += '<ul>';
        inUl = true;
      }
      html += `<li>${inline(ulMatch[1])}</li>`;
      continue;
    }

    // Ordered list
    const olMatch = trimmed.match(/^\d+\.\s+(.+)$/);
    if (olMatch) {
      closeBlockquote();
      if (inUl) {
        html += '</ul>';
        inUl = false;
      }
      if (!inOl) {
        html += '<ol>';
        inOl = true;
      }
      html += `<li>${inline(olMatch[1])}</li>`;
      continue;
    }

    // Close lists before non-list content
    closeLists();
    closeBlockquote();

    // Blank line
    if (trimmed === '') {
      html += '<div style="height:0.75rem;"></div>';
      continue;
    }

    // Paragraph
    html += `<p>${inline(trimmed)}</p>`;
  }

  closeLists();
  closeTable();
  closeBlockquote();
  return html;
}

/**
 * Render the final travel report.
 * @param {string} markdownText  Markdown report text
 * @returns {HTMLElement}
 */
export function renderReport(markdownText) {
  const container = document.createElement('div');
  container.className = 'animate-fadeSlideUp w-full max-w-3xl';

  const html = markdownToHtml(markdownText || '');

  container.innerHTML = `
    <div class="report-accent bg-white rounded-2xl shadow-md border border-blue-100 overflow-hidden">
      <div class="bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-4 flex items-center gap-3">
        <span class="text-2xl">📋</span>
        <h2 class="text-white font-semibold text-lg">${escapeHtml(t('yourTravelReport'))}</h2>
      </div>
      <div class="px-6 py-6 markdown-body">
        ${html}
      </div>
    </div>
  `;

  return container;
}

/**
 * Render a live-streaming preview of the report as markdown chunks arrive.
 * Shows a typing cursor and a character count indicator.
 * @param {string} partialMarkdown  Accumulated markdown text so far
 * @param {number} totalChars       Total character count from the backend
 * @returns {HTMLElement}
 */
export function renderStreamingReport(partialMarkdown, totalChars) {
  const container = document.createElement('div');
  container.className = 'w-full max-w-3xl';

  const html = markdownToHtml(partialMarkdown || '');

  container.innerHTML = `
    <div class="bg-white rounded-2xl shadow-md border border-blue-200 overflow-hidden streaming-report">
      <div class="bg-gradient-to-r from-blue-500 to-indigo-500 px-6 py-3 flex items-center justify-between">
        <div class="flex items-center gap-3">
          <span class="text-xl">📝</span>
          <h2 class="text-white font-semibold text-sm">${escapeHtml(t('streamingReportTitle') || 'Generating report...')}</h2>
        </div>
        <div class="flex items-center gap-2">
          <span class="streaming-cursor">▊</span>
          <span class="text-white/80 text-xs streaming-chars">${totalChars || 0} chars</span>
        </div>
      </div>
      <div class="px-6 py-6 markdown-body streaming-report-body">
        ${html}
      </div>
    </div>
  `;

  // NOTE: auto-scroll is handled in app.js after this element is appended to the DOM.
  return container;
}

// ---- RetryButton ---------------------------------------------------------

/**
 * Render a retry button for stream timeout/error recovery.
 * @param {() => void} onRetry  Callback when the retry button is clicked.
 * @returns {HTMLElement}
 */
export function renderRetryButton(onRetry) {
  const container = document.createElement('div');
  container.className = 'mt-6 flex justify-center';

  const btn = document.createElement('button');
  btn.className =
    'btn-gradient px-6 py-2.5 rounded-xl text-white font-medium text-sm transition';
  btn.textContent = t('retry');
  btn.addEventListener('click', onRetry);

  container.appendChild(btn);
  return container;
}
