/**
 * Internationalization module - supports Chinese (zh) and English (en).
 * Default language: zh
 * Persists user choice in localStorage.
 */

const translations = {
  zh: {
    // ---- Input view ----
    appTitle: '✈️ TravelCrew',
    appSubtitle: '多智能体协作旅行规划系统',
    formLabel: '你想去哪里旅行？',
    formHelper: '描述你的行程——目的地、天数、预算、同行人员及任何特殊偏好。',
    inputPlaceholder: '例如：我想去武汉玩3天，预算5000元，喜欢历史文化和美食...',
    inputGuide: `为了帮你创建个性化行程，你可以包含以下任何信息：
• 目的地（如：东京、巴黎、武汉）
• 出行日期或天数（如：5天、7月15-20日）
• 预算（如：5000元、150,000日元）
• 兴趣爱好（美食、博物馆、动漫、购物、自然、夜生活等）
• 节奏偏好（休闲、适中、紧凑）
• 饮食偏好（素食、清真、寿司、拉面等）
• 体力水平（低、中、高）
• 想避免的事物（人群、徒步、夜生活、海鲜等）`,
    startPlanning: '开始规划',
    emptyInputHint: '请先输入你的旅行需求',
    creatingPlan: '正在创建你的方案…',
    poweredBy: '由 LangGraph 驱动 · 实时数据 · 多币种支持',
    examples: [
      '🇨🇳 计划10月1日去北京4天，预算5000元，想去故宫、长城，品尝地道北京菜。节奏适中，体力中等，不想去拥挤的购物区。',
      '🇨🇳 想在成都待5天，预算6000元，喜欢熊猫、火锅、茶馆和四川文化。节奏悠闲，体力偏低，不想爬山。',
      '🇯🇵 计划7月15日去东京6天，预算18万日元，喜欢动漫、拉面、神社和购物。节奏适中，体力中等，想避开拥挤的夜生活区。',
      '🇫🇷 计划去巴黎5天，预算2500欧元，想参观博物馆、著名地标和法国美食。节奏悠闲，素食主义者，体力偏低，不想去游乐园。',
    ],

    // ---- Streaming view ----
    planningTitle: '正在为你规划行程…',
    finalizingTitle: '正在生成最终方案…',
    liveIndicator: '实时',
    analyzingTrip: '正在分析你的行程',
    finalizingPlan: '正在完善方案',
    retry: '重试',

    // ---- Node labels ----
    nodes: {
      intentparser: '解析旅行意图',
      information: '收集目的地信息',
      recommendation: '生成推荐方案',
      user_review: '等待用户确认',
      routing: '路线规划',
      critic: '质量审核',
      synthenrich: '加载图片和地图',
      synthesizer: '生成最终报告',
    },

    // ---- Review view ----
    reviewTitle: '行程预览',
    reviewSubtitle: '请查看推荐方案，确认或提出修改意见',
    dayLabel: '第 {n} 天',
    attractions: '景点',
    dining: '美食',
    hotel: '住宿',
    perNight: '/晚',
    rating: '评分',
    estimatedCost: '预计花费',
    free: '免费',
    noItinerary: '暂无行程数据',
    reviewPrompt: '需要调整吗？',
    reviewHelper: '告诉我们需要修改的地方，或确认方案以生成最终报告。',
    feedbackPlaceholder: '输入你的修改建议，例如：我不想去博物馆，想多加一些美食体验...',
    confirmPlan: '✅ 确认方案，生成详细行程',
    submitFeedback: '✏️ 提交修改意见',

    // ---- Shared UI labels (POI cards / map) ----
    websiteLabel: '官网',
    mapLabel: '地图',
    expandMapLabel: '点击展开地图',
    viewInteractiveMapLabel: '点击查看交互地图',
    submitting: '提交中…',
    confirming: '确认中…',

    // ---- Completed view ----
    doneTitle: '规划完成！',
    newTrip: '规划新行程',
    backHome: '← 新行程',
    downloadReport: '📥 下载 PDF',
    generatingPdf: '生成中…',
    copyShareLink: '🔗 复制分享链接',
    linkCopied: '✓ 已复制',
    loadingReport: '加载报告中…',
    failedToLoadReport: '报告加载失败',
    yourTravelReport: '你的旅行报告',
        streamingReportTitle: '📝 报告生成中…',

    // ---- Progress ----
    progressTitle: '实时进度',
    progressStep: '步骤 {current}/{total}',
    progressPercent: '{percent}%',
    progressElapsed: '已耗时 {time}',

    // ---- Language ----
    langSwitch: 'EN',
    langLabel: '语言',
  },

  en: {
    // ---- Input view ----
    appTitle: '✈️ TravelCrew',
    appSubtitle: 'Multi-Agent Collaborative Travel Planning',
    formLabel: 'Where do you want to go?',
    formHelper: 'Describe your trip — destination, duration, budget, travel companions, or any special preferences.',
    inputPlaceholder: 'e.g., I want to visit Tokyo for 5 days, budget $3000, interested in culture and food...',
    inputGuide: `To help me create a personalized itinerary, you can include any of the following:
• Destination (e.g., Tokyo, Paris)
• Travel dates or duration (e.g., 5 days, July 15–20)
• Budget (e.g., 150,000 JPY)
• Interests (food, museums, anime, shopping, nature, nightlife, etc.)
• Preferred pace (relaxed, moderate, intensive)
• Dietary preferences (vegetarian, halal, sushi, ramen, etc.)
• Physical activity level (low, moderate, high)
• Things you'd like to avoid (crowds, hiking, nightlife, seafood, etc.)`,
    startPlanning: 'Start Planning',
    emptyInputHint: 'Please enter your travel request first.',
    creatingPlan: 'Creating your plan…',
    poweredBy: 'Powered by LangGraph · Real-time data · Multi-currency support',
    examples: [
      "🇨🇳 I'm planning a 4-day trip to Beijing starting on October 1st with a budget of 5,000 RMB. I'd like to visit the Forbidden City, the Great Wall, and try local Beijing cuisine. I prefer a moderate pace, have a moderate fitness level, and would like to avoid crowded shopping areas.",
      "🇨🇳 I want to spend 5 days in Chengdu with a budget of 6,000 RMB. I'm interested in pandas, hot pot, tea houses, and Sichuan culture. I prefer a relaxed itinerary, have a low physical activity level, and don't want to do mountain hiking.",
      "🇯🇵 I'm planning a 6-day trip to Tokyo starting on July 15th with a budget of 180,000 JPY. I'm interested in anime, ramen, shrines, and shopping. I prefer a moderate pace, have a moderate fitness level, and would like to avoid crowded nightlife areas.",
      "🇫🇷 I'm planning a 5-day trip to Paris with a budget of 2,500 EUR. I'd like to visit museums, famous landmarks, and enjoy French cuisine. I prefer a relaxed itinerary, I'm vegetarian, have a low physical activity level, and don't want to visit amusement parks.",
    ],

    // ---- Streaming view ----
    planningTitle: 'Planning your trip…',
    finalizingTitle: 'Finalising your plan…',
    liveIndicator: 'Live',
    analyzingTrip: 'Analysing your trip',
    finalizingPlan: 'Finalising your plan',
    retry: 'Retry',

    // ---- Node labels ----
    nodes: {
      intentparser: 'Parsing Intent',
      information: 'Gathering Information',
      recommendation: 'Generating Recommendations',
      user_review: 'Awaiting Review',
      routing: 'Route Planning',
      critic: 'Quality Review',
      synthenrich: 'Loading Images & Maps',
      synthesizer: 'Generating Report',
    },

    // ---- Review view ----
    reviewTitle: 'Trip Preview',
    reviewSubtitle: 'Review the recommended plan, confirm or suggest changes',
    dayLabel: 'Day {n}',
    attractions: 'Attractions',
    dining: 'Dining',
    hotel: 'Hotel',
    perNight: '/night',
    rating: 'Rating',
    estimatedCost: 'Est. Cost',
    free: 'Free',
    noItinerary: 'No itinerary data available.',
    reviewPrompt: 'Would you like any changes?',
    reviewHelper: 'Tell us what to adjust, or confirm the plan to generate your final report.',
    feedbackPlaceholder: 'e.g., I don\'t want museums, add more food experiences...',
    confirmPlan: '✅ Looks Great, Generate Report',
    submitFeedback: '✏️ Revise Plan',

    // ---- Shared UI labels (POI cards / map) ----
    websiteLabel: 'Website',
    mapLabel: 'Map',
    expandMapLabel: 'Click to expand map',
    viewInteractiveMapLabel: 'View interactive map',
    submitting: 'Submitting…',
    confirming: 'Confirming…',

    // ---- Completed view ----
    doneTitle: 'Done!',
    newTrip: 'Plan New Trip',
    backHome: '← New trip',
    downloadReport: '📥 Download PDF',
    generatingPdf: 'Generating…',
    copyShareLink: '🔗 Copy Share Link',
    linkCopied: '✓ Copied',
    loadingReport: 'Loading report…',
    failedToLoadReport: 'Failed to load report',
    yourTravelReport: 'Your Travel Report',
        streamingReportTitle: '📝 Generating report…',

    // ---- Progress ----
    progressTitle: 'Live Progress',
    progressStep: 'Step {current}/{total}',
    progressPercent: '{percent}%',
    progressElapsed: 'Elapsed {time}',

    // ---- Language ----
    langSwitch: '中文',
    langLabel: 'Language',
  },
};

let currentLang = localStorage.getItem('lang') || 'zh';
const _instanceId = Math.random().toString(36).slice(2, 8);
console.log('[i18n] module loaded, instance:', _instanceId, '| initial lang:', currentLang);

/**
 * Look up a translation by dot-separated key, e.g. t('nodes.intentparser').
 * Falls back to the key itself if not found.
 * @param {string} key
 * @returns {string}
 */
export function t(key) {
  const keys = key.split('.');
  let val = translations[currentLang];
  for (const k of keys) {
    if (val && typeof val === 'object') val = val[k];
    else {
      if (key === 'langSwitch') console.log('[i18n] t(langSwitch) lang=' + currentLang + ' result=' + key + ' (fallback)');
      return key;
    }
  }
  const result = val || key;
  if (key === 'langSwitch') console.log('[i18n] t(langSwitch) lang=' + currentLang + ' result=' + result);
  return result;
}

/**
 * Return the localised example queries array.
 * @returns {string[]}
 */
export function getExamples() {
  return translations[currentLang].examples || translations.zh.examples;
}

/**
 * Get the current language code.
 * @returns {'zh'|'en'}
 */
export function getLang() {
  return currentLang;
}

/**
 * Set the language and persist to localStorage.
 * @param {'zh'|'en'} lang
 */
export function setLang(lang) {
  currentLang = lang;
  localStorage.setItem('lang', lang);
}

/**
 * Toggle between zh and en, persist, and return the new code.
 * @returns {'zh'|'en'}
 */
export function toggleLang() {
  const oldLang = currentLang;
  const newLang = currentLang === 'zh' ? 'en' : 'zh';
  setLang(newLang);
  console.log('[i18n] toggleLang:', oldLang, '->', newLang, '| t(langSwitch):', translations[newLang].langSwitch);
  return newLang;
}
