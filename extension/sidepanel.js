/* ────────────────────────────────────────────────────────────
   AIshield Extension — Sidepanel JS
   Talks to the ADK api_server (adk api_server) natively.
   Endpoints used:
     GET  /version                                    ← health
     POST /apps/{app}/users/{uid}/sessions/{sid}      ← create session
     POST /run                                        ← chat
──────────────────────────────────────────────────────────── */

const DEFAULT_FRONTEND_URL = window.TIBRAIN_EXTENSION_CONFIG?.defaultFrontendUrl || '';
const DEFAULT_API_URL      = window.TIBRAIN_EXTENSION_CONFIG?.defaultApiUrl || 'http://localhost:8000';
const APP_NAME             = window.TIBRAIN_EXTENSION_CONFIG?.appName || 'tri_model_agent';

/* ── Utilities ─────────────────────────────────────────── */
const normalizeUrl = (value) => {
  const url = String(value || '').trim().replace(/\/+$/, '');
  if (!url) return '';
  if (
    url.startsWith('https://') ||
    url.startsWith('http://localhost') ||
    url.startsWith('http://127.0.0.1')
  ) return url;
  return '';
};

const generateId = () =>
  'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });

const escapeHtml = (str) =>
  String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

const formatText = (text) =>
  escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');

const timeLabel = () => {
  const d = new Date();
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

/* ── DOM refs ──────────────────────────────────────────── */
const setupEl        = document.getElementById('setup');
const frameEl        = document.getElementById('app-frame');
const chatbotEl      = document.getElementById('chatbot');
const messagesEl     = document.getElementById('chat-messages');
const typingEl       = document.getElementById('typing');
const chatInput      = document.getElementById('chat-input');
const sendBtn        = document.getElementById('chat-send');
const connDot        = document.getElementById('conn-dot');
const connLabel      = document.getElementById('conn-label');
const footerSession  = document.getElementById('footer-session');

/* ── State ─────────────────────────────────────────────── */
let apiUrl    = DEFAULT_API_URL;
let userId    = '';
let sessionId = '';
let isSending = false;

/* ── Connection status ─────────────────────────────────── */
const setConnStatus = (state, text) => {
  connDot.className = `status-dot ${state}`;
  connLabel.textContent = text;
};

/* ── Verdict card config ───────────────────────────────── */
const VERDICT_CFG = {
  high_risk:      { color: '#ef4444', bg: 'rgba(239,68,68,0.10)',   border: '#ef4444', label: '🔴 HIGH RISK' },
  needs_review:   { color: '#f59e0b', bg: 'rgba(245,158,11,0.10)',  border: '#f59e0b', label: '🟡 NEEDS REVIEW' },
  low_risk:       { color: '#22c55e', bg: 'rgba(34,197,94,0.10)',   border: '#22c55e', label: '🟢 LOW RISK' },
  analysis_ready: { color: '#22d3ee', bg: 'rgba(34,211,238,0.10)', border: '#22d3ee', label: '🔵 ANALYZED' },
};

const getVerdictCfg = (v) => VERDICT_CFG[v] || VERDICT_CFG.analysis_ready;

/* ── Render helpers ────────────────────────────────────── */
function renderAnalysisCard(analysis) {
  if (!analysis || !analysis.verdict) return '';
  const cfg = getVerdictCfg(analysis.verdict);

  const scorePart = analysis.risk_score != null
    ? `<span class="card-score">${Math.round(analysis.risk_score)}% risk</span>`
    : '';

  const confPart = typeof analysis.confidence === 'number'
    ? `<div class="card-row"><span>Confidence</span><span>${(analysis.confidence * 100).toFixed(1)}%</span></div>`
    : '';

  const reasonPart = analysis.reason
    ? `<div class="card-reason">${escapeHtml(analysis.reason)}</div>`
    : '';

  const actions = Array.isArray(analysis.recommended_actions) && analysis.recommended_actions.length
    ? `<div class="card-actions-list">${
        analysis.recommended_actions
          .slice(0, 3)
          .map((a) => `<div class="card-action">• ${escapeHtml(a)}</div>`)
          .join('')
      }</div>`
    : '';

  return `
    <div class="analysis-card"
         style="border-left-color:${cfg.border};background:${cfg.bg}">
      <div class="card-verdict" style="color:${cfg.color}">
        ${cfg.label} ${scorePart}
      </div>
      ${confPart}
      ${reasonPart}
      ${actions}
    </div>`;
}

function renderSuggestions(suggestions) {
  if (!suggestions || !suggestions.length) return '';
  return `<div class="suggestions">${suggestions
    .slice(0, 4)
    .map((s) => `<button class="suggestion-chip" type="button">${escapeHtml(s)}</button>`)
    .join('')}</div>`;
}

/* ── Append a message ──────────────────────────────────── */
function appendMessage({ role, content, analysis, suggestions, time }) {
  const isUser = role === 'user';
  const wrap = document.createElement('div');
  wrap.className = `message-wrap ${isUser ? 'user' : 'assistant'}`;

  let html = `
    <div class="message ${isUser ? 'message--user' : 'message--assistant'}">
      ${formatText(content)}
    </div>
    <div class="msg-time">${time || timeLabel()}</div>`;

  if (!isUser) {
    if (analysis && analysis.verdict) html += renderAnalysisCard(analysis);
    if (suggestions && suggestions.length) html += renderSuggestions(suggestions);
  }

  wrap.innerHTML = html;

  // Wire suggestion chips
  wrap.querySelectorAll('.suggestion-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      chatInput.value = chip.textContent.trim();
      adjustTextareaHeight();
      sendBtn.disabled = false;
      sendMessage();
    });
  });

  messagesEl.appendChild(wrap);
  requestAnimationFrame(() => {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  });
}

/* ── Welcome message ───────────────────────────────────── */
function appendWelcome() {
  appendMessage({
    role: 'assistant',
    content:
      '🛡️ Welcome to **AIshield** — your AI-powered cybersecurity analyst.\n\n' +
      'I use three local threat-detection models (DistilBERT · LightURLNet · Embedding) combined with a Google ADK agent to detect:\n\n' +
      '• **Phishing** links and fake login pages\n' +
      '• **Malicious URLs** and drive-by downloads\n' +
      '• **Social engineering** in emails and messages\n' +
      '• **Brand impersonation** (Google, Microsoft, PayPal…)\n\n' +
      'Paste a URL, forward an email, or ask me anything about a suspicious message.',
    suggestions: [
      'Analyze this URL: http://paypal-secure-login.xyz/verify',
      'Is this email a phishing attempt?',
      'What is my current threat risk score?',
      'Explain how you detect phishing',
    ],
  });
}

/* ── API: health — uses ADK /version endpoint ──────────── */
async function checkHealth() {
  setConnStatus('checking', 'Connecting…');
  try {
    const res = await fetch(`${apiUrl}/version`, {
      signal: AbortSignal.timeout(6000),
    });
    if (res.ok) {
      const data = await res.json();
      const ver = data.version || data.adk_version || '';
      setConnStatus('online', `Online · ADK${ver ? ' ' + ver : ''}`);
      return true;
    }
    setConnStatus('offline', `Error ${res.status}`);
  } catch (_) {
    setConnStatus('offline', 'API offline');
  }
  return false;
}

/* ── API: ensure ADK session exists ────────────────────── */
async function ensureSession() {
  const url = `${apiUrl}/apps/${APP_NAME}/users/${encodeURIComponent(userId)}/sessions/${encodeURIComponent(sessionId)}`;
  try {
    // POST creates it if absent; GET returns it if it exists — both are safe to call
    await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
      signal: AbortSignal.timeout(8000),
    });
  } catch (_) {
    // Non-fatal — /run will also auto-create sessions in most ADK versions
  }
}

/* ── API: send chat message via ADK /run ───────────────── */
async function postChat(message) {
  await ensureSession();
  const res = await fetch(`${apiUrl}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      app_name: APP_NAME,
      user_id: userId,
      session_id: sessionId,
      new_message: {
        role: 'user',
        parts: [{ text: message }],
      },
    }),
    signal: AbortSignal.timeout(60000),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  return res.json(); // array of ADK events
}

/* ── Parse ADK events → {text, analysis} ──────────────── */
function parseAdkEvents(events) {
  if (!Array.isArray(events)) return { text: String(events), analysis: null };

  // Collect all text parts from model/agent turns
  const textParts = [];
  let analysis = null;

  for (const evt of events) {
    const parts = evt?.content?.parts || [];
    for (const p of parts) {
      if (p.text && p.text.trim()) textParts.push(p.text.trim());
    }
    // ADK agents can embed structured data in state_delta or custom fields
    const stateDelta = evt?.actions?.state_delta || {};
    if (stateDelta.last_analysis && !analysis) {
      analysis = stateDelta.last_analysis;
    }
    // Also check for function_response parts carrying analysis
    for (const p of parts) {
      if (p.function_response?.response?.analysis && !analysis) {
        analysis = p.function_response.response.analysis;
      }
    }
  }

  // Heuristic: try to extract a verdict from the text if no structured analysis
  if (!analysis && textParts.length) {
    const combined = textParts.join(' ').toLowerCase();
    const verdictMatch = combined.match(/verdict[:\s]+(high_risk|needs_review|low_risk|analysis_ready)/i);
    const scoreMatch   = combined.match(/risk[\s_]score[:\s]+(\d+(?:\.\d+)?)/i);
    if (verdictMatch) {
      analysis = {
        verdict:    verdictMatch[1].toLowerCase(),
        risk_score: scoreMatch ? parseFloat(scoreMatch[1]) : null,
        reason:     null,
      };
    }
  }

  return { text: textParts.join('\n\n') || 'Analysis complete.', analysis };
}

/* ── Send message flow ─────────────────────────────────── */
async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text || isSending) return;

  isSending = true;
  chatInput.value = '';
  adjustTextareaHeight();
  sendBtn.disabled = true;

  appendMessage({ role: 'user', content: text });

  typingEl.hidden = false;
  messagesEl.scrollTop = messagesEl.scrollHeight;

  try {
    const events = await postChat(text);   // ADK returns array of events
    typingEl.hidden = true;

    const { text: replyText, analysis } = parseAdkEvents(events);

    appendMessage({
      role: 'assistant',
      content: replyText,
      analysis,
      suggestions: null,
    });

    setConnStatus('online', 'Online · ADK');
  } catch (err) {
    typingEl.hidden = true;
    appendMessage({
      role: 'assistant',
      content:
        '⚠️ **Could not reach the ADK server.**\n\n' +
        `${err.message}\n\n` +
        'Make sure the server is running:\n' +
        '`adk api_server`\n\n' +
        'Then check the API URL in **Options → AI Chatbot**.',
    });
    setConnStatus('offline', 'Connection error');
  }

  isSending = false;
  sendBtn.disabled = false;
  chatInput.focus();
}

/* ── New conversation ──────────────────────────────────── */
function newConversation() {
  sessionId = generateId();
  chrome.storage.local.set({ sessionId });
  messagesEl.innerHTML = '';
  footerSession.textContent = `Session: ${sessionId.slice(0, 8)}…`;
  appendWelcome();
  checkHealth();
}

/* ── Textarea auto-resize ──────────────────────────────── */
function adjustTextareaHeight() {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
}

/* ── Input events ──────────────────────────────────────── */
chatInput.addEventListener('input', () => {
  adjustTextareaHeight();
  sendBtn.disabled = chatInput.value.trim() === '' || isSending;
});

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);

/* ── New chat button ───────────────────────────────────── */
document.getElementById('new-chat-btn').addEventListener('click', () => {
  if (confirm('Start a new conversation? Current chat will be cleared.')) {
    newConversation();
  }
});

/* ── Options buttons ───────────────────────────────────── */
document.getElementById('open-options').addEventListener('click', () => {
  chrome.runtime.openOptionsPage();
});
document.getElementById('open-options-chat').addEventListener('click', () => {
  chrome.runtime.openOptionsPage();
});

/* ── Scan page button ──────────────────────────────────── */
const scanBtn = document.getElementById('scan-page-btn');

const SOURCE_LABELS = {
  whatsapp: '📱 WhatsApp',
  gmail:    '📧 Gmail',
  twitter:  '🐦 Twitter/X',
  page:     '🌐 Page',
  error:    '⚠️ Error',
};

scanBtn.addEventListener('click', async () => {
  if (scanBtn.classList.contains('scanning')) return;

  scanBtn.classList.add('scanning');
  scanBtn.title = 'Extracting content…';

  try {
    // Get the current active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab?.id) {
      appendMessage({
        role: 'assistant',
        content: '⚠️ Could not find the active tab. Make sure a web page is open.',
      });
      return;
    }

    // Inject content script if not already present (covers pages loaded before extension)
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ['content.js'],
      });
    } catch (_) {
      // Already injected or restricted page — continue anyway
    }

    // Ask the content script to extract page content
    const response = await new Promise((resolve) => {
      chrome.tabs.sendMessage(
        tab.id,
        { type: 'AISHIELD_EXTRACT_CONTENT' },
        (res) => resolve(res || { ok: false, error: 'No response from page.' }),
      );
    });

    if (!response.ok) {
      appendMessage({
        role: 'assistant',
        content:
          `⚠️ **Could not extract content from this page.**\n\n` +
          `${response.error || 'Unknown error.'}\n\n` +
          `Try navigating to WhatsApp Web, Gmail, or Twitter/X first.`,
      });
      return;
    }

    const sourceLabel = SOURCE_LABELS[response.source] || '🌐 Page';
    const extracted   = (response.text || '').trim();

    if (!extracted) {
      appendMessage({
        role: 'assistant',
        content: `ℹ️ No content found on this page to analyze.`,
      });
      return;
    }

    // Build the message and send it directly to the agent
    const prompt =
      `[${sourceLabel} content detected — please analyze for threats]\n\n` +
      extracted;

    // Show what we grabbed in the UI
    appendMessage({
      role: 'user',
      content: `🔍 Scanning ${sourceLabel} content…`,
    });

    typingEl.hidden = false;
    messagesEl.scrollTop = messagesEl.scrollHeight;
    isSending = true;
    sendBtn.disabled = true;

    try {
      const events = await postChat(prompt);
      typingEl.hidden = true;
      const { text: replyText, analysis } = parseAdkEvents(events);
      appendMessage({ role: 'assistant', content: replyText, analysis });
      setConnStatus('online', 'Online · ADK');
    } catch (err) {
      typingEl.hidden = true;
      appendMessage({
        role: 'assistant',
        content: `⚠️ **Agent error:** ${err.message}`,
      });
      setConnStatus('offline', 'Connection error');
    } finally {
      isSending = false;
      sendBtn.disabled = false;
    }

  } catch (err) {
    appendMessage({
      role: 'assistant',
      content: `⚠️ **Scan failed:** ${err.message}`,
    });
  } finally {
    scanBtn.classList.remove('scanning');
    scanBtn.title = 'Scan page content (WhatsApp · Gmail · Twitter)';
  }
});


/* ── Init ──────────────────────────────────────────────── */
async function init() {
  const stored = await chrome.storage.local.get([
    'frontendUrl',
    'apiUrl',
    'panelMode',
    'userId',
    'sessionId',
  ]);

  const mode = stored.panelMode || 'chatbot';
  apiUrl = normalizeUrl(stored.apiUrl) || DEFAULT_API_URL;

  if (mode === 'chatbot') {
    /* ── Chatbot mode ── */
    frameEl.hidden = true;
    setupEl.hidden = true;
    chatbotEl.hidden = false;

    // Auto-generate identity (fake login — no auth needed)
    userId = stored.userId || `ext_${generateId().replace(/-/g, '').slice(0, 12)}`;
    sessionId = stored.sessionId || generateId();
    await chrome.storage.local.set({ userId, sessionId });

    footerSession.textContent = `Session: ${sessionId.slice(0, 8)}…`;

    // Render welcome & check health in parallel
    appendWelcome();
    checkHealth();

  } else {
    /* ── Iframe mode (existing behavior) ── */
    chatbotEl.hidden = true;
    const frontendUrl = normalizeUrl(stored.frontendUrl || DEFAULT_FRONTEND_URL);

    if (!frontendUrl) {
      setupEl.hidden = false;
      frameEl.hidden = true;
    } else {
      frameEl.src = frontendUrl;
      frameEl.hidden = false;
      setupEl.hidden = true;
    }
  }
}

init();
