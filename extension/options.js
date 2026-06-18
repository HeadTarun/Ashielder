const DEFAULT_FRONTEND_URL = window.TIBRAIN_EXTENSION_CONFIG?.defaultFrontendUrl || '';
const DEFAULT_API_URL = window.TIBRAIN_EXTENSION_CONFIG?.defaultApiUrl || 'http://localhost:8000';

const frontendInput = document.getElementById('frontend-url');
const apiInput = document.getElementById('api-url');
const statusEl = document.getElementById('status');
const iframeSettings = document.getElementById('iframe-settings');
const chatbotSettings = document.getElementById('chatbot-settings');
const apiStatusContainer = document.getElementById('api-status-container');
const apiStatusDot = document.getElementById('api-status-dot');
const apiStatusText = document.getElementById('api-status-text');

const normalizeUrl = (value) => String(value || '').trim().replace(/\/+$/, '');

const isAllowedUrl = (url) =>
  url.startsWith('https://') ||
  url.startsWith('http://localhost') ||
  url.startsWith('http://127.0.0.1');

const setStatus = (message, ok = true) => {
  statusEl.textContent = message;
  statusEl.style.color = ok ? '#22d3ee' : '#f87171';
  window.setTimeout(() => {
    if (statusEl.textContent === message) statusEl.textContent = '';
  }, 3000);
};

const updateModeUI = (mode) => {
  if (mode === 'chatbot') {
    iframeSettings.hidden = true;
    chatbotSettings.hidden = false;
    document.getElementById('mode-opt-chatbot').classList.add('selected');
    document.getElementById('mode-opt-iframe').classList.remove('selected');
  } else {
    iframeSettings.hidden = false;
    chatbotSettings.hidden = true;
    document.getElementById('mode-opt-iframe').classList.add('selected');
    document.getElementById('mode-opt-chatbot').classList.remove('selected');
  }
};

// Mode radio buttons
document.querySelectorAll('input[name="panel-mode"]').forEach((radio) => {
  radio.addEventListener('change', async (e) => {
    const mode = e.target.value;
    updateModeUI(mode);
    await chrome.storage.local.set({ panelMode: mode });
    setStatus(`Mode saved: ${mode === 'chatbot' ? 'AI Chatbot' : 'Iframe App'}`);
  });
});

// Save frontend URL
document.getElementById('save').addEventListener('click', async () => {
  const normalized = normalizeUrl(frontendInput.value);
  if (!isAllowedUrl(normalized)) {
    setStatus('Use an https:// URL, localhost, or 127.0.0.1.', false);
    return;
  }
  await chrome.storage.local.set({ frontendUrl: normalized });
  frontendInput.value = normalized;
  setStatus('Frontend URL saved.');
});

document.getElementById('use-local').addEventListener('click', async () => {
  const url = 'http://localhost:5173';
  await chrome.storage.local.set({ frontendUrl: url });
  frontendInput.value = url;
  setStatus('Using localhost:5173.');
});

// Save API URL
document.getElementById('save-api').addEventListener('click', async () => {
  const normalized = normalizeUrl(apiInput.value);
  if (!isAllowedUrl(normalized)) {
    setStatus('Use http://localhost or http://127.0.0.1 URL.', false);
    return;
  }
  await chrome.storage.local.set({ apiUrl: normalized });
  apiInput.value = normalized;
  setStatus('API URL saved.');
});

document.getElementById('use-local-api').addEventListener('click', async () => {
  const url = 'http://localhost:8000';
  await chrome.storage.local.set({ apiUrl: url });
  apiInput.value = url;
  setStatus('Using localhost:8000.');
});

// Test connection
document.getElementById('test-connection').addEventListener('click', async () => {
  const url = normalizeUrl(apiInput.value) || DEFAULT_API_URL;
  const testLabel = document.getElementById('test-label');
  testLabel.textContent = '⏳ Testing...';
  apiStatusContainer.hidden = false;
  apiStatusDot.className = 'api-status__dot checking';
  apiStatusText.textContent = 'Connecting to ADK server...';

  try {
    const res = await fetch(`${url}/version`, {
      signal: AbortSignal.timeout(5000),
    });
    if (res.ok) {
      const data = await res.json();
      apiStatusDot.className = 'api-status__dot online';
      const ver = data.version || data.adk_version || '?';
      apiStatusText.textContent = `✓ Connected — ADK ${ver}`;
      testLabel.textContent = '✓ Connection OK';
    } else {
      apiStatusDot.className = 'api-status__dot offline';
      apiStatusText.textContent = `✗ Error ${res.status}: ${res.statusText}`;
      testLabel.textContent = '⚡ Test Connection';
    }
  } catch (err) {
    apiStatusDot.className = 'api-status__dot offline';
    apiStatusText.textContent = `✗ Unreachable — ${err.message}`;
    testLabel.textContent = '⚡ Test Connection';
  }
});

// Load saved settings on page open
chrome.storage.local.get(['frontendUrl', 'apiUrl', 'panelMode']).then((stored) => {
  frontendInput.value = stored.frontendUrl || DEFAULT_FRONTEND_URL;
  apiInput.value = stored.apiUrl || DEFAULT_API_URL;
  const mode = stored.panelMode || 'chatbot';
  const radio = document.querySelector(`input[name="panel-mode"][value="${mode}"`);
  if (radio) radio.checked = true;
  updateModeUI(mode);
});
