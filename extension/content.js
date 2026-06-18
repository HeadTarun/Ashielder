/**
 * AIshield Content Script
 * Runs on WhatsApp Web, Gmail, Twitter/X.
 * Extracts visible text content from the active conversation/email/tweet
 * and returns it to the sidepanel on request.
 */

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type !== 'AISHIELD_EXTRACT_CONTENT') return;

  try {
    const result = extractContent();
    sendResponse({ ok: true, ...result });
  } catch (err) {
    sendResponse({ ok: false, error: err.message, text: '', source: 'error' });
  }

  return true; // keep channel open for async
});

/* ── Per-site extractors ──────────────────────────────────── */

function extractContent() {
  const host = location.hostname;

  if (host.includes('web.whatsapp.com')) return extractWhatsApp();
  if (host.includes('mail.google.com'))  return extractGmail();
  if (host.includes('twitter.com') || host.includes('x.com')) return extractTwitter();

  return extractGeneric();
}

/* WhatsApp Web — last 10 visible messages in the open chat */
function extractWhatsApp() {
  const bubbles = [
    ...document.querySelectorAll(
      'div.message-in .selectable-text span[dir], ' +
      'div.message-out .selectable-text span[dir]'
    ),
  ];

  if (!bubbles.length) {
    // Fallback: any copyable text
    const fallback = [...document.querySelectorAll('.copyable-text')];
    const text = fallback.map(el => el.innerText.trim()).filter(Boolean).slice(-10).join('\n');
    return { source: 'whatsapp', text: text || 'No messages found in the open chat.' };
  }

  const messages = bubbles
    .map(el => el.innerText.trim())
    .filter(Boolean)
    .slice(-10)   // last 10 messages only — enough for context
    .join('\n');

  return { source: 'whatsapp', text: messages };
}

/* Gmail — subject + body of the open email */
function extractGmail() {
  const subject = document.querySelector('h2.hP')?.innerText?.trim() || '';
  const body    = document.querySelector('div.a3s.aiL')?.innerText?.trim()
               || document.querySelector('[role="main"]')?.innerText?.trim()
               || '';

  const text = [subject ? `Subject: ${subject}` : '', body].filter(Boolean).join('\n\n');
  return { source: 'gmail', text: text || 'No email open or content not accessible.' };
}

/* Twitter/X — focused tweet + replies visible on screen */
function extractTwitter() {
  // Primary tweet (article at top)
  const articles = [...document.querySelectorAll('article[data-testid="tweet"]')];
  const texts = articles
    .map(a => {
      const tweetText = a.querySelector('[data-testid="tweetText"]')?.innerText?.trim() || '';
      const author    = a.querySelector('[data-testid="User-Name"]')?.innerText?.trim() || '';
      return author ? `${author}:\n${tweetText}` : tweetText;
    })
    .filter(Boolean)
    .slice(0, 5);   // first 5 (main + replies)

  return {
    source: 'twitter',
    text: texts.join('\n\n---\n\n') || 'No tweets found on this page.',
  };
}

/* Generic fallback — clean visible body text */
function extractGeneric() {
  // Remove scripts, styles, nav, footer etc.
  const clone = document.body.cloneNode(true);
  ['script', 'style', 'noscript', 'svg', 'nav', 'footer', 'header'].forEach(tag => {
    clone.querySelectorAll(tag).forEach(el => el.remove());
  });

  const text = (clone.innerText || clone.textContent || '')
    .replace(/\n{3,}/g, '\n\n')  // collapse blank lines
    .trim()
    .slice(0, 3000);             // cap at 3 000 chars

  return { source: 'page', text: text || 'Could not extract content from this page.' };
}
