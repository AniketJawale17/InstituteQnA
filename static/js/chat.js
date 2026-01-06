const chatWindow = document.getElementById('chat-window');
const chatForm = document.getElementById('chat-form');
const questionInput = document.getElementById('question');
const includeSourcesToggle = document.getElementById('include-sources');
const sendButton = document.getElementById('send-btn');
const statusPill = document.getElementById('status-pill');
const modelPill = document.getElementById('model-pill');
const promptGrid = document.getElementById('prompt-grid');

function appendMessage(role, text, sources) {
  const message = document.createElement('div');
  message.className = `message message--${role}`;

  const roleEl = document.createElement('div');
  roleEl.className = 'message__role';
  roleEl.textContent = role === 'user' ? 'You' : 'Assistant';

  const contentEl = document.createElement('div');
  contentEl.className = 'message__content';
  // For user messages, use plain text. For AI messages, render markdown as HTML
  if (role === 'user') {
    contentEl.textContent = text;
  } else {
    // Check if marked is available, fallback to plain text if not
    if (typeof marked !== 'undefined' && marked.parse) {
      contentEl.innerHTML = marked.parse(text);
    } else {
      console.error('Marked.js not loaded');
      contentEl.textContent = text;
    }
  }

  message.appendChild(roleEl);
  message.appendChild(contentEl);

  if (sources && sources.length) {
    const list = document.createElement('ul');
    list.className = 'source-list';
    sources.slice(0, 4).forEach((source, idx) => {
      const item = document.createElement('li');
      item.className = 'source-item';
      const title = source.metadata?.source || `Source ${idx + 1}`;
      const snippet = (source.content || '').slice(0, 140);
      item.textContent = `${title}: ${snippet}${snippet.length === 140 ? '...' : ''}`;
      list.appendChild(item);
    });
    message.appendChild(list);
  }

  chatWindow.appendChild(message);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function setLoading(isLoading) {
  sendButton.disabled = isLoading;
  if (isLoading) {
    sendButton.classList.add('loading');
  } else {
    sendButton.classList.remove('loading');
  }
}

async function sendQuestion(question) {
  appendMessage('user', question);
  setLoading(true);

  try {
    const response = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        return_sources: includeSourcesToggle.checked,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      appendMessage('ai', `Error: ${errorText}`);
      return;
    }

    const data = await response.json();
    appendMessage('ai', data.answer || 'No answer returned.', data.sources);
  } catch (err) {
    appendMessage('ai', `Request failed: ${err}`);
  } finally {
    setLoading(false);
  }
}

chatForm.addEventListener('submit', (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  questionInput.value = '';
  sendQuestion(question);
});

if (promptGrid) {
  promptGrid.addEventListener('click', (event) => {
    const target = event.target;
    if (target.classList.contains('prompt-chip')) {
      questionInput.value = target.textContent;
      questionInput.focus();
    }
  });
}

async function refreshStatus() {
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error('Unavailable');
    const data = await res.json();
    if (data.status === 'healthy') {
      statusPill.textContent = 'System ready';
      statusPill.style.borderColor = 'rgba(74, 222, 128, 0.6)';
      statusPill.style.background = 'rgba(74, 222, 128, 0.08)';
    } else {
      statusPill.textContent = 'RAG not ready';
      statusPill.style.borderColor = 'rgba(239, 68, 68, 0.5)';
      statusPill.style.background = 'rgba(239, 68, 68, 0.08)';
    }
  } catch (err) {
    statusPill.textContent = 'Status unknown';
    statusPill.style.borderColor = 'rgba(239, 68, 68, 0.5)';
    statusPill.style.background = 'rgba(239, 68, 68, 0.08)';
  }
}

async function loadMeta() {
  try {
    const res = await fetch('/ui/meta');
    if (!res.ok) throw new Error('No meta');
    const data = await res.json();
    modelPill.textContent = `Model: ${data.provider} · ${data.model}`;
  } catch (err) {
    modelPill.textContent = 'Model info unavailable';
  }
}

refreshStatus();
loadMeta();
