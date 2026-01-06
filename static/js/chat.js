const chatWindow = document.getElementById('chat-window');
const chatForm = document.getElementById('chat-form');
const questionInput = document.getElementById('question');
const includeSourcesToggle = document.getElementById('include-sources');
const sendButton = document.getElementById('send-btn');
const statusPill = document.getElementById('status-pill');
const modelPill = document.getElementById('model-pill');
const promptGrid = document.getElementById('prompt-grid');
const autocompleteList = document.getElementById('autocomplete-list');

let autocompleteTimeout;
let selectedSuggestionIndex = -1;

// Autocomplete functionality
questionInput.addEventListener('input', async (event) => {
  const text = event.target.value;
  const cursorPos = event.target.selectionStart;
  
  // Get the word/phrase being typed
  const beforeCursor = text.substring(0, cursorPos);
  const lastNewline = beforeCursor.lastIndexOf('\n');
  const currentLine = lastNewline === -1 ? beforeCursor : beforeCursor.substring(lastNewline + 1);
  
  clearTimeout(autocompleteTimeout);
  selectedSuggestionIndex = -1;
  
  if (currentLine.length < 2) {
    autocompleteList.innerHTML = '';
    return;
  }
  
  autocompleteTimeout = setTimeout(async () => {
    try {
      const res = await fetch(`/autocomplete?query=${encodeURIComponent(currentLine)}`);
      const data = await res.json();
      
      autocompleteList.innerHTML = '';
      selectedSuggestionIndex = -1;
      
      if (data.suggestions && data.suggestions.length > 0) {
        data.suggestions.forEach((suggestion, index) => {
          const li = document.createElement('li');
          li.className = 'autocomplete-item';
          li.textContent = suggestion;
          li.dataset.index = index;
          li.addEventListener('click', () => {
            selectSuggestion(suggestion);
          });
          li.addEventListener('mouseenter', () => {
            highlightSuggestion(index);
          });
          autocompleteList.appendChild(li);
        });
      }
    } catch (err) {
      console.error('Autocomplete error:', err);
    }
  }, 300);
});

// Keyboard navigation for autocomplete
questionInput.addEventListener('keydown', (event) => {
  const items = autocompleteList.querySelectorAll('.autocomplete-item');
  
  if (items.length === 0) return;
  
  switch (event.key) {
    case 'Tab':
      event.preventDefault();
      if (selectedSuggestionIndex === -1) {
        highlightSuggestion(0);
      } else {
        selectSuggestion(items[selectedSuggestionIndex].textContent);
      }
      break;
      
    case 'ArrowDown':
      event.preventDefault();
      if (selectedSuggestionIndex < items.length - 1) {
        highlightSuggestion(selectedSuggestionIndex + 1);
      }
      break;
      
    case 'ArrowUp':
      event.preventDefault();
      if (selectedSuggestionIndex > 0) {
        highlightSuggestion(selectedSuggestionIndex - 1);
      }
      break;
      
    case 'Enter':
      if (selectedSuggestionIndex !== -1) {
        event.preventDefault();
        selectSuggestion(items[selectedSuggestionIndex].textContent);
      }
      break;
      
    case 'Escape':
      autocompleteList.innerHTML = '';
      selectedSuggestionIndex = -1;
      break;
  }
});

function highlightSuggestion(index) {
  const items = autocompleteList.querySelectorAll('.autocomplete-item');
  items.forEach((item, i) => {
    if (i === index) {
      item.classList.add('autocomplete-item--selected');
    } else {
      item.classList.remove('autocomplete-item--selected');
    }
  });
  selectedSuggestionIndex = index;
}

function selectSuggestion(suggestion) {
  questionInput.value = suggestion;
  autocompleteList.innerHTML = '';
  selectedSuggestionIndex = -1;
  questionInput.focus();
}

// Hide autocomplete when clicking outside
document.addEventListener('click', (event) => {
  if (event.target !== questionInput) {
    autocompleteList.innerHTML = '';
    selectedSuggestionIndex = -1;
  }
});

function appendMessage(role, text, sources) {
  const message = document.createElement('div');
  message.className = `message message--${role}`;
  message.style.animation = 'slideIn 0.4s ease';

  // Avatar
  const avatarEl = document.createElement('div');
  avatarEl.className = 'message__avatar';
  avatarEl.textContent = role === 'user' ? '👤' : '🤖';
  message.appendChild(avatarEl);

  // Body container
  const bodyEl = document.createElement('div');
  bodyEl.className = 'message__body';

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

  bodyEl.appendChild(contentEl);

  if (sources && sources.length && role === 'ai') {
    const list = document.createElement('ul');
    list.className = 'source-list';
    sources.slice(0, 4).forEach((source, idx) => {
      const item = document.createElement('li');
      item.className = 'source-item';
      const title = source.metadata?.source || `Source ${idx + 1}`;
      const snippet = (source.content || '').slice(0, 140);
      item.textContent = `📄 ${title}: ${snippet}${snippet.length === 140 ? '...' : ''}`;
      list.appendChild(item);
    });
    bodyEl.appendChild(list);
  }

  message.appendChild(bodyEl);
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
