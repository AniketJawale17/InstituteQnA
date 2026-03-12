const chatWindow = document.getElementById('chat-window');
const chatForm = document.getElementById('chat-form');
const questionInput = document.getElementById('question');
const includeSourcesToggle = document.getElementById('include-sources');
const sendButton = document.getElementById('send-btn');
const statusPill = document.getElementById('status-pill');
const modelPill = document.getElementById('model-pill');
const promptGrid = document.getElementById('prompt-grid');
const autocompleteList = document.getElementById('autocomplete-list');
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebarBackdrop = document.getElementById('sidebar-backdrop');
const clearChatButton = document.getElementById('clear-chat');
const charCount = document.getElementById('char-count');
const welcomeTime = document.getElementById('welcome-time');
const toastContainer = document.getElementById('toast-container');

let autocompleteTimeout;
let selectedSuggestionIndex = -1;
let typingMessage = null;

function formatTime(date = new Date()) {
  return new Intl.DateTimeFormat([], {
    hour: 'numeric',
    minute: '2-digit',
  }).format(date);
}

function scrollChatToEnd() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function resetTextareaHeight() {
  questionInput.style.height = 'auto';
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 220)}px`;
}

function updateCharCount() {
  const length = questionInput.value.length;
  charCount.textContent = `${length} / 2000`;
  charCount.classList.toggle('char-count--warn', length >= 1600 && length < 2000);
  charCount.classList.toggle('char-count--limit', length >= 2000);
}

function showToast(message) {
  if (!toastContainer) return;

  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.textContent = message;
  toastContainer.appendChild(toast);

  window.setTimeout(() => {
    toast.remove();
  }, 2600);
}

function setStatus(state, label) {
  if (!statusPill) return;

  const textNode = statusPill.querySelector('.status-text');
  statusPill.dataset.state = state;

  if (textNode) {
    textNode.textContent = label;
  } else {
    statusPill.textContent = label;
  }
}

function createMessageMeta(role) {
  const metaEl = document.createElement('div');
  metaEl.className = 'message__meta';

  const roleEl = document.createElement('span');
  roleEl.className = 'message__role';
  roleEl.textContent = role === 'user' ? 'You' : 'Assistant';

  const timeEl = document.createElement('span');
  timeEl.className = 'message__time';
  timeEl.textContent = formatTime();

  metaEl.append(roleEl, timeEl);
  return metaEl;
}

function createTypingIndicator() {
  const message = document.createElement('div');
  message.className = 'message message--ai';
  message.dataset.typing = 'true';

  const avatarEl = document.createElement('div');
  avatarEl.className = 'message__avatar';
  avatarEl.textContent = 'AI';

  const bodyEl = document.createElement('div');
  bodyEl.className = 'message__body';

  const indicatorEl = document.createElement('div');
  indicatorEl.className = 'typing-indicator';
  indicatorEl.innerHTML = [
    '<span class="typing-indicator__dot"></span>',
    '<span class="typing-indicator__dot"></span>',
    '<span class="typing-indicator__dot"></span>',
  ].join('');

  bodyEl.appendChild(indicatorEl);
  message.append(avatarEl, bodyEl);
  return message;
}

function removeTypingIndicator() {
  if (typingMessage) {
    typingMessage.remove();
    typingMessage = null;
  }
}

function appendSources(bodyEl, sources) {
  if (!sources || !sources.length) return;

  const list = document.createElement('ul');
  list.className = 'source-list';

  sources.slice(0, 4).forEach((source, index) => {
    const item = document.createElement('li');
    item.className = 'source-item';

    const title = document.createElement('span');
    title.className = 'source-item__title';
    title.textContent = source.metadata?.source || `Source ${index + 1}`;

    const snippet = document.createElement('span');
    const content = (source.content || '').trim();
    snippet.textContent = content.length > 180 ? `${content.slice(0, 180)}...` : content || 'No snippet available.';

    item.append(title, snippet);
    list.appendChild(item);
  });

  bodyEl.appendChild(list);
}

function appendMessage(role, text, sources = []) {
  const message = document.createElement('div');
  message.className = `message message--${role}`;

  const avatarEl = document.createElement('div');
  avatarEl.className = 'message__avatar';
  avatarEl.textContent = role === 'user' ? 'YOU' : 'AI';

  const bodyEl = document.createElement('div');
  bodyEl.className = 'message__body';

  const contentEl = document.createElement('div');
  contentEl.className = 'message__content';

  if (role === 'user') {
    contentEl.textContent = text;
  } else if (typeof marked !== 'undefined' && marked.parse) {
    contentEl.innerHTML = marked.parse(text);
  } else {
    contentEl.textContent = text;
  }

  bodyEl.appendChild(contentEl);

  if (role === 'ai') {
    appendSources(bodyEl, sources);
  }

  bodyEl.appendChild(createMessageMeta(role));
  message.append(avatarEl, bodyEl);
  chatWindow.appendChild(message);
  scrollChatToEnd();
}

function setLoading(isLoading) {
  sendButton.disabled = isLoading;
  sendButton.classList.toggle('loading', isLoading);
  sendButton.setAttribute('aria-busy', String(isLoading));
}

function closeAutocomplete() {
  autocompleteList.innerHTML = '';
  selectedSuggestionIndex = -1;
}

function highlightSuggestion(index) {
  const items = autocompleteList.querySelectorAll('.autocomplete-item');
  items.forEach((item, itemIndex) => {
    item.classList.toggle('autocomplete-item--selected', itemIndex === index);
  });
  selectedSuggestionIndex = index;
}

function selectSuggestion(suggestion) {
  questionInput.value = suggestion;
  closeAutocomplete();
  updateCharCount();
  resetTextareaHeight();
  questionInput.focus();
}

function openSidebar() {
  if (!sidebar) return;
  sidebar.classList.remove('sidebar--collapsed');
  sidebarBackdrop?.classList.add('sidebar-backdrop--visible');
}

function closeSidebar() {
  if (!sidebar) return;
  sidebar.classList.add('sidebar--collapsed');
  sidebarBackdrop?.classList.remove('sidebar-backdrop--visible');
}

function toggleSidebar() {
  if (!sidebar) return;
  sidebar.classList.contains('sidebar--collapsed') ? openSidebar() : closeSidebar();
}

function syncInputUI() {
  updateCharCount();
  resetTextareaHeight();
}

questionInput.addEventListener('input', async (event) => {
  syncInputUI();

  const text = event.target.value;
  const cursorPos = event.target.selectionStart;
  const beforeCursor = text.substring(0, cursorPos);
  const lastNewline = beforeCursor.lastIndexOf('\n');
  const currentLine = lastNewline === -1 ? beforeCursor : beforeCursor.substring(lastNewline + 1);

  clearTimeout(autocompleteTimeout);
  selectedSuggestionIndex = -1;

  if (currentLine.trim().length < 2) {
    closeAutocomplete();
    return;
  }

  autocompleteTimeout = window.setTimeout(async () => {
    try {
      const res = await fetch(`/autocomplete?query=${encodeURIComponent(currentLine)}`);
      const data = await res.json();

      closeAutocomplete();

      if (!data.suggestions?.length) {
        return;
      }

      data.suggestions.forEach((suggestion, index) => {
        const li = document.createElement('li');
        li.className = 'autocomplete-item';
        li.textContent = suggestion;
        li.dataset.index = index;
        li.addEventListener('click', () => selectSuggestion(suggestion));
        li.addEventListener('mouseenter', () => highlightSuggestion(index));
        autocompleteList.appendChild(li);
      });
    } catch (error) {
      console.error('Autocomplete error:', error);
    }
  }, 220);
});

questionInput.addEventListener('keydown', (event) => {
  const items = autocompleteList.querySelectorAll('.autocomplete-item');

  switch (event.key) {
    case 'Tab':
      if (!items.length) return;
      event.preventDefault();
      if (selectedSuggestionIndex === -1) {
        highlightSuggestion(0);
      } else {
        selectSuggestion(items[selectedSuggestionIndex].textContent);
      }
      break;
    case 'ArrowDown':
      if (!items.length) return;
      event.preventDefault();
      highlightSuggestion(Math.min(selectedSuggestionIndex + 1, items.length - 1));
      break;
    case 'ArrowUp':
      if (!items.length) return;
      event.preventDefault();
      highlightSuggestion(Math.max(selectedSuggestionIndex - 1, 0));
      break;
    case 'Enter':
      if (selectedSuggestionIndex !== -1 && items.length) {
        event.preventDefault();
        selectSuggestion(items[selectedSuggestionIndex].textContent);
      } else if (!event.shiftKey) {
        event.preventDefault();
        chatForm.requestSubmit();
      }
      break;
    case 'Escape':
      closeAutocomplete();
      break;
    default:
      break;
  }
});

document.addEventListener('click', (event) => {
  if (event.target !== questionInput && !autocompleteList.contains(event.target)) {
    closeAutocomplete();
  }
});

sidebarToggle?.addEventListener('click', toggleSidebar);
sidebarBackdrop?.addEventListener('click', closeSidebar);

if (sidebar && window.matchMedia('(max-width: 960px)').matches) {
  sidebar.classList.add('sidebar--collapsed');
}

clearChatButton?.addEventListener('click', () => {
  const messageNodes = Array.from(chatWindow.querySelectorAll('.message'));
  messageNodes.forEach((node) => {
    if (!node.classList.contains('welcome-message')) {
      node.remove();
    }
  });

  removeTypingIndicator();
  showToast('Conversation cleared');
  scrollChatToEnd();
});

async function sendQuestion(question) {
  appendMessage('user', question);
  setLoading(true);

  typingMessage = createTypingIndicator();
  chatWindow.appendChild(typingMessage);
  scrollChatToEnd();

  try {
    const response = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        return_sources: includeSourcesToggle.checked,
      }),
    });

    removeTypingIndicator();

    if (!response.ok) {
      const errorText = await response.text();
      appendMessage('ai', `Error: ${errorText}`);
      showToast('Request failed');
      return;
    }

    const data = await response.json();
    appendMessage('ai', data.answer || 'No answer returned.', data.sources || []);
  } catch (error) {
    removeTypingIndicator();
    appendMessage('ai', `Request failed: ${error}`);
    showToast('Network error while sending question');
  } finally {
    setLoading(false);
  }
}

chatForm.addEventListener('submit', (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();

  if (!question) {
    return;
  }

  questionInput.value = '';
  syncInputUI();
  closeAutocomplete();

  if (window.matchMedia('(max-width: 960px)').matches) {
    closeSidebar();
  }

  sendQuestion(question);
});

promptGrid?.addEventListener('click', (event) => {
  const target = event.target.closest('.prompt-chip');
  if (!target) return;

  questionInput.value = target.textContent.trim();
  syncInputUI();
  questionInput.focus();

  if (window.matchMedia('(max-width: 960px)').matches) {
    closeSidebar();
  }
});

async function refreshStatus() {
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error('Unavailable');

    const data = await res.json();

    if (data.status === 'healthy') {
      setStatus('ready', 'System ready');
    } else {
      setStatus('offline', 'RAG not ready');
    }
  } catch (error) {
    setStatus('unknown', 'Status unknown');
  }
}

async function loadMeta() {
  try {
    const res = await fetch('/ui/meta');
    if (!res.ok) throw new Error('No meta');

    const data = await res.json();
    modelPill.textContent = `Model: ${data.provider} · ${data.model}`;
  } catch (error) {
    modelPill.textContent = 'Model info unavailable';
  }
}

welcomeTime.textContent = formatTime();
syncInputUI();
refreshStatus();
loadMeta();
questionInput.focus();