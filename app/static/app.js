/**
 * Multimodal Agentic Assistant — Client Application Logic (Vanilla JS)
 */

document.addEventListener('DOMContentLoaded', () => {
  // Global State
  let activeThreadId = null;
  let isAwaitingClarification = false;
  let selectedFiles = [];

  // DOM Elements
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const fileList = document.getElementById('file-list');
  const userQueryInput = document.getElementById('user-query-input');
  const queryLabel = document.getElementById('query-label');
  const agentForm = document.getElementById('agent-form');
  const submitBtn = document.getElementById('submit-btn');
  const submitBtnText = document.getElementById('submit-btn-text');
  const submitSpinner = document.getElementById('submit-spinner');
  const clearBtn = document.getElementById('clear-btn');
  const chatThread = document.getElementById('chat-thread');
  const statusPill = document.getElementById('status-pill');
  const statusText = document.getElementById('status-text');
  const sessionThreadLabel = document.getElementById('session-thread-label');
  const fusedContextText = document.getElementById('fused-context-text');
  const traceJsonText = document.getElementById('trace-json-text');
  const refreshTraceBtn = document.getElementById('refresh-trace-btn');
  const toggleFusedContext = document.getElementById('toggle-fused-context');
  const toggleTraceJson = document.getElementById('toggle-trace-json');

  // ---------------------------------------------------------------------------
  // Drag & Drop File Handlers
  // ---------------------------------------------------------------------------

  dropZone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = Array.from(e.dataTransfer.files);
    addFiles(files);
  });

  fileInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    addFiles(files);
  });

  function addFiles(files) {
    files.forEach(file => {
      if (!selectedFiles.some(f => f.name === file.name && f.size === file.size)) {
        selectedFiles.push(file);
      }
    });
    renderFileList();
  }

  function removeFile(index) {
    selectedFiles.splice(index, 1);
    renderFileList();
  }

  function renderFileList() {
    fileList.innerHTML = '';
    selectedFiles.forEach((file, index) => {
      const item = document.createElement('div');
      item.className = 'file-item';
      
      const ext = file.name.split('.').pop().toLowerCase();
      let fileType = 'DOC';
      if (['png', 'jpg', 'jpeg'].includes(ext)) fileType = 'IMG';
      if (['wav', 'mp3', 'm4a'].includes(ext)) fileType = 'AUD';

      item.innerHTML = `
        <div class="file-item-info">
          <span class="file-type">${fileType}</span>
          <span class="file-name" title="${file.name}">${file.name}</span>
          <span class="file-size" style="color: var(--text-dim); font-size: 10px;">(${(file.size / 1024).toFixed(1)} KB)</span>
        </div>
        <button type="button" class="file-remove-btn" data-index="${index}">&times;</button>
      `;
      fileList.appendChild(item);
    });

    // Wire remove buttons
    fileList.querySelectorAll('.file-remove-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const idx = parseInt(e.target.getAttribute('data-index'), 10);
        removeFile(idx);
      });
    });
  }

  clearBtn.addEventListener('click', resetAll);

  function resetAll() {
    activeThreadId = null;
    isAwaitingClarification = false;
    selectedFiles = [];
    renderFileList();
    userQueryInput.value = '';
    queryLabel.textContent = 'User Query / Instructions';
    userQueryInput.placeholder = "e.g., 'Summarize this document', 'Analyse tone', or leave empty for auto-planning...";
    submitBtnText.textContent = 'Submit Request';
    dropZone.style.pointerEvents = 'auto';
    dropZone.style.opacity = '1';
    sessionThreadLabel.textContent = 'No active session';
    updateStatusPill('ready', 'Ready');
    refreshTraceBtn.disabled = true;
    fusedContextText.textContent = 'No extraction performed yet.';
    traceJsonText.textContent = 'No active session trace.';
  }

  // ---------------------------------------------------------------------------
  // Form Submission (POST /session or POST /session/{thread_id}/reply)
  // ---------------------------------------------------------------------------

  agentForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const queryText = userQueryInput.value.trim();

    if (!isAwaitingClarification && selectedFiles.length === 0 && !queryText) {
      alert('Please upload at least one file or enter a query.');
      return;
    }

    if (isAwaitingClarification && !queryText) {
      alert('Please enter your clarification response.');
      return;
    }

    // Clear empty thread placeholder if present
    const placeholder = chatThread.querySelector('.empty-thread-placeholder');
    if (placeholder) placeholder.remove();

    if (!isAwaitingClarification) {
      // Step 1: Start New Session
      renderUserMsg(queryText, selectedFiles);
      await executeSessionRequest(queryText, selectedFiles);
    } else {
      // Step 2: Reply to Paused Clarification Session
      renderUserMsg(queryText, []);
      await executeReplyRequest(queryText);
    }
  });

  async function executeSessionRequest(userQuery, files) {
    setLoadingState(true, 'Extracting & Planning...');

    const formData = new FormData();
    formData.append('user_query', userQuery);
    files.forEach(file => formData.append('files', file));

    try {
      const response = await fetch('/session', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok || data.status === 'error') {
        renderErrorMsg(data.message || 'An error occurred during execution.');
        updateStatusPill('error', 'Error');
        return;
      }

      activeThreadId = data.thread_id;
      sessionThreadLabel.textContent = `Session Thread: ${activeThreadId.slice(0, 8)}...`;
      refreshTraceBtn.disabled = false;

      if (data.status === 'awaiting_clarification') {
        handleClarificationState(data);
      } else if (data.status === 'done') {
        handleDoneState(data);
      }

      await fetchTraceTelemetry();

    } catch (err) {
      renderErrorMsg(`Network or server error: ${err.message}`);
      updateStatusPill('error', 'Network Error');
    } finally {
      setLoadingState(false);
    }
  }

  async function executeReplyRequest(replyText) {
    setLoadingState(true, 'Resuming Execution...');

    const formData = new FormData();
    formData.append('reply', replyText);

    try {
      const response = await fetch(`/session/${activeThreadId}/reply`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok || data.status === 'error') {
        renderErrorMsg(data.message || 'Error resuming session.');
        updateStatusPill('error', 'Error');
        return;
      }

      if (data.status === 'awaiting_clarification') {
        handleClarificationState(data);
      } else if (data.status === 'done') {
        handleDoneState(data);
      }

      await fetchTraceTelemetry();

    } catch (err) {
      renderErrorMsg(`Failed to send reply: ${err.message}`);
      updateStatusPill('error', 'Network Error');
    } finally {
      setLoadingState(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Response Handlers
  // ---------------------------------------------------------------------------

  function handleClarificationState(data) {
    isAwaitingClarification = true;
    updateStatusPill('warning', 'Awaiting Clarification');

    // Lock file dropzone during reply
    dropZone.style.pointerEvents = 'none';
    dropZone.style.opacity = '0.5';

    queryLabel.textContent = 'Clarification Reply';
    userQueryInput.placeholder = 'Type your answer to the question above...';
    userQueryInput.value = '';
    submitBtnText.textContent = 'Send Clarification Reply';

    const card = document.createElement('div');
    card.className = 'chat-msg chat-msg-agent';
    card.innerHTML = `
      <div class="msg-card">
        <div class="clarify-banner">
          <div class="clarify-header">
            <span>Clarification Required</span>
          </div>
          <p class="clarify-question">${escapeHtml(data.question)}</p>
        </div>
      </div>
    `;
    chatThread.appendChild(card);
    chatThread.scrollTop = chatThread.scrollHeight;
  }

  function handleDoneState(data) {
    isAwaitingClarification = false;
    updateStatusPill('ready', 'Completed');

    // Unlock file dropzone
    dropZone.style.pointerEvents = 'auto';
    dropZone.style.opacity = '1';
    queryLabel.textContent = 'User Query / Instructions';
    userQueryInput.placeholder = "e.g., 'Summarize this document', 'Analyse tone'...";
    userQueryInput.value = '';
    submitBtnText.textContent = 'Submit Request';
    selectedFiles = [];
    renderFileList();

    const output = data.output || {};
    const trace = data.trace || [];

    const card = document.createElement('div');
    card.className = 'chat-msg chat-msg-agent';

    let contentHtml = '';

    if (output.summary) {
      const sum = output.summary;
      const bulletsHtml = sum.bullets.map(b => `<li>${escapeHtml(b)}</li>`).join('');
      contentHtml = `
        <div class="summary-one-line">${escapeHtml(sum.one_line)}</div>
        <ul class="summary-bullets">${bulletsHtml}</ul>
        <p class="summary-paragraph">${escapeHtml(sum.five_sentence)}</p>
      `;
    } else if (output.sentiment) {
      const sent = output.sentiment;
      contentHtml = `
        <div class="summary-one-line">Sentiment: <strong>${escapeHtml(sent.label.toUpperCase())}</strong> (Confidence: ${(sent.confidence * 100).toFixed(0)}%)</div>
        <p class="summary-paragraph">${escapeHtml(sent.justification)}</p>
      `;
    } else if (output.raw_text) {
      contentHtml = `<pre class="code-block">${escapeHtml(output.raw_text)}</pre>`;
    } else {
      contentHtml = `<p class="summary-paragraph">Task executed successfully (${escapeHtml(output.task_type || 'unspecified')}).</p>`;
    }

    const drawerHtml = renderTraceDrawerHtml(trace);

    card.innerHTML = `
      <div class="msg-card">
        <div class="msg-content">${contentHtml}</div>
        ${drawerHtml}
      </div>
    `;

    chatThread.appendChild(card);
    chatThread.scrollTop = chatThread.scrollHeight;

    // Attach drawer toggle listener
    const drawerHeader = card.querySelector('.trace-drawer-header');
    if (drawerHeader) {
      drawerHeader.addEventListener('click', () => {
        const body = card.querySelector('.trace-drawer-body');
        const icon = card.querySelector('.drawer-chevron');
        if (body.classList.contains('hidden')) {
          body.classList.remove('hidden');
          icon.textContent = '▲';
        } else {
          body.classList.add('hidden');
          icon.textContent = '▼';
        }
      });
    }
  }

  function renderTraceDrawerHtml(trace) {
    if (!trace || trace.length === 0) return '';

    const rows = trace.map((t, idx) => `
      <tr>
        <td>${idx + 1}</td>
        <td><strong>${escapeHtml(t.tool_name)}</strong></td>
        <td><span class="badge-${t.status}">${escapeHtml(t.status)}</span></td>
        <td>${t.latency_ms} ms</td>
        <td style="max-width: 200px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${escapeHtml(t.output_summary)}</td>
      </tr>
    `).join('');

    return `
      <div class="trace-drawer">
        <div class="trace-drawer-header">
          <span>Plan &amp; Execution Trace (${trace.length} step${trace.length > 1 ? 's' : ''})</span>
          <span class="drawer-chevron">▼</span>
        </div>
        <div class="trace-drawer-body hidden">
          <table class="trace-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Tool</th>
                <th>Status</th>
                <th>Latency</th>
                <th>Output Summary</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </div>
    `;
  }

  function renderUserMsg(queryText, files) {
    const card = document.createElement('div');
    card.className = 'chat-msg chat-msg-user';
    
    let fileBadgeHtml = '';
    if (files.length > 0) {
      fileBadgeHtml = `<div class="uploaded-files">UPLOADED: ${files.map(f => escapeHtml(f.name)).join(', ')}</div>`;
    }

    card.innerHTML = `
      <div class="msg-bubble">
        ${fileBadgeHtml}
        <div>${escapeHtml(queryText || '(Bare media upload — requesting auto-plan)')}</div>
      </div>
    `;
    chatThread.appendChild(card);
    chatThread.scrollTop = chatThread.scrollHeight;
  }

  function renderErrorMsg(errorText) {
    const card = document.createElement('div');
    card.className = 'chat-msg chat-msg-agent';
    card.innerHTML = `
      <div class="msg-card error-card">
        <div class="error-title">Error</div>
        <p class="error-message">${escapeHtml(errorText)}</p>
      </div>
    `;
    chatThread.appendChild(card);
    chatThread.scrollTop = chatThread.scrollHeight;
  }

  // ---------------------------------------------------------------------------
  // Inspection Panel (GET /session/{thread_id}/trace)
  // ---------------------------------------------------------------------------

  async function fetchTraceTelemetry() {
    if (!activeThreadId) return;
    try {
      const res = await fetch(`/session/${activeThreadId}/trace`);
      if (res.ok) {
        const data = await res.json();
        traceJsonText.textContent = JSON.stringify(data, null, 2);
      }
    } catch (err) {
      traceJsonText.textContent = `Error fetching trace: ${err.message}`;
    }
  }

  refreshTraceBtn.addEventListener('click', fetchTraceTelemetry);

  toggleFusedContext.addEventListener('click', () => {
    const body = document.getElementById('fused-context-body');
    body.classList.toggle('collapsed');
  });

  toggleTraceJson.addEventListener('click', () => {
    const body = document.getElementById('trace-json-body');
    body.classList.toggle('collapsed');
  });

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  function setLoadingState(isLoading, statusMsg = 'Processing...') {
    if (isLoading) {
      submitBtn.disabled = true;
      submitSpinner.classList.remove('hidden');
      submitBtnText.textContent = statusMsg;
      updateStatusPill('processing', statusMsg);
    } else {
      submitBtn.disabled = false;
      submitSpinner.classList.add('hidden');
    }
  }

  function updateStatusPill(stateClass, label) {
    statusPill.className = `status-pill status-${stateClass}`;
    statusText.textContent = label;
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
