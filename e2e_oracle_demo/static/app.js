/*
 * Interface notes: same-origin /api; static assets served at /static/.
 * Only explicit submissions may invoke a model/embedding provider or mutate data.
 * Local storage allowlist: selected mode/session and manual scenario verdicts/notes.
 * Session bodies, API payloads, messages, document text and keys are never persisted here.
 * Learning checklist (manual, not evidence of mastery):
 * [ ] Explain requested vs effective mode and synthetic model/embedding provenance.
 * [ ] Explain demo user scoping, session lifecycle and deterministic helper actions.
 * [ ] Verify approvals, retrieval, persisted evidence and failure/partial-stream cases.
 * [ ] Compare STATIC/LIVE outcomes and manually record each scenario's evidence.
 */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const UI_KEY = 'agno-oracle-fresh.ui.v1';
  const CHECKLIST_KEY = 'agno-oracle-fresh.checklist.v1';
  const state = {
    status: null, statusError: false, sessionId: null, sessionMode: null,
    sessionName: '', sessions: [], scenarios: [], checklist: { static: {}, live: {} },
    busy: false, controller: null, approval: null, contextVersion: 0,
    historyVersion: 0, sessionsVersion: 0, scenarioVersion: 0, statusVersion: 0,
    observedMode: null, payload: null, panel: 'chat', dialogResolve: null,
    liveCall: 'unverified',
  };
  const actionHelp = {
    auto: 'STATIC uses deterministic routing; LIVE can use the configured model. Nothing is submitted until you send.',
    add_numbers: 'Deterministic helper: add the supplied numbers. No LLM is required.',
    remember_note: 'Deterministic helper: store a note in the selected demo user’s memory.',
    recall_notes: 'Deterministic helper: retrieve saved notes for the selected demo user.',
    save_learning: 'Deterministic helper: persist a reusable learning for later retrieval.',
    find_learning: 'Deterministic helper: look up previously saved learnings.',
    search_documents: 'Deterministic helper: retrieve indexed documents. Embedding provenance follows the effective mode.',
    publish_report: 'Deterministic helper: request report publication. Review any approval requirements returned by the server.',
    make_report: 'Deterministic helper: run the report action for this component.',
  };
  const argumentHelp = {
    add_numbers: 'Example: {"a": 12, "b": 30}. Both values must be finite numbers.',
    remember_note: 'Example: {"note": "My unique demo marker"}. Scoped to the selected user.',
    recall_notes: 'Use {} to retrieve the selected user’s saved notes.',
    save_learning: 'Example: {"rule": "Verify Oracle evidence before marking a pass"}.',
    find_learning: 'Example: {"query": "Oracle evidence"}.',
    search_documents: 'Example: {"query": "team handbook"}. Uses this mode’s vector space.',
    publish_report: 'Use {}. Pauses for approval of a fictional report; no external publication occurs.',
    make_report: 'Example: {"rows": 500}. Default 500; allowed range 1–1500.',
  };
  const argumentDefaults = {
    auto: {}, add_numbers: { a: 12, b: 30 }, remember_note: { note: 'My unique demo marker' },
    recall_notes: {}, save_learning: { rule: 'Verify Oracle evidence before marking a pass' },
    find_learning: { query: 'Oracle evidence' }, search_documents: { query: 'team handbook' },
    publish_report: {}, make_report: { rows: 500 },
  };

  function printable(value) {
    if (typeof value === 'string') return value;
    if (value === undefined || value === null) return '';
    try { return JSON.stringify(value, null, 2); } catch { return String(value); }
  }
  function node(tag, className, value) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (value !== undefined) element.textContent = printable(value);
    return element;
  }
  function modeName(value) { return value === 'live' ? 'LIVE' : 'STATIC'; }
  function validMode(value) { return value === 'static' || value === 'live'; }
  function describeLiveStatus(requested, ready, call) {
    if (requested !== 'live') return { tone: 'neutral', text: 'Static selected · no LLM call' };
    if (ready === false) return { tone: 'neutral', text: 'Live not configured · static fallback' };
    if (ready !== true) return { tone: 'neutral', text: 'Live availability unknown · refresh status' };
    if (call === 'verified') return { tone: 'verified', text: 'Live response verified · this page session' };
    if (call === 'failed') return { tone: 'failed', text: 'Live request incomplete · inspect the error' };
    if (call === 'pending') return { tone: 'pending', text: 'Live request in progress · not yet verified' };
    return { tone: 'pending', text: 'Live configured · no verified response yet' };
  }
  function updateLiveIndicator() {
    const indicator = describeLiveStatus(requestedMode(), state.status?.model?.ready, state.liveCall);
    $('live-indicator').dataset.tone = indicator.tone;
    $('live-indicator-text').textContent = indicator.text;
  }
  function requestedMode() { return $('mode').value; }
  function effectiveMode() {
    if (state.observedMode) return state.observedMode;
    if (requestedMode() === 'static') return 'static';
    return state.status?.model?.ready === false ? 'static' : 'live';
  }
  function scope() {
    return { user_id: $('user').value, mode: effectiveMode(), component: $('component').value };
  }
  function scopeKey(value = scope()) { return `${value.user_id}/${value.mode}/${value.component}`; }
  function query(values) { return new URLSearchParams(values).toString(); }
  function sessionPath(id, context = scope()) {
    return `/api/sessions/${encodeURIComponent(id)}?${query(context)}`;
  }
  function rememberSelection() {
    try {
      // A session is deliberately restored only in the default Alice/Agent scope.
      const restorable = $('user').value === 'alice' && $('component').value === 'agent';
      localStorage.setItem(UI_KEY, JSON.stringify({
        mode: requestedMode(), session_id: restorable ? state.sessionId : null,
      }));
    } catch (error) { showError('Browser preferences could not be saved', error); }
  }
  function saveChecklist() {
    try { localStorage.setItem(CHECKLIST_KEY, JSON.stringify(state.checklist)); }
    catch (error) { showError('Checklist could not be saved locally', error); }
  }
  function readPreferences() {
    let savedSession = null;
    try {
      const saved = JSON.parse(localStorage.getItem(UI_KEY) || 'null');
      if (validMode(saved?.mode)) $('mode').value = saved.mode;
      if (typeof saved?.session_id === 'string') savedSession = saved.session_id;
      const savedChecklist = JSON.parse(localStorage.getItem(CHECKLIST_KEY) || 'null');
      for (const mode of ['static', 'live']) {
        if (savedChecklist?.[mode] && typeof savedChecklist[mode] === 'object' && !Array.isArray(savedChecklist[mode])) {
          // Copy only the intended fields; ignore inherited or unexpected properties.
          for (const [id, record] of Object.entries(savedChecklist[mode])) {
            if (['__proto__', 'prototype', 'constructor'].includes(id) || !record) continue;
            state.checklist[mode][id] = {
              verdict: ['pass', 'fail', 'not-run'].includes(record.verdict) ? record.verdict : 'not-run',
              notes: typeof record.notes === 'string' ? record.notes.slice(0, 10000) : '',
              updated_at: typeof record.updated_at === 'string' ? record.updated_at : '',
            };
          }
        }
      }
    } catch (error) { showError('Browser preferences could not be restored', error); }
    return savedSession;
  }
  function showError(stage, error) {
    const message = error?.message || printable(error) || 'Unknown error';
    const item = node('li', '', `${new Date().toLocaleTimeString()} · ${stage}: ${message}`);
    $('error-list').append(item);
    $('error-region').hidden = false;
  }
  function recordPayload(label, payload) {
    state.payload = payload;
    $('payload-meta').textContent = `${label} · ${new Date().toLocaleTimeString()} · ${modeName(effectiveMode())} context`;
    $('last-payload').textContent = printable(payload);
  }
  function errorMessage(payload, fallback) {
    return printable(payload?.detail ?? payload?.error?.message ?? payload?.error ?? payload?.message ?? payload) || fallback;
  }
  async function api(path, { method = 'GET', body, signal, record = true, allowError = false } = {}) {
    const response = await fetch(path, {
      method, signal, credentials: 'same-origin', cache: 'no-store',
      headers: body === undefined ? { Accept: 'application/json' } : { Accept: 'application/json', 'Content-Type': 'application/json' },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
    const raw = await response.text();
    let payload = null;
    if (raw) {
      try { payload = JSON.parse(raw); }
      catch {
        recordPayload(`${method} ${path} · invalid JSON`, { status: response.status, body: raw });
        throw new Error(`HTTP ${response.status}: expected JSON; received ${raw.slice(0, 400)}`);
      }
    }
    if (record || !response.ok) recordPayload(`${method} ${path} · HTTP ${response.status}`, payload);
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${errorMessage(payload, response.statusText)}`);
    if (!allowError && (payload?.error || payload?.status === 'error' || payload?.status === 'failed')) throw new Error(errorMessage(payload, 'API reported an error'));
    return payload;
  }
  function guard(stage, fn) {
    return async (event) => {
      if (event?.type === 'submit') event.preventDefault();
      try { await fn(event); }
      catch (error) { showError(stage, error); }
    };
  }

  function updateContextUI() {
    const requested = requestedMode();
    const effective = effectiveMode();
    const fallback = requested === 'live' && effective === 'static';
    const unverified = requested === 'live' && !state.status;
    $('effective-badge').textContent = unverified ? 'LIVE · availability unverified' : `Effective · ${modeName(effective)}`;
    $('effective-badge').className = `pill${fallback || unverified ? ' warning-pill' : effective === 'live' ? ' live' : ''}`;
    $('fallback-notice').hidden = !fallback;
    if (fallback) {
      $('fallback-title').textContent = 'LIVE requested → STATIC effective';
      $('fallback-detail').textContent = `${printable(state.status?.model?.reason) || 'The live model is unavailable, or the server returned a static fallback.'} Results use deterministic helpers; they do not verify a live model. History, evidence and the checklist use STATIC.`;
    }
    $('status-notice').hidden = !state.statusError && !unverified;
    if (!$('status-notice').hidden) $('status-notice').textContent = 'Environment status is unavailable. Model availability and embedding provenance are unverified. Refresh status before interpreting results.';
    const user = $('user').selectedOptions[0].textContent;
    const component = $('component').selectedOptions[0].textContent;
    const mode = modeName(effective);
    $('history-scope').textContent = `${user} · ${mode} · ${component}`;
    $('session-meta').textContent = `${state.sessionId ? state.sessionId : 'Unsaved'} · ${user} · ${component} · ${mode}`;
    $('conversation-title').textContent = state.sessionName || (state.sessionId ? 'Saved conversation' : 'A fresh conversation');
    $('documents-scope').textContent = `${user} · ${mode} effective mode · document/search scope`;
    $('checklist-scope').textContent = `Checklist · ${mode} effective mode${fallback ? ' · LIVE request fell back' : ''} · all components and demo users`;
    $('component-help').textContent = {
      agent: 'One agent, with tools and persisted history.',
      team: 'A coordinating team with shared run evidence.',
      workflow: 'Draft → Review. SSE frames completed content; it is not live model-token streaming.',
    }[$('component').value];
    if (effective === 'static') {
      $('document-provenance').textContent = 'STATIC uses synthetic demo embeddings. Search results exercise the persistence and retrieval plumbing; inspect evidence before claiming semantic quality.';
      $('embedding-state').textContent = 'Synthetic · STATIC';
      $('embedding-detail').textContent = 'Demo vectors; not semantic-quality evidence';
    } else if (state.status?.embeddings?.ready === false) {
      $('embedding-state').textContent = 'Live embeddings not configured';
      $('embedding-detail').textContent = state.status.embeddings.reason;
      $('document-provenance').textContent = state.status.embeddings.reason;
    } else {
      const embedding = state.status?.embeddings ?? state.status?.capabilities?.embeddings;
      const synthetic = embedding?.synthetic ?? state.status?.capabilities?.synthetic_embeddings;
      $('embedding-state').textContent = synthetic === true ? 'Static embeddings · local' : synthetic === false ? 'Provider-backed · reported' : 'LIVE · provenance unverified';
      $('embedding-detail').textContent = printable(embedding?.provider ?? embedding?.model_id) || 'Inspect ingestion/search response for provider';
      $('document-provenance').textContent = synthetic === true ? 'LIVE chat uses the configured LLM; document embeddings stay local and synthetic. These searches test Oracle storage and retrieval, not semantic embedding quality. No embedding API is called.' : 'LIVE documents and vector queries use the configured OCI embedding model. These requests may incur charges. Inspect provider/model/dimensions in the response evidence; upload existing static documents again to use them here.';
    }
    $('rename-session').disabled = state.busy || !state.sessionId;
    $('delete-session').disabled = state.busy || !state.sessionId;
    $('load-evidence').disabled = state.busy || !state.sessionId;
    const documentsUnavailable = effective === 'live' && state.status?.embeddings?.ready === false;
    for (const id of ['save-document', 'search-documents', 'create-indexes']) $(id).disabled = state.busy || documentsUnavailable;
    $('action-help').textContent = actionHelp[$('action').value];
    $('action').disabled = state.busy || effective === 'live';
    if (effective === 'live') $('action-help').textContent = 'LIVE uses the configured model and free-text tool selection. Choose STATIC to run an explicit helper without an LLM.';
    $('arguments-help').textContent = argumentHelp[$('action').value] || 'Explicit helper actions run without an LLM. Argument keys must match the backend helper.';
    updateLiveIndicator();
    updateChecklistSummary();
  }
  async function loadStatus() {
    const version = ++state.statusVersion;
    $('refresh-status').disabled = true;
    const previousScope = scopeKey();
    try {
      const result = await api('/api/status', { allowError: true });
      if (version !== state.statusVersion) return;
      if (!result || !result.database || !result.model) throw new Error('Status response is missing database or model information.');
      if (state.status?.model?.ready !== result.model.ready) {
        state.observedMode = null;
        state.liveCall = 'unverified';
      }
      state.status = result;
      if (result.error) showError('Environment status', new Error(errorMessage(result.error, 'Environment unavailable')));
      state.statusError = false;
      const db = result.database;
      const identity = db.identity || {};
      $('db-state').textContent = db.connected === true ? 'Oracle connected' : db.connected === false ? 'Oracle disconnected' : 'Connection unverified';
      $('db-detail').textContent = [identity.user, identity.container, identity.schema, identity.version, db.prefix ? `prefix ${db.prefix}` : ''].filter(Boolean).map(printable).join(' · ') || 'Identity not reported';
      $('model-state').textContent = effectiveMode() === 'static' ? 'Synthetic / deterministic · no LLM' : result.model.ready === true ? 'Live configured · inspect response evidence' : result.model.ready === false ? 'Live model unavailable' : 'Model readiness unverified';
      $('model-detail').textContent = [result.model.provider, result.model.model_id, result.model.max_output_tokens ? `Output limit ${result.model.max_output_tokens} tokens per model response` : '', result.model.reason].filter(Boolean).map(printable).join(' · ') || result.static?.label || 'Provider not reported';
      $('status-json').textContent = printable(result);
    } catch (error) {
      if (version !== state.statusVersion) return;
      state.status = null;
      state.statusError = true;
      $('db-state').textContent = 'Status unavailable';
      $('db-detail').textContent = error.message;
      $('model-state').textContent = 'Availability unverified';
      $('model-detail').textContent = 'Refresh status to retry';
      $('status-json').textContent = `Status could not be loaded: ${error.message}`;
      showError('Load environment status', error);
    } finally {
      if (version === state.statusVersion) {
        $('refresh-status').disabled = false;
        updateContextUI();
        if (scopeKey() !== previousScope) {
          clearSession(false);
          renderScenarios();
          await loadSessions();
        }
      }
    }
  }

  function formatTime(value) {
    if (!value) return 'Time unavailable';
    const date = new Date(typeof value === 'number' && value < 1e12 ? value * 1000 : value);
    return Number.isNaN(date.getTime()) ? printable(value) : date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  }
  function renderSessions() {
    const list = $('session-list');
    list.replaceChildren();
    for (const session of state.sessions) {
      if (typeof session?.session_id !== 'string') continue;
      const item = node('li');
      const button = node('button', `session-button${session.session_id === state.sessionId ? ' active' : ''}`);
      button.type = 'button';
      button.disabled = state.busy;
      button.setAttribute('aria-current', session.session_id === state.sessionId ? 'true' : 'false');
      button.append(node('strong', '', session.name || session.session_id), node('span', '', formatTime(session.updated_at)));
      button.addEventListener('click', guard('Load session', () => openSession(session)));
      item.append(button);
      list.append(item);
    }
  }
  async function loadSessions() {
    const version = ++state.sessionsVersion;
    const context = scope();
    $('session-state').textContent = 'Loading history…';
    $('refresh-sessions').disabled = true;
    try {
      const sessions = await api(`/api/sessions?${query(context)}`, { record: false });
      if (version !== state.sessionsVersion || scopeKey(context) !== scopeKey()) return false;
      if (!Array.isArray(sessions)) throw new Error('Expected a sessions array.');
      state.sessions = sessions;
      renderSessions();
      $('session-state').textContent = sessions.length ? `${sessions.length} saved session${sessions.length === 1 ? '' : 's'}` : 'No saved sessions in this scope.';
      return true;
    } catch (error) {
      if (version === state.sessionsVersion && scopeKey(context) === scopeKey()) {
        $('session-state').textContent = `History unavailable: ${error.message}`;
        showError('Load session history', error);
      }
      return false;
    } finally { if (version === state.sessionsVersion) $('refresh-sessions').disabled = state.busy; }
  }
  function clearSession(persist = true) {
    state.historyVersion++;
    state.sessionId = null; state.sessionMode = null; state.sessionName = ''; state.approval = null;
    $('approval-region').hidden = true;
    for (const message of $('transcript').querySelectorAll('.message')) message.remove();
    $('empty-chat').hidden = false;
    $('session-evidence').textContent = 'Select a session and load its evidence.';
    $('chat-state').textContent = `New session · ${modeName(effectiveMode())} · no messages cached`;
    $('message').value = '';
    updateContextUI();
    renderSessions();
    if (persist) rememberSelection();
  }
  function textContent(value) {
    if (Array.isArray(value)) return value.map((part) => part?.text ?? part?.content ?? printable(part)).join('\n');
    if (value && typeof value === 'object') return printable(value.input_content ?? value.content ?? value.message ?? value.text ?? value);
    return printable(value);
  }
  function normalizeHistory(raw) {
    if (Array.isArray(raw?.messages)) return raw.messages;
    const messages = [];
    for (const run of Array.isArray(raw?.runs) ? raw.runs : []) {
      if (run.parent_run_id) continue;
      const input = run.input?.input_content ?? run.input?.content ?? run.input ?? run.message;
      const runMessages = Array.isArray(run.messages) ? run.messages.filter((message) => ['user', 'assistant'].includes(message?.role)) : [];
      if (input !== undefined) messages.push({ role: 'user', content: input });
      else if (runMessages.length) messages.push(...runMessages);
      const output = run.content ?? run.response?.content ?? run.output;
      if (output !== undefined && (input !== undefined || !runMessages.length)) messages.push({ role: 'assistant', content: output });
    }
    return messages;
  }
  function addMessage(role, content, meta = '') {
    $('empty-chat').hidden = true;
    const article = node('article', `message ${role === 'user' ? 'user' : 'assistant'}`);
    const label = node('p', 'message-label', role === 'user' ? $('user').selectedOptions[0].textContent : role === 'assistant' ? 'Agno' : role);
    const body = node('div', 'message-body', textContent(content));
    const metadata = node('p', 'message-meta', meta);
    article.append(label, body, metadata);
    $('transcript').append(article);
    scrollTranscript();
    return { article, body, metadata };
  }
  function scrollTranscript() { $('transcript').scrollTop = $('transcript').scrollHeight; }
  async function openSession(session) {
    if (state.busy) return;
    clearSession(false);
    const version = ++state.historyVersion;
    const context = scope();
    setBusy(true);
    $('chat-state').textContent = 'Loading conversation…';
    try {
      const raw = await api(sessionPath(session.session_id, context));
      if (version !== state.historyVersion || scopeKey(context) !== scopeKey()) return;
      if (!raw || typeof raw !== 'object') throw new Error('Expected a session object.');
      state.sessionId = session.session_id;
      state.sessionMode = context.mode;
      state.sessionName = printable(session.name || raw.name || '');
      const messages = normalizeHistory(raw);
      for (const message of messages) {
        if (!message || !['user', 'assistant', 'tool', 'system'].includes(message.role)) continue;
        addMessage(message.role, message.content, `Stored history · ${modeName(context.mode)} scope · original run provenance: inspect evidence`);
      }
      if (!messages.length && Array.isArray(raw.runs) && raw.runs.length) {
        addMessage('system', 'This session contains runs, but no displayable message fields were found. Inspect the raw session payload in Evidence & data.');
      }
      const pausedRun = [...(Array.isArray(raw.runs) ? raw.runs : [])].reverse().find((run) => !run.parent_run_id && /paused|pending|requires|awaiting/i.test(printable(run.status)));
      if (pausedRun) renderApproval({ ...pausedRun, session_id: state.sessionId }, { ...context, requested: context.mode });
      $('chat-state').textContent = `Loaded ${messages.length} message${messages.length === 1 ? '' : 's'} · ${modeName(context.mode)}`;
      $('session-evidence').textContent = 'Session selected. Load raw evidence to inspect requirements and run details.';
      rememberSelection();
      switchPanel('chat');
    } catch (error) {
      $('chat-state').textContent = `Conversation could not be loaded: ${error.message}`;
      throw error;
    } finally { setBusy(false); updateContextUI(); renderSessions(); }
  }
  function setBusy(busy) {
    state.busy = busy;
    for (const id of ['mode', 'user', 'component', 'new-session', 'send', 'stream', 'action', 'arguments', 'message', 'refresh-sessions', 'refresh-status', 'approve-run', 'reject-run', 'save-document', 'search-documents', 'create-indexes', 'load-inventory', 'refresh-scenarios']) $(id).disabled = busy;
    $('stop-stream').hidden = !busy || !state.controller;
    $('chat-form').setAttribute('aria-busy', String(busy));
    updateContextUI(); renderSessions();
  }
  function resultProvenance(result, context) {
    const requested = result.requested_mode ?? context.requested;
    const effective = validMode(result.effective_mode) ? result.effective_mode : null;
    return `${modeName(requested)} requested → ${effective ? modeName(effective) : 'UNREPORTED'} effective${result.fallback ? ' · FALLBACK' : ''}${result.status ? ` · ${printable(result.status)}` : ''}${result.notice ? ` · ${printable(result.notice)}` : ''}`;
  }
  function renderApproval(result, context) {
    state.approval = null;
    $('approval-region').hidden = true;
    const requirements = result.requirements;
    const status = printable(result.status);
    const pending = /paused|pending|requires|awaiting/i.test(status);
    if (!pending && (status || !Array.isArray(requirements) || !requirements.length)) return;
    $('approval-region').hidden = false;
    $('approval-requirements').textContent = printable(requirements ?? []);
    const resumable = !!(result.session_id && result.run_id && context.component === 'agent');
    $('approval-help').textContent = resumable ? 'Review the server requirements. Approve or reject explicitly; the UI never chooses for you.' : 'Continuation requires an Agent run with session_id and run_id. Inspect the evidence and use the documented API for other requirements.';
    if (resumable) state.approval = { session_id: result.session_id, run_id: result.run_id, mode: validMode(result.effective_mode) ? result.effective_mode : context.mode, user_id: context.user_id };
    $('approve-run').hidden = $('reject-run').hidden = !resumable;
  }
  function applyResult(result, assistant, context) {
    if (!result || typeof result !== 'object' || Array.isArray(result)) throw new Error('Expected a chat result object.');
    if (result.error || result.status === 'error' || result.status === 'failed') throw new Error(errorMessage(result, 'Run failed'));
    if (result.content !== undefined) assistant.body.textContent = textContent(result.content);
    else if (!assistant.body.textContent) assistant.body.textContent = '(No content returned. Inspect the API evidence.)';
    const streaming = result.evidence?.streaming;
    const provenance = `${resultProvenance(result, context)}${streaming ? ` · ${printable(streaming)}` : ''}`;
    assistant.metadata.textContent = provenance;
    if (Array.isArray(result.tools) && result.tools.length) {
      const details = node('details');
      details.append(node('summary', '', `${result.tools.length} tool record${result.tools.length === 1 ? '' : 's'} · inspect`), node('pre', 'json-output', printable(result.tools)));
      assistant.article.append(details);
    }
    const oldMode = effectiveMode();
    if (result.effective_mode === 'live' && result.fallback !== true) {
      state.liveCall = /^completed$/i.test(printable(result.status)) && result.evidence?.simulated === false ? 'verified' : 'unverified';
    }
    if (validMode(result.effective_mode)) state.observedMode = result.effective_mode;
    else showError('Run provenance', new Error('The server did not report effective_mode. Inspect the payload before classifying this run.'));
    if (typeof result.session_id === 'string' && result.session_id) {
      state.sessionId = result.session_id; state.sessionMode = effectiveMode();
      state.sessionName = state.sessions.find((session) => session.session_id === result.session_id)?.name || '';
      rememberSelection();
    }
    renderApproval(result, { ...context, mode: effectiveMode() });
    updateContextUI();
    if (oldMode !== effectiveMode()) renderScenarios();
    $('chat-state').textContent = provenance;
    recordPayload('Chat result · server-reported provenance', result);
    scrollTranscript();
  }
  async function streamChat(body, assistant, signal) {
    const response = await fetch('/api/chat', {
      method: 'POST', credentials: 'same-origin', signal,
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' }, body: JSON.stringify(body),
    });
    if (!response.ok) {
      const raw = await response.text();
      let payload; try { payload = JSON.parse(raw); } catch { payload = raw; }
      recordPayload(`POST /api/chat · HTTP ${response.status}`, payload);
      throw new Error(`HTTP ${response.status}: ${errorMessage(payload, response.statusText)}`);
    }
    if ((response.headers.get('content-type') || '').includes('application/json')) {
      const result = await response.json();
      recordPayload('POST /api/chat · JSON response to streaming request', result);
      return result;
    }
    if (!response.body) throw new Error('Streaming response body is unavailable in this browser. Disable streaming and retry.');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '', eventLines = [], finalResult = null, accumulated = '';
    function dispatch() {
      if (!eventLines.length) return;
      const raw = eventLines.join('\n'); eventLines = [];
      if (raw.trim() === '[DONE]') return;
      let event;
      try { event = JSON.parse(raw); }
      catch { recordPayload('SSE · invalid event JSON', raw); throw new Error('Stream contained an invalid JSON data event.'); }
      if (event.type === 'error') {
        recordPayload('SSE · error', event);
        throw new Error(errorMessage(event.payload ?? event, 'Stream reported an error'));
      }
      if (event.type === 'content') {
        const chunk = event.chunk?.text ?? event.content?.text ?? event.text ?? (typeof event.content === 'string' ? event.content : '');
        accumulated += printable(chunk);
        assistant.body.textContent = accumulated;
        assistant.metadata.textContent = 'Streaming · waiting for final provenance';
        scrollTranscript();
      } else if (event.type === 'result') {
        finalResult = event.payload ?? event.result ?? event.data ?? event;
        recordPayload('SSE · final result', finalResult);
      }
    }
    function line(value) {
      if (value === '') dispatch();
      else if (value.startsWith('data:')) eventLines.push(value.slice(5).replace(/^ /, ''));
    }
    function consume(final = false) {
      // SSE permits CR, LF and CRLF; a CRLF can cross network chunks.
      let start = 0;
      for (let i = 0; i < buffer.length; i++) {
        if (buffer[i] !== '\n' && buffer[i] !== '\r') continue;
        if (buffer[i] === '\r' && i === buffer.length - 1 && !final) break;
        line(buffer.slice(start, i));
        if (buffer[i] === '\r' && buffer[i + 1] === '\n') i++;
        start = i + 1;
      }
      buffer = buffer.slice(start);
      if (final) { if (buffer) line(buffer); buffer = ''; dispatch(); }
    }
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) { buffer += decoder.decode(); consume(true); break; }
        buffer += decoder.decode(value, { stream: true });
        consume();
      }
    } catch (error) {
      try { await reader.cancel(); } catch { /* Original stream error is kept. */ }
      throw error;
    } finally { reader.releaseLock(); }
    if (!finalResult) throw new Error('Stream ended without a final result. Partial text is retained; completion and effective provenance are unverified.');
    return finalResult;
  }
  async function sendChat() {
    if (state.busy) return;
    const message = $('message').value.trim();
    const action = $('action').value;
    if (!message) throw new Error('Write a message before sending, including when selecting a helper action.');
    let args;
    try { args = JSON.parse($('arguments').value.trim() || JSON.stringify(argumentDefaults[action])); }
    catch { $('arguments-details').open = true; $('arguments').focus(); throw new Error('Action arguments must be valid JSON. Nothing was sent.'); }
    if (!args || typeof args !== 'object' || Array.isArray(args)) throw new Error('Action arguments must be a JSON object, such as {}. Nothing was sent.');
    const context = { ...scope(), requested: requestedMode() };
    const body = { user_id: context.user_id, mode: context.requested, component: context.component, session_id: state.sessionId, message, stream: $('stream').checked, action: action === 'auto' ? null : action, arguments: args };
    const oldApproval = state.approval;
    state.approval = null; $('approval-region').hidden = true;
    addMessage('user', message || `[Helper action: ${action}]`, `Requested · ${modeName(context.requested)}${action !== 'auto' ? ` · explicit ${action} helper` : ''}`);
    const assistant = addMessage('assistant', '', 'Waiting for response…');
    state.controller = new AbortController();
    if (context.requested === 'live' && context.mode === 'live') state.liveCall = 'pending';
    setBusy(true); $('chat-state').textContent = 'Running…';
    try {
      const result = body.stream ? await streamChat(body, assistant, state.controller.signal) : await api('/api/chat', { method: 'POST', body, signal: state.controller.signal });
      applyResult(result, assistant, context);
      $('message').value = '';
      await loadSessions();
    } catch (error) {
      if (context.requested === 'live' && context.mode === 'live') state.liveCall = 'failed';
      assistant.article.classList.add('error');
      assistant.metadata.textContent = error.name === 'AbortError' ? 'Stopped waiting · server may still complete; refresh history before retrying.' : `Request incomplete · ${error.message}`;
      if (!assistant.body.textContent) assistant.body.textContent = 'No completed response was received.';
      $('chat-state').textContent = assistant.metadata.textContent;
      if (oldApproval) { state.approval = oldApproval; $('approval-region').hidden = false; }
      throw error;
    } finally { state.controller = null; setBusy(false); scrollTranscript(); }
  }
  async function continueRun(approved) {
    if (!state.approval || state.busy) return;
    const approval = { ...state.approval };
    const confirmed = await confirmAction({ title: approved ? 'Approve this run?' : 'Reject this run?', description: `Continue run ${approval.run_id} in ${modeName(approval.mode)} mode with approved=${approved}. Review the requirements above before confirming.`, confirm: approved ? 'Approve & continue' : 'Reject & continue' });
    if (!confirmed) return;
    if (approval.mode === 'live') state.liveCall = 'pending';
    setBusy(true); $('chat-state').textContent = 'Continuing run…';
    const assistant = addMessage('assistant', '', 'Waiting for continuation…');
    try {
      const result = await api('/api/continue', { method: 'POST', body: { ...approval, approved } });
      applyResult(result, assistant, { ...scope(), requested: approval.mode });
      await loadSessions();
    } catch (error) {
      if (approval.mode === 'live') state.liveCall = 'failed';
      assistant.article.classList.add('error'); assistant.body.textContent = `Continuation failed: ${error.message}`;
      $('chat-state').textContent = 'Continuation failed. Requirements retained for review.';
      throw error;
    } finally { setBusy(false); }
  }

  function confirmAction({ title, description, confirm, input }) {
    if (state.dialogResolve) return Promise.resolve(false);
    $('dialog-title').textContent = title;
    $('dialog-description').textContent = description;
    $('dialog-confirm').textContent = confirm;
    $('dialog-input-wrap').hidden = input === undefined;
    $('dialog-input').required = input !== undefined;
    $('dialog-input').value = input ?? '';
    $('confirm-dialog').showModal();
    (input === undefined ? $('dialog-cancel') : $('dialog-input')).focus();
    if (input !== undefined) $('dialog-input').select();
    return new Promise((resolve) => { state.dialogResolve = resolve; });
  }
  function closeDialog(value) {
    const resolve = state.dialogResolve; state.dialogResolve = null;
    $('confirm-dialog').close(); resolve?.(value);
  }
  async function renameSession() {
    if (!state.sessionId || state.busy) return;
    const context = scope(), id = state.sessionId;
    const name = await confirmAction({ title: 'Rename session', description: 'Save a new name for this session in the current user, mode and component scope.', confirm: 'Save name', input: state.sessionName });
    if (name === false || !name.trim()) return;
    setBusy(true);
    try {
      await api(`/api/sessions/${encodeURIComponent(id)}`, { method: 'PATCH', body: { ...context, name: name.trim() } });
      state.sessionName = name.trim(); updateContextUI(); await loadSessions();
    } finally { setBusy(false); }
  }
  async function deleteSession() {
    if (!state.sessionId || state.busy) return;
    const context = scope(), id = state.sessionId;
    if (!await confirmAction({ title: 'Delete this session?', description: `Delete “${state.sessionName || id}” for ${context.user_id} · ${modeName(context.mode)} · ${context.component}. This removes saved session history on the server.`, confirm: 'Delete session' })) return;
    setBusy(true);
    try {
      await api(sessionPath(id, context), { method: 'DELETE' });
      clearSession(); await loadSessions();
      $('chat-state').textContent = 'Session deleted on the server. A fresh session is ready.';
    } finally { setBusy(false); }
  }

  function checklistRecord(id, mode = effectiveMode()) {
    return Object.hasOwn(state.checklist[mode], id) ? state.checklist[mode][id] : { verdict: 'not-run', notes: '', updated_at: '' };
  }
  function updateChecklistSummary() {
    const counts = { pass: 0, fail: 0, 'not-run': 0 };
    for (const scenario of state.scenarios) counts[checklistRecord(String(scenario.id)).verdict]++;
    $('checklist-summary').textContent = `${counts.pass} pass · ${counts.fail} fail · ${counts['not-run']} not run · ${modeName(effectiveMode())}`;
    $('scenario-count').textContent = state.scenarios.length ? `${counts.pass}/${state.scenarios.length}` : '—';
  }
  function isAdvanced(scenario) {
    if (['concurrency', 'async', 'queue', 'schedule', 'observability', 'metrics', 'bulk'].includes(String(scenario.id))) return true;
    if (scenario.id === 'text') return false;
    return !/^\s*(browser|documents? panel|ui)/i.test(printable(scenario.surface));
  }
  function renderScenarios() {
    $('scenario-list').replaceChildren(); $('advanced-scenario-list').replaceChildren();
    let advanced = 0;
    state.scenarios.forEach((scenario, index) => {
      const id = String(scenario.id), mode = effectiveMode();
      if (['__proto__', 'prototype', 'constructor'].includes(id)) return;
      const record = checklistRecord(id, mode);
      const card = node('details', 'scenario-card'); card.dataset.verdict = record.verdict;
      const summary = node('summary');
      const chip = node('span', 'verdict-chip', record.verdict === 'not-run' ? 'NOT RUN' : record.verdict.toUpperCase());
      summary.append(node('strong', '', `${String(index + 1).padStart(2, '0')} · ${scenario.title || id}`), chip);
      if (scenario.surface) summary.append(node('span', 'scenario-surface', printable(scenario.surface)));
      const content = node('div', 'scenario-content'); content.append(node('h4', '', 'Steps'));
      const steps = node('ol');
      for (const step of Array.isArray(scenario.steps) ? scenario.steps : [scenario.steps ?? 'Consult the backend scenario instructions.']) steps.append(node('li', '', printable(step)));
      content.append(steps, node('h4', '', 'Expected result'), node('p', 'scenario-expected', printable(scenario.expected) || 'No expected result was supplied. Inspect the scenario API payload.'));
      if (scenario.static_scope) content.append(node('h4', '', 'STATIC scope'), node('p', 'scenario-expected', scenario.static_scope));
      if (scenario.live_scope) content.append(node('h4', '', 'LIVE scope'), node('p', 'scenario-expected', scenario.live_scope));
      if (!isAdvanced(scenario)) content.append(node('p', 'field-help', 'Some branches require a manual API call or process restart. UI output alone does not complete those branches.'));
      const controls = node('div', 'scenario-controls');
      const verdictWrap = node('div'), noteWrap = node('div');
      const verdictLabel = node('label', '', 'Your manual verdict'); verdictLabel.htmlFor = `verdict-${index}`;
      const verdict = node('select'); verdict.id = `verdict-${index}`;
      for (const [value, label] of [['not-run', 'Not run'], ['pass', 'Pass · verified manually'], ['fail', 'Fail · observed mismatch']]) {
        const option = node('option', '', label); option.value = value; verdict.append(option);
      }
      verdict.value = record.verdict;
      const noteLabel = node('label', '', 'Local notes / observed evidence'); noteLabel.htmlFor = `notes-${index}`;
      const notes = node('textarea'); notes.id = `notes-${index}`; notes.rows = 2; notes.maxLength = 10000;
      notes.placeholder = 'What did you observe? No secrets or conversation content.'; notes.value = record.notes;
      const saved = node('p', 'field-help', record.updated_at ? `Saved locally · ${formatTime(record.updated_at)}` : `Not recorded · ${modeName(mode)} mode`);
      function save() {
        const value = { verdict: verdict.value, notes: notes.value, updated_at: new Date().toISOString() };
        state.checklist[mode][id] = value;
        saveChecklist(); card.dataset.verdict = value.verdict;
        chip.textContent = value.verdict === 'not-run' ? 'NOT RUN' : value.verdict.toUpperCase();
        saved.textContent = `Saved locally · ${formatTime(value.updated_at)} · ${modeName(mode)}`;
        updateChecklistSummary();
      }
      verdict.addEventListener('change', save); notes.addEventListener('input', save);
      verdictWrap.append(verdictLabel, verdict); noteWrap.append(noteLabel, notes);
      controls.append(verdictWrap, noteWrap); content.append(controls, saved); card.append(summary, content);
      const apiScenario = isAdvanced(scenario); if (apiScenario) advanced++;
      $(apiScenario ? 'advanced-scenario-list' : 'scenario-list').append(card);
    });
    if (!state.scenarios.length) $('scenario-list').append(node('p', 'field-help', 'Scenarios are unavailable. Reload to retry; no verdicts have been inferred.'));
    if (state.scenarios.length === advanced) $('scenario-list').append(node('p', 'field-help', 'The server categorized all scenarios as API/advanced. See the separate section below.'));
    if (!advanced) $('advanced-scenario-list').append(node('p', 'field-help', 'No scenarios were categorized as API/advanced by the server. Use the backend documentation for additional API coverage.'));
    $('advanced-count').textContent = `· ${advanced}`;
    updateChecklistSummary();
  }
  async function loadScenarios() {
    const version = ++state.scenarioVersion;
    $('scenario-state').textContent = 'Loading scenario definitions…'; $('refresh-scenarios').disabled = true;
    try {
      const result = await api('/api/scenarios');
      if (version !== state.scenarioVersion) return;
      if (!Array.isArray(result)) throw new Error('Expected a scenarios array.');
      state.scenarios = result.filter((scenario) => scenario && (typeof scenario.id === 'string' || typeof scenario.id === 'number'));
      renderScenarios();
      $('scenario-state').textContent = `${state.scenarios.length} scenarios supplied by the backend.${state.scenarios.length !== 20 ? ' Expected 20; check backend scenario coverage.' : ''}`;
    } catch (error) {
      $('scenario-state').textContent = `Scenario definitions unavailable: ${error.message}`; showError('Load scenarios', error);
      renderScenarios();
    } finally { if (version === state.scenarioVersion) $('refresh-scenarios').disabled = state.busy; }
  }
  async function resetChecklist() {
    const mode = effectiveMode();
    if (!await confirmAction({ title: `Reset ${modeName(mode)} checklist?`, description: 'Remove the manual verdicts and local notes for this effective mode in this browser. This does not change server data.', confirm: 'Reset checklist' })) return;
    state.checklist[mode] = {}; saveChecklist(); renderScenarios();
  }

  async function documentRequest(kind) {
    if (state.busy) return;
    const context = scope();
    const isSearch = kind === 'search';
    const path = isSearch ? '/api/search' : '/api/documents';
    const body = isSearch ? { user_id: context.user_id, mode: context.mode, query: $('search-query').value.trim(), search_type: $('search-type').value, limit: 5 } : { user_id: context.user_id, mode: context.mode, name: $('document-name').value.trim(), content: $('document-content').value.trim() };
    if (isSearch ? !body.query : !body.name || !body.content) throw new Error(isSearch ? 'Enter a search query.' : 'Document name and content are required.');
    const status = $(isSearch ? 'search-state' : 'document-state');
    setBusy(true); status.textContent = `${isSearch ? 'Searching' : 'Storing document'} · ${modeName(context.mode)}…`;
    try {
      const result = await api(path, { method: 'POST', body });
      $('document-result').textContent = printable(result);
      status.textContent = `${isSearch ? 'Search completed' : 'Document request completed'} · ${modeName(context.mode)} requested · inspect returned evidence`;
    } catch (error) {
      status.textContent = `Failed: ${error.message}`;
      $('document-result').textContent = `Request failed: ${error.message}\n\nSee the last API payload for returned details.`;
      throw error;
    } finally { setBusy(false); }
  }
  async function createIndexes() {
    if (state.busy) return;
    const mode = effectiveMode();
    if (!await confirmAction({ title: 'Create database indexes?', description: `Request index creation for ${modeName(mode)} data. This is an explicit database change. Inspect the response for unsupported indexes or errors.`, confirm: 'Create indexes' })) return;
    setBusy(true); $('index-state').textContent = 'Creating indexes…';
    try {
      const result = await api('/api/indexes', { method: 'POST', body: { mode } });
      $('document-result').textContent = printable(result);
      $('index-state').textContent = 'Request completed. Inspect the response for each index outcome.';
    } catch (error) { $('index-state').textContent = `Index creation failed: ${error.message}`; throw error; }
    finally { setBusy(false); }
  }
  async function loadEvidence() {
    if (!state.sessionId || state.busy) return;
    const context = scope(), id = state.sessionId;
    setBusy(true); $('evidence-state').textContent = 'Loading session evidence…';
    try {
      const result = await api(`/api/evidence/${encodeURIComponent(id)}?${query(context)}`);
      $('session-evidence').textContent = printable(result);
      $('session-evidence').parentElement.open = true;
      $('evidence-state').textContent = `Evidence loaded · ${id} · ${modeName(context.mode)}`;
    } catch (error) {
      $('evidence-state').textContent = `Evidence unavailable: ${error.message}`;
      $('session-evidence').textContent = `Evidence request failed: ${error.message}`;
      throw error;
    } finally { setBusy(false); }
  }
  async function loadInventory() {
    if (state.busy) return;
    setBusy(true); $('inventory-meta').textContent = 'Loading inventory…';
    try {
      const result = await api('/api/inventory');
      if (!Array.isArray(result?.tables)) throw new Error('Inventory response is missing a tables array.');
      $('inventory-identity').textContent = printable(result.identity || {});
      const rows = result.tables.map((table) => {
        const row = node('tr'); row.append(node('td', '', table.name), node('td', '', table.rows ?? 'unreported')); return row;
      });
      if (!rows.length) { const row = node('tr'); const cell = node('td', '', 'No tables reported.'); cell.colSpan = 2; row.append(cell); rows.push(row); }
      $('inventory-body').replaceChildren(...rows);
      $('inventory-meta').textContent = `${rows.length && result.tables.length} tables · ${new Date().toLocaleTimeString()} · server inventory`;
    } catch (error) { $('inventory-meta').textContent = `Inventory failed: ${error.message} · previous snapshot, if any, retained`; throw error; }
    finally { setBusy(false); }
  }
  function switchPanel(panel) {
    state.panel = panel;
    for (const tab of document.querySelectorAll('[data-panel]')) {
      const selected = tab.dataset.panel === panel;
      tab.classList.toggle('active', selected); tab.setAttribute('aria-selected', String(selected)); tab.tabIndex = selected ? 0 : -1;
      $(`panel-${tab.dataset.panel}`).hidden = !selected;
    }
  }
  async function changeContext() {
    state.contextVersion++; state.observedMode = null; state.liveCall = 'unverified';
    if (effectiveMode() === 'live') $('action').value = 'auto';
    clearSession();
    $('document-result').textContent = 'Context changed. Submit a new document/search request for this user and mode.';
    $('document-state').textContent = 'No document submitted in this context';
    $('search-state').textContent = 'No search submitted in this context';
    if (state.status) {
      $('model-state').textContent = effectiveMode() === 'static' ? 'Synthetic / deterministic · no LLM' : state.status.model.ready === true ? 'Live configured · inspect response evidence' : 'Model availability unverified';
    }
    renderScenarios(); await loadSessions();
  }
  function bindEvents() {
    $('chat-form').addEventListener('submit', guard('Send message', sendChat));
    $('document-form').addEventListener('submit', guard('Store document', () => documentRequest('document')));
    $('search-form').addEventListener('submit', guard('Search documents', () => documentRequest('search')));
    for (const id of ['mode', 'user', 'component']) $(id).addEventListener('change', guard('Change context', changeContext));
    $('action').addEventListener('change', () => {
      $('arguments').value = '';
      $('arguments').placeholder = JSON.stringify(argumentDefaults[$('action').value], null, 2);
      $('arguments-details').open = $('action').value !== 'auto';
      updateContextUI();
    });
    $('new-session').addEventListener('click', () => { if (!state.busy) { clearSession(); $('message').focus(); } });
    $('refresh-status').addEventListener('click', guard('Refresh status', loadStatus));
    $('refresh-sessions').addEventListener('click', guard('Refresh sessions', loadSessions));
    $('rename-session').addEventListener('click', guard('Rename session', renameSession));
    $('delete-session').addEventListener('click', guard('Delete session', deleteSession));
    $('approve-run').addEventListener('click', guard('Approve run', () => continueRun(true)));
    $('reject-run').addEventListener('click', guard('Reject run', () => continueRun(false)));
    $('stop-stream').addEventListener('click', () => state.controller?.abort());
    $('refresh-scenarios').addEventListener('click', guard('Reload scenarios', loadScenarios));
    $('reset-checklist').addEventListener('click', guard('Reset checklist', resetChecklist));
    $('create-indexes').addEventListener('click', guard('Create indexes', createIndexes));
    $('load-evidence').addEventListener('click', guard('Load evidence', loadEvidence));
    $('load-inventory').addEventListener('click', guard('Load inventory', loadInventory));
    $('dismiss-errors').addEventListener('click', () => { $('error-list').replaceChildren(); $('error-region').hidden = true; });
    $('dialog-cancel').addEventListener('click', () => closeDialog(false));
    $('confirm-dialog').addEventListener('cancel', (event) => { event.preventDefault(); closeDialog(false); });
    $('dialog-form').addEventListener('submit', (event) => {
      event.preventDefault();
      const value = $('dialog-input-wrap').hidden ? true : $('dialog-input').value.trim();
      if (value === '') { $('dialog-input').focus(); return; }
      closeDialog(value);
    });
    for (const tab of document.querySelectorAll('[data-panel]')) {
      tab.addEventListener('click', () => switchPanel(tab.dataset.panel));
      tab.addEventListener('keydown', (event) => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        const tabs = [...document.querySelectorAll('[data-panel]')];
        const current = tabs.indexOf(tab);
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
        switchPanel(tabs[next].dataset.panel); tabs[next].focus();
      });
    }
    window.addEventListener('error', (event) => showError('Unexpected UI error', event.error || event.message));
    window.addEventListener('unhandledrejection', (event) => { event.preventDefault(); showError('Unexpected request error', event.reason); });
  }
  async function start() {
    bindEvents();
    const savedSession = readPreferences();
    setBusy(true);
    // These GETs are read-only. No chat, indexing, ingestion or search runs at startup.
    await Promise.allSettled([loadStatus(), loadScenarios()]);
    const loaded = await loadSessions();
    setBusy(false);
    if (savedSession && loaded) {
      const matching = state.sessions.find((session) => session.session_id === savedSession);
      if (matching) { try { await openSession(matching); } catch (error) { showError('Restore selected session', error); } }
      else rememberSelection();
    }
  }
  start().catch((error) => showError('Initialize demo', error));
})();
