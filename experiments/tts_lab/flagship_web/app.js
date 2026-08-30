import {
  finiteNumber,
  formatDate,
  formatMemory,
  formatRtf,
  formatSeconds,
  normalizeMetrics,
  unwrapList
} from './formatters.js';

const $ = id => document.getElementById(id);
const API = '/api';
const MODEL_FALLBACK = 'MOSS-TTS v1.5';
const MAX_MEMORY_MB = 16 * 1024;
const DEFAULT_PARAMETERS = Object.freeze({temperatureValue: 1.7, topP: 0.8, topK: 25, repetitionPenalty: 1});
const FALLBACK_CASES = [
  {id: 'welcome', label: 'Karşılama', text: 'Merhaba, ben Fatma. Size nasıl yardımcı olabilirim?'},
  {id: 'empathy', label: 'Empati', text: 'Sizi anlıyorum. Bu durumun can sıkıcı olduğunu biliyorum; birlikte bir çözüm bulalım.'},
  {id: 'numbers', label: 'Sayı ve tarih', text: 'Randevunuz yirmi dokuz Ağustos günü saat on dört otuz için oluşturuldu. Tutar bin iki yüz kırk dokuz lira doksan kuruş.'},
  {id: 'pronunciation', label: 'Telaffuz', text: 'Değerlendirmenizi doğruladıktan sonra güncel bilgileri sizinle paylaşacağım.'},
  {id: 'emotion', label: 'Duygu geçişi', text: 'Harika, işleminiz tamamlandı! Beklediğiniz için teşekkür ederim; şimdi içiniz rahat olabilir.'}
];

const state = {
  ready: false,
  busy: false,
  generating: false,
  uploading: false,
  audioObjectUrl: null,
  statusTimer: null
};

function setText(id, value) {
  const element = $(id);
  if (element) element.textContent = value ?? '—';
}

function showError(message) {
  setText('error-message', message || 'Beklenmeyen bir hata oluştu.');
  $('error-notice').hidden = false;
  $('error-notice').focus?.();
}

function clearError() {
  $('error-notice').hidden = true;
  setText('error-message', '');
}

async function apiRequest(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API}${path}`, {
      ...options,
      headers: options.body instanceof FormData
        ? {'X-Fatma-Lab': 'flagship-v1', ...options.headers}
        : {'Content-Type': 'application/json', 'X-Fatma-Lab': 'flagship-v1', ...options.headers}
    });
  } catch (error) {
    throw new Error(`Yerel sunucuya ulaşılamadı: ${error.message}`);
  }

  if (!response.ok) {
    let detail = '';
    try {
      const payload = await response.json();
      detail = payload.detail || payload.error || payload.message || '';
    } catch {
      detail = await response.text().catch(() => '');
    }
    throw new Error(detail || `Sunucu ${response.status} hatası döndürdü.`);
  }

  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('audio/')) {
    return {
      audio_blob: await response.blob(),
      metrics: {
        latency_seconds: response.headers.get('x-latency-seconds'),
        audio_seconds: response.headers.get('x-audio-seconds'),
        rtf: response.headers.get('x-rtf'),
        rss_mb: response.headers.get('x-rss-mb')
      },
      created_at: new Date().toISOString()
    };
  }
  if (response.status === 204) return {};
  return response.json();
}

function updateGenerateState() {
  const blocked = !state.ready || state.busy || state.generating || state.uploading;
  $('generate-button').disabled = blocked;
  $('generate-button').classList.toggle('is-busy', state.generating);
  $('voice-file').disabled = blocked;
  $('voice-select').disabled = state.generating || state.uploading;
  $('generate-form').setAttribute('aria-busy', String(state.generating));

  if (state.generating) setText('form-status', 'Ses üretiliyor. Bu işlem model hızına göre sürebilir.');
  else if (state.uploading) setText('form-status', 'Referans ses yerel sunucuya yükleniyor.');
  else if (state.busy) setText('form-status', 'Model başka bir üretimi tamamlıyor.');
  else if (state.ready) setText('form-status', 'Hazır. Üretim başladığında gerçek süre ölçülecek.');
  else setText('form-status', 'Model hazır değil. Sistem durumunu kontrol edin.');
}

function renderStatus(data) {
  state.ready = Boolean(data.ready);
  state.busy = Boolean(data.busy);
  const connectionState = data.error ? 'error' : state.ready ? (state.busy ? 'busy' : 'ready') : 'offline';
  $('connection-pill').dataset.state = connectionState;
  setText('connection-label', data.error ? 'Hata' : state.ready ? (state.busy ? 'Meşgul' : 'Yerelde hazır') : 'Hazır değil');
  setText('model-name', data.model || MODEL_FALLBACK);
  setText('model-language', data.language || 'Turkish (tr)');
  setText('runtime-name', data.runtime || 'Metal · yerel');
  setText('cpu-threads', data.cpu_threads ?? '4');

  const modelState = $('model-state');
  modelState.replaceChildren();
  const dot = document.createElement('span');
  dot.className = `mini-dot ${connectionState}`;
  dot.setAttribute('aria-hidden', 'true');
  modelState.append(dot, document.createTextNode(data.error ? ' Hata' : state.ready ? (state.busy ? ' Meşgul' : ' Hazır') : ' Kapalı'));

  const memory = finiteNumber(data.backend_rss_mb);
  setText('memory-value', formatMemory(memory));
  $('memory-bar').style.width = memory === null ? '0%' : `${Math.min(100, (memory / MAX_MEMORY_MB) * 100)}%`;
  $('memory-bar').classList.toggle('high', memory !== null && memory > 12 * 1024);

  if (Array.isArray(data.voices)) renderVoices(data.voices);
  if (data.error && !state.generating) showError(data.error);
  updateGenerateState();
}

function renderVoices(voices, preferredId) {
  const select = $('voice-select');
  const current = preferredId || select.value;
  select.replaceChildren();
  if (!voices.length) {
    const option = new Option('Kullanılabilir ses yok', '');
    select.append(option);
    return;
  }
  for (const voice of voices) {
    const id = String(voice.id ?? voice.value ?? voice.label ?? '');
    const label = voice.label || voice.name || id;
    select.append(new Option(label, id));
  }
  if ([...select.options].some(option => option.value === current)) select.value = current;
}

async function refreshStatus({announceError = false} = {}) {
  $('refresh-status').classList.add('rotating');
  try {
    renderStatus(await apiRequest('/status'));
  } catch (error) {
    renderStatus({ready: false, busy: false, error: null});
    if (announceError) showError(error.message);
  } finally {
    $('refresh-status').classList.remove('rotating');
  }
}

function renderCases(cases) {
  const list = $('case-list');
  list.replaceChildren();
  for (const testCase of cases) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'case-button';
    button.textContent = testCase.label || testCase.name || 'Test cümlesi';
    button.dataset.caseId = testCase.id || '';
    button.addEventListener('click', () => {
      $('speech-text').value = testCase.text || '';
      updateCharacterCount();
      validateText();
      document.querySelectorAll('.case-button').forEach(item => item.classList.toggle('selected', item === button));
      $('speech-text').focus();
    });
    list.append(button);
  }
}

async function loadCases() {
  try {
    const payload = await apiRequest('/cases');
    const cases = unwrapList(payload, ['cases', 'items']);
    renderCases(cases.length ? cases : FALLBACK_CASES);
  } catch {
    renderCases(FALLBACK_CASES);
  }
}

function resolveAudioSource(item) {
  const url = item.audio_url || item.audioUrl || item.url;
  if (url) return url;
  if (item.audio_blob instanceof Blob) {
    if (state.audioObjectUrl) URL.revokeObjectURL(state.audioObjectUrl);
    state.audioObjectUrl = URL.createObjectURL(item.audio_blob);
    return state.audioObjectUrl;
  }
  const encoded = item.audio_base64 || item.audioBase64 || item.audio;
  if (typeof encoded === 'string' && encoded.length > 100) {
    return encoded.startsWith('data:') ? encoded : `data:audio/wav;base64,${encoded}`;
  }
  return '';
}

function renderMetrics(item) {
  const metrics = normalizeMetrics(item);
  setText('metric-latency', formatSeconds(metrics.latency));
  setText('metric-duration', formatSeconds(metrics.duration));
  setText('metric-rtf', formatRtf(metrics.rtf));
  setText('metric-memory', formatMemory(metrics.memory));
}

function createWaveArt() {
  const wave = $('wave-art');
  if (wave.childElementCount) return;
  const heights = [18, 34, 24, 52, 42, 68, 38, 76, 56, 30, 66, 44, 72, 34, 54, 24, 48, 18, 36, 26, 58, 42, 68, 30, 52, 22, 40, 16];
  for (const height of heights) {
    const bar = document.createElement('span');
    bar.style.height = `${height}%`;
    wave.append(bar);
  }
}

function showResult(item) {
  const audioSource = resolveAudioSource(item);
  if (!audioSource) throw new Error('Üretim tamamlandı ancak oynatılabilir ses adresi dönmedi.');
  $('result-audio').src = audioSource;
  setText('result-text', item.text || $('speech-text').value.trim());
  setText('result-time', formatDate(item.created_at || new Date().toISOString()));
  renderMetrics(item);
  createWaveArt();
  $('result-section').hidden = false;
  $('result-section').scrollIntoView({behavior: 'smooth', block: 'nearest'});
}

function createMetricTag(label, value) {
  const span = document.createElement('span');
  span.textContent = `${label} ${value}`;
  return span;
}

function renderHistory(items) {
  const list = $('history-list');
  list.replaceChildren();
  list.setAttribute('aria-busy', 'false');
  if (!items.length) {
    const empty = document.createElement('div');
    empty.className = 'history-empty';
    const title = document.createElement('strong');
    title.textContent = 'Henüz üretim yok';
    const copy = document.createElement('p');
    copy.textContent = 'İlk sesi ürettiğinizde ölçümler burada saklanacak.';
    empty.append(title, copy);
    list.append(empty);
    return;
  }

  for (const item of items) {
    const row = document.createElement('article');
    row.className = 'history-item';

    const body = document.createElement('div');
    body.className = 'history-body';
    const meta = document.createElement('div');
    meta.className = 'history-meta';
    const voice = document.createElement('strong');
    voice.textContent = item.voice_label || item.voice || 'Ses';
    const date = document.createElement('time');
    date.dateTime = item.created_at || '';
    date.textContent = formatDate(item.created_at);
    meta.append(voice, date);
    const text = document.createElement('p');
    text.textContent = item.text || 'Metin bilgisi yok.';
    body.append(meta, text);

    const metrics = normalizeMetrics(item);
    const tags = document.createElement('div');
    tags.className = 'history-metrics';
    tags.append(
      createMetricTag('Gecikme', formatSeconds(metrics.latency)),
      createMetricTag('RTF', formatRtf(metrics.rtf)),
      createMetricTag('RSS', formatMemory(metrics.memory))
    );
    body.append(tags);

    const player = document.createElement('audio');
    player.controls = true;
    player.preload = 'none';
    player.src = item.audio_url || item.audioUrl || item.url || '';
    player.setAttribute('aria-label', `${voice.textContent} sonucunu dinle`);
    if (!player.src) player.hidden = true;
    row.append(body, player);
    list.append(row);
  }
}

async function loadHistory({announceError = false} = {}) {
  $('history-list').setAttribute('aria-busy', 'true');
  try {
    const payload = await apiRequest('/history');
    renderHistory(unwrapList(payload, ['history', 'items']));
  } catch (error) {
    renderHistory([]);
    if (announceError) showError(error.message);
  }
}

function validateText() {
  const text = $('speech-text').value.trim();
  let message = '';
  if (!text) message = 'Seslendirilecek metni yazın.';
  else if (text.length < 2) message = 'Metin en az iki karakter olmalı.';
  else if (text.length > 600) message = 'Metin 600 karakteri geçemez.';
  setText('text-error', message);
  $('speech-text').setAttribute('aria-invalid', String(Boolean(message)));
  return !message;
}

function updateCharacterCount() {
  const length = $('speech-text').value.length;
  setText('character-count', `${length} / 600`);
  $('character-count').classList.toggle('near-limit', length > 540);
}

function generationPayload() {
  const payload = {
    text: $('speech-text').value.trim(),
    voice: $('voice-select').value,
    instruction: $('instruction').value.trim(),
    top_p: Number($('top-p').value),
    top_k: Number($('top-k').value),
    repetition_penalty: Number($('repetition-penalty').value)
  };
  if ($('temperature-enabled').checked) payload.temperature = Number($('temperature').value);
  return payload;
}

async function generate(event) {
  event.preventDefault();
  clearError();
  if (!validateText()) {
    $('speech-text').focus();
    return;
  }
  if (!$('voice-select').value) {
    showError('Önce kullanılacak sesi seçin.');
    $('voice-select').focus();
    return;
  }

  state.generating = true;
  updateGenerateState();
  const startedAt = performance.now();
  let completed = false;
  try {
    const item = await apiRequest('/generate', {method: 'POST', body: JSON.stringify(generationPayload())});
    if (!item.metrics) item.metrics = {};
    if (finiteNumber(item.metrics.latency_seconds) === null) {
      item.metrics.latency_seconds = (performance.now() - startedAt) / 1000;
    }
    showResult(item);
    completed = true;
    await loadHistory();
  } catch (error) {
    showError(error.message);
    setText('form-status', 'Üretim başarısız oldu. Hata ayrıntısını kontrol edin.');
  } finally {
    state.generating = false;
    await refreshStatus();
    updateGenerateState();
    if (completed) {
      setText('form-status', 'Üretim tamamlandı. Telaffuz ve doğallığı kulaklıkla değerlendirin.');
    }
  }
}

async function uploadVoice() {
  const file = $('voice-file').files?.[0];
  if (!file) return;
  clearError();
  if (!$('voice-rights').checked) {
    showError('Referans sesi yüklemeden önce kullanım ve klonlama hakkını doğrulayın.');
    $('voice-file').value = '';
    return;
  }
  if (!/\.wav$/i.test(file.name) && file.type !== 'audio/wav' && file.type !== 'audio/x-wav') {
    showError('Referans ses WAV biçiminde olmalı.');
    $('voice-file').value = '';
    return;
  }
  if (file.size > 25 * 1024 * 1024) {
    showError('Referans ses 25 MB’tan büyük olamaz.');
    $('voice-file').value = '';
    return;
  }

  state.uploading = true;
  setText('upload-label', file.name);
  updateGenerateState();
  const formData = new FormData();
  formData.append('file', file);
  formData.append('rights_confirmed', 'true');
  try {
    const payload = await apiRequest('/voices', {method: 'POST', body: formData});
    const voice = payload.voice || payload;
    if (!voice?.id) throw new Error('Sunucu yüklenen ses için bir kimlik döndürmedi.');
    const options = [...$('voice-select').options]
      .filter(option => option.value)
      .map(option => ({id: option.value, label: option.textContent}));
    if (!options.some(option => option.id === String(voice.id))) {
      options.push({id: String(voice.id), label: voice.label || voice.name || file.name});
    }
    renderVoices(options, String(voice.id));
    setText('upload-label', voice.label || voice.name || file.name);
    setText('form-status', 'Referans ses hazır ve seçildi.');
  } catch (error) {
    showError(error.message);
    setText('upload-label', 'WAV dosyası seç');
    $('voice-file').value = '';
  } finally {
    state.uploading = false;
    updateGenerateState();
  }
}

function bindParameter(id, outputId, digits = 1) {
  const input = $(id);
  const update = () => setText(outputId, Number(input.value).toFixed(digits));
  input.addEventListener('input', update);
  update();
}

function resetParameters() {
  $('temperature-enabled').checked = false;
  $('temperature').disabled = true;
  $('temperature').value = DEFAULT_PARAMETERS.temperatureValue;
  setText('temperature-output', 'Model varsayılanı');
  $('top-p').value = DEFAULT_PARAMETERS.topP;
  $('top-k').value = DEFAULT_PARAMETERS.topK;
  $('repetition-penalty').value = DEFAULT_PARAMETERS.repetitionPenalty;
  for (const input of document.querySelectorAll('.parameter input')) input.dispatchEvent(new Event('input'));
}

function resolvedTheme() {
  const saved = localStorage.getItem('fatma-theme');
  if (saved === 'light' || saved === 'dark') return saved;
  return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const dark = theme === 'dark';
  $('theme-toggle').setAttribute('aria-label', dark ? 'Açık temaya geç' : 'Koyu temaya geç');
  $('theme-toggle').title = dark ? 'Açık temaya geç' : 'Koyu temaya geç';
}

function initialise() {
  applyTheme(resolvedTheme());
  $('temperature-enabled').addEventListener('change', () => {
    const enabled = $('temperature-enabled').checked;
    $('temperature').disabled = !enabled;
    setText('temperature-output', enabled ? Number($('temperature').value).toFixed(1) : 'Model varsayılanı');
  });
  $('temperature').addEventListener('input', () => {
    if ($('temperature-enabled').checked) {
      setText('temperature-output', Number($('temperature').value).toFixed(1));
    }
  });
  bindParameter('top-p', 'top-p-output');
  bindParameter('top-k', 'top-k-output', 0);
  bindParameter('repetition-penalty', 'repetition-penalty-output');
  updateCharacterCount();

  $('speech-text').addEventListener('input', () => { updateCharacterCount(); validateText(); });
  $('generate-form').addEventListener('submit', generate);
  $('voice-file').addEventListener('change', uploadVoice);
  $('reset-parameters').addEventListener('click', resetParameters);
  $('refresh-status').addEventListener('click', () => refreshStatus({announceError: true}));
  $('refresh-history').addEventListener('click', () => loadHistory({announceError: true}));
  $('dismiss-error').addEventListener('click', clearError);
  $('theme-toggle').addEventListener('click', () => {
    const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    localStorage.setItem('fatma-theme', theme);
    applyTheme(theme);
  });
  window.addEventListener('beforeunload', () => {
    if (state.audioObjectUrl) URL.revokeObjectURL(state.audioObjectUrl);
    clearInterval(state.statusTimer);
  });

  Promise.all([refreshStatus({announceError: true}), loadCases(), loadHistory()]);
  state.statusTimer = window.setInterval(() => refreshStatus(), 5000);
}

initialise();
