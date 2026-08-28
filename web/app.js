const $ = (selector) => document.querySelector(selector);
const ui = {
  intro: $("#intro"), conversation: $("#conversation"), orb: $("#orbWrap"),
  mic: $("#micButton"), state: $("#stateText"), detail: $("#detailText"),
  form: $("#textForm"), input: $("#textInput"), send: $("#sendButton"), metrics: $("#metrics"),
  healthButton: $("#healthButton"), healthText: $("#healthText"),
  healthDialog: $("#healthDialog"), healthDetails: $("#healthDetails"),
  healthSummary: $("#healthSummary"), toast: $("#toast"), closeHealth: $("#closeHealth"),
};

const LANGUAGE_STORAGE_KEY = "aicom.language";
const STATE_KEYS = {
  idle: "stateIdle", connecting: "stateConnecting", switching: "stateSwitching",
  listening: "stateListening", recording: "stateRecording", transcribing: "stateTranscribing",
  thinking: "stateThinking", speaking: "stateSpeaking", error: "stateError",
};
const ERROR_KEYS = {
  invalid_message: "invalidMessage", unsupported_language: "unsupportedLanguage",
  response_failed: "responseFailed", speech_not_understood: "speechNotUnderstood",
  speech_synthesis_failed: "speechSynthesisFailed",
};
const COMPONENT_KEYS = {
  llm: "componentLLM", "stt-whisper": "componentWhisper", "stt-nemotron": "componentNemotron",
  "tts-freya": "componentFreya", "tts-macos": "componentMacOS", "tts-none": "componentSilent",
};
const ignoredTurnIds = new Set();
const MAX_IGNORED_TURNS = 128;

let language = preferredLanguage();
let acknowledgedLanguage = null;
let pendingLanguage = language;
let receivedInitialLanguage = false;
let languageRequests = [];
let socket;
let sessionId;
let reconnectTimer;
let connectionAttempt = 0;
let unloading = false;
let micEnabled = false;
let micStarting = false;
let audioContext;
let workletReady;
let micStream;
let micSource;
let processor;
let silentOutput;
let capture = false;
let captureStartedAt = 0;
let speechFrames = 0;
let silenceFrames = 0;
let calibrationFrames = 0;
let noiseFloor = 0.006;
let preRoll = [];
let audioQueue = [];
let activeAudio = null;
let activeAudioEvent = null;
let assistantSpeaking = false;
let currentAssistant = null;
let activeTurnId = null;
let serverState = "listening";
let currentState = "idle";
let currentDetail = { key: "detailPrivacy", values: {} };
let latestMetrics = { type: "ready" };
let healthState = "loading";
let healthData = null;
let healthRequest = 0;
let toastTimer;

function preferredLanguage() {
  try {
    const saved = localStorage.getItem(LANGUAGE_STORAGE_KEY);
    if (saved === "en" || saved === "tr") return saved;
  } catch {
    // Storage can be unavailable in private or restricted browser contexts.
  }
  return (navigator.language || "").toLowerCase().startsWith("tr") ? "tr" : "en";
}

function t(key, values = {}) {
  const template = AICOM_COPY[language][key] ?? AICOM_COPY.en[key] ?? key;
  return template.replace(/\{(\w+)\}/g, (match, name) => String(values[name] ?? match));
}

function formatTime(milliseconds) {
  if (!Number.isFinite(milliseconds)) return "—";
  return new Intl.NumberFormat(language).format(milliseconds) + " ms";
}

function canInteract() {
  return socket?.readyState === WebSocket.OPEN
    && pendingLanguage === null && acknowledgedLanguage === language;
}

function renderControls() {
  ui.mic.disabled = micStarting;
  ui.mic.setAttribute("aria-pressed", String(micEnabled));
  ui.mic.setAttribute("aria-label", t(micEnabled ? "microphoneStop" : "microphoneStart"));
  ui.mic.title = t(micEnabled ? "microphoneStop" : "microphoneStart");
  ui.send.disabled = !canInteract();
  ui.form.setAttribute("aria-busy", String(pendingLanguage !== null));
  document.querySelectorAll("[data-language]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.language === language));
  });
}

function renderState() {
  ui.orb.dataset.state = currentState;
  ui.state.textContent = t(STATE_KEYS[currentState] || "stateIdle");
  ui.detail.textContent = t(currentDetail.key, currentDetail.values);
}

function setState(state, detailKey = "detailLocalProcessing", values = {}) {
  currentState = state;
  currentDetail = { key: detailKey, values };
  renderState();
}

function renderMetrics() {
  if (latestMetrics.type === "transcript") {
    ui.metrics.textContent = Number.isFinite(latestMetrics.transcription_ms)
      ? t("metricsRecognition", { time: formatTime(latestMetrics.transcription_ms) })
      : t("metricsTextInput");
  } else if (latestMetrics.type === "metrics") {
    ui.metrics.textContent = t("metricsResponse", {
      text: formatTime(latestMetrics.first_token_ms),
      voice: latestMetrics.first_audio_ms == null
        ? t("metricsVoiceOff") : formatTime(latestMetrics.first_audio_ms),
      total: formatTime(latestMetrics.total_ms),
    });
  } else {
    ui.metrics.textContent = t("metricsReady");
  }
}

function applyLanguage() {
  document.documentElement.lang = language;
  document.title = t("pageTitle");
  ui.input.lang = language;
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    element.textContent = t(element.dataset.i18n);
  });
  for (const attribute of ["aria-label", "placeholder", "title"]) {
    document.querySelectorAll("[data-i18n-" + attribute + "]").forEach((element) => {
      element.setAttribute(attribute, t(element.getAttribute("data-i18n-" + attribute)));
    });
  }
  ui.conversation.querySelectorAll(".message").forEach((element) => {
    element.querySelector(".role").textContent = t(
      element.dataset.role === "user" ? "userRole" : "assistantRole",
    );
  });
  renderControls();
  renderState();
  renderMetrics();
  renderHealth();
}

function toast(key, values = {}) {
  ui.toast.textContent = t(key, values);
  ui.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ui.toast.classList.remove("show"), 4000);
}

function send(event) {
  if (socket?.readyState !== WebSocket.OPEN) return false;
  try {
    socket.send(JSON.stringify(event));
    return true;
  } catch {
    return false;
  }
}

function sendPCM(samples) {
  if (!canInteract()) return false;
  try {
    socket.send(samples.buffer);
    return true;
  } catch {
    return false;
  }
}

function ignoreTurn(turnId) {
  if (!turnId) return;
  ignoredTurnIds.add(turnId);
  while (ignoredTurnIds.size > MAX_IGNORED_TURNS) {
    ignoredTurnIds.delete(ignoredTurnIds.values().next().value);
  }
}

function resetCapture() {
  capture = false;
  captureStartedAt = 0;
  speechFrames = 0;
  silenceFrames = 0;
  preRoll = [];
}

function stopAssistant({ notify = true } = {}) {
  ignoreTurn(activeTurnId);
  ignoreTurn(currentAssistant?.dataset.turnId);
  ignoreTurn(activeAudioEvent?.turn_id);
  audioQueue.forEach((event) => ignoreTurn(event.turn_id));
  audioQueue = [];
  if (activeAudio) {
    activeAudio.onended = null;
    activeAudio.onerror = null;
    activeAudio.pause();
    activeAudio.src = "";
    activeAudio = null;
  }
  activeAudioEvent = null;
  assistantSpeaking = false;
  currentAssistant = null;
  activeTurnId = null;
  serverState = "listening";
  if (notify) send({ type: "interrupt" });
}

function requestLanguageSync() {
  pendingLanguage = language;
  if (send({ type: "language.set", language })) languageRequests.push(language);
  renderControls();
}

function acknowledgeLanguage(event) {
  if (event.language !== "en" && event.language !== "tr") return;
  acknowledgedLanguage = event.language;
  if (!receivedInitialLanguage) {
    // The first language event is the connection welcome, not a language.set acknowledgement.
    receivedInitialLanguage = true;
    return;
  }
  const expected = languageRequests.shift();
  if (expected && expected !== event.language && languageRequests.length === 0) {
    requestLanguageSync();
    return;
  }
  // Drain every queued acknowledgement before accepting turns, even when an earlier
  // acknowledgement happens to match the latest target after a rapid EN/TR/EN switch.
  if (languageRequests.length === 0 && event.language === language) {
    pendingLanguage = null;
    serverState = "listening";
    resetCapture();
    renderControls();
    setState(micEnabled ? "listening" : "idle", "detailLanguageReady");
  }
}

function switchLanguage(nextLanguage) {
  if (!["en", "tr"].includes(nextLanguage) || nextLanguage === language) return;
  language = nextLanguage;
  try {
    localStorage.setItem(LANGUAGE_STORAGE_KEY, language);
  } catch {
    // The selected language still applies for this page when persistence is unavailable.
  }
  stopAssistant({ notify: false });
  resetCapture();
  calibrationFrames = 0;
  pendingLanguage = language;
  clearTimeout(toastTimer);
  ui.toast.classList.remove("show");
  applyLanguage();
  setState("switching", "detailSwitching");
  requestLanguageSync();
  loadHealth();
}

async function createSession() {
  const response = await fetch("/api/sessions", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ language }),
  });
  if (!response.ok) throw new Error("sessionFailed");
  const data = await response.json();
  if (typeof data.id !== "string") throw new Error("sessionFailed");
  sessionId = data.id;
}

function scheduleReconnect() {
  clearTimeout(reconnectTimer);
  if (!unloading) reconnectTimer = setTimeout(connect, 1200);
}

async function connect() {
  if (unloading || socket?.readyState === WebSocket.OPEN
    || socket?.readyState === WebSocket.CONNECTING) return;
  clearTimeout(reconnectTimer);
  const attempt = ++connectionAttempt;
  pendingLanguage = language;
  acknowledgedLanguage = null;
  receivedInitialLanguage = false;
  languageRequests = [];
  setState("connecting", "detailReconnecting");
  renderControls();
  try {
    if (!sessionId) await createSession();
    if (unloading || attempt !== connectionAttempt) return;
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const connection = new WebSocket(
      scheme + "://" + location.host + "/api/realtime/" + sessionId,
    );
    socket = connection;
    connection.binaryType = "arraybuffer";
    connection.onopen = () => {
      if (socket !== connection) return;
      requestLanguageSync();
      setState("connecting", "detailSwitching");
    };
    connection.onmessage = (message) => {
      if (socket !== connection) return;
      try {
        handleEvent(JSON.parse(message.data));
      } catch {
        toast("invalidResponse");
      }
    };
    connection.onerror = () => {
      if (socket === connection) setState("error", "detailCheckServer");
    };
    connection.onclose = (event) => {
      if (socket !== connection) return;
      socket = null;
      stopAssistant({ notify: false });
      resetCapture();
      pendingLanguage = language;
      acknowledgedLanguage = null;
      if (event.code === 4404) sessionId = null;
      renderControls();
      setState("connecting", "detailReconnecting");
      scheduleReconnect();
    };
  } catch {
    if (unloading || attempt !== connectionAttempt) return;
    setState("error", sessionId ? "detailCheckServer" : "sessionFailed");
    renderControls();
    scheduleReconnect();
  }
}

function addMessage(role, text = "", turnId = "", messageLanguage = language) {
  ui.intro.classList.add("compact");
  const element = document.createElement("article");
  element.className = "message " + role;
  element.dataset.role = role;
  element.lang = messageLanguage;
  if (turnId) element.dataset.turnId = turnId;
  const label = document.createElement("span");
  label.className = "role";
  label.textContent = t(role === "user" ? "userRole" : "assistantRole");
  const body = document.createElement("span");
  body.className = "body";
  body.textContent = text;
  element.append(label, body);
  ui.conversation.append(element);
  ui.conversation.scrollTop = ui.conversation.scrollHeight;
  return element;
}

function assistantFor(turnId) {
  if (!currentAssistant || currentAssistant.dataset.turnId !== turnId) {
    currentAssistant = addMessage("assistant", "", turnId);
  }
  return currentAssistant.querySelector(".body");
}

function handleServerError(event) {
  const key = ERROR_KEYS[event.code]
    || (event.type === "warning" ? "serverWarning" : "responseFailed");
  toast(key);
  if (event.type === "error") setState("error", key);
}

function handleEvent(event) {
  if (!event || typeof event !== "object") return;
  if (event.type === "language") {
    acknowledgeLanguage(event);
    return;
  }
  if (event.type === "error" && !event.turn_id
    && ["invalid_message", "unsupported_language"].includes(event.code)) {
    handleServerError(event);
    return;
  }
  if (!canInteract()) {
    ignoreTurn(event.turn_id);
    return;
  }
  if (event.turn_id && ignoredTurnIds.has(event.turn_id)) return;
  if (event.language && event.language !== language) {
    ignoreTurn(event.turn_id);
    return;
  }
  if (event.turn_id) activeTurnId = event.turn_id;
  if (event.type === "state") {
    serverState = event.state;
    const visibleState = event.state === "listening" && !micEnabled ? "idle" : event.state;
    if (!assistantSpeaking && !capture) setState(visibleState, "detailLocalProcessing");
  } else if (event.type === "transcript") {
    addMessage("user", event.text, event.turn_id, event.language || language);
    latestMetrics = event;
    renderMetrics();
  } else if (event.type === "text_delta") {
    assistantFor(event.turn_id).textContent += event.delta;
    ui.conversation.scrollTop = ui.conversation.scrollHeight;
  } else if (event.type === "text_done") {
    assistantFor(event.turn_id).textContent = event.text;
    currentAssistant = null;
  } else if (event.type === "audio") {
    if (typeof event.url !== "string"
      || new URL(event.url, location.href).origin !== location.origin) return;
    audioQueue.push(event);
    playNextAudio();
  } else if (event.type === "metrics") {
    latestMetrics = event;
    renderMetrics();
  } else if (event.type === "warning" || event.type === "error") {
    handleServerError(event);
  } else if (event.type === "interrupted") {
    stopAssistant({ notify: false });
  }
}

function playNextAudio() {
  if (activeAudio || !canInteract()) return;
  while (audioQueue.length && ignoredTurnIds.has(audioQueue[0].turn_id)) audioQueue.shift();
  if (!audioQueue.length) {
    assistantSpeaking = false;
    if (!capture) {
      const state = serverState === "listening" ? (micEnabled ? "listening" : "idle") : serverState;
      setState(state, micEnabled ? "detailInterrupt" : "detailMicrophoneOff");
    }
    return;
  }
  const event = audioQueue.shift();
  const audio = new Audio(event.url);
  activeAudio = audio;
  activeAudioEvent = event;
  assistantSpeaking = true;
  setState("speaking", "detailVoiceTiming", {
    provider: event.provider, time: formatTime(event.synthesis_ms),
  });
  const finish = (failed = false) => {
    if (activeAudio !== audio) return;
    audio.onended = null;
    audio.onerror = null;
    activeAudio = null;
    activeAudioEvent = null;
    if (failed) toast("audioFailed");
    playNextAudio();
  };
  audio.onended = () => finish();
  audio.onerror = () => finish(true);
  audio.play().catch((error) => {
    if (activeAudio !== audio) return;
    toast(error.name === "NotAllowedError" ? "audioPermission" : "audioFailed");
    finish();
  });
}

function rmsOf(samples) {
  let sum = 0;
  for (const sample of samples) sum += sample * sample;
  return Math.sqrt(sum / samples.length);
}

function floatToPCM(samples) {
  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const value = Math.max(-1, Math.min(1, samples[i]));
    pcm[i] = value < 0 ? value * 32768 : value * 32767;
  }
  return pcm;
}

function handleMicFrame(samples) {
  if (!micEnabled || !canInteract()) return;
  const rms = rmsOf(samples);
  if (!capture && !assistantSpeaking && calibrationFrames < 50) {
    noiseFloor = noiseFloor * 0.92 + rms * 0.08;
    calibrationFrames++;
  } else if (!capture && !assistantSpeaking && rms < noiseFloor * 1.8) {
    noiseFloor = noiseFloor * 0.995 + rms * 0.005;
  }
  const threshold = assistantSpeaking
    ? Math.max(noiseFloor * 3.4, 0.028)
    : Math.max(noiseFloor * 2.25, 0.012);
  preRoll.push(floatToPCM(samples));
  if (preRoll.length > 14) preRoll.shift();
  if (!capture) {
    speechFrames = rms > threshold ? speechFrames + 1 : Math.max(0, speechFrames - 1);
    if (speechFrames >= (assistantSpeaking ? 14 : 6)) beginCapture();
    return;
  }
  if (!sendPCM(floatToPCM(samples))) {
    resetCapture();
    return;
  }
  silenceFrames = rms < threshold * 0.68 ? silenceFrames + 1 : 0;
  if (silenceFrames >= 45 || performance.now() - captureStartedAt > 30000) endCapture();
}

function beginCapture() {
  if (!canInteract()) return;
  stopAssistant();
  if (!send({ type: "audio.start", language })) return;
  capture = true;
  captureStartedAt = performance.now();
  silenceFrames = 0;
  speechFrames = 0;
  for (const frame of preRoll) {
    if (!sendPCM(frame)) {
      resetCapture();
      return;
    }
  }
  preRoll = [];
  setState("recording", "detailRecording");
}

function endCapture() {
  if (!capture) return;
  resetCapture();
  if (canInteract() && send({ type: "audio.commit" })) {
    serverState = "transcribing";
    setState("transcribing", "detailTranscribing");
  }
}

const WORKLET_SOURCE = [
  "class AicomPCM extends AudioWorkletProcessor {",
  "  constructor() { super(); this.source = []; this.position = 0; this.output = []; }",
  "  process(inputs) {",
  "    const input = inputs[0] && inputs[0][0];",
  "    if (!input) return true;",
  "    for (let i = 0; i < input.length; i++) this.source.push(input[i]);",
  "    const ratio = sampleRate / 16000;",
  "    while (this.position + 1 < this.source.length) {",
  "      const left = Math.floor(this.position), fraction = this.position - left;",
  "      this.output.push(this.source[left] * (1 - fraction) + this.source[left + 1] * fraction);",
  "      this.position += ratio;",
  "      if (this.output.length >= 320) {",
  "        this.port.postMessage(new Float32Array(this.output.splice(0, 320)));",
  "      }",
  "    }",
  "    const consumed = Math.floor(this.position);",
  "    if (consumed > 0) { this.source.splice(0, consumed); this.position -= consumed; }",
  "    return true;",
  "  }",
  "}",
  "registerProcessor('aicom-pcm', AicomPCM);",
].join("\n");

async function prepareAudioContext() {
  if (!audioContext || audioContext.state === "closed") {
    audioContext = new AudioContext();
    workletReady = null;
  }
  await audioContext.resume();
  if (!workletReady) {
    const workletUrl = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: "text/javascript" }));
    workletReady = audioContext.audioWorklet.addModule(workletUrl)
      .finally(() => URL.revokeObjectURL(workletUrl));
  }
  try {
    await workletReady;
  } catch (error) {
    workletReady = null;
    throw error;
  }
}

function disconnectMicrophone() {
  if (processor) {
    processor.port.onmessage = null;
    processor.disconnect();
  }
  micSource?.disconnect();
  silentOutput?.disconnect();
  micStream?.getTracks().forEach((track) => track.stop());
  processor = null;
  micSource = null;
  silentOutput = null;
  micStream = null;
}

async function toggleMicrophone() {
  if (micStarting) return;
  if (micEnabled) {
    endCapture();
    micEnabled = false;
    disconnectMicrophone();
    resetCapture();
    renderControls();
    if (!assistantSpeaking) setState("idle", "detailMicrophoneOff");
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia || typeof AudioContext === "undefined") {
    toast("microphoneUnsupported");
    return;
  }
  micStarting = true;
  renderControls();
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
    await prepareAudioContext();
    micSource = audioContext.createMediaStreamSource(micStream);
    processor = new AudioWorkletNode(audioContext, "aicom-pcm");
    silentOutput = audioContext.createGain();
    silentOutput.gain.value = 0;
    micSource.connect(processor).connect(silentOutput).connect(audioContext.destination);
    processor.port.onmessage = (event) => handleMicFrame(event.data);
    micEnabled = true;
    calibrationFrames = 0;
    resetCapture();
    setState(canInteract() ? "listening" : "connecting",
      canInteract() ? "detailStartSpeaking" : "detailSwitching");
  } catch (error) {
    disconnectMicrophone();
    const errors = {
      NotAllowedError: "microphoneDenied", SecurityError: "microphoneDenied",
      NotFoundError: "microphoneMissing", NotReadableError: "microphoneBusy",
    };
    toast(errors[error.name] || "microphoneFailed");
  } finally {
    micStarting = false;
    renderControls();
  }
}

function renderHealth() {
  const statusKeys = {
    loading: "systemLoading", ready: "systemReady",
    setup: "systemSetup", unavailable: "systemUnavailable",
  };
  ui.healthButton.className = "health " + (
    healthState === "ready" ? "ready" : healthState === "loading" ? "" : "error"
  );
  ui.healthText.textContent = t(statusKeys[healthState]);
  ui.healthButton.title = t("showSystem");
  ui.healthButton.setAttribute("aria-label", t(statusKeys[healthState]) + ". " + t("showSystem"));
  ui.healthDetails.replaceChildren();
  ui.healthSummary.textContent = "";
  if (!healthData) {
    const placeholder = document.createElement("p");
    placeholder.className = "health-placeholder";
    placeholder.textContent = t(statusKeys[healthState]);
    ui.healthDetails.append(placeholder);
    return;
  }
  for (const item of healthData.components) {
    const row = document.createElement("div");
    row.className = "component " + (item.ready ? "ready" : "");
    const dot = document.createElement("i");
    dot.setAttribute("aria-hidden", "true");
    const content = document.createElement("div");
    const title = document.createElement("strong");
    const status = document.createElement("span");
    const detail = document.createElement("span");
    title.textContent = COMPONENT_KEYS[item.name] ? t(COMPONENT_KEYS[item.name]) : item.name;
    status.className = "component-status";
    status.textContent = t(item.ready ? "componentReady" : "componentUnavailable");
    detail.textContent = item.detail;
    detail.lang = "en";
    content.append(title, status, detail);
    row.append(dot, content);
    ui.healthDetails.append(row);
  }
  ui.healthSummary.textContent = t("knowledgeCount", {
    count: new Intl.NumberFormat(language).format(healthData.knowledge_documents || 0),
  });
}

async function loadHealth(showDialog = false) {
  const request = ++healthRequest;
  const requestedLanguage = language;
  healthState = "loading";
  healthData = null;
  renderHealth();
  if (showDialog && !ui.healthDialog.open) ui.healthDialog.showModal();
  try {
    const response = await fetch("/api/health?language=" + requestedLanguage);
    if (!response.ok) throw new Error("Health request failed");
    const data = await response.json();
    if (request !== healthRequest || requestedLanguage !== language) return;
    if (!Array.isArray(data.components)) throw new Error("Invalid health response");
    healthData = data;
    healthState = data.ready ? "ready" : "setup";
  } catch {
    if (request !== healthRequest || requestedLanguage !== language) return;
    healthState = "unavailable";
  }
  renderHealth();
}

ui.mic.addEventListener("click", toggleMicrophone);
ui.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = ui.input.value.trim();
  if (!text) return;
  if (!canInteract()) {
    toast(socket?.readyState === WebSocket.OPEN ? "waitForLanguage" : "waitForConnection");
    return;
  }
  resetCapture();
  stopAssistant();
  if (send({ type: "text", text, language })) {
    ui.input.value = "";
    serverState = "thinking";
    setState("thinking", "detailLocalProcessing");
  }
});
document.querySelectorAll("[data-language]").forEach((button) => {
  button.addEventListener("click", () => switchLanguage(button.dataset.language));
});
ui.healthButton.addEventListener("click", () => loadHealth(true));
ui.closeHealth.addEventListener("click", () => ui.healthDialog.close());
window.addEventListener("beforeunload", () => {
  unloading = true;
  clearTimeout(reconnectTimer);
  disconnectMicrophone();
  socket?.close();
  if (sessionId) {
    fetch("/api/sessions/" + sessionId, { method: "DELETE", keepalive: true }).catch(() => {});
  }
});

applyLanguage();
connect();
loadHealth();
