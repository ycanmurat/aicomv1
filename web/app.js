const $ = (selector) => document.querySelector(selector);
const ui = {
  intro: $("#intro"), conversation: $("#conversation"), orb: $("#orbWrap"),
  mic: $("#micButton"), state: $("#stateText"), detail: $("#detailText"),
  form: $("#textForm"), input: $("#textInput"), metrics: $("#metrics"),
  healthButton: $("#healthButton"), healthText: $("#healthText"),
  healthDialog: $("#healthDialog"), healthDetails: $("#healthDetails"),
  toast: $("#toast"), closeHealth: $("#closeHealth"),
};

let socket;
let sessionId;
let micEnabled = false;
let audioContext;
let micStream;
let processor;
let capture = false;
let captureStartedAt = 0;
let speechFrames = 0;
let silenceFrames = 0;
let calibrationFrames = 0;
let noiseFloor = 0.006;
let preRoll = [];
let audioQueue = [];
let activeAudio = null;
let assistantSpeaking = false;
let currentAssistant = null;
let reconnectTimer;

function setState(state, detail) {
  ui.orb.dataset.state = state;
  const labels = {
    idle: "Başlamak için dokun", connecting: "Yerel hat bağlanıyor…",
    listening: "Dinliyorum", recording: "Seni duyuyorum…",
    transcribing: "Söylediğini çözüyorum…", thinking: "Düşünüyorum…",
    speaking: "Konuşuyorum", error: "Bağlantı sorunu",
  };
  ui.state.textContent = labels[state] || state;
  if (detail) ui.detail.textContent = detail;
}

function toast(message) {
  ui.toast.textContent = message;
  ui.toast.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => ui.toast.classList.remove("show"), 3200);
}

async function createSession() {
  const response = await fetch("/api/sessions", { method: "POST" });
  if (!response.ok) throw new Error("Yerel oturum açılamadı");
  sessionId = (await response.json()).id;
}

async function connect() {
  clearTimeout(reconnectTimer);
  if (!sessionId) await createSession();
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${location.host}/api/realtime/${sessionId}`);
  socket.binaryType = "arraybuffer";
  setState("connecting");
  socket.onopen = () => setState(micEnabled ? "listening" : "idle", "Yerel hat hazır");
  socket.onmessage = (message) => handleEvent(JSON.parse(message.data));
  socket.onerror = () => setState("error", "Uygulama sunucusunu kontrol et");
  socket.onclose = (event) => {
    if (event.code === 4404) sessionId = null;
    setState("connecting", "Yeniden bağlanıyor…");
    reconnectTimer = setTimeout(connect, 1200);
  };
}

function send(event) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(event));
}

function addMessage(role, text = "", turnId = "") {
  ui.intro.classList.add("compact");
  const element = document.createElement("article");
  element.className = `message ${role}`;
  if (turnId) element.dataset.turnId = turnId;
  const label = document.createElement("span");
  label.className = "role";
  label.textContent = role === "user" ? "SEN" : "AICOM";
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

function handleEvent(event) {
  if (event.type === "state") {
    const visibleState = event.state === "listening" && !micEnabled ? "idle" : event.state;
    if (!assistantSpeaking && !capture) setState(visibleState, "İşlem tamamen bu cihazda");
  } else if (event.type === "transcript") {
    addMessage("user", event.text, event.turn_id);
    ui.metrics.textContent = event.transcription_ms ? `STT ${event.transcription_ms} ms` : "Metin girişi";
  } else if (event.type === "text_delta") {
    assistantFor(event.turn_id).textContent += event.delta;
    ui.conversation.scrollTop = ui.conversation.scrollHeight;
  } else if (event.type === "text_done") {
    assistantFor(event.turn_id).textContent = event.text;
    currentAssistant = null;
  } else if (event.type === "audio") {
    audioQueue.push(event);
    playNextAudio();
  } else if (event.type === "metrics") {
    const token = event.first_token_ms ?? "—";
    const voice = event.first_audio_ms ?? "kapalı";
    ui.metrics.textContent = `Metin ${token} ms · ses ${voice} ms · toplam ${event.total_ms} ms`;
  } else if (event.type === "warning" || event.type === "error") {
    toast(event.message);
    if (event.type === "error") setState("error", event.message);
  } else if (event.type === "interrupted") {
    currentAssistant = null;
  }
}

function playNextAudio() {
  if (activeAudio || !audioQueue.length) {
    if (!activeAudio && !audioQueue.length && assistantSpeaking) {
      assistantSpeaking = false;
      if (!capture) {
        setState(
          micEnabled ? "listening" : "idle",
          micEnabled ? "Sözünü kesmek için doğrudan konuş" : "Mikrofon kapalı",
        );
      }
    }
    return;
  }
  const event = audioQueue.shift();
  activeAudio = new Audio(event.url);
  assistantSpeaking = true;
  setState("speaking", `${event.provider} · ${event.synthesis_ms} ms`);
  activeAudio.onended = activeAudio.onerror = () => {
    activeAudio = null;
    playNextAudio();
  };
  activeAudio.play().catch(() => {
    activeAudio = null;
    toast("Sesi duymak için sayfaya bir kez dokun.");
    playNextAudio();
  });
}

function stopAssistant() {
  audioQueue = [];
  if (activeAudio) {
    activeAudio.pause();
    activeAudio.src = "";
    activeAudio = null;
  }
  assistantSpeaking = false;
  send({ type: "interrupt" });
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
  if (!micEnabled) return;
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
    const needed = assistantSpeaking ? 14 : 6;
    if (speechFrames >= needed) beginCapture();
    return;
  }

  socket?.send(floatToPCM(samples).buffer);
  silenceFrames = rms < threshold * 0.68 ? silenceFrames + 1 : 0;
  if (silenceFrames >= 45 || performance.now() - captureStartedAt > 30000) endCapture();
}

function beginCapture() {
  if (assistantSpeaking) stopAssistant();
  capture = true;
  captureStartedAt = performance.now();
  silenceFrames = 0;
  speechFrames = 0;
  send({ type: "audio.start" });
  for (const frame of preRoll) socket?.send(frame.buffer);
  preRoll = [];
  setState("recording", "Bitirdiğinde otomatik anlayacağım");
}

function endCapture() {
  if (!capture) return;
  capture = false;
  silenceFrames = 0;
  send({ type: "audio.commit" });
  setState("transcribing", "Sesi yerel olarak çözüyorum");
}

async function enableMicrophone() {
  if (micEnabled) {
    micEnabled = false;
    micStream?.getTracks().forEach((track) => track.stop());
    processor?.disconnect();
    if (capture) endCapture();
    setState("idle", "Mikrofon kapalı");
    return;
  }
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
    audioContext = audioContext || new AudioContext();
    await audioContext.resume();
    const workletCode = `
      class AicomPCM extends AudioWorkletProcessor {
        constructor() { super(); this.source = []; this.position = 0; this.output = []; }
        process(inputs) {
          const input = inputs[0] && inputs[0][0];
          if (!input) return true;
          for (let i = 0; i < input.length; i++) this.source.push(input[i]);
          const ratio = sampleRate / 16000;
          while (this.position + 1 < this.source.length) {
            const left = Math.floor(this.position), fraction = this.position - left;
            this.output.push(this.source[left] * (1 - fraction) + this.source[left + 1] * fraction);
            this.position += ratio;
            if (this.output.length >= 320) {
              this.port.postMessage(new Float32Array(this.output.splice(0, 320)));
            }
          }
          const consumed = Math.floor(this.position);
          if (consumed > 0) { this.source.splice(0, consumed); this.position -= consumed; }
          return true;
        }
      }
      registerProcessor('aicom-pcm', AicomPCM);
    `;
    const workletUrl = URL.createObjectURL(new Blob([workletCode], { type: "text/javascript" }));
    await audioContext.audioWorklet.addModule(workletUrl);
    URL.revokeObjectURL(workletUrl);
    const source = audioContext.createMediaStreamSource(micStream);
    processor = new AudioWorkletNode(audioContext, "aicom-pcm");
    const silent = audioContext.createGain();
    silent.gain.value = 0;
    source.connect(processor).connect(silent).connect(audioContext.destination);
    processor.port.onmessage = (event) => handleMicFrame(event.data);
    micEnabled = true;
    calibrationFrames = 0;
    setState("listening", "Konuşmaya başlayabilirsin");
  } catch (error) {
    toast(`Mikrofon açılamadı: ${error.message}`);
  }
}

async function loadHealth(showDialog = false) {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    ui.healthButton.className = `health ${health.ready ? "ready" : "error"}`;
    ui.healthText.textContent = health.ready ? "Tüm yerel bileşenler hazır" : "Kurulum gerekli";
    ui.healthDetails.replaceChildren();
    for (const item of health.components) {
      const row = document.createElement("div");
      row.className = `component ${item.ready ? "ready" : ""}`;
      const dot = document.createElement("i");
      const content = document.createElement("div");
      const title = document.createElement("strong");
      const detail = document.createElement("span");
      title.textContent = item.name;
      detail.textContent = item.detail;
      content.append(title, detail);
      row.append(dot, content);
      ui.healthDetails.append(row);
    }
    if (showDialog) ui.healthDialog.showModal();
  } catch {
    ui.healthButton.className = "health error";
    ui.healthText.textContent = "Sunucu erişilemiyor";
  }
}

ui.mic.addEventListener("click", enableMicrophone);
ui.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = ui.input.value.trim();
  if (!text) return;
  stopAssistant();
  send({ type: "text", text });
  ui.input.value = "";
});
ui.healthButton.addEventListener("click", () => loadHealth(true));
ui.closeHealth.addEventListener("click", () => ui.healthDialog.close());
window.addEventListener("beforeunload", () => {
  if (sessionId) {
    fetch(`/api/sessions/${sessionId}`, { method: "DELETE", keepalive: true }).catch(() => {});
  }
});

connect().catch((error) => setState("error", error.message));
loadHealth();
