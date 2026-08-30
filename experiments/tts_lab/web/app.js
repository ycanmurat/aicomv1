import {readEvents, scheduleChunk, wavBlob} from './stream.js';

const $ = id => document.getElementById(id);
const headers = {'X-Fatma-Lab': '1'};
let mode = 'tts', running = false, healthy = false, micAvailable = false;
let stopping = false, backendBusy = false;
let controller, audioContext, nextTime = 0, sourceNodes = new Set();
let recording, recordingChunks = [], recordTimer, recordingStream;
let recordingActive = false, transcribing = false, replayUrl;

function status(text, error = false) {
  $('status').textContent = text;
  $('status').parentElement.classList.toggle('error', error);
}
function setBusy(value) {
  running = value;
  const blocked = value || stopping || backendBusy;
  $('start').disabled = blocked || !healthy || recordingActive || transcribing;
  $('stop').disabled = !value && !recordingActive && !transcribing;
  $('reference-file').disabled = blocked || transcribing || recordingActive;
  $('record').disabled = blocked || !micAvailable || transcribing;
  $('tts-tab').disabled = blocked || transcribing || recordingActive;
  $('ask-tab').disabled = blocked || transcribing || recordingActive;
  $('activity').classList.toggle('active', value || transcribing || recordingActive);
}
function stopPlayback() {
  for (const node of sourceNodes) {try {node.stop();} catch {} node.disconnect();}
  sourceNodes.clear(); nextTime = 0;
  $('replay').pause(); $('reference-player').pause();
}
async function cancel() {
  stopping = true;
  stopPlayback();
  controller?.abort();
  if (recordingActive) {recording.onstop = null; recording.stop(); recordingStream.getTracks().forEach(t => t.stop()); recordingActive = false; clearTimeout(recordTimer); $('record').textContent = 'Mikrofonla yazdır';}
  await fetch('/api/cancel', {method: 'POST', headers}).catch(() => {});
  transcribing = false; stopping = false;
  setBusy(false); status('Durduruldu. Önceki işlem kapanıyor olabilir.');
  await refreshHealth();
}
$('stop').addEventListener('click', cancel);

async function refreshHealth() {
  try {
    const result = await fetch('/api/health');
    const data = await result.json();
    healthy = data.ready;
    backendBusy = data.busy;
    micAvailable = data.microphone_available && Boolean(navigator.mediaDevices?.getUserMedia);
    $('connection').textContent = healthy ? '● Yerelde hazır' : 'Hazır değil';
    $('reference-name').textContent = data.reference;
    $('llm-name').textContent = data.llm_model + (data.llm_available ? '' : ' · hazır değil');
    $('memory').textContent = `Deney süreci: ${(data.process_rss_mb / 1024).toFixed(2)} GB RSS · Ollama yüklü modelleri: ${(data.ollama_loaded_model_mb / 1024).toFixed(2)} GB. Bu değerler toplam sistem belleği değildir.`;
    if (data.has_reference && !$('reference-player').getAttribute('src')) {
      $('reference-player').src = '/api/reference'; $('reference-player').hidden = false;
    }
    if (!running && !recordingActive && !transcribing) {
      setBusy(false);
      if (data.error) status(data.error, true);
      else if ($('status').textContent === 'Ses modeli hazırlanıyor.') status('Hazır. Bir cümle seçin veya kendi metninizi yazın.');
    }
  } catch {
    healthy = false; $('connection').textContent = 'Bağlantı yok';
    if (!running) {setBusy(false); status('Yerel sunucuya ulaşılamıyor. run_moss.sh ile başlatın.', true);}
  }
}

for (const selectedMode of ['tts', 'ask']) {
  $(selectedMode + '-tab').addEventListener('click', () => {
    mode = selectedMode;
    $('tts-tab').setAttribute('aria-selected', String(mode === 'tts'));
    $('ask-tab').setAttribute('aria-selected', String(mode === 'ask'));
    $('text-label').textContent = mode === 'tts' ? 'Fatma ne söylesin?' : 'Fatma’ya ne sormak istersiniz?';
    $('text').value = mode === 'tts' ? 'Merhaba, ben Fatma. Size nasıl yardımcı olabilirim?' : 'Müşteriyle konuşurken iyi bir dinleyici olmak neden önemli?';
    $('research-options').hidden = mode !== 'ask'; $('samples').hidden = mode !== 'tts';
    $('start').textContent = mode === 'tts' ? 'Seslendir ↗' : 'Sor ve dinle ↗';
  });
}
document.querySelectorAll('[data-sample]').forEach(button => button.addEventListener('click', () => {$('text').value = button.dataset.sample;}));
$('use-web').addEventListener('change', () => {$('privacy').textContent = $('use-web').checked ? 'Arama açık: bu soru internete gönderilecek.' : 'Ses ve yanıt üretimi tamamen yerel.';});

function playPacket(event, trialStarted) {
  const bytes = Uint8Array.from(atob(event.pcm), c => c.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  const frames = bytes.byteLength / 2 / event.channels;
  const buffer = audioContext.createBuffer(event.channels, frames, event.sample_rate);
  for (let channel = 0; channel < event.channels; channel++) {
    const target = buffer.getChannelData(channel);
    for (let sample = 0; sample < frames; sample++) target[sample] = view.getInt16((sample * event.channels + channel) * 2, true) / 32768;
  }
  const {when, gap} = scheduleChunk(nextTime, audioContext.currentTime, nextTime > 0);
  const node = audioContext.createBufferSource(); node.buffer = buffer; node.connect(audioContext.destination);
  sourceNodes.add(node); node.onended = () => {node.disconnect(); sourceNodes.delete(node);};
  node.start(when); nextTime = when + buffer.duration;
  return {bytes, gap, scheduledMs: performance.now() - trialStarted + (when - audioContext.currentTime) * 1000};
}

$('trial-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (running || !$('text').value.trim()) return;
  stopPlayback();
  audioContext ??= new AudioContext();
  await audioContext.resume();
  setBusy(true); status(mode === 'tts' ? 'Ses üretiliyor…' : 'Fatma soruyu değerlendiriyor…');
  $('answer').textContent = ''; $('sources').replaceChildren(); $('answer-panel').hidden = mode !== 'ask';
  $('playback').hidden = true;
  for (const id of ['first-pcm', 'playback-start', 'rtf', 'chunks']) $(id).textContent = '—';
  controller = new AbortController();
  const trialController = controller;
  const started = performance.now();
  const captured = []; let firstAnswer = false, gaps = 0, done = false, warning = false;
  let sampleRate = 48000, channels = 2;
  try {
    const response = await fetch('/api/stream', {method: 'POST', headers: {...headers, 'Content-Type': 'application/json'}, signal: trialController.signal,
      body: JSON.stringify({text: $('text').value, mode, use_web: mode === 'ask' && $('use-web').checked, seed: 42})});
    for await (const data of readEvents(response)) {
      if (trialController.signal.aborted) break;
      if (data.type === 'status') status(data.message);
      if (data.type === 'warning') {warning = true; status(data.message, true);}
      if (data.type === 'text_delta') $('answer').textContent += data.text;
      if (data.type === 'sources') {
        const note = document.createElement('p'); note.textContent = 'Arama sonuçlarının özetleri kullanıldı; tam sayfalar okunmadı.'; $('sources').append(note);
        for (const source of data.sources) {
          const link = document.createElement('a'); link.textContent = source.title || source.url; link.href = source.url; link.target = '_blank'; link.rel = 'noopener noreferrer'; $('sources').append(link);
        }
      }
      if (data.type === 'audio') {
        const played = playPacket(data, started);
        sampleRate = data.sample_rate; channels = data.channels;
        captured.push(played.bytes);
        if (played.gap > 0.02) gaps++;
        if (data.role === 'answer' && !firstAnswer) {firstAnswer = true; $('playback-start').textContent = `${Math.round(played.scheduledMs)} ms`;}
        status(data.role === 'filler' ? 'İşlem sürüyor; aynı sesle kısa geri bildirim.' : 'Fatma konuşuyor…');
      }
      if (data.type === 'error') throw new Error(data.message);
      if (data.type === 'done') {
        done = true;
        $('first-pcm').textContent = data.first_answer_pcm_ms == null ? '—' : `${Math.round(data.first_answer_pcm_ms)} ms`;
        $('rtf').textContent = data.rtf == null ? '—' : data.rtf.toFixed(2);
        $('chunks').textContent = data.pcm_chunks;
        $('gaps').textContent = `Oynatımda ${gaps} tampon boşluğu. Süreç tepe RSS: ${(data.process_peak_rss_mb / 1024).toFixed(2)} GB. Bekleme ifadesi yanıt olarak sayılmaz.`;
        if (data.frame_limit_hit) {warning = true; status('Ses süresi sınırına ulaşıldı; metin kesilmiş olabilir.', true);}
      }
    }
    if (!done && !trialController.signal.aborted) throw new Error('Akış tamamlanmadan bağlantı kapandı.');
    if (!trialController.signal.aborted && captured.length) {
      if (replayUrl) URL.revokeObjectURL(replayUrl);
      replayUrl = URL.createObjectURL(wavBlob(captured, sampleRate, channels));
      $('replay').src = replayUrl; $('download').href = replayUrl; $('playback').hidden = false;
      // Retain Stop until already-scheduled audio finishes playing.
      while (sourceNodes.size && !trialController.signal.aborted) await new Promise(resolve => setTimeout(resolve, 80));
    }
    if (!warning && !trialController.signal.aborted) status('Deney tamamlandı. Doğallığı ve telaffuzu dinleyerek değerlendirin.');
  } catch (error) {
    if (error.name !== 'AbortError') {stopPlayback(); status(error.message, true);}
  } finally {
    if (controller === trialController) {controller = null; setBusy(false); await refreshHealth();}
  }
});

$('reference-file').addEventListener('change', async () => {
  const file = $('reference-file').files[0]; if (!file) return;
  stopPlayback(); setBusy(true); status('Yeni referans sesi hazırlanıyor…');
  const data = new FormData(); data.append('file', file);
  try {
    const response = await fetch('/api/reference', {method: 'POST', headers, body: data});
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Referans yüklenemedi.');
    $('reference-player').src = `/api/reference?v=${Date.now()}`; $('reference-player').hidden = false;
    status('Referans hazır. Sonraki deneyler bu sesle yapılacak.');
  } catch (error) {status(error.message, true);}
  finally {setBusy(false); await refreshHealth();}
});

$('record').addEventListener('click', async () => {
  if (recordingActive) {recording.stop(); return;}
  try {
    stopPlayback();
    recordingStream = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true, noiseSuppression: true}});
    recording = new MediaRecorder(recordingStream); recordingChunks = []; recordingActive = true;
    $('record').textContent = 'Kaydı bitir'; setBusy(false); status('Dinliyorum. Bitince “Kaydı bitir”e basın.');
    recording.ondataavailable = event => {if (event.data.size) recordingChunks.push(event.data);};
    recording.onstop = async () => {
      clearTimeout(recordTimer); recordingStream.getTracks().forEach(track => track.stop());
      recordingActive = false; transcribing = true; $('record').textContent = 'Mikrofonla yazdır';
      setBusy(false); status('Kayıt yerel Whisper ile yazıya çevriliyor…');
      const form = new FormData(); form.append('file', new Blob(recordingChunks, {type: recording.mimeType}), 'recording');
      controller = new AbortController();
      try {
        const response = await fetch('/api/transcribe', {method: 'POST', headers, body: form, signal: controller.signal});
        const data = await response.json(); if (!response.ok) throw new Error(data.detail);
        $('text').value = data.text; status(`Metin hazır (${(data.elapsed_ms / 1000).toFixed(1)} sn). Kontrol edip “Sor ve dinle”ye basın.`);
      } catch (error) {if (error.name !== 'AbortError') status(error.message, true);}
      finally {transcribing = false; controller = null; setBusy(false);}
    };
    recording.start(); recordTimer = setTimeout(() => {if (recordingActive) recording.stop();}, 20000);
  } catch (error) {recordingActive = false; recordingStream?.getTracks().forEach(track => track.stop()); setBusy(false); status(`Mikrofon açılamadı: ${error.message}`, true);}
});
window.addEventListener('beforeunload', () => {controller?.abort(); recordingStream?.getTracks().forEach(t => t.stop());});
for (const id of ['replay', 'reference-player']) {
  $(id).addEventListener('play', () => {
    if (running) $(id).pause();
    else $(id === 'replay' ? 'reference-player' : 'replay').pause();
  });
}
await refreshHealth(); setInterval(refreshHealth, 4000);
