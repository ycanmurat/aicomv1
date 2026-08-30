export function finiteNumber(value) {
  if (value === null || value === undefined || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export function formatSeconds(value) {
  const number = finiteNumber(value);
  return number === null ? '—' : `${number.toFixed(number < 10 ? 2 : 1)} sn`;
}

export function formatRtf(value) {
  const number = finiteNumber(value);
  return number === null ? '—' : number.toFixed(2);
}

export function formatMemory(value) {
  const megabytes = finiteNumber(value);
  if (megabytes === null) return '—';
  return megabytes >= 1024 ? `${(megabytes / 1024).toFixed(2)} GB` : `${Math.round(megabytes)} MB`;
}

export function formatDate(value) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat('tr-TR', {
    day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit'
  }).format(date);
}

export function unwrapList(payload, keys) {
  if (Array.isArray(payload)) return payload;
  for (const key of keys) {
    if (Array.isArray(payload?.[key])) return payload[key];
  }
  return [];
}

export function normalizeMetrics(item = {}) {
  const source = item.metrics ?? item;
  return {
    latency: finiteNumber(source.latency_seconds ?? source.latency ?? source.elapsed_seconds),
    duration: finiteNumber(source.audio_seconds ?? source.duration_seconds ?? source.audio_duration),
    rtf: finiteNumber(source.rtf ?? source.real_time_factor),
    memory: finiteNumber(source.rss_mb ?? source.backend_rss_mb ?? source.memory_mb)
  };
}
