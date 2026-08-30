import assert from 'node:assert/strict';
import {
  finiteNumber,
  formatMemory,
  formatRtf,
  formatSeconds,
  normalizeMetrics,
  unwrapList
} from './formatters.js';

assert.equal(finiteNumber('3.25'), 3.25);
assert.equal(finiteNumber(''), null);
assert.equal(finiteNumber('geçersiz'), null);
assert.equal(formatSeconds(3.245), '3.25 sn');
assert.equal(formatSeconds(null), '—');
assert.equal(formatRtf('3.071'), '3.07');
assert.equal(formatMemory(10600), '10.35 GB');
assert.deepEqual(unwrapList({cases: [{id: 1}]}, ['cases']), [{id: 1}]);
assert.deepEqual(unwrapList([{id: 2}], ['cases']), [{id: 2}]);
assert.deepEqual(normalizeMetrics({metrics: {
  latency_seconds: '11.67', audio_seconds: 3.8, rtf: 3.07, rss_mb: 10600
}}), {latency: 11.67, duration: 3.8, rtf: 3.07, memory: 10600});

console.log('formatters: bütün testler geçti');
