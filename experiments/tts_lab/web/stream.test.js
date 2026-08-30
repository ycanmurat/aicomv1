import test from 'node:test';
import assert from 'node:assert/strict';
import {readEvents, scheduleChunk, wavBlob} from './stream.js';

test('NDJSON survives split UTF-8 Turkish letters and partial records', async () => {
  const bytes = new TextEncoder().encode('{"type":"text_delta","text":"İyi günler."}\n{"type":"done"}\n');
  const body = new ReadableStream({start(controller) {for (const byte of bytes) controller.enqueue(Uint8Array.of(byte)); controller.close();}});
  const events = []; for await (const event of readEvents(new Response(body))) events.push(event);
  assert.equal(events[0].text, 'İyi günler.'); assert.equal(events[1].type, 'done');
});
test('HTTP error is not treated as audio', async () => {
  await assert.rejects(async () => {for await (const event of readEvents(new Response('{"detail":"Meşgul"}', {status: 409}))) void event;}, /Meşgul/);
});
test('Audio is scheduled continuously without a new prebuffer on every packet', () => {
  assert.equal(scheduleChunk(0, 1, false).when, 1.12);
  assert.equal(scheduleChunk(1.2, 1.03, true).when, 1.2);
  assert.equal(scheduleChunk(1.2, 1.4, true).gap.toFixed(2), '0.20');
});
test('WAV header matches stereo PCM data', async () => {
  const bytes = new Uint8Array(await wavBlob([new Uint8Array(16)], 48000, 2).arrayBuffer());
  const view = new DataView(bytes.buffer);
  assert.equal(bytes.length, 60); assert.equal(view.getUint32(40, true), 16);
  assert.equal(view.getUint16(22, true), 2); assert.equal(view.getUint32(24, true), 48000);
});
