// Speech playback for the per-message "read aloud" button.
//
// The backend streams the answer as raw little-endian 16-bit mono PCM, one
// clip at a time, so sound starts after the first sentence instead of after
// the whole answer. That rules out an <audio> element: raw PCM has no
// container, and appending to a MediaSource would mean muxing in the browser.
// Instead each arriving chunk becomes an AudioBuffer scheduled back-to-back on
// a shared clock, which is gapless as long as synthesis outruns playback
// (SILMA runs at ~0.46x real time, so it does).
//
// Exactly one utterance plays at a time: starting a new one, or unmounting the
// button that owns it, aborts the fetch and stops every scheduled source.

// The spoken text is the answer minus markdown furniture. Deterministic text
// transform — links keep their label, tables/code are dropped.
export function stripForSpeech(md) {
  return String(md || "")
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`([^`]*)`/g, "$1")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/(\*\*|__|\*|_|~~)(.*?)\1/g, "$2")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/^\s*\|.*\|\s*$/gm, " ")
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// Don't schedule buffers smaller than this — a TCP chunk can be a few hundred
// bytes, and one AudioBuffer per fragment is needless churn. The first flush
// ignores it so the opening sound is never held back.
const MIN_SAMPLES = 2400; // 100ms at 24kHz

// The one live utterance, or null. Module scope rather than component state:
// the owning button unmounts whenever a newer answer arrives, and playback has
// to be stoppable from that cleanup path.
let current = null;

export function stopSpeech() {
  const session = current;
  if (!session) return;
  current = null;
  session.stopped = true;
  session.controller.abort();
  for (const source of session.sources) {
    try {
      source.onended = null;
      source.stop();
    } catch {
      /* already finished */
    }
  }
  session.sources.length = 0;
  if (session.ctx && session.ctx.state !== "closed") session.ctx.close();
}

function pcm16ToFloat32(bytes) {
  const samples = bytes.byteLength >> 1;
  const view = new DataView(bytes.buffer, bytes.byteOffset, samples * 2);
  const out = new Float32Array(samples);
  for (let i = 0; i < samples; i++) out[i] = view.getInt16(i * 2, true) / 32768;
  return out;
}

/**
 * Synthesize `text` and play it as it arrives.
 *
 * @param {string} text  raw markdown of the assistant message
 * @param {(state: "loading"|"playing"|"idle"|"error") => void} onState
 */
export async function speak(text, { onState } = {}) {
  stopSpeech();

  const speakable = stripForSpeech(text);
  if (!speakable) {
    onState?.("idle");
    return;
  }

  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) {
    onState?.("error");
    return;
  }

  // Built before the first await so it is still inside the click gesture —
  // browsers refuse to start an AudioContext created outside one.
  const ctx = new AudioCtx();
  const session = {
    controller: new AbortController(),
    ctx,
    stopped: false,
    sources: [],
  };
  current = session;
  onState?.("loading");

  try {
    if (ctx.state === "suspended") await ctx.resume();

    const res = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: speakable }),
      signal: session.controller.signal,
    });
    if (!res.ok || !res.body) throw new Error(`tts failed: ${res.status}`);

    const sampleRate = Number(res.headers.get("X-TTS-Sample-Rate")) || 24000;
    const reader = res.body.getReader();

    let playhead = 0;      // ctx time at which the next buffer should start
    let pending = null;    // bytes not yet long enough to be worth scheduling
    let playing = false;
    let lastSource = null;

    const schedule = (bytes) => {
      const channel = pcm16ToFloat32(bytes);
      if (!channel.length) return;
      const buffer = ctx.createBuffer(1, channel.length, sampleRate);
      buffer.copyToChannel(channel, 0);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      // max() covers an underrun: if the network fell behind the clock we
      // resume at "now" rather than trying to start in the past
      const startAt = Math.max(ctx.currentTime, playhead);
      source.start(startAt);
      playhead = startAt + buffer.duration;
      session.sources.push(source);
      lastSource = source;
      if (!playing) {
        playing = true;
        onState?.("playing");
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (session.stopped) return;
      if (done) break;
      if (!value || !value.byteLength) continue;

      // a chunk boundary can fall inside a 16-bit sample, so carry the odd byte
      let bytes = value;
      if (pending && pending.byteLength) {
        const merged = new Uint8Array(pending.byteLength + value.byteLength);
        merged.set(pending, 0);
        merged.set(value, pending.byteLength);
        bytes = merged;
      }
      const whole = bytes.byteLength - (bytes.byteLength % 2);
      const usable = bytes.subarray(0, whole);
      pending = bytes.subarray(whole);

      if (!playing || usable.byteLength >= MIN_SAMPLES * 2) {
        schedule(usable.slice());
      } else {
        // too small to bother with — hold it with the odd byte for next time
        const merged = new Uint8Array(usable.byteLength + pending.byteLength);
        merged.set(usable, 0);
        merged.set(pending, usable.byteLength);
        pending = merged;
      }
    }

    if (pending && pending.byteLength > 1) schedule(pending.slice());
    if (session.stopped) return;

    if (!lastSource) {
      // stream produced nothing playable
      if (current === session) stopSpeech();
      onState?.("idle");
      return;
    }

    const finish = () => {
      if (current === session) stopSpeech();
      onState?.("idle");
    };
    // A short answer can finish playing before the stream is fully read, in
    // which case "ended" already fired and attaching a handler would hang the
    // button on "playing" forever.
    if (ctx.currentTime >= playhead) finish();
    else lastSource.onended = finish;
  } catch (err) {
    const aborted = err?.name === "AbortError" || session.stopped;
    if (current === session) stopSpeech();
    if (!aborted) onState?.("error");
  }
}
