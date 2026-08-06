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

// ---- barge-in: cut at a word boundary rather than mid-syllable ----------
//
// Speech energy dips between words. As each chunk is scheduled we measure its
// RMS in short windows and keep that envelope against the playback clock, so
// an interrupt can look ahead for the next quiet point and stop there instead
// of wherever the user happened to click.
//
// Measured on real SILMA Arabic output (AR3/AR4 bench clips): at a 5% of-peak
// threshold the median wait to the next trough is ~70ms and p90 is ~410ms.
// MAX_WAIT_MS trades cut quality against responsiveness — past the cap we cut
// immediately rather than let the agent talk over the user.
// Measured across four real SILMA clips: a qualifying gap is only within reach
// ~57% of the time at these settings, and tightening the rule to exclude
// stop-consonant closures pushes the miss rate past 65%. So the gap search is
// now opportunistic — take a clean cut when one is genuinely imminent, and
// ramp out otherwise rather than truncating a waveform mid-cycle (the click).
export const interruptConfig = {
  // Stays at 10ms. A longer window does NOT find longer pauses — it averages
  // them away: a 1000ms window over a 150ms pause plus 850ms of speech barely
  // dips, and measured across four SILMA clips it produced ZERO windows under
  // 2% of peak in three of them. Pause LENGTH is what distinguishes a clause
  // boundary from a word gap, so minGapMs is the knob, not windowMs.
  windowMs: 100,        // RMS window; 10ms ≈ 240 samples at 24kHz
  // Depth separates real pauses from stop-consonant closures: measured on four
  // SILMA clips, genuine pauses sit at 0.8-1.3% of peak while closures inside a
  // word sit at 3.3-4.4%. 5% let the closures through, which is what made cuts
  // land mid-word.
  thresholdPct: 0.02,
  // Pauses in this audio are bimodal: a cluster at 80-120ms (phrase/clause
  // boundaries) and then nothing until 300ms+ (clip ends). 80ms therefore
  // targets "let it finish the phrase" without waiting for the chunk to end —
  // ~41 such pauses per minute, median 650ms away from any random instant.
  minGapMs: 100,
  // Don't even start looking until the word in progress has had time to finish.
  // An Arabic word runs roughly 300-400ms, so searching from t+dwell means the
  // first candidate is a boundary rather than a closure inside the current word.
  minDwellMs: 10,
  // How far past the dwell a phrase boundary may sit and still be worth
  // waiting for. The search itself is instant (the envelope is already
  // buffered) — this only bounds how far ahead the cut may be scheduled.
  maxWaitMs: 12000,
  fadeMs: 400,         // ramp length when no gap is reachable (0 = hard cut)
  debug: true,
};

/** Tune from the browser console while listening, e.g.
 *  setInterruptConfig({ thresholdPct: 0.10, maxWaitMs: 200 }) */
export function setInterruptConfig(patch) {
  Object.assign(interruptConfig, patch);
  console.log("[tts] interrupt config →", { ...interruptConfig });
}

const log = (...a) => interruptConfig.debug && console.log("[tts]", ...a);

/**
 * Cut playback the instant the user commits a message, not once the run
 * starts. CopilotKit's own submit path awaits `waitForActiveRunToSettle()`
 * before the agent run begins, so hooking `onRunInitialized` reacts visibly
 * late. These listeners run in the CAPTURE phase, ahead of the chat input's
 * own Enter handler, so the cut is scheduled on the same keystroke.
 *
 * Returns a teardown function.
 */
export function installSubmitInterrupt() {
  const onKeyDown = (e) => {
    // matches CopilotChatInput's own rule: Enter without Shift submits
    if (e.key !== "Enter" || e.shiftKey || e.isComposing) return;
    if (!(e.target instanceof HTMLTextAreaElement)) return;
    if (!e.target.value.trim()) return; // empty box won't send
    interruptSpeech("enter");
  };

  const onPointerDown = (e) => {
    const button = e.target?.closest?.('[data-testid="copilot-send-button"]');
    if (button) interruptSpeech("send-button");
  };

  document.addEventListener("keydown", onKeyDown, true);
  document.addEventListener("pointerdown", onPointerDown, true);
  log("submit interrupt armed (Enter + send button, capture phase)");

  return () => {
    document.removeEventListener("keydown", onKeyDown, true);
    document.removeEventListener("pointerdown", onPointerDown, true);
  };
}

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
  log("hard stop (immediate)");
}

/**
 * Barge-in. Stops at the next inter-word gap if one is close enough, otherwise
 * cuts immediately. Unlike stopSpeech() the audio keeps playing until the cut
 * instant, so the current word finishes.
 */
export function interruptSpeech(reason = "user") {
  const session = current;
  if (!session || !session.ctx) {
    log(`interrupt (${reason}): nothing playing`);
    return;
  }

  const { ctx, envelope, peak } = session;
  const now = ctx.currentTime;
  const threshold = peak * interruptConfig.thresholdPct;
  // Nothing before this instant is a candidate — let the current word land.
  const searchFrom = now + interruptConfig.minDwellMs / 1000;
  const deadline = searchFrom + interruptConfig.maxWaitMs / 1000;

  log(`── interrupt (${reason}) at ${now.toFixed(3)}s ──`);
  log(
    `   peak RMS ${peak.toFixed(4)} | threshold ${threshold.toFixed(4)} ` +
      `(${(interruptConfig.thresholdPct * 100).toFixed(1)}% of peak) | ` +
      `gap ≥${interruptConfig.minGapMs}ms`
  );
  log(
    `   dwell ${interruptConfig.minDwellMs}ms → searching ` +
      `${searchFrom.toFixed(3)}s to ${deadline.toFixed(3)}s`
  );

  // A single quiet window is not a word boundary — it is just as likely to be
  // a stop-consonant closure or a low-amplitude moment inside a syllable. Only
  // quiet that PERSISTS for minGapMs is a gap worth cutting in. The cut lands
  // at the start of that run, i.e. the instant the word finishes.
  const minWindows = Math.max(
    1,
    Math.ceil(interruptConfig.minGapMs / interruptConfig.windowMs)
  );
  let trough = null;
  let scanned = 0;
  let shortDips = 0; // dips rejected for being too brief
  let runStart = -1;
  let runLen = 0;

  for (let i = 0; i < envelope.length; i++) {
    const point = envelope[i];
    if (point.t < searchFrom) continue;
    if (point.t > deadline && runLen === 0) break; // nothing pending, out of time
    scanned++;
    if (point.rms < threshold) {
      if (runLen === 0) {
        if (point.t > deadline) break; // a gap starting this late is too late
        runStart = i;
      }
      runLen++;
      if (runLen >= minWindows) {
        trough = envelope[runStart];
        break;
      }
    } else {
      if (runLen > 0) shortDips++;
      runLen = 0;
      runStart = -1;
    }
  }

  // Gap in reach -> clean cut, exactly at the gap, no ramp.
  // No gap -> ramp to silence over fadeMs so the truncation doesn't click.
  const fadeSec = trough ? 0 : interruptConfig.fadeMs / 1000;
  // The dwell is already committed, so a fade begins where the search did.
  const fadeFrom = searchFrom;
  const cutAt = trough ? trough.t : fadeFrom + fadeSec;
  const waitMs = Math.max(0, Math.round((cutAt - now) * 1000));

  if (trough) {
    log(
      `   ✓ CLEAN CUT — gap opens at ${trough.t.toFixed(3)}s ` +
        `(rms ${trough.rms.toFixed(4)} = ${((trough.rms / peak) * 100).toFixed(1)}% of peak, ` +
        `${runLen * interruptConfig.windowMs}ms quiet) after ${scanned} window(s), ` +
        `${shortDips} short dip(s) skipped → letting it run ${waitMs}ms`
    );
  } else {
    log(
      `   ~ FADE — no ${interruptConfig.minGapMs}ms gap in the ` +
        `${interruptConfig.maxWaitMs}ms after the dwell (scanned ${scanned}, ` +
        `${shortDips} short dip(s) rejected) → full volume for ` +
        `${interruptConfig.minDwellMs}ms, then ramping out over ${interruptConfig.fadeMs}ms`
    );
    if (session.gain && fadeSec > 0) {
      const g = session.gain.gain;
      g.cancelScheduledValues(now);
      g.setValueAtTime(g.value, fadeFrom); // full volume through the dwell
      g.linearRampToValueAtTime(0.0001, cutAt);
    }
  }

  // Output latency is what you actually hear on top of the numbers above —
  // WASAPI shared mode is commonly 100-200ms.
  const outMs = Math.round((ctx.outputLatency || ctx.baseLatency || 0) * 1000);
  log(`   ⏱ output latency ${outMs}ms → perceived tail ≈ ${waitMs + outMs}ms`);
  log(
    `   ✂ cut at ${cutAt.toFixed(3)}s | spoke ${cutAt.toFixed(2)}s of ` +
      `${session.playhead.toFixed(2)}s buffered`
  );

  // Detach before mutating so a later stopSpeech() can't hard-kill the pending
  // cut, and so `speak()`'s catch treats the aborted fetch as expected.
  current = null;
  session.stopped = true;
  session.controller.abort();

  for (const source of session.sources) {
    try {
      source.onended = null;
      source.stop(cutAt); // absolute context time; sources starting later never sound
    } catch {
      /* already finished */
    }
  }

  setTimeout(() => {
    session.sources.length = 0;
    if (session.ctx.state !== "closed") session.ctx.close();
    session.onState?.("idle");
    log(`   ● stopped (${reason})`);
  }, waitMs + 30);
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

  const t0 = performance.now();
  const speakable = stripForSpeech(text);
  if (!speakable) {
    onState?.("idle");
    return;
  }
  log(`speak: ${speakable.length} chars`);

  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) {
    onState?.("error");
    return;
  }

  // Built before the first await so it is still inside the click gesture —
  // browsers refuse to start an AudioContext created outside one.
  const ctx = new AudioCtx();
  // Everything plays through one gain node so an interrupt can ramp the whole
  // utterance down in a single automation, regardless of how many buffers are
  // currently scheduled.
  const gain = ctx.createGain();
  gain.connect(ctx.destination);
  const session = {
    controller: new AbortController(),
    ctx,
    gain,
    stopped: false,
    sources: [],
    onState,
    envelope: [], // [{t: absolute ctx time, rms}] — drives interruptSpeech()
    peak: 0,
    playhead: 0,
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
    log(
      `stream open | ${sampleRate}Hz | route=${res.headers.get("X-TTS-Route")} ` +
        `seg=${res.headers.get("X-TTS-Segmenter")} clips=${res.headers.get("X-TTS-Clips")} ` +
        `cache=${res.headers.get("X-TTS-Cache") || "miss"}`
    );

    let playhead = 0;      // ctx time at which the next buffer should start
    let pending = null;    // bytes not yet long enough to be worth scheduling
    let playing = false;
    let lastSource = null;
    let clips = 0;

    // RMS per short window, stamped with the absolute time each window will be
    // heard. Cheap: one pass over samples we have already decoded.
    const measure = (channel, startAt) => {
      const win = Math.max(1, Math.round((sampleRate * interruptConfig.windowMs) / 1000));
      for (let i = 0; i + win <= channel.length; i += win) {
        let sum = 0;
        for (let j = i; j < i + win; j++) sum += channel[j] * channel[j];
        const rms = Math.sqrt(sum / win);
        if (rms > session.peak) session.peak = rms;
        session.envelope.push({ t: startAt + i / sampleRate, rms });
      }
    };

    const schedule = (bytes) => {
      const channel = pcm16ToFloat32(bytes);
      if (!channel.length) return;
      const buffer = ctx.createBuffer(1, channel.length, sampleRate);
      buffer.copyToChannel(channel, 0);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(gain);
      // max() covers an underrun: if the network fell behind the clock we
      // resume at "now" rather than trying to start in the past
      const startAt = Math.max(ctx.currentTime, playhead);
      source.start(startAt);
      measure(channel, startAt);
      playhead = startAt + buffer.duration;
      session.playhead = playhead;
      session.sources.push(source);
      lastSource = source;
      clips++;
      log(
        `clip ${clips}: ${buffer.duration.toFixed(2)}s @ ${startAt.toFixed(3)}s ` +
          `→ buffered to ${playhead.toFixed(2)}s | envelope ${session.envelope.length} ` +
          `windows, peak ${session.peak.toFixed(4)}`
      );
      if (!playing) {
        playing = true;
        onState?.("playing");
        log(`first audio after ${Math.round(performance.now() - t0)}ms`);
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

    log(
      `stream complete: ${clips} buffer(s), ${playhead.toFixed(2)}s audio, ` +
        `${session.envelope.length} envelope windows, peak RMS ${session.peak.toFixed(4)}`
    );

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
