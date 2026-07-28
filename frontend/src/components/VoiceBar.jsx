// Push-to-talk mic: records, sends to /api/stt, drops the transcript into the
// composer. Playback lives with the messages themselves — see SpeakButton.
import { useCallback, useRef, useState } from "react";

// CopilotKit's composer only registers values that arrive as real input
// events — set via the native setter, then dispatch.
function fillComposer(text) {
  const ta = document.querySelector("textarea");
  if (!ta) return;
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLTextAreaElement.prototype,
    "value"
  ).set;
  setter.call(ta, text);
  ta.dispatchEvent(new Event("input", { bubbles: true }));
  ta.focus();
}

export default function VoiceBar() {
  const [micState, setMicState] = useState("idle"); // idle | rec | busy
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);

  const startRec = useCallback(async () => {
    if (micState !== "idle") return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream, { mimeType: "audio/webm" });
      chunksRef.current = [];
      rec.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data);
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        setMicState("busy");
        try {
          const blob = new Blob(chunksRef.current, { type: "audio/webm" });
          const form = new FormData();
          form.append("audio", blob, "speech.webm");
          const res = await fetch("/api/stt", { method: "POST", body: form });
          const data = await res.json();
          if (data.text) fillComposer(data.text);
        } finally {
          setMicState("idle");
        }
      };
      recorderRef.current = rec;
      rec.start();
      setMicState("rec");
    } catch {
      setMicState("idle"); // mic permission denied
    }
  }, [micState]);

  const stopRec = useCallback(() => {
    if (recorderRef.current && micState === "rec") {
      recorderRef.current.stop();
      recorderRef.current = null;
    }
  }, [micState]);

  return (
    <button
      className={micState === "rec" ? "ghost voice-rec" : "ghost"}
      title="Hold to talk — release to transcribe into the input"
      onMouseDown={startRec}
      onMouseUp={stopRec}
      onMouseLeave={stopRec}
      onTouchStart={startRec}
      onTouchEnd={stopRec}
      disabled={micState === "busy"}
    >
      {micState === "rec" ? "🎙 Listening…" : micState === "busy" ? "… " : "🎙 Talk"}
    </button>
  );
}
