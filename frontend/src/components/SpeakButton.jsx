// Speaker button in the assistant message toolbar. Click to synthesize and
// play that answer; click again to stop. Nothing is spoken until asked for.
import { useEffect, useRef, useState } from "react";
import { CopilotChatAssistantMessage } from "@copilotkit/react-core/v2";
import { speak, stopSpeech } from "../api/tts.js";

const SVG = {
  width: 20,
  height: 20,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round",
  strokeLinejoin: "round",
};

// Matches the stroke weight of CopilotKit's built-in toolbar icons.
const SpeakerIcon = () => (
  <svg {...SVG} aria-hidden="true">
    <path d="M11 5 6 9H2v6h4l5 4V5z" />
    <path d="M15.5 8.5a5 5 0 0 1 0 7" />
    <path d="M19 5a9 9 0 0 1 0 14" />
  </svg>
);

const StopIcon = () => (
  <svg {...SVG} aria-hidden="true">
    <rect x="6" y="6" width="12" height="12" rx="2" />
  </svg>
);

const SpinnerIcon = () => (
  <svg {...SVG} className="tts-spinner" aria-hidden="true">
    <path d="M21 12a9 9 0 1 1-6.219-8.56" />
  </svg>
);

const LABEL = {
  idle: "Read this answer aloud",
  loading: "Synthesizing — click to cancel",
  playing: "Stop playback",
  error: "Could not play this answer",
};

export default function SpeakButton({ text }) {
  const [state, setState] = useState("idle");
  const liveRef = useRef(true);

  // This button only exists on the newest answer, so it unmounts as soon as a
  // newer one lands — that is exactly when in-flight audio should be dropped.
  useEffect(() => {
    liveRef.current = true;
    return () => {
      liveRef.current = false;
      stopSpeech();
    };
  }, []);

  // Failures are transient: show the state briefly, then let the user retry.
  useEffect(() => {
    if (state !== "error") return undefined;
    const t = setTimeout(() => liveRef.current && setState("idle"), 2500);
    return () => clearTimeout(t);
  }, [state]);

  const busy = state === "loading" || state === "playing";

  const onClick = () => {
    if (busy) {
      stopSpeech();
      setState("idle");
      return;
    }
    speak(text, {
      onState: (next) => liveRef.current && setState(next),
    });
  };

  return (
    <CopilotChatAssistantMessage.ToolbarButton
      data-testid="tts-speak-button"
      className={state === "error" ? "tts-button tts-error" : "tts-button"}
      title={LABEL[state]}
      onClick={onClick}
    >
      {state === "loading" ? (
        <SpinnerIcon />
      ) : state === "playing" ? (
        <StopIcon />
      ) : (
        <SpeakerIcon />
      )}
    </CopilotChatAssistantMessage.ToolbarButton>
  );
}
