// Transport layer between the UI and the agent backend.
//
// This is the single file to replace when migrating to AG-UI:
// swap the raw fetch/stream calls for an @ag-ui/client HttpAgent and map
// its events (TEXT_MESSAGE_CONTENT, RUN_FINISHED, ...) onto the same
// callbacks the components already consume.

const BASE = "/api";

/**
 * Stream an answer for `question`, invoking `onToken` per text chunk.
 * Returns an abort function; callbacks: onToken(text), onDone(), onError(err).
 */
export function streamChat(question, { onToken, onDone, onError }) {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) {
        throw new Error(`Request failed with status ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value, { stream: true });
        if (text) onToken(text);
      }
      onDone();
    } catch (err) {
      if (err.name === "AbortError") {
        onDone();
      } else {
        onError(err);
      }
    }
  })();

  return () => controller.abort();
}

export async function getProfile() {
  const res = await fetch(`${BASE}/profile`);
  if (!res.ok) throw new Error(`Request failed with status ${res.status}`);
  return res.json();
}

export async function resetProfile() {
  const res = await fetch(`${BASE}/profile/reset`, { method: "POST" });
  if (!res.ok) throw new Error(`Request failed with status ${res.status}`);
  return res.json();
}
