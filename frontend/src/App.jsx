import { useCallback, useEffect, useState } from "react";
import {
  CopilotKit,
  CopilotChat,
  defineToolCallRenderer,
} from "@copilotkit/react-core/v2";
import { HttpAgent } from "@ag-ui/client";
import { Observable } from "rxjs";
import ProfilePanel from "./components/ProfilePanel.jsx";
import Sidebar from "./components/Sidebar.jsx";
import ServicePanel from "./components/ServicePanel.jsx";
import ServiceComparison from "./components/ServiceComparison.jsx";
import DynamicUI from "./components/DynamicUI.jsx";
import VoiceBar from "./components/VoiceBar.jsx";
import AssistantMessage from "./components/AssistantMessage.jsx";

// CopilotKit calls agent.connect() when it attaches to a thread, expecting a
// history replay (stock HttpAgent doesn't implement it). We replay the thread's
// checkpointed messages from the backend as a MESSAGES_SNAPSHOT event, so
// switching conversations restores their history through the AG-UI protocol.
class ThreadAgent extends HttpAgent {
  connect() {
    return new Observable((subscriber) => {
      const runId = crypto.randomUUID();
      (async () => {
        let history = [];
        try {
          const res = await fetch(`/api/threads/${this.threadId}/messages`);
          if (res.ok) history = await res.json();
        } catch {
          /* no history available — treat as a fresh thread */
        }
        subscriber.next({ type: "RUN_STARTED", threadId: this.threadId, runId });
        subscriber.next({ type: "MESSAGES_SNAPSHOT", messages: history });
        subscriber.next({ type: "RUN_FINISHED", threadId: this.threadId, runId });
        subscriber.complete();
      })();
    });
  }
}

// Without a registered renderer, CopilotChat falls back to dumping tool calls
// and their raw results as text. Render `retrieve` as a compact chip, the
// service panel as its card template, and hide anything else.
const toolCallRenderer = defineToolCallRenderer({
  name: "*",
  render: ({ name, args, status }) => {
    if (name === "retrieve") {
      return (
        <div className="tool-chip">
          {status === "complete" ? "🔎 Searched" : "🔎 Searching"}
          {args?.query ? `: ${args.query}` : "…"}
        </div>
      );
    }
    if (name === "show_service_panel") {
      return <ServicePanel data={args} />;
    }
    if (name === "show_service_comparison") {
      return <ServiceComparison data={args} />;
    }
    if (name === "show_dynamic_ui") {
      return <DynamicUI data={args} />;
    }
    return <></>;
  },
});

// Direct browser -> AG-UI connection (no CopilotKit runtime server). Each
// conversation gets its own agent instance carrying its threadId; history is
// restored by ThreadAgent.connect() above.
const makeAgent = (threadId) => new ThreadAgent({ url: "/api/agui", threadId });

export default function App() {
  const [showProfile, setShowProfile] = useState(false);
  const [threads, setThreads] = useState([]);
  const [activeId, setActiveId] = useState(() => crypto.randomUUID());
  const [agent, setAgent] = useState(() => makeAgent(activeId));

  const refreshThreads = useCallback(async () => {
    try {
      const res = await fetch("/api/threads");
      if (res.ok) setThreads(await res.json());
    } catch {
      /* backend not up yet */
    }
  }, []);

  // Event-driven refresh (no polling): the thread list only changes when a
  // run finishes (new title / recency), so refresh then, on tab focus, and
  // after explicit sidebar actions.
  useEffect(() => {
    refreshThreads();
    window.addEventListener("focus", refreshThreads);
    return () => window.removeEventListener("focus", refreshThreads);
  }, [refreshThreads]);

  useEffect(() => {
    const sub = agent.subscribe({
      onRunFinalized: () => refreshThreads(),
    });
    return () => sub?.unsubscribe?.();
  }, [agent, refreshThreads]);

  const newChat = useCallback(() => {
    const id = crypto.randomUUID();
    setActiveId(id);
    setAgent(makeAgent(id));
  }, []);

  const selectThread = useCallback((id) => {
    setActiveId(id);
    setAgent(makeAgent(id));
  }, []);

  const renameThread = useCallback(
    async (id, title) => {
      await fetch(`/api/threads/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
      refreshThreads();
    },
    [refreshThreads]
  );

  const deleteThread = useCallback(
    async (id) => {
      await fetch(`/api/threads/${id}`, { method: "DELETE" });
      if (id === activeId) {
        newChat();
      }
      refreshThreads();
    },
    [activeId, newChat, refreshThreads]
  );

  return (
    <CopilotKit
      key={activeId}
      threadId={activeId}
      agents__unsafe_dev_only={{ rag_agent: agent }}
      agent="rag_agent"
      renderToolCalls={[toolCallRenderer]}
    >
      <div className="app dark">
        <header>
          <span className="dot" />
          <h1>Agentic RAG</h1>
          <div className="header-actions">
            <VoiceBar />
            <button
              className={showProfile ? "ghost active" : "ghost"}
              onClick={() => setShowProfile((v) => !v)}
            >
              Memory
            </button>
          </div>
        </header>
        <div className="body">
          <Sidebar
            threads={threads}
            activeId={activeId}
            onNew={newChat}
            onSelect={selectThread}
            onRename={renameThread}
            onDelete={deleteThread}
          />
          <main className="chat">
            <CopilotChat
              labels={{ chatInputPlaceholder: "Type your question…" }}
              // Swap in our assistant bubble so the newest answer gets a
              // speaker button in its toolbar (see AssistantMessage.jsx).
              messageView={{ assistantMessage: AssistantMessage }}
            />
          </main>
          {showProfile && (
            <ProfilePanel onClose={() => setShowProfile(false)} />
          )}
        </div>
      </div>
    </CopilotKit>
  );
}
