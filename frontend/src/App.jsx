import { useState } from "react";
import { CopilotKit, CopilotChat } from "@copilotkit/react-core/v2";
import { HttpAgent } from "@ag-ui/client";
import ProfilePanel from "./components/ProfilePanel.jsx";

// Direct browser -> AG-UI connection to the FastAPI endpoint (no CopilotKit
// runtime server). The Vite dev proxy rewrites /api/agui -> localhost:8000/agui.
const ragAgent = new HttpAgent({ url: "/api/agui" });

export default function App() {
  const [showProfile, setShowProfile] = useState(false);

  return (
    <CopilotKit agents__unsafe_dev_only={{ rag_agent: ragAgent }} agent="rag_agent">
      <div className="app dark">
        <header>
          <span className="dot" />
          <h1>Agentic RAG</h1>
          <div className="header-actions">
            <button
              className={showProfile ? "ghost active" : "ghost"}
              onClick={() => setShowProfile((v) => !v)}
            >
              Memory
            </button>
          </div>
        </header>
        <div className="body">
          <main className="chat">
            <CopilotChat
              labels={{ chatInputPlaceholder: "Type your question…" }}
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
