# RAG Agent — React frontend

Vite + React chat UI for the FastAPI agent in the repo root.

## Run (dev)

1. Start the backend from the repo root:

   ```
   .venv\Scripts\python -m uvicorn main:app --port 8000
   ```

2. Start the frontend:

   ```
   cd frontend
   npm install
   npm run dev
   ```

   Open http://localhost:5173. The dev server proxies `/api/*` to
   `http://localhost:8000` (see `vite.config.js`), so no CORS setup is needed.

## Build

`npm run build` outputs static files to `dist/`.

## Structure (AG-UI / CopilotKit)

The chat now runs on the AG-UI protocol via CopilotKit:

- Backend: `main.py` exposes the LangGraph agent at `POST /agui`
  (`ag_ui_langgraph.add_langgraph_fastapi_endpoint`), streaming AG-UI events
  as SSE.
- `src/App.jsx` — wraps the app in `<CopilotKit>` with an `@ag-ui/client`
  `HttpAgent` pointed at `/api/agui` (direct browser -> agent connection via
  `agents__unsafe_dev_only`; for production move to `selfManagedAgents` or a
  CopilotKit runtime). `<CopilotChat>` renders the conversation, including
  markdown and reasoning sections.
- `src/api/agentClient.js` — still used by `ProfilePanel` for `/profile` and
  `/profile/reset`. Its `streamChat` (plain `/chat/stream` transport) is no
  longer used by the UI but kept as a reference/fallback, along with the
  custom `Chat.jsx`/`Message.jsx` components from the pre-AG-UI version.
- `src/components/ProfilePanel.jsx` — view/clear what the agent remembers.
