import { useState } from "react";

export default function Sidebar({ threads, activeId, onNew, onSelect, onRename, onDelete }) {
  const [editingId, setEditingId] = useState(null);
  const [draft, setDraft] = useState("");

  const startEdit = (t) => {
    setEditingId(t.id);
    setDraft(t.title);
  };

  const commitEdit = () => {
    if (editingId && draft.trim()) onRename(editingId, draft.trim());
    setEditingId(null);
  };

  return (
    <nav className="sidebar">
      <button className="new-chat" onClick={onNew}>
        + New chat
      </button>
      <div className="thread-list">
        {threads.length === 0 && (
          <p className="muted small">No conversations yet</p>
        )}
        {threads.map((t) => (
          <div
            key={t.id}
            className={t.id === activeId ? "thread active" : "thread"}
            onClick={() => t.id !== activeId && onSelect(t.id)}
          >
            {editingId === t.id ? (
              <input
                className="thread-rename"
                value={draft}
                autoFocus
                onChange={(e) => setDraft(e.target.value)}
                onBlur={commitEdit}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commitEdit();
                  if (e.key === "Escape") setEditingId(null);
                }}
                onClick={(e) => e.stopPropagation()}
              />
            ) : (
              <span className="thread-title" title={t.title}>
                {t.title}
              </span>
            )}
            <span className="thread-actions">
              <button
                className="icon"
                title="Rename"
                onClick={(e) => {
                  e.stopPropagation();
                  startEdit(t);
                }}
              >
                ✎
              </button>
              <button
                className="icon"
                title="Delete"
                onClick={(e) => {
                  e.stopPropagation();
                  if (window.confirm(`Delete "${t.title}"?`)) onDelete(t.id);
                }}
              >
                🗑
              </button>
            </span>
          </div>
        ))}
      </div>
    </nav>
  );
}
