import { useEffect, useState } from "react";
import { getProfile, resetProfile } from "../api/agentClient.js";

export default function ProfilePanel({ onClose }) {
  const [profile, setProfile] = useState(null);
  const [error, setError] = useState(null);

  const load = () => {
    setError(null);
    getProfile().then(setProfile).catch(setError);
  };

  useEffect(load, []);

  const reset = async () => {
    try {
      await resetProfile();
      load();
    } catch (err) {
      setError(err);
    }
  };

  const isEmpty =
    profile && typeof profile === "object" && Object.keys(profile).length === 0;

  return (
    <aside className="profile-panel">
      <div className="panel-header">
        <h2>Agent memory</h2>
        <button className="ghost" onClick={onClose}>
          ✕
        </button>
      </div>
      {error && <p className="error">Could not load profile.</p>}
      {!error && !profile && <p className="muted">Loading…</p>}
      {!error && isEmpty && <p className="muted">Nothing remembered yet.</p>}
      {!error && profile && !isEmpty && (
        <pre className="profile-json">{JSON.stringify(profile, null, 2)}</pre>
      )}
      <div className="panel-footer">
        <button className="ghost" onClick={load}>
          Refresh
        </button>
        <button className="danger" onClick={reset}>
          Clear memory
        </button>
      </div>
    </aside>
  );
}
