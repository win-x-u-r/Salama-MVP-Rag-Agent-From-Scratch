// Recursive renderer for `show_dynamic_ui` (dynamic-schema generative UI).
// The backend LLM composes a tree of catalog components; this maps each node
// type to a React component. Unknown types/props were already stripped by the
// backend validator — this renderer still guards every access, because a
// malformed prop must degrade to "nothing", never to a crash.

const CHART_COLORS = {
  blue: "#3b82f6",
  amber: "#d97706",
  teal: "#14b8a6",
  green: "#22c55e",
};

const TONES = {
  info: { bg: "rgba(59,130,246,0.15)", fg: "#a8c4f9" },
  success: { bg: "rgba(34,197,94,0.15)", fg: "#9be3b5" },
  warning: { bg: "rgba(217,119,6,0.18)", fg: "#f0bd7f" },
  danger: { bg: "rgba(239,68,68,0.15)", fg: "#f3a0a0" },
  neutral: { bg: "rgba(255,255,255,0.08)", fg: "#aab1c2" },
};

const arr = (v) => (Array.isArray(v) ? v : []);
const str = (v) => (v === null || v === undefined ? "" : String(v));

// Experimental: chips send their text as the next user message by driving the
// CopilotKit composer (its React state only listens to real input events).
function sendChip(text) {
  const ta = document.querySelector("textarea");
  if (!ta) return;
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLTextAreaElement.prototype,
    "value"
  ).set;
  setter.call(ta, text);
  ta.dispatchEvent(new Event("input", { bubbles: true }));
  ta.focus();
  ta.dispatchEvent(
    new KeyboardEvent("keydown", {
      key: "Enter",
      code: "Enter",
      keyCode: 13,
      bubbles: true,
    })
  );
}

function BarChart({ props }) {
  const items = arr(props.items).filter(
    (i) => i && typeof i.value === "number"
  );
  if (items.length === 0) return null;
  const max = Math.max(1, ...items.map((i) => i.value));
  const color = CHART_COLORS[props.color] || CHART_COLORS.blue;
  return (
    <div className="dyn-plot">
      {items.map((item, i) => (
        <div
          key={i}
          className="dyn-col"
          title={`${str(item.label)}: ${item.value.toLocaleString("en-US")}${
            props.unit ? " " + str(props.unit) : ""
          }`}
        >
          <span className="dyn-colvalue">
            {item.value.toLocaleString("en-US")}
          </span>
          <span
            className="dyn-vbar"
            style={{
              height: `${Math.max(2, (item.value / max) * 100)}px`,
              background: color,
            }}
          />
          <span className="dyn-collabel">{str(item.label)}</span>
        </div>
      ))}
    </div>
  );
}

// Fixed slice order, CVD-validated for the dark surface (dataviz six-checks).
const PIE_COLORS = ["#3b82f6", "#d97706", "#0d9488", "#e24b4a", "#8b5cf6", "#d4537e"];

function polar(cx, cy, r, angle) {
  return [cx + r * Math.cos(angle), cy + r * Math.sin(angle)];
}

function PieChart({ props }) {
  const items = arr(props.items).filter(
    (i) => i && typeof i.value === "number" && i.value > 0
  );
  if (items.length < 2) return null;
  const total = items.reduce((sum, i) => sum + i.value, 0);
  const cx = 50, cy = 50, r = 46;
  const inner = props.donut ? 24 : 0;
  let angle = -Math.PI / 2;
  const slices = items.map((item, i) => {
    const sweep = (item.value / total) * Math.PI * 2;
    const a0 = angle, a1 = angle + sweep;
    angle = a1;
    const large = sweep > Math.PI ? 1 : 0;
    const [x0, y0] = polar(cx, cy, r, a0);
    const [x1, y1] = polar(cx, cy, r, a1);
    let d;
    if (inner > 0) {
      const [ix0, iy0] = polar(cx, cy, inner, a1);
      const [ix1, iy1] = polar(cx, cy, inner, a0);
      d = `M ${x0} ${y0} A ${r} ${r} 0 ${large} 1 ${x1} ${y1} L ${ix0} ${iy0} A ${inner} ${inner} 0 ${large} 0 ${ix1} ${iy1} Z`;
    } else {
      d = `M ${cx} ${cy} L ${x0} ${y0} A ${r} ${r} 0 ${large} 1 ${x1} ${y1} Z`;
    }
    return { d, color: PIE_COLORS[i % PIE_COLORS.length], item };
  });
  return (
    <div className="dyn-pie">
      <svg viewBox="0 0 100 100" className="dyn-pie-svg" aria-hidden="true">
        {slices.map((s, i) => (
          <path key={i} d={s.d} fill={s.color} stroke="#1e2330" strokeWidth="1.5" />
        ))}
      </svg>
      <div className="dyn-pie-legend">
        {slices.map((s, i) => (
          <div key={i} className="dyn-legend-row">
            <span className="dyn-legend-swatch" style={{ background: s.color }} />
            <span className="dyn-legend-label">{str(s.item.label)}</span>
            <span className="dyn-legend-value">
              {s.item.value.toLocaleString("en-US")}
              {props.unit ? ` ${str(props.unit)}` : ""}
              <span className="dyn-legend-pct">
                {" "}
                {Math.round((s.item.value / total) * 100)}%
              </span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function LineChart({ props }) {
  const series = arr(props.series).filter(
    (s) => s && Array.isArray(s.points) && s.points.length >= 2
  );
  if (series.length === 0) return null;
  const labels = arr(props.labels).map(str);
  const all = series.flatMap((s) => s.points);
  const min = Math.min(...all), max = Math.max(...all);
  const span = max - min || 1;
  const W = 300, H = 110, PAD = 8;
  const nPoints = Math.max(...series.map((s) => s.points.length));
  const x = (i) => PAD + (i / Math.max(1, nPoints - 1)) * (W - PAD * 2);
  const y = (v) => H - PAD - ((v - min) / span) * (H - PAD * 2);
  return (
    <div className="dyn-line">
      <svg viewBox={`0 0 ${W} ${H}`} className="dyn-line-svg" aria-hidden="true">
        <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="rgba(255,255,255,0.18)" />
        {series.map((s, si) => (
          <g key={si}>
            <polyline
              points={s.points.map((v, i) => `${x(i)},${y(v)}`).join(" ")}
              fill="none"
              stroke={PIE_COLORS[si % PIE_COLORS.length]}
              strokeWidth="2"
              strokeLinejoin="round"
            />
            {s.points.map((v, i) => (
              <circle
                key={i}
                cx={x(i)}
                cy={y(v)}
                r="2.6"
                fill={PIE_COLORS[si % PIE_COLORS.length]}
                stroke="#1e2330"
                strokeWidth="1"
              />
            ))}
          </g>
        ))}
      </svg>
      {labels.length > 0 && (
        <div className="dyn-line-labels">
          {labels.slice(0, nPoints).map((l, i) => (
            <span key={i}>{l}</span>
          ))}
        </div>
      )}
      {series.length > 1 && (
        <div className="dyn-pie-legend dyn-line-legend">
          {series.map((s, i) => (
            <div key={i} className="dyn-legend-row">
              <span
                className="dyn-legend-swatch"
                style={{ background: PIE_COLORS[i % PIE_COLORS.length] }}
              />
              <span className="dyn-legend-label">{str(s.label) || `Series ${i + 1}`}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Node({ node }) {
  if (!node || typeof node !== "object") return null;
  const props = node.props || {};
  const children = arr(node.children).map((c, i) => <Node key={i} node={c} />);

  switch (node.type) {
    case "panel":
      return (
        <div className="dyn-panel">
          {(props.title || props.subtitle) && (
            <div className="dyn-panel-head">
              {props.title && <span className="dyn-title">{str(props.title)}</span>}
              {props.subtitle && (
                <span className="dyn-subtitle">{str(props.subtitle)}</span>
              )}
            </div>
          )}
          {children}
        </div>
      );
    case "card":
      return (
        <div className="dyn-card">
          {props.title && <div className="dyn-card-title">{str(props.title)}</div>}
          {children}
        </div>
      );
    case "columns":
      return <div className="dyn-columns">{children}</div>;
    case "section":
      return (
        <div className="dyn-section">
          {props.title && <div className="dyn-section-title">{str(props.title)}</div>}
          {children}
        </div>
      );
    case "heading": {
      const level = [1, 2, 3].includes(props.level) ? props.level : 2;
      return <div className={`dyn-heading dyn-h${level}`}>{str(props.text)}</div>;
    }
    case "text":
      return (
        <p className={props.muted ? "dyn-text dyn-muted" : "dyn-text"}>
          {str(props.text)}
        </p>
      );
    case "badge": {
      const tone = TONES[props.tone] || TONES.neutral;
      return (
        <span className="dyn-badge" style={{ background: tone.bg, color: tone.fg }}>
          {str(props.text)}
        </span>
      );
    }
    case "alert": {
      const tone = TONES[props.tone] || TONES.info;
      return (
        <div className="dyn-alert" style={{ background: tone.bg, color: tone.fg }}>
          {str(props.text)}
        </div>
      );
    }
    case "keyvalue":
      return (
        <div className="dyn-kv">
          {arr(props.items).map((item, i) => (
            <div key={i} className="dyn-kv-row">
              <span className="dyn-kv-label">{str(item?.label)}</span>
              <span className="dyn-kv-value">{str(item?.value)}</span>
            </div>
          ))}
        </div>
      );
    case "list": {
      const Tag = props.ordered ? "ol" : "ul";
      return (
        <Tag className="dyn-list">
          {arr(props.items).map((item, i) => (
            <li key={i}>{str(item)}</li>
          ))}
        </Tag>
      );
    }
    case "steps":
      return (
        <div className="dyn-steps">
          {arr(props.items).map((item, i) => (
            <div key={i} className="dyn-step">
              <span className="dyn-step-num">{i + 1}</span>
              <span>
                <span className="dyn-step-title">{str(item?.title)}</span>
                {item?.detail && (
                  <span className="dyn-step-detail">{str(item.detail)}</span>
                )}
              </span>
            </div>
          ))}
        </div>
      );
    case "table":
      return (
        <table className="dyn-table">
          {arr(props.columns).length > 0 && (
            <thead>
              <tr>
                {arr(props.columns).map((c, i) => (
                  <th key={i}>{str(c)}</th>
                ))}
              </tr>
            </thead>
          )}
          <tbody>
            {arr(props.rows).map((row, i) => (
              <tr key={i}>
                {arr(row).map((cell, j) => (
                  <td key={j}>{str(cell)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      );
    case "barchart":
      return <BarChart props={props} />;
    case "piechart":
      return <PieChart props={props} />;
    case "linechart":
      return <LineChart props={props} />;
    case "stat":
      return (
        <div className="dyn-stat">
          <span className="dyn-stat-label">{str(props.label)}</span>
          <span className="dyn-stat-value">{str(props.value)}</span>
          {props.hint && <span className="dyn-stat-hint">{str(props.hint)}</span>}
        </div>
      );
    case "progress": {
      const value = Number(props.value) || 0;
      const max = Number(props.max) || 100;
      const pct = Math.max(0, Math.min(100, (value / max) * 100));
      return (
        <div className="dyn-progress">
          {props.label && <span className="dyn-progress-label">{str(props.label)}</span>}
          <span className="dyn-progress-track">
            <span className="dyn-progress-fill" style={{ width: `${pct}%` }} />
          </span>
        </div>
      );
    }
    case "timeline":
      return (
        <div className="dyn-timeline">
          {arr(props.items).map((item, i) => (
            <div key={i} className="dyn-tl-item">
              <span className="dyn-tl-dot" />
              <span>
                <span className="dyn-step-title">{str(item?.title)}</span>
                {item?.detail && (
                  <span className="dyn-step-detail">{str(item.detail)}</span>
                )}
              </span>
            </div>
          ))}
        </div>
      );
    case "divider":
      return <hr className="dyn-divider" />;
    case "link":
      return str(props.url).startsWith("https://") ? (
        <a className="dyn-link" href={props.url} target="_blank" rel="noreferrer">
          {str(props.text) || props.url} ↗
        </a>
      ) : null;
    case "chips":
      return (
        <div className="dyn-chips">
          {arr(props.items).map((item, i) => (
            <button key={i} className="dyn-chip" onClick={() => sendChip(str(item))}>
              {str(item)}
            </button>
          ))}
        </div>
      );
    default:
      return null;
  }
}

export default function DynamicUI({ data }) {
  if (!data || !data.tree) return null;
  return (
    <div className="dyn-root" dir="auto">
      <Node node={data.tree} />
    </div>
  );
}
