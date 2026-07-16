// Fixed-template renderer for the backend's `show_service_panel` tool call —
// "ledger sheet" style: one continuous surface, left label rail, hairline
// dividers. The payload is assembled deterministically from the GDRFA service
// record; this component owns all presentation, the data arrives verbatim.

const WHEN_LABELS = {
  in_country: "inside UAE only",
  optional: "optional",
};

const CHANNEL_LABELS = {
  digital_web: "Website",
  smart_app: "Smart app",
  amer_center: "Amer centre",
};

// Unknown channel codes still arrive snake_cased ("customer_happiness_centre")
// — humanize them instead of leaking raw identifiers into the UI.
const channelLabel = (c) =>
  CHANNEL_LABELS[c] ||
  c.replace(/_/g, " ").replace(/^\w/, (m) => m.toUpperCase());

function feeAmount(fee) {
  if (fee.amount === null || fee.amount === undefined) {
    return <span className="svc-varies">varies</span>;
  }
  return `${fee.currency} ${fee.amount.toLocaleString("en-US")}`;
}

// Label-rail row: label cell + content cell inside the sheet grid.
const Row = ({ label, children }) => (
  <>
    <div className="svc-k">{label}</div>
    <div className="svc-vcell">{children}</div>
  </>
);

// Items joined inline with dot separators; one block per route section.
function InlineItems({ sections, render }) {
  return sections.map((section, i) => (
    <div key={i} className="svc-inline-section">
      {section.route && <span className="svc-route">{section.route} — </span>}
      {section.items.map((item, j) => (
        <span key={j}>
          {render(item)}
          {j < section.items.length - 1 && (
            <span className="svc-sep"> · </span>
          )}
        </span>
      ))}
    </div>
  ));
}

export default function ServicePanel({ data }) {
  // Args can arrive incomplete while the run is streaming — render nothing
  // until the payload has its identity.
  if (!data || !data.name) return null;

  const fees = data.fees || [];
  const docSections = data.documentSections || [];
  const condSections = data.conditionSections || [];
  const channels = (data.channels || []).map(channelLabel).join(" · ");
  const sla = (data.slaText || "").replace(/^Expected Completion Time:\s*/i, "");

  return (
    <div className="svc-panel" dir="auto">
      <div className="svc-header">
        <span className="svc-title">{data.name}</span>
        <span className="svc-crumb">
          {[data.category, data.group].filter(Boolean).join(" · ")}
        </span>
      </div>
      <div className="svc-sheet">
        {data.description && <Row label="What">{data.description}</Row>}
        {data.audience && <Row label="Who">{data.audience}</Row>}
        {(sla || channels) && (
          <Row label="Apply">
            {[channels, sla && `~${sla}`].filter(Boolean).join("  ·  ")}
          </Row>
        )}
        {fees.length > 0 && (
          <Row label="Fees">
            <div className="svc-fees">
              {fees.map((fee, i) => (
                <div key={i} className="svc-fee-row">
                  <span className="svc-fee-name">
                    {fee.name}
                    {WHEN_LABELS[fee.appliesWhen] && (
                      <span className="svc-pill">
                        {WHEN_LABELS[fee.appliesWhen]}
                      </span>
                    )}
                  </span>
                  <span className="svc-amount">{feeAmount(fee)}</span>
                </div>
              ))}
            </div>
          </Row>
        )}
        {docSections.length > 0 && (
          <Row label="Bring">
            <InlineItems
              sections={docSections}
              render={(doc) => (
                <>
                  {doc.name}
                  {!doc.mandatory && <span className="svc-pill">optional</span>}
                </>
              )}
            />
          </Row>
        )}
        {condSections.length > 0 && (
          <Row label="Eligible if">
            <InlineItems sections={condSections} render={(cond) => cond.text} />
          </Row>
        )}
      </div>
      <div className="svc-footer">
        <span>From the official GDRFA service record</span>
        {data.url && (
          <a href={data.url} target="_blank" rel="noreferrer">
            gdrfad.gov.ae ↗
          </a>
        )}
      </div>
    </div>
  );
}
