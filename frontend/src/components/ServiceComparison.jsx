// Fixed-template renderer for `show_service_comparison` — two services side
// by side, each as a vertical bar chart with ONE BAR PER FEE, on a shared
// scale so heights are comparable across the two charts. Amounts arrive
// precomputed by backend code; this template never does fee arithmetic.
// Series colors are CVD-validated for the dark surface.

const SERIES = ["#3b82f6", "#d97706"];

const fmt = (n) => `AED ${Number(n).toLocaleString("en-US")}`;

const PLOT_HEIGHT = 110;

export default function ServiceComparison({ data }) {
  const services = data?.services || [];
  if (services.length < 2) return null;

  // Shared scale: the tallest fee across BOTH services is the 100% mark.
  const maxAmount = Math.max(
    1,
    ...services.flatMap((s) => (s.fees || []).map((f) => f.amount))
  );
  const hasInCountry = services.some((s) =>
    (s.fees || []).some((f) => f.appliesWhen === "in_country")
  );

  const totalLine = (s) => {
    const bars = s.bars || [];
    if (bars.length === 0) return null;
    if (bars.length === 1) return `Total ${fmt(bars[0].amount)}`;
    return bars.map((b) => `${b.label}: ${fmt(b.amount)}`).join(" · ");
  };

  const factRows = [
    {
      label: "Processing",
      values: services.map(
        (s) =>
          (s.slaText || "").replace(/^Expected Completion Time:\s*/i, "") || "—"
      ),
      show: services.some((s) => s.slaText),
    },
    {
      label: "Documents",
      values: services.map((s) => (s.docCount ? `${s.docCount} required` : "—")),
      show: services.some((s) => s.docCount),
    },
    {
      label: "Refundable deposit",
      values: services.map((s) => (s.deposit ? fmt(s.deposit) : "—")),
      show: services.some((s) => s.deposit),
    },
  ].filter((r) => r.show);

  return (
    <div className="svc-panel svc-cmp" dir="auto">
      <div className="svc-cmp-caption">
        Fees per service — bars share one scale
      </div>
      <div className="svc-cmp-charts">
        {services.map((s, i) => (
          <div key={i} className="svc-cmp-chart">
            <div className="svc-cmp-legend">
              <span
                className="svc-cmp-swatch"
                style={{ background: SERIES[i] }}
              />
              <span className="svc-cmp-name">{s.name}</span>
            </div>
            {(s.fees || []).length === 0 ? (
              <div className="svc-cmp-note">fees vary — see record</div>
            ) : (
              <div className="svc-cmp-plot" style={{ height: PLOT_HEIGHT + 38 }}>
                {(s.fees || []).map((fee, j) => (
                  <div
                    key={j}
                    className="svc-cmp-col"
                    title={`${fee.name}: ${fmt(fee.amount)}${
                      fee.appliesWhen === "in_country"
                        ? " (inside UAE only)"
                        : fee.appliesWhen === "optional"
                        ? " (optional)"
                        : ""
                    }`}
                  >
                    <span className="svc-cmp-colvalue">
                      {Number(fee.amount).toLocaleString("en-US")}
                    </span>
                    <span
                      className="svc-cmp-vbar"
                      style={{
                        height: `${Math.max(
                          2,
                          (fee.amount / maxAmount) * PLOT_HEIGHT
                        )}px`,
                        background: SERIES[i],
                      }}
                    />
                    <span className="svc-cmp-collabel">
                      {fee.name}
                      {fee.appliesWhen === "in_country" && " †"}
                      {fee.appliesWhen === "optional" && " (opt.)"}
                    </span>
                  </div>
                ))}
              </div>
            )}
            {totalLine(s) && (
              <div className="svc-cmp-total">{totalLine(s)}</div>
            )}
            {(s.variableFees || []).length > 0 && (
              <div className="svc-cmp-note">
                + {s.variableFees.join(", ")} (varies)
              </div>
            )}
          </div>
        ))}
      </div>
      {hasInCountry && (
        <div className="svc-cmp-note">† applies only inside the UAE</div>
      )}
      {factRows.length > 0 && (
        <table className="svc-cmp-facts">
          <tbody>
            {factRows.map((row, i) => (
              <tr key={i}>
                <td className="svc-cmp-facts-label">{row.label}</td>
                {row.values.map((v, j) => (
                  <td key={j}>
                    <span
                      className="svc-cmp-swatch"
                      style={{ background: SERIES[j] }}
                    />
                    {v}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div className="svc-footer">
        <span>From official GDRFA service records</span>
        <span className="svc-cmp-links">
          {services.map(
            (s, i) =>
              s.url && (
                <a key={i} href={s.url} target="_blank" rel="noreferrer">
                  <span
                    className="svc-cmp-swatch"
                    style={{ background: SERIES[i] }}
                  />
                  record ↗
                </a>
              )
          )}
        </span>
      </div>
    </div>
  );
}
