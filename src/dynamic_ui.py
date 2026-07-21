"""Dynamic-schema generative UI (experimental, for testing/learning).

Instead of code filling a fixed template, an LLM composes a UI *tree* from a
catalog of allowed components; the frontend renders the tree recursively.
The tradeoffs vs the fixed pipeline (service_panel.py):
  + the agent can invent layouts per question (comparison grids, timelines,
    stat rows) without new frontend templates
  - an extra LLM call per turn (post-stream), and the tree can be wrong in
    ways a fixed template never is — validation only bounds the damage.
Data honesty rule stays: the composer receives the structured records and is
instructed to copy values verbatim; the validator can't check that, so this
mode is strictly weaker on false-precision defenses than the fixed one.

Toggle via PANEL_UI_MODE=dynamic|fixed (service_panel.py reads it).
"""
import json
import logging
import re
import time

logger = logging.getLogger("uvicorn.error")

DYNAMIC_TOOL_NAME = "show_dynamic_ui"

MAX_NODES = 80
MAX_DEPTH = 8

# ---- the component catalog ----
# type -> (allowed prop keys, takes children?)
COMPONENTS: dict[str, tuple[set, bool]] = {
    "panel":    ({"title", "subtitle"}, True),   # top-level surface
    "card":     ({"title"}, True),               # bordered sub-surface
    "columns":  (set(), True),                   # children side by side
    "section":  ({"title"}, True),               # titled block, no border
    "heading":  ({"text", "level"}, False),      # level 1..3
    "text":     ({"text", "muted"}, False),
    "badge":    ({"text", "tone"}, False),       # tone: info|success|warning|danger|neutral
    "alert":    ({"text", "tone"}, False),
    "keyvalue": ({"items"}, False),              # items: [{label, value}]
    "list":     ({"items", "ordered"}, False),   # items: [string]
    "steps":    ({"items"}, False),              # items: [{title, detail?}]
    "table":    ({"columns", "rows"}, False),    # columns: [str], rows: [[str]]
    "barchart": ({"items", "unit", "color"}, False),  # items: [{label, value}]
    "piechart": ({"items", "unit", "donut"}, False),  # items: [{label, value}]
    "linechart": ({"labels", "series", "unit"}, False),  # series: [{label?, points: [num]}]
    "stat":     ({"label", "value", "hint"}, False),
    "progress": ({"label", "value", "max"}, False),
    "timeline": ({"items"}, False),              # items: [{title, detail?}]
    "divider":  (set(), False),
    "link":     ({"text", "url"}, False),        # https only
    "chips":    ({"items"}, False),              # tappable follow-up questions
}

CATALOG_DOC = """Available components (type -> props):
- panel {title?, subtitle?} + children: the outer surface. Use exactly one as root.
- card {title?} + children: bordered grouping inside a panel/columns.
- columns {} + children: lays its children out side by side (2-3 children).
- section {title} + children: titled block without a border.
- heading {text, level: 1|2|3}
- text {text, muted?: true}
- badge {text, tone: "info"|"success"|"warning"|"danger"|"neutral"}: small pill.
- alert {text, tone}: highlighted callout line.
- keyvalue {items: [{label, value}]}: aligned label/value rows.
- list {items: [string], ordered?: true}
- steps {items: [{title, detail?}]}: numbered process steps.
- table {columns: [string], rows: [[string]]}
- barchart {items: [{label, value}], unit?: "AED", color?: "blue"|"amber"|"teal"|"green"}:
  vertical bars, value labels shown; use numbers only (no strings) for value.
  Use for comparing MAGNITUDES (which is bigger).
- piechart {items: [{label, value}], unit?, donut?: true}: 2-6 slices, numbers
  only. Use ONLY for part-of-whole shares (how a TOTAL breaks down, e.g. what
  makes up the full cost). Never for comparing separate services.
- linechart {labels: [string], series: [{label?, points: [numbers]}], unit?}:
  max 3 series. Use ONLY when data is ordered along an axis (durations,
  years); this knowledge base rarely has such data — prefer barchart.
- stat {label, value, hint?}: one big number tile.
- progress {label?, value, max}
- timeline {items: [{title, detail?}]}
- divider {}
- link {text, url}: https URLs only.
- chips {items: [string]}: tappable follow-up questions the user might ask next.
"""

COMPOSER_PROMPT = (
    "You are a UI composer for a GDRFA Dubai visa assistant. Given the user's "
    "question, the assistant's answer, and STRUCTURED SERVICE RECORDS, design "
    "a small UI that presents the most decision-relevant data visually.\n\n"
    + CATALOG_DOC
    + "\nOutput format: ONLY a JSON object, no prose, shaped as\n"
    '{"root": {"type": "panel", "props": {...}, "children": [{"type": ..., '
    '"props": {...}, "children": [...]}, ...]}}\n\n'
    "Hard rules:\n"
    "- Every number, fee, document name, and URL must be copied VERBATIM from "
    "the records. Never invent, round, or total numbers yourself unless the "
    "record provides the total.\n"
    "- Amounts with null values are 'varies' — show them as text/badges, "
    "never as chart values.\n"
    "- Prefer compact: one panel, 2-6 components. A comparison question wants "
    "columns with one card per service; a single-service question wants the "
    "key facts + one chart or table, not everything.\n"
    "- Refundable deposits/guarantees must not be mixed into fee charts; show "
    "them separately (stat or keyvalue) and label them refundable.\n"
    "- Include one 'link' to each record's official url.\n"
    "- Optionally end with 'chips' suggesting 2-3 natural follow-up questions.\n"
    "- If the records don't contain data worth visualizing, output "
    '{"root": null}.'
)


_COLOR_ALLOWED = {"blue", "amber", "teal", "green"}
_TONE_ALLOWED = {"info", "success", "warning", "danger", "neutral"}


def _validate(node, depth: int, budget: list) -> dict | None:
    """Whitelist filter: unknown types/props are dropped, depth and node count
    bounded, link URLs restricted to https. Returns a clean node or None."""
    if not isinstance(node, dict) or depth > MAX_DEPTH or budget[0] <= 0:
        return None
    ntype = node.get("type")
    if ntype not in COMPONENTS:
        return None
    budget[0] -= 1
    allowed_props, takes_children = COMPONENTS[ntype]
    props_in = node.get("props") or {}
    props = {k: v for k, v in props_in.items() if k in allowed_props} if isinstance(props_in, dict) else {}
    if ntype == "link":
        url = str(props.get("url", ""))
        if not url.startswith("https://"):
            return None
    if ntype == "barchart":
        items = props.get("items")
        if not isinstance(items, list):
            return None
        clean = [
            {"label": str(i.get("label", "")), "value": i["value"]}
            for i in items
            if isinstance(i, dict) and isinstance(i.get("value"), (int, float))
        ]
        if not clean:
            return None
        props["items"] = clean[:12]
        if props.get("color") not in _COLOR_ALLOWED:
            props.pop("color", None)
    if ntype == "piechart":
        items = props.get("items")
        if not isinstance(items, list):
            return None
        clean = [
            {"label": str(i.get("label", "")), "value": i["value"]}
            for i in items
            if isinstance(i, dict)
            and isinstance(i.get("value"), (int, float))
            and i["value"] > 0
        ]
        if len(clean) < 2:
            return None
        props["items"] = clean[:6]
    if ntype == "linechart":
        series = props.get("series")
        if not isinstance(series, list):
            return None
        clean_series = []
        for s in series[:3]:
            if not isinstance(s, dict):
                continue
            points = [p for p in (s.get("points") or []) if isinstance(p, (int, float))]
            if len(points) >= 2:
                clean_series.append({"label": str(s.get("label", "")), "points": points[:24]})
        if not clean_series:
            return None
        props["series"] = clean_series
        labels = props.get("labels")
        props["labels"] = [str(x) for x in labels[:24]] if isinstance(labels, list) else []
    if ntype in ("badge", "alert") and props.get("tone") not in _TONE_ALLOWED:
        props.pop("tone", None)
    out = {"type": ntype, "props": props}
    if takes_children:
        children = node.get("children")
        if isinstance(children, list):
            cleaned = [
                c for c in (_validate(ch, depth + 1, budget) for ch in children)
                if c is not None
            ]
            out["children"] = cleaned
        else:
            out["children"] = []
    return out


def _parse_tree(raw: str) -> dict | None:
    text = raw.strip()
    # tolerate ```json fences and leading prose
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        if start > 0:
            text = text[start:]
    try:
        data = json.loads(text)
    except Exception:
        return None
    root = data.get("root") if isinstance(data, dict) else None
    if root is None:
        return None
    return _validate(root, 0, [MAX_NODES])


def compose_tree(question: str, answer: str, records: list[dict]) -> dict | None:
    """One LLM call -> validated component tree, or None (caller falls back)."""
    from langchain_core.messages import SystemMessage, HumanMessage
    from .backend_setup import llm
    from .system_prompt_setup import NO_THINK

    t0 = time.perf_counter()
    view = HumanMessage(
        content=(
            f"USER QUESTION: {question}\n\n"
            f"ASSISTANT ANSWER (already shown; the UI supplements it):\n{answer[:800]}\n\n"
            f"STRUCTURED SERVICE RECORDS:\n{json.dumps(records, ensure_ascii=False)}"
        )
    )
    try:
        response = llm.invoke(
            [SystemMessage(content=COMPOSER_PROMPT + NO_THINK), view]
        )
        tree = _parse_tree(str(response.content))
        logger.info(
            f"[timing] ui composer: {time.perf_counter() - t0:.2f}s | "
            + ("tree ok" if tree else "no tree (parse/validate failed or root null)")
        )
        return tree
    except Exception as e:
        logger.error(f"Error in ui composer: {e}")
        return None
