"""Deterministic service-panel builder (generative-UI v1).

Renders GDRFA service records AS-IS: the panel payload is assembled from
gdrfa_records.json by code — no LLM touches the numbers. The LLM's only
influence is indirect: the retriever resolved which service the user means.
Emitted as a synthetic `show_service_panel` tool call (same mechanism as the
router's `retrieve` call), which the frontend renders as a 2x2 card panel
(overview / fees / documents / eligibility).
"""
import json
import logging
import os
import time
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

logger = logging.getLogger("uvicorn.error")

PANEL_TOOL_NAME = "show_service_panel"
# Neo4j cosine score of the top vector hit must clear this for a panel to show;
# tune via env after watching the logged scores for real queries.
PANEL_MIN_SCORE = float(os.getenv("PANEL_MIN_SCORE", "0.60"))

_RECORDS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gdrfa_records.json"
)


def _load_records() -> dict:
    try:
        with open(_RECORDS_PATH, encoding="utf-8") as f:
            records = json.load(f)
        by_name = {r["name"]: r for r in records}
        logger.info(f"service panel: loaded {len(by_name)} records")
        return by_name
    except Exception as e:
        logger.error(f"service panel: could not load records ({e}) — panels disabled")
        return {}


RECORDS_BY_NAME = _load_records()


def _route_sections(routes: list[dict], field: str) -> list[dict]:
    """Items grouped by route, collapsed to one unlabeled section when every
    route's list is identical (e.g. 30/60-day routes sharing the same docs)."""
    if not routes:
        return []

    def items(route):
        if field == "documents":
            return [
                {"name": d.get("name", ""), "mandatory": bool(d.get("mandatory"))}
                for d in route.get("documents") or []
            ]
        return [
            {"text": c.get("text", "")}
            for c in route.get("conditions") or []
            if c.get("text")
        ]

    per_route = [(r.get("name") or "General", items(r)) for r in routes]
    identical = len({json.dumps(i, sort_keys=True) for _, i in per_route}) <= 1
    if identical:
        return [{"route": None, "items": per_route[0][1]}] if per_route[0][1] else []
    return [{"route": name, "items": i} for name, i in per_route if i]


def build_panel_payload(record: dict) -> dict:
    """Fixed-schema payload, verbatim from the record. Null fee amounts stay
    null (rendered as 'varies') — never dropped."""
    return {
        "name": record.get("name", ""),
        "category": record.get("category") or "",
        "group": record.get("group") or "",
        "description": record.get("description") or "",
        "audience": record.get("audience") or "",
        "slaText": record.get("slaText") or "",
        "channels": record.get("channels") or [],
        "url": record.get("url") or "",
        "fees": [
            {
                "name": f.get("name", ""),
                "amount": f.get("amount"),
                "currency": f.get("currency") or "AED",
                "appliesWhen": f.get("appliesWhen") or "always",
            }
            for f in record.get("fees") or []
            if f.get("name")
        ],
        "documentSections": _route_sections(record.get("routes") or [], "documents"),
        "conditionSections": _route_sections(record.get("routes") or [], "conditions"),
    }


def _turn_retrieval(state) -> dict | None:
    """This turn's retrieve artifact ({'services': [{name, score}, ...]}), or
    None if the turn had no retrieval. Scans back only to the user message so
    a previous turn's retrieval can never leak a stale panel."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return None
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "retrieve":
            artifact = getattr(msg, "artifact", None)
            return artifact if isinstance(artifact, dict) else None
    return None


def service_panel(state):
    """Post-responder node: attach the resolved service's panel when this turn
    retrieved and the top hit is confident. Pure code — no LLM call."""
    try:
        t0 = time.perf_counter()
        if state.get("mode") == "ask":
            return {}  # clarifying question turn — no panel alongside it
        artifact = _turn_retrieval(state)
        services = (artifact or {}).get("services") or []
        if not services:
            return {}
        top = services[0]
        name, score = top.get("name"), top.get("score") or 0.0
        if score < PANEL_MIN_SCORE:
            logger.info(
                f"[panel] skip: top hit {name!r} score {score:.3f} < {PANEL_MIN_SCORE}"
            )
            return {}
        record = RECORDS_BY_NAME.get(name)
        if record is None:
            logger.info(f"[panel] skip: no record for {name!r}")
            return {}
        payload = build_panel_payload(record)
        call_id = f"panel-{uuid4().hex[:8]}"
        logger.info(
            f"[timing] panel node: {time.perf_counter() - t0:.2f}s "
            f"| show {name!r} (score {score:.3f})"
        )
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": PANEL_TOOL_NAME,
                        "args": payload,
                        "id": call_id,
                        "type": "tool_call",
                    }],
                ),
                ToolMessage(content="rendered", tool_call_id=call_id, name=PANEL_TOOL_NAME),
            ]
        }
    except Exception as e:
        logger.error(f"Error in service panel node: {e}")
        return {}  # a missing panel is never worth failing the turn
