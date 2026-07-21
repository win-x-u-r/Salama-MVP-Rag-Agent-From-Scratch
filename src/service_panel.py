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
import re
import time
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

logger = logging.getLogger("uvicorn.error")

PANEL_TOOL_NAME = "show_service_panel"
COMPARE_TOOL_NAME = "show_service_comparison"

# "fixed" = deterministic templates (panel/comparison); "dynamic" = an LLM
# composes a component tree from a catalog (see dynamic_ui.py), falling back
# to the fixed templates when composition fails.
PANEL_UI_MODE = os.getenv("PANEL_UI_MODE", "dynamic").lower()
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


# ---- comparison payload (two services, chartable) ----
# Same bucketing rules as GraphRetriever._fee_totals, kept structured:
# deposits/guarantees are refundable (never in totals), in-country fees are an
# add-on, optional fees are listed not summed, per-duration fees ("30-day ...")
# are alternatives — one total per duration.
def _fee_buckets(fees: list[dict]):
    variants: dict[str, float] = {}
    common = in_ctry = deposit = 0.0
    variable, optional = [], []
    for f in fees:
        amt = f.get("amount")
        name = f.get("name") or ""
        ftype = f.get("feeType") or ""
        when = f.get("appliesWhen")
        if amt is None:
            variable.append(name)
        elif "deposit" in ftype or "guarantee" in ftype:
            deposit += amt
        elif when == "in_country":
            in_ctry += amt
        elif when == "optional":
            optional.append(name)
        else:
            m = re.search(r"(\d+)[- ]?day", name.lower())
            if m:
                variants[m.group(1)] = variants.get(m.group(1), 0.0) + amt
            else:
                common += amt
    return variants, common, in_ctry, deposit, variable, optional


def _comparison_entry(record: dict) -> dict:
    variants, common, in_ctry, deposit, variable, optional = _fee_buckets(
        record.get("fees") or []
    )
    if variants:
        bars = [
            {"label": f"{day}-day", "amount": common + variants[day]}
            for day in sorted(variants, key=int)
        ]
    elif common:
        bars = [{"label": "Total", "amount": common}]
    else:
        bars = []
    doc_names = {
        d.get("name")
        for r in record.get("routes") or []
        for d in r.get("documents") or []
        if d.get("name")
    }
    # Raw fee rows for the per-fee bar chart: everything with an amount except
    # refundable deposits/guarantees (they'd distort the scale; shown in facts).
    fee_bars = [
        {
            "name": f.get("name", ""),
            "amount": f.get("amount"),
            "appliesWhen": f.get("appliesWhen") or "always",
        }
        for f in record.get("fees") or []
        if f.get("amount") is not None
        and "deposit" not in (f.get("feeType") or "")
        and "guarantee" not in (f.get("feeType") or "")
    ]
    return {
        "name": record.get("name", ""),
        "url": record.get("url") or "",
        "bars": bars,
        "fees": fee_bars,
        "inCountry": in_ctry,
        "deposit": deposit,
        "variableFees": variable,
        "optionalFees": optional,
        "slaText": record.get("slaText") or "",
        "docCount": len(doc_names),
    }


def build_comparison_payload(records: list[dict]) -> dict:
    """Fixed-schema comparison of two services — totals precomputed by code so
    the template (and the LLM) never do fee arithmetic."""
    return {"services": [_comparison_entry(r) for r in records]}


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


def _resolve_compare_targets(targets: list[str]) -> list[dict]:
    """One vector search per named target (embeds are cached; runs post-stream).
    Returns the distinct, confidently-resolved records — may be fewer than 2."""
    if len(targets) != 2:
        return []
    try:
        from .backend_setup import graph_retriever
    except Exception:
        return []
    if graph_retriever is None:
        return []
    resolved: list[dict] = []
    for target in targets:
        try:
            docs = graph_retriever.invoke(target)
        except Exception as e:
            logger.error(f"[panel] target resolution failed for {target!r}: {e}")
            continue
        if not docs:
            continue
        name = docs[0].metadata.get("name")
        score = docs[0].metadata.get("score") or 0.0
        rec = RECORDS_BY_NAME.get(name)
        logger.info(f"[panel] target {target!r} -> {name!r} (score {score:.3f})")
        if rec is not None and score >= PANEL_MIN_SCORE:
            if all(r["name"] != rec["name"] for r in resolved):
                resolved.append(rec)
    return resolved


def _try_dynamic(state, services: list[dict]):
    """Dynamic-schema path: hand the resolved records to the LLM composer.
    Returns an emission dict, or None to fall back to fixed templates."""
    from .dynamic_ui import compose_tree, DYNAMIC_TOOL_NAME

    # Records: resolved compare targets when comparing, else confident top hits.
    records: list[dict] = []
    if state.get("compare_services"):
        records = _resolve_compare_targets(state.get("compare_targets") or [])
    if not records:
        for s in services[:2]:
            rec = RECORDS_BY_NAME.get(s.get("name"))
            if rec is not None and (s.get("score") or 0.0) >= PANEL_MIN_SCORE:
                if all(r["name"] != rec["name"] for r in records):
                    records.append(rec)
        if not state.get("compare_services"):
            records = records[:1]
    if not records:
        return None

    question = answer = ""
    for msg in reversed(state["messages"]):
        if not answer and isinstance(msg, AIMessage) and msg.content:
            answer = str(msg.content)
        if isinstance(msg, HumanMessage):
            question = str(msg.content)
            break

    payloads = [
        {**build_panel_payload(r), "fees": _comparison_entry(r)["fees"]}
        for r in records
    ]
    tree = compose_tree(question, answer, payloads)
    if tree is None:
        return None
    return _emit(DYNAMIC_TOOL_NAME, {"tree": tree}, "dyn")


def _emit(tool_name: str, payload: dict, prefix: str) -> dict:
    call_id = f"{prefix}-{uuid4().hex[:8]}"
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": tool_name,
                    "args": payload,
                    "id": call_id,
                    "type": "tool_call",
                }],
            ),
            ToolMessage(content="rendered", tool_call_id=call_id, name=tool_name),
        ]
    }


def service_panel(state):
    """Post-responder node: attach the resolved service's panel — or, when the
    router flagged a comparative question and two distinct services resolved
    confidently, a two-service comparison chart. Pure code — no LLM call."""
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

        # Dynamic mode: an LLM composes a component tree from the resolved
        # records; any failure falls through to the fixed templates below.
        if PANEL_UI_MODE == "dynamic":
            emitted = _try_dynamic(state, services)
            if emitted is not None:
                return emitted

        # Comparison path: resolve EACH named target with its own vector search
        # (top-2 of one blended query often returns two flavors of the same
        # service instead of the two the user named). Falls back to the top-2
        # distinct retrieval hits when targets are missing or don't resolve.
        if state.get("compare_services"):
            confident = _resolve_compare_targets(state.get("compare_targets") or [])
            if len(confident) < 2:
                for s in services:
                    rec = RECORDS_BY_NAME.get(s.get("name"))
                    if rec is not None and (s.get("score") or 0.0) >= PANEL_MIN_SCORE:
                        if all(r["name"] != rec["name"] for r in confident):
                            confident.append(rec)
                    if len(confident) == 2:
                        break
            if len(confident) == 2:
                logger.info(
                    f"[timing] panel node: {time.perf_counter() - t0:.2f}s "
                    f"| compare {confident[0]['name']!r} vs {confident[1]['name']!r}"
                )
                return _emit(
                    COMPARE_TOOL_NAME, build_comparison_payload(confident), "cmp"
                )
            logger.info("[panel] compare flagged but <2 confident hits -> single panel")

        record = RECORDS_BY_NAME.get(name)
        if record is None:
            logger.info(f"[panel] skip: no record for {name!r}")
            return {}
        logger.info(
            f"[timing] panel node: {time.perf_counter() - t0:.2f}s "
            f"| show {name!r} (score {score:.3f})"
        )
        return _emit(PANEL_TOOL_NAME, build_panel_payload(record), "panel")
    except Exception as e:
        logger.error(f"Error in service panel node: {e}")
        return {}  # a missing panel is never worth failing the turn
