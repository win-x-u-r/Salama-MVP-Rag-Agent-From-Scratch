"""
Ingest gdrfa_records.json into Neo4j as a knowledge graph.

Graph schema
------------
(:Service {guid, name, url, description, audience, slaText, notes,
           permitType, action, sponsorType, verified, searchText, embedding})
(:Category {name})            (:Group {name})
(:Route {key, name})          (:Document {name})
(:Condition {key, text, threshold})
(:Fee {key, name, amount, currency, feeType, appliesWhen})
(:Channel {name})             (:GovEntity {name})

(Service)-[:IN_CATEGORY]->(Category)
(Service)-[:IN_GROUP]->(Group)
(Service)-[:HAS_ROUTE]->(Route)
(Route)-[:REQUIRES_DOCUMENT {mandatory}]->(Document)   # Document deduped globally
(Route)-[:HAS_CONDITION]->(Condition)
(Service)-[:HAS_FEE]->(Fee)
(Service)-[:AVAILABLE_VIA]->(Channel)
(Service)-[:INVOLVES {aspect}]->(GovEntity)
(Service)-[:RELATED_TO {relation}]->(Service)          # matched by name

A vector index `service_embedding` (3072 dims, cosine) is created over
Service.embedding so retrieval can find entry-point services semantically,
then expand through the graph.

Usage:
    python graph_ingest.py            # full ingest (wipes existing graph data)
    python graph_ingest.py --dry-run  # parse + build statements, no DB writes
"""

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("graph_ingest")

JSON_PATH = os.path.join(os.path.dirname(__file__), "gdrfa_records.json")
EMBEDDING_DIM = 3072
VECTOR_INDEX = "service_embedding"


# ----------------------------------------------------------------------
# Build the text that gets embedded for each service.
# Richer than raw JSON: a readable summary of everything reachable from
# the node, so vector search has good recall.
# ----------------------------------------------------------------------
def build_search_text(rec: dict) -> str:
    parts = [
        f"Service: {rec['name']}",
        f"Category: {rec['category']} | Group: {rec['group']}",
        f"Action: {rec['action']} | Permit type: {rec['permitType']}",
        f"Description: {rec['description']}",
        f"Audience: {rec['audience']}",
        f"Sponsor type: {rec['sponsorType']}",
    ]
    for route in rec.get("routes") or []:
        docs = ", ".join(d["name"] for d in route.get("documents") or [])
        conds = "; ".join(c["text"] for c in route.get("conditions") or [])
        parts.append(f"Route '{route['name']}': documents required: {docs}. Conditions: {conds}")
    fees = "; ".join(
        f"{f['name']}: "
        + (f"{f['amount']} {f['currency']}" if f["amount"] is not None else "variable amount")
        + f" ({f['feeType']}, {f['appliesWhen']})"
        for f in rec.get("fees") or []
    )
    if fees:
        parts.append(f"Fees: {fees}")
    if rec.get("slaText"):
        parts.append(rec["slaText"])
    if rec.get("notes"):
        parts.append(f"Notes: {rec['notes']}")
    return "\n".join(parts)


# ----------------------------------------------------------------------
# Transform records into parameter dicts for batched Cypher UNWIND writes
# ----------------------------------------------------------------------
def prepare(records: list[dict]) -> dict:
    services, routes, fees, related = [], [], [], []
    for rec in records:
        guid = rec["guid"]
        services.append(
            {
                "guid": guid,
                "name": rec["name"],
                "url": rec["url"],
                "category": rec["category"],
                "group": rec["group"],
                "description": rec["description"],
                "action": rec["action"],
                "permitType": rec["permitType"],
                "audience": rec["audience"],
                "sponsorType": str(rec["sponsorType"]),
                "slaText": rec.get("slaText") or "",
                "notes": rec.get("notes") or "",
                "verified": bool(rec.get("verified")),
                "channels": rec.get("channels") or [],
                "govEntities": rec.get("govEntities") or [],
                "searchText": build_search_text(rec),
            }
        )
        for route in rec.get("routes") or []:
            routes.append(
                {
                    "serviceGuid": guid,
                    "key": f"{guid}::{route['name']}",
                    "name": route["name"],
                    "documents": [
                        {"name": d["name"], "mandatory": bool(d.get("mandatory"))}
                        for d in route.get("documents") or []
                    ],
                    "conditions": [
                        {
                            "key": f"{guid}::{route['name']}::{c['text']}",
                            "text": c["text"],
                            "threshold": c.get("threshold"),
                        }
                        for c in route.get("conditions") or []
                    ],
                }
            )
        for i, fee in enumerate(rec.get("fees") or []):
            fees.append(
                {
                    "serviceGuid": guid,
                    "key": f"{guid}::fee::{i}",
                    "name": fee["name"],
                    "amount": fee["amount"],
                    "currency": fee["currency"],
                    "feeType": fee["feeType"],
                    "appliesWhen": fee["appliesWhen"],
                }
            )
        for rel in rec.get("relatedServices") or []:
            related.append(
                {
                    "serviceGuid": guid,
                    "targetName": rel["name"],
                    "relation": rel.get("relation") or "related",
                }
            )
    return {"services": services, "routes": routes, "fees": fees, "related": related}


CONSTRAINTS = [
    "CREATE CONSTRAINT service_guid IF NOT EXISTS FOR (s:Service) REQUIRE s.guid IS UNIQUE",
    "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT group_name IF NOT EXISTS FOR (g:Group) REQUIRE g.name IS UNIQUE",
    "CREATE CONSTRAINT route_key IF NOT EXISTS FOR (r:Route) REQUIRE r.key IS UNIQUE",
    "CREATE CONSTRAINT document_name IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT condition_key IF NOT EXISTS FOR (c:Condition) REQUIRE c.key IS UNIQUE",
    "CREATE CONSTRAINT fee_key IF NOT EXISTS FOR (f:Fee) REQUIRE f.key IS UNIQUE",
    "CREATE CONSTRAINT channel_name IF NOT EXISTS FOR (c:Channel) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT gov_name IF NOT EXISTS FOR (g:GovEntity) REQUIRE g.name IS UNIQUE",
]

INGEST_SERVICES = """
UNWIND $services AS s
MERGE (svc:Service {guid: s.guid})
SET svc.name = s.name, svc.url = s.url, svc.description = s.description,
    svc.action = s.action, svc.permitType = s.permitType,
    svc.audience = s.audience, svc.sponsorType = s.sponsorType,
    svc.slaText = s.slaText, svc.notes = s.notes, svc.verified = s.verified,
    svc.searchText = s.searchText
MERGE (cat:Category {name: s.category})
MERGE (svc)-[:IN_CATEGORY]->(cat)
MERGE (grp:Group {name: s.group})
MERGE (svc)-[:IN_GROUP]->(grp)
WITH svc, s
FOREACH (ch IN s.channels |
    MERGE (c:Channel {name: ch})
    MERGE (svc)-[:AVAILABLE_VIA]->(c))
FOREACH (ge IN s.govEntities |
    MERGE (g:GovEntity {name: ge.name})
    MERGE (svc)-[r:INVOLVES]->(g)
    SET r.aspect = ge.aspect)
"""

INGEST_ROUTES = """
UNWIND $routes AS r
MATCH (svc:Service {guid: r.serviceGuid})
MERGE (route:Route {key: r.key})
SET route.name = r.name
MERGE (svc)-[:HAS_ROUTE]->(route)
WITH route, r
FOREACH (d IN r.documents |
    MERGE (doc:Document {name: d.name})
    MERGE (route)-[rd:REQUIRES_DOCUMENT]->(doc)
    SET rd.mandatory = d.mandatory)
FOREACH (c IN r.conditions |
    MERGE (cond:Condition {key: c.key})
    SET cond.text = c.text, cond.threshold = c.threshold
    MERGE (route)-[:HAS_CONDITION]->(cond))
"""

INGEST_FEES = """
UNWIND $fees AS f
MATCH (svc:Service {guid: f.serviceGuid})
MERGE (fee:Fee {key: f.key})
SET fee.name = f.name, fee.amount = f.amount, fee.currency = f.currency,
    fee.feeType = f.feeType, fee.appliesWhen = f.appliesWhen
MERGE (svc)-[:HAS_FEE]->(fee)
"""

# RELATED_TO only when the target service actually exists (matched by name)
INGEST_RELATED = """
UNWIND $related AS rel
MATCH (src:Service {guid: rel.serviceGuid})
MATCH (dst:Service {name: rel.targetName})
MERGE (src)-[r:RELATED_TO]->(dst)
SET r.relation = rel.relation
"""

SET_EMBEDDINGS = """
UNWIND $rows AS row
MATCH (svc:Service {guid: row.guid})
CALL db.create.setNodeVectorProperty(svc, 'embedding', row.embedding)
"""

CREATE_VECTOR_INDEX = f"""
CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS
FOR (s:Service) ON (s.embedding)
OPTIONS {{indexConfig: {{
    `vector.dimensions`: {EMBEDDING_DIM},
    `vector.similarity_function`: 'cosine'
}}}}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="parse only, no DB writes")
    parser.add_argument("--skip-embeddings", action="store_true", help="ingest graph without embeddings")
    args = parser.parse_args()

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info("Loaded %d records", len(records))

    params = prepare(records)
    logger.info(
        "Prepared %d services, %d routes, %d fees, %d related links",
        len(params["services"]), len(params["routes"]),
        len(params["fees"]), len(params["related"]),
    )

    if args.dry_run:
        logger.info("Dry run complete. Sample searchText:\n%s",
                    params["services"][0]["searchText"][:500])
        return

    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=auth)
    driver.verify_connectivity()
    logger.info("Connected to Neo4j at %s", uri)

    with driver.session(database=database) as session:
        # wipe previous graph data so re-ingest is idempotent
        session.run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared existing graph")

        for stmt in CONSTRAINTS:
            session.run(stmt)
        logger.info("Constraints ensured")

        session.run(INGEST_SERVICES, services=params["services"])
        session.run(INGEST_ROUTES, routes=params["routes"])
        session.run(INGEST_FEES, fees=params["fees"])
        session.run(INGEST_RELATED, related=params["related"])
        logger.info("Graph nodes and relationships created")

        counts = session.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY c DESC"
        ).data()
        for row in counts:
            logger.info("  %-12s %d", row["label"], row["c"])

        if not args.skip_embeddings:
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(
                model="openai/text-embedding-3-large",
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            )
            texts = [s["searchText"] for s in params["services"]]
            logger.info("Embedding %d service texts...", len(texts))
            vectors = embeddings.embed_documents(texts)
            if len(vectors[0]) != EMBEDDING_DIM:
                logger.error("Embedding dim %d != expected %d", len(vectors[0]), EMBEDDING_DIM)
                sys.exit(1)
            rows = [
                {"guid": s["guid"], "embedding": v}
                for s, v in zip(params["services"], vectors)
            ]
            session.run(SET_EMBEDDINGS, rows=rows)
            session.run(CREATE_VECTOR_INDEX)
            logger.info("Embeddings stored and vector index '%s' created", VECTOR_INDEX)

    driver.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
