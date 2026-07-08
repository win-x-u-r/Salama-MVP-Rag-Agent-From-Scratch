"""
Hybrid graph retrieval over the Neo4j GDRFA knowledge graph.

How it works:
1. The query is embedded (same model used at ingest time).
2. Neo4j's vector index finds the top-k most similar Service nodes (entry points).
3. One Cypher query expands each entry point through the graph — routes,
   documents, conditions, fees, channels, gov entities, related services —
   and the result is formatted into a readable context block for the LLM.

This gives vector-search recall plus graph-structured, complete answers:
a service hit always comes back with ALL of its fees and conditions, not
just whichever chunk happened to match.
"""
import logging
import os

from neo4j import GraphDatabase
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import PrivateAttr
from src.config import RETRIEVAL_K


logger = logging.getLogger("uvicorn.error")

VECTOR_INDEX = "service_embedding"

# Vector search + full subgraph expansion in a single query.
RETRIEVAL_QUERY = """
CALL db.index.vector.queryNodes($index, $k, $embedding)
YIELD node AS svc, score
OPTIONAL MATCH (svc)-[:IN_CATEGORY]->(cat:Category)
OPTIONAL MATCH (svc)-[:IN_GROUP]->(grp:Group)
OPTIONAL MATCH (svc)-[:AVAILABLE_VIA]->(ch:Channel)
OPTIONAL MATCH (svc)-[inv:INVOLVES]->(gov:GovEntity)
OPTIONAL MATCH (svc)-[:HAS_FEE]->(fee:Fee)
OPTIONAL MATCH (svc)-[:HAS_ROUTE]->(route:Route)
OPTIONAL MATCH (route)-[rd:REQUIRES_DOCUMENT]->(doc:Document)
OPTIONAL MATCH (route)-[:HAS_CONDITION]->(cond:Condition)
OPTIONAL MATCH (svc)-[rel:RELATED_TO]->(related:Service)
WITH svc, score, cat, grp,
     collect(DISTINCT ch.name) AS channels,
     collect(DISTINCT {name: gov.name, aspect: inv.aspect}) AS govEntities,
     collect(DISTINCT {name: fee.name, amount: fee.amount, currency: fee.currency,
                       feeType: fee.feeType, appliesWhen: fee.appliesWhen}) AS fees,
     collect(DISTINCT {route: route.name, doc: doc.name, mandatory: rd.mandatory}) AS routeDocs,
     collect(DISTINCT {route: route.name, text: cond.text, threshold: cond.threshold}) AS routeConds,
     collect(DISTINCT {name: related.name, relation: rel.relation}) AS relatedServices
RETURN svc.name AS name, svc.url AS url, svc.description AS description,
       svc.audience AS audience, svc.sponsorType AS sponsorType,
       svc.action AS action, svc.permitType AS permitType,
       svc.slaText AS slaText, svc.notes AS notes,
       cat.name AS category, grp.name AS group,
       channels, govEntities, fees, routeDocs, routeConds, relatedServices,
       score
ORDER BY score DESC
"""


# class GraphRetriever:
#     def __init__(self, embeddings):
#         self.embeddings = embeddings
#         uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
#         auth = (os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))
#         self.database = os.getenv("NEO4J_DATABASE", "neo4j")
#         self.driver = GraphDatabase.driver(uri, auth=auth)
#         self.driver.verify_connectivity()
#         logger.info(f"GraphRetriever connected to Neo4j at {uri}")

#     def close(self):
#         self.driver.close()

#     def retrieve(self, query: str, k: int = 3) -> str:
#         query_vector = self.embeddings.embed_query(query)
#         with self.driver.session(database=self.database) as session:
#             rows = session.run(
#                 RETRIEVAL_QUERY, index=VECTOR_INDEX, k=k, embedding=query_vector
#             ).data()
#         if not rows:
#             return "No matching services found in the knowledge graph."
#         return "\n\n---\n\n".join(self._format(row) for row in rows)

class GraphRetriever(BaseRetriever):
    """Neo4j hybrid retriever that returns one Document per matched service."""

    k: int = RETRIEVAL_K

    _embeddings: object = PrivateAttr()
    _driver: object = PrivateAttr()
    _database: str = PrivateAttr()

    def __init__(self, embeddings, k: int=RETRIEVAL_K, **kwargs):
        super().__init__(k=k, **kwargs)
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        auth = (os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))
        self._database = os.getenv("NEO4J_DATABASE", "neo4j")
        self._embeddings = embeddings
        self._driver = GraphDatabase.driver(uri, auth=auth)
        self._driver.verify_connectivity()
        logger.info(f"GraphRetriever connected to Neo4j at {uri}")

    def close(self):
        self._driver.close()

    # NEW: the LangChain retriever interface — returns Documents (one per service block)
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        query_vector = self._embeddings.embed_query(query)
        with self._driver.session(database=self._database) as session:
            rows = session.run(
                RETRIEVAL_QUERY, index=VECTOR_INDEX, k=self.k, embedding=query_vector
            ).data()
        return [
            Document(
                page_content=self._format(row),
                metadata={"name": row["name"], "score": row["score"]},
            )
            for row in rows
        ]

    # BACKWARD-COMPAT: keeps your current tool (graph_retriever.retrieve(query, k=...)) working.
    # Remove this once the tool switches to the reranker/compression path.
    def retrieve(self, query: str, k: int | None = None) -> str:
        if k is not None:
            self.k = k
        docs = self.invoke(query)
        if not docs:
            return "No matching services found in the knowledge graph."
        return "\n\n---\n\n".join(d.page_content for d in docs)


    @staticmethod
    def _format(row: dict) -> str:
        lines = [
            f"SERVICE: {row['name']}  (relevance: {row['score']:.3f})",
            f"Category: {row['category']} | Group: {row['group']}",
            f"Action: {row['action']} | Permit type: {row['permitType']}",
            f"Description: {row['description']}",
            f"Audience: {row['audience']}",
            f"Sponsor type: {row['sponsorType']}",
        ]

        # routes -> documents
        routes: dict[str, dict] = {}
        for rd in row["routeDocs"]:
            if rd["route"] is None:
                continue
            routes.setdefault(rd["route"], {"docs": [], "conds": []})
            if rd["doc"]:
                flag = "mandatory" if rd["mandatory"] else "optional"
                routes[rd["route"]]["docs"].append(f"{rd['doc']} ({flag})")
        for rc in row["routeConds"]:
            if rc["route"] is None:
                continue
            routes.setdefault(rc["route"], {"docs": [], "conds": []})
            if rc["text"]:
                cond = rc["text"]
                if rc["threshold"] is not None:
                    cond += f" [threshold: {rc['threshold']}]"
                routes[rc["route"]]["conds"].append(cond)
        for rname, data in routes.items():
            lines.append(f"ROUTE: {rname}")
            if data["docs"]:
                lines.append("  Documents: " + "; ".join(data["docs"]))
            if data["conds"]:
                lines.append("  Conditions: " + "; ".join(data["conds"]))

        fees = [f for f in row["fees"] if f["name"]]
        if fees:
            lines.append("FEES:")
            for f in fees:
                amount = (
                    f"{f['amount']} {f['currency']}"
                    if f["amount"] is not None
                    else "variable amount (e.g. percentage-based)"
                )
                lines.append(
                    f"  - {f['name']}: {amount} ({f['feeType']}, applies: {f['appliesWhen']})"
                )

        if row["channels"]:
            lines.append("Channels: " + ", ".join(row["channels"]))
        govs = [g for g in row["govEntities"] if g["name"]]
        if govs:
            lines.append(
                "Government entities: "
                + ", ".join(f"{g['name']} ({g['aspect']})" for g in govs)
            )
        if row["slaText"]:
            lines.append(row["slaText"])
        related = [r for r in row["relatedServices"] if r["name"]]
        if related:
            lines.append(
                "Related services: "
                + "; ".join(f"{r['name']} ({r['relation']})" for r in related)
            )
        if row["notes"]:
            lines.append(f"Notes: {row['notes']}")
        lines.append(f"URL: {row['url']}")
        return "\n".join(lines)
# end of module
