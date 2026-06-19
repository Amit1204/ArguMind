"""
Semantic Agent — clusters evidence by semantic similarity.

Uses TF-IDF cosine similarity (pure Python + stdlib, no sklearn needed).
Optional: sentence-transformers for better embeddings if installed.

Responsibilities:
  1. Cluster claims from evidence_set by topic similarity
  2. Label each cluster with topic + consensus direction (via LLM)
  3. Enrich the citation DAG with 'extends' edges between semantically
     related claims from different papers
"""
import math
import re
from collections import Counter

from langchain_core.prompts import ChatPromptTemplate
from utils.json_parser import RobustJsonOutputParser
from utils.retry import invoke_with_retry
from agents.llm_factory import get_llm
from graph.citation_dag import CitationDAG
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

CLUSTER_LABEL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research taxonomy expert.
Given a cluster of related research claims, label it and assess consensus.
Return JSON only:
{{
  "topic": "short topic label (3-6 words)",
  "consensus_direction": "positive|negative|neutral|mixed",
  "agreement_score": 0.0,
  "summary": "one sentence summarizing what these claims collectively say"
}}"""),
    ("human", "Research question: {query}\n\nClaims in this cluster:\n{claims}"),
])


# ── TF-IDF cosine similarity (no external deps) ───────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase tokenize, 3+ char alphabetic tokens."""
    return re.findall(r'\b[a-z]{3,}\b', text.lower())


def _tfidf_vectors(docs: list[str]) -> list[dict]:
    """Return L2-normalized TF-IDF dicts for each document."""
    N = len(docs)
    tokenized = [_tokenize(d) for d in docs]

    # Document frequency
    df: Counter = Counter()
    for tokens in tokenized:
        for t in set(tokens):
            df[t] += 1

    # IDF with smoothing
    idf = {t: math.log((N + 1) / (cnt + 1)) + 1 for t, cnt in df.items()}

    vectors = []
    for tokens in tokenized:
        tf = Counter(tokens)
        n_tokens = max(len(tokens), 1)
        vec = {t: (cnt / n_tokens) * idf.get(t, 1.0) for t, cnt in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append({t: v / norm for t, v in vec.items()})

    return vectors


def _cosine(v1: dict, v2: dict) -> float:
    return sum(v1[t] * v2[t] for t in set(v1) & set(v2))


def _greedy_cluster(claims: list[dict], threshold: float = 0.25) -> list[list[int]]:
    """
    Greedy single-linkage clustering by cosine similarity.
    Returns list of clusters (each a list of claim indices).
    """
    if not claims:
        return []

    texts = [c.get("text", "") or c.get("claim_text", "") for c in claims]
    vecs = _tfidf_vectors(texts)
    n = len(claims)
    assigned = [-1] * n
    clusters: list[list[int]] = []

    for i in range(n):
        if assigned[i] >= 0:
            continue
        cid = len(clusters)
        clusters.append([i])
        assigned[i] = cid
        for j in range(i + 1, n):
            if assigned[j] >= 0:
                continue
            if _cosine(vecs[i], vecs[j]) >= threshold:
                clusters[cid].append(j)
                assigned[j] = cid

    return clusters


# ── Agent ─────────────────────────────────────────────────────────────────────

class SemanticAgent:
    name = "semantic_agent"

    def __init__(self):
        self.llm = get_llm()
        self._label_chain = CLUSTER_LABEL_PROMPT | self.llm | RobustJsonOutputParser()

    def run(self, state: dict) -> dict:
        """
        Cluster evidence semantically, label clusters, enrich citation DAG.

        Returns dict with:
          semantic_clusters: list of cluster records
          citation_graph: updated DAG (new 'extends' edges)
          reasoning_trace: step description
        """
        evidence_set = state.get("evidence_set", [])
        citation_graph = state.get("citation_graph", {"nodes": [], "edges": []})
        query = state.get("query", "")

        if len(evidence_set) < 2:
            logger.info("SemanticAgent: too few claims to cluster")
            return {
                "semantic_clusters": [],
                "reasoning_trace": ["SemanticAgent: < 2 claims, skipping clustering"],
            }

        logger.info(f"SemanticAgent: clustering {len(evidence_set)} claims")

        raw_clusters = _greedy_cluster(evidence_set, threshold=0.25)
        dag = CitationDAG.from_dict(citation_graph)
        semantic_clusters = []

        for raw in raw_clusters:
            if len(raw) < 2:
                continue  # singletons carry no clustering signal

            cluster_claims = [evidence_set[i] for i in raw]
            claims_text = "\n".join(
                f"- {c.get('text', '')[:160]}" for c in cluster_claims
            )

            # Label cluster with LLM
            try:
                label = invoke_with_retry(self._label_chain, {
                    "query": query,
                    "claims": claims_text,
                })
            except Exception as e:
                logger.warning(f"Cluster label LLM failed: {e}")
                label = {
                    "topic": f"topic_group_{len(semantic_clusters)}",
                    "consensus_direction": "mixed",
                    "agreement_score": 0.5,
                    "summary": f"Group of {len(cluster_claims)} semantically related claims.",
                }

            # Enrich DAG: add 'extends' edges between claims in different papers
            claim_ids = [
                c.get("claim_id", "") for c in cluster_claims if c.get("claim_id")
            ]
            for i, cid_a in enumerate(claim_ids):
                for cid_b in claim_ids[i + 1:]:
                    paper_a = next(
                        (c.get("paper_id", "") for c in cluster_claims if c.get("claim_id") == cid_a), ""
                    )
                    paper_b = next(
                        (c.get("paper_id", "") for c in cluster_claims if c.get("claim_id") == cid_b), ""
                    )
                    # Only link cross-paper claims that exist in DAG
                    if (
                        paper_a != paper_b
                        and dag.graph.has_node(cid_a)
                        and dag.graph.has_node(cid_b)
                        and not dag.graph.has_edge(cid_a, cid_b)
                    ):
                        dag.link(
                            source_id=cid_a,
                            target_id=cid_b,
                            edge_type="extends",
                            explanation=f"Semantically related [{label.get('topic', '')}]",
                            weight=float(label.get("agreement_score", 0.5)),
                        )

            cluster_record = {
                "cluster_id": f"cluster_{len(semantic_clusters)}",
                "topic": label.get("topic", ""),
                "consensus_direction": label.get("consensus_direction", "mixed"),
                "agreement_score": float(label.get("agreement_score", 0.5)),
                "summary": label.get("summary", ""),
                "claim_ids": claim_ids,
                "size": len(cluster_claims),
            }
            semantic_clusters.append(cluster_record)

        logger.info(f"SemanticAgent: {len(semantic_clusters)} multi-claim clusters formed")

        return {
            "semantic_clusters": semantic_clusters,
            "citation_graph": dag.to_dict(),
            "reasoning_trace": [
                f"SemanticAgent: {len(semantic_clusters)} clusters, "
                f"added extends edges to DAG"
            ],
        }
