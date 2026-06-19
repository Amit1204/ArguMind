"""
ArguMind API routes.
"""
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=500, description="Research question")
    user_id: str = Field(default="anonymous", description="Optional user identifier")


class EvidenceItem(BaseModel):
    claim_id: str
    text: str
    confidence: float
    source: str = ""
    agent: str = ""


class ClusterItem(BaseModel):
    cluster_id: str
    topic: str
    consensus_direction: str
    agreement_score: float
    summary: str
    size: int


class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    consensus_strength: str
    evidence_count: int
    conflict_count: int
    cluster_count: int
    iteration_count: int
    key_agreements: list[str]
    key_disagreements: list[str]
    research_gaps: list[str]
    reasoning_trace: list[str]
    semantic_clusters: list[ClusterItem]
    citation_graph_summary: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, summary="Run ArguMind pipeline")
async def run_query(request: QueryRequest):
    """
    Submit a research question to the full ArguMind pipeline.
    Returns a structured response with evidence, consensus, and reasoning trace.

    Takes 30-90 seconds depending on complexity.
    """
    try:
        from agents.orchestrator import ArguMindOrchestrator
        orchestrator = ArguMindOrchestrator()

        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: orchestrator.run(request.query, user_id=request.user_id),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    # Unpack state
    from graph.citation_dag import CitationDAG
    dag = CitationDAG.from_dict(
        result.get("citation_graph", {"nodes": [], "edges": []})
    )
    conflict_report = result.get("conflict_report", {})
    consensus_report = result.get("consensus_report", {})
    semantic_clusters_raw = result.get("semantic_clusters", [])

    return QueryResponse(
        query=request.query,
        answer=result.get("final_response", ""),
        confidence=result.get("confidence", 0.0),
        consensus_strength=consensus_report.get("consensus_strength", "unknown"),
        evidence_count=len(result.get("evidence_set", [])),
        conflict_count=conflict_report.get("conflict_count", 0),
        cluster_count=len(semantic_clusters_raw),
        iteration_count=result.get("iteration_count", 1),
        key_agreements=consensus_report.get("key_agreements", []),
        key_disagreements=consensus_report.get("key_disagreements", []),
        research_gaps=consensus_report.get("research_gaps", []),
        reasoning_trace=result.get("reasoning_trace", []),
        semantic_clusters=[
            ClusterItem(
                cluster_id=c.get("cluster_id", ""),
                topic=c.get("topic", ""),
                consensus_direction=c.get("consensus_direction", "mixed"),
                agreement_score=c.get("agreement_score", 0.5),
                summary=c.get("summary", ""),
                size=c.get("size", 0),
            )
            for c in semantic_clusters_raw
        ],
        citation_graph_summary=dag.summary(),
    )


@router.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "service": "argumind"}


@router.get("/graph/{query_hash}", summary="Get raw citation graph")
async def get_graph(query_hash: str):
    """Placeholder — in production, cache results by query hash."""
    raise HTTPException(status_code=404, detail="Query not cached. Re-run /query.")
