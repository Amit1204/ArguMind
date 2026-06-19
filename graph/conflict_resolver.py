"""
Conflict Resolution Engine.

Resolves disagreements between agents using:
  1. Recency bias     — newer papers win
  2. Source authority — ranked source types
  3. Consensus score  — majority agreement
"""
from langchain_core.prompts import ChatPromptTemplate
from utils.json_parser import RobustJsonOutputParser
from agents.llm_factory import get_llm
from graph.citation_dag import CitationDAG
from utils.logger import get_logger

logger = get_logger(__name__)

SOURCE_AUTHORITY = {
    "arxiv_rct": 1.0,
    "arxiv_review": 0.9,
    "arxiv_preprint": 0.75,
    "web": 0.4,
    "unknown": 0.3,
}

RESOLVE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a scientific evidence analyst resolving conflicts between research claims.
Given conflicting claims from different papers, analyse and return a JSON with:
{{
  "winner": "claim_id of stronger claim or 'inconclusive'",
  "reasoning": "why this claim wins",
  "minority_report": "what the losing claim says",
  "confidence": 0.0-1.0
}}
Be objective. If evidence is genuinely equal, return winner: 'inconclusive'."""),
    ("human", """Conflicting claims:
{claims}

Source metadata:
{sources}

Resolve the conflict.""")
])


class ConflictResolver:

    def __init__(self):
        self.llm = get_llm()
        self._chain = RESOLVE_PROMPT | self.llm | RobustJsonOutputParser()

    def resolve(self, evidence_set: list[dict], dag: CitationDAG) -> dict:
        """
        Resolve all conflicts in the evidence set.
        Returns a conflict report with weighted claims and minority reports.
        """
        conflicts = dag.get_conflicts()
        resolutions = []
        weighted_claims = []
        minority_reports = []

        for conflict in conflicts:
            resolution = self._resolve_single(conflict, dag)
            resolutions.append(resolution)
            if resolution.get("winner") != "inconclusive":
                weighted_claims.append({
                    "claim_id": resolution["winner"],
                    "confidence": resolution.get("confidence", 0.5),
                    "reasoning": resolution.get("reasoning", ""),
                })
                minority_reports.append({
                    "claim_id": conflict["claim_id"],
                    "minority_view": resolution.get("minority_report", ""),
                })

        # Score uncontested claims at full confidence
        contested_ids = {c["claim_id"] for c in conflicts}
        for ev in evidence_set:
            if ev.get("claim_id") not in contested_ids:
                weighted_claims.append({
                    "claim_id": ev.get("claim_id", ""),
                    "confidence": ev.get("confidence", 0.7),
                    "reasoning": "Uncontested claim",
                })

        return {
            "weighted_claims": weighted_claims,
            "minority_reports": minority_reports,
            "conflict_count": len(conflicts),
            "resolution_count": len(resolutions),
        }

    def _resolve_single(self, conflict: dict, dag: CitationDAG) -> dict:
        """Resolve a single conflict using LLM + heuristics."""
        claim_node = dag.graph.nodes.get(conflict["claim_id"], {})

        # Build claims text
        claims_text = f"Claim: {claim_node.get('title', conflict['claim_id'])}\n"
        claims_text += f"  Supported by: {conflict['supporting']}\n"
        claims_text += f"  Refuted by: {conflict['refuting']}\n"

        # Build source metadata
        sources_text = ""
        for pid in conflict["supporting"] + conflict["refuting"]:
            node = dag.graph.nodes.get(pid, {})
            sources_text += (
                f"- {pid}: year={node.get('year', 'unknown')}, "
                f"confidence={node.get('confidence', 0.5):.2f}\n"
            )

        try:
            result = self._chain.invoke({
                "claims": claims_text,
                "sources": sources_text,
            })
            return result
        except Exception as e:
            logger.warning(f"LLM conflict resolution failed: {e} — using heuristic")
            return self._heuristic_resolve(conflict, dag)

    def _heuristic_resolve(self, conflict: dict, dag: CitationDAG) -> dict:
        """Fallback: pick side with newer + higher-confidence papers."""
        def side_score(paper_ids):
            total = 0.0
            for pid in paper_ids:
                node = dag.graph.nodes.get(pid, {})
                year_score = min((node.get("year", 2000) - 2000) / 25.0, 1.0)
                conf = node.get("confidence", 0.5)
                total += 0.5 * year_score + 0.5 * conf
            return total / max(len(paper_ids), 1)

        support_score = side_score(conflict["supporting"])
        refute_score = side_score(conflict["refuting"])

        if abs(support_score - refute_score) < 0.1:
            winner = "inconclusive"
        elif support_score > refute_score:
            winner = conflict["claim_id"]
        else:
            winner = conflict["refuting"][0] if conflict["refuting"] else "inconclusive"

        return {
            "winner": winner,
            "reasoning": f"Heuristic: support_score={support_score:.2f}, refute_score={refute_score:.2f}",
            "minority_report": "See refuting papers",
            "confidence": abs(support_score - refute_score),
        }
