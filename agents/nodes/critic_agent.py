"""
Critic Agent — global quality gatekeeper.
Checks: citation validity, logical consistency, evidence sufficiency.
Triggers back-edge to planner if evidence is insufficient.
"""
from langchain_core.prompts import ChatPromptTemplate
from utils.json_parser import RobustJsonOutputParser
from utils.retry import invoke_with_retry
from agents.llm_factory import get_llm
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a rigorous scientific critic. Evaluate the evidence gathered for a query.
Return JSON:
{{
  "passed": true/false,
  "confidence": 0.0-1.0,
  "issues": ["list of specific problems"],
  "sufficient_evidence": true/false,
  "logical_consistency": true/false,
  "recommendation": "pass | retry | inconclusive",
  "reasoning": "brief explanation"
}}

Pass only if:
1. At least {min_evidence} distinct evidence items
2. No major logical contradictions
3. Overall confidence >= {threshold}
Otherwise recommend 'retry' or 'inconclusive'."""),
    ("human", """Query: {query}

Evidence count: {evidence_count}
Conflict count: {conflict_count}
Average confidence: {avg_confidence:.2f}
Weighted claims: {claims_summary}

Evaluate this evidence.""")
])


class CriticAgent:
    name = "critic_agent"

    def __init__(self):
        self.llm = get_llm()
        self._chain = CRITIC_PROMPT | self.llm | RobustJsonOutputParser()

    def evaluate(self, state: dict) -> dict:
        """
        Evaluate the current evidence state.
        Returns a critic result dict.
        """
        evidence_set = state.get("evidence_set", [])
        conflict_report = state.get("conflict_report", {})
        iteration = state.get("iteration_count", 0)

        evidence_count = len(evidence_set)
        conflict_count = conflict_report.get("conflict_count", 0)
        weighted_claims = conflict_report.get("weighted_claims", [])

        if evidence_count == 0:
            return {
                "passed": False,
                "confidence": 0.0,
                "issues": ["No evidence gathered"],
                "sufficient_evidence": False,
                "logical_consistency": True,
                "recommendation": "retry" if iteration < settings.max_iterations else "inconclusive",
                "reasoning": "No evidence found",
            }

        avg_confidence = (
            sum(c.get("confidence", 0.5) for c in weighted_claims) / len(weighted_claims)
            if weighted_claims else 0.5
        )

        claims_summary = "; ".join(
            c.get("claim_id", "")[:30] for c in weighted_claims[:5]
        )

        try:
            result = invoke_with_retry(self._chain, {
                "query": state.get("query", ""),
                "evidence_count": evidence_count,
                "conflict_count": conflict_count,
                "avg_confidence": avg_confidence,
                "claims_summary": claims_summary,
                "min_evidence": settings.min_evidence_count,
                "threshold": settings.confidence_threshold,
            })
            result["overall_confidence"] = avg_confidence
            return result
        except Exception as e:
            logger.warning(f"Critic LLM failed: {e} — using heuristic")
            passed = (
                evidence_count >= settings.min_evidence_count
                and avg_confidence >= settings.confidence_threshold
            )
            return {
                "passed": passed,
                "confidence": avg_confidence,
                "issues": [] if passed else ["Insufficient evidence"],
                "sufficient_evidence": evidence_count >= settings.min_evidence_count,
                "logical_consistency": conflict_count == 0,
                "recommendation": "pass" if passed else (
                    "retry" if iteration < settings.max_iterations else "inconclusive"
                ),
                "reasoning": f"Heuristic: {evidence_count} items, confidence={avg_confidence:.2f}",
                "overall_confidence": avg_confidence,
            }
