"""
Consensus Agent — synthesizes majority and minority positions.

Inputs (from state):
  evidence_set     — normalized claims
  semantic_clusters — output of SemanticAgent
  conflict_report  — output of ConflictResolver
  query            — original research question

Outputs:
  consensus_report — structured consensus with majority view,
                     agreement strength, key agreements/disagreements,
                     and research gaps
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents.llm_factory import get_llm
from utils.json_parser import parse_json as _parse_json
from utils.retry import invoke_with_retry
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

CONSENSUS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a systematic review expert synthesizing research evidence.
Given semantic clusters of claims and conflict information, produce a consensus assessment.
Return JSON only:
{{
  "overall_consensus": "one sentence: what the body of evidence collectively says",
  "consensus_strength": "strong|moderate|weak|absent",
  "key_agreements": ["up to 4 points most evidence agrees on"],
  "key_disagreements": ["up to 3 contested points"],
  "research_gaps": ["up to 3 unanswered questions"],
  "confidence": 0.0
}}

Be honest. If evidence is thin or contradictory, say so."""),
    ("human", """Research question: {query}

Semantic clusters ({cluster_count} found):
{clusters_text}

Conflict summary: {conflict_summary}
Total evidence items: {evidence_count}
"""),
])

MAJORITY_VIEW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Write the single most evidence-supported position on this research question.
Direct, factual prose. One paragraph, max 4 sentences.
Do not fabricate; if evidence is weak, say so."""),
    ("human", """Question: {query}
Overall consensus: {consensus}
Key agreements: {agreements}
Confidence: {confidence:.0%}"""),
])


# ── Agent ─────────────────────────────────────────────────────────────────────

class ConsensusAgent:
    name = "consensus_agent"

    def __init__(self):
        self.llm = get_llm()
        self._consensus_chain = CONSENSUS_PROMPT | self.llm | StrOutputParser()
        self._majority_chain = MAJORITY_VIEW_PROMPT | self.llm | StrOutputParser()

    def run(self, state: dict) -> dict:
        """
        Build a structured consensus report from semantic clusters + evidence.

        Returns dict with:
          consensus_report: full consensus analysis
          reasoning_trace:  step description
        """
        evidence_set = state.get("evidence_set", [])
        semantic_clusters = state.get("semantic_clusters", [])
        conflict_report = state.get("conflict_report", {})
        query = state.get("query", "")

        if not evidence_set:
            logger.info("ConsensusAgent: no evidence")
            return {
                "consensus_report": {
                    "overall_consensus": "No evidence was gathered.",
                    "consensus_strength": "absent",
                    "key_agreements": [],
                    "key_disagreements": [],
                    "research_gaps": ["The query produced no retrievable evidence."],
                    "confidence": 0.0,
                    "majority_view": "Insufficient evidence to answer this question.",
                    "cluster_count": 0,
                    "evidence_count": 0,
                },
                "reasoning_trace": ["ConsensusAgent: no evidence — skipped"],
            }

        logger.info(
            f"ConsensusAgent: {len(semantic_clusters)} clusters, "
            f"{len(evidence_set)} claims"
        )

        # Format clusters for LLM
        if semantic_clusters:
            clusters_text = "\n".join(
                f"• [{cl['topic']}] direction={cl['consensus_direction']}, "
                f"agreement={cl['agreement_score']:.2f}, size={cl['size']}\n"
                f"  {cl['summary']}"
                for cl in semantic_clusters
            )
        else:
            # Fallback: use raw claims if no clusters formed
            clusters_text = "\n".join(
                f"• {ev.get('text', '')[:120]}" for ev in evidence_set[:12]
            )

        conflict_summary = (
            f"{conflict_report.get('conflict_count', 0)} conflicts detected, "
            f"{conflict_report.get('resolution_count', 0)} resolved."
        )

        # ── Build consensus via LLM ────────────────────────────────────────────
        try:
            raw = invoke_with_retry(self._consensus_chain, {
                "query": query,
                "cluster_count": len(semantic_clusters),
                "clusters_text": clusters_text,
                "conflict_summary": conflict_summary,
                "evidence_count": len(evidence_set),
            })
            logger.info(f"ConsensusAgent raw LLM output: {raw[:300]}")
            consensus = _parse_json(raw)
        except Exception as e:
            logger.warning(f"ConsensusAgent LLM failed: {e} — raw was: {repr(raw) if 'raw' in dir() else 'N/A'} — using fallback")
            consensus = {
                "overall_consensus": "Evidence available but automated consensus synthesis failed.",
                "consensus_strength": "weak",
                "key_agreements": [],
                "key_disagreements": [],
                "research_gaps": [],
                "confidence": 0.4,
            }

        # Clamp confidence to [0, 1]
        raw_conf = consensus.get("confidence", 0.5)
        confidence = max(0.0, min(1.0, float(raw_conf) if raw_conf else 0.5))
        consensus["confidence"] = confidence

        # ── Majority view narrative ────────────────────────────────────────────
        try:
            majority_view = invoke_with_retry(self._majority_chain, {
                "query": query,
                "consensus": consensus.get("overall_consensus", ""),
                "agreements": "; ".join(consensus.get("key_agreements", [])[:4]),
                "confidence": confidence,
            })
        except Exception as e:
            logger.warning(f"Majority view LLM failed: {e}")
            majority_view = consensus.get("overall_consensus", "")

        consensus_report = {
            **consensus,
            "majority_view": majority_view,
            "cluster_count": len(semantic_clusters),
            "evidence_count": len(evidence_set),
        }

        strength = consensus.get("consensus_strength", "unknown")
        logger.info(
            f"ConsensusAgent: strength={strength}, confidence={confidence:.2f}, "
            f"{len(consensus.get('key_agreements', []))} agreements"
        )

        return {
            "consensus_report": consensus_report,
            "reasoning_trace": [
                f"ConsensusAgent: strength={strength}, conf={confidence:.2f}, "
                f"{len(consensus.get('key_agreements', []))} agreements, "
                f"{len(consensus.get('key_disagreements', []))} disagreements"
            ],
        }
