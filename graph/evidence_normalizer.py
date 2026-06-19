"""
Evidence Normalizer — unifies raw agent outputs into a standard claim format.
Steps: deduplication → entity normalization → confidence standardization.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents.llm_factory import get_llm
from utils.json_parser import parse_json as _parse_json
from utils.retry import invoke_with_retry
from utils.logger import get_logger

logger = get_logger(__name__)

EXTRACT_CLAIMS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Extract the key factual claims from this research summary.
Return JSON:
{{
  "claims": [
    {{
      "claim_id": "unique short id like claim_1",
      "text": "the factual claim",
      "confidence": 0.0-1.0,
      "evidence_type": "empirical|theoretical|review|opinion"
    }}
  ]
}}
Extract 2-5 concrete, specific claims. Skip vague statements."""),
    ("human", "Summary:\n{summary}\n\nSource: {source}")
])


class EvidenceNormalizer:

    def __init__(self):
        self.llm = get_llm()
        self._extract_chain = EXTRACT_CLAIMS_PROMPT | self.llm | StrOutputParser()

    def normalize(self, agent_outputs: list[dict]) -> list[dict]:
        """
        Convert raw agent outputs → unified evidence set.
        Each item: {claim_id, text, confidence, source, agent, evidence_type}
        """
        all_claims = []
        seen_texts = set()

        for output in agent_outputs:
            claims = self._extract_claims(output)
            for claim in claims:
                # Deduplication by approximate text match
                key = claim["text"][:60].lower().strip()
                if key in seen_texts:
                    continue
                seen_texts.add(key)
                claim["source"] = output.get("source", "")
                claim["agent"] = output.get("agent", "")
                claim["paper_id"] = output.get("paper_id", "")
                all_claims.append(claim)

        logger.info(f"Normalized {len(agent_outputs)} agent outputs → {len(all_claims)} claims")
        return all_claims

    def _extract_claims(self, output: dict) -> list[dict]:
        summary = output.get("summary", "")
        if not summary:
            return []
        try:
            raw = invoke_with_retry(self._extract_chain, {
                "summary": summary,
                "source": output.get("source", "unknown"),
            })
            result = _parse_json(raw)
            return result.get("claims", [])
        except Exception as e:
            logger.warning(f"Claim extraction failed: {e}")
            # Fallback: treat the summary itself as one claim
            return [{
                "claim_id": f"claim_{hash(summary) % 10000}",
                "text": summary[:200],
                "confidence": output.get("confidence", 0.5),
                "evidence_type": "review",
            }]
