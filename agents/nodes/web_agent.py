"""
Web Agent — supplements arXiv evidence with Wikipedia and DuckDuckGo.

Used alongside ArxivAgent in agent_executor. Provides broader context
for sub-questions that are general-knowledge heavy or lack arXiv coverage.

Confidence is lower than arXiv (0.5 vs 0.75) — web sources are less
peer-reviewed — so ConflictResolver will weight them accordingly.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents.llm_factory import get_llm
from tools.web_search import search_web, web_result_to_agent_output
from utils.logger import get_logger

logger = get_logger(__name__)

SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research analyst. Summarize web sources relevant to the question.
Note the source type (Wikipedia, etc.) and any reliability caveats.
Be factual. 2-3 sentences."""),
    ("human", "Question: {question}\n\nWeb sources:\n{sources}"),
])


class WebAgent:
    name = "web_agent"

    def __init__(self):
        self.llm = get_llm()
        self._chain = SUMMARIZE_PROMPT | self.llm | StrOutputParser()

    def run(self, sub_question: dict) -> list[dict]:
        """
        Search web for the sub-question.
        Returns list of agent output dicts (standard format).
        """
        question = sub_question.get("question", "")
        logger.info(f"WebAgent: {question[:60]}")

        results = search_web(question, max_results=4)
        if not results:
            logger.warning("WebAgent: no results")
            return []

        sources_text = "\n".join(
            f"[{r.source}] {r.title}: {r.snippet[:200]}" for r in results
        )
        try:
            summary = self._chain.invoke({
                "question": question,
                "sources": sources_text,
            })
        except Exception as e:
            logger.warning(f"WebAgent summarization failed: {e}")
            summary = f"Found {len(results)} web sources on: {question[:80]}"

        outputs = []
        for result in results:
            out = web_result_to_agent_output(result, agent_name=self.name)
            out["summary"] = summary
            out["sub_question"] = question
            outputs.append(out)

        logger.info(f"WebAgent: returning {len(outputs)} outputs")
        return outputs
