"""
arXiv Agent — searches and summarizes papers for a given sub-question.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents.llm_factory import get_llm
from tools.arxiv_search import search_arxiv, paper_to_agent_output
from utils.logger import get_logger

logger = get_logger(__name__)

SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research analyst. Summarize the key findings of these papers
relevant to the question. Be specific and factual. Include confidence caveats where appropriate.
Keep to 3-4 sentences."""),
    ("human", "Question: {question}\n\nPapers:\n{papers}")
])


class ArxivAgent:
    name = "arxiv_agent"

    def __init__(self):
        self.llm = get_llm()
        self._chain = SUMMARIZE_PROMPT | self.llm | StrOutputParser()

    def run(self, sub_question: dict) -> list[dict]:
        """
        Search arXiv for the sub-question and return agent outputs.
        sub_question: {question, domain, dependencies}
        """
        question = sub_question.get("question", "")
        logger.info(f"ArxivAgent: {question[:60]}")

        papers = search_arxiv(question, max_results=5)
        if not papers:
            logger.warning("ArxivAgent: no papers found")
            return []

        # Summarize papers in context of the question
        papers_text = "\n\n".join(
            f"[{p.paper_id}] {p.title} ({p.year})\n{p.abstract[:400]}"
            for p in papers
        )
        try:
            summary = self._chain.invoke({
                "question": question,
                "papers": papers_text,
            })
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            summary = f"Found {len(papers)} papers on: {question}"

        outputs = []
        for paper in papers:
            out = paper_to_agent_output(paper, agent_name=self.name)
            out["summary"] = summary  # shared summary for this question
            out["sub_question"] = question
            outputs.append(out)

        return outputs
