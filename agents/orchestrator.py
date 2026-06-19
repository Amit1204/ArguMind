"""
ArguMind Orchestrator — LangGraph StateGraph.

Full pipeline:
  planner
    → decomposer
    → agent_executor          (arXiv + Web agents in parallel per sub-question)
    → normalizer              (claim extraction + dedup)
    → conflict_resolver       (recency + authority + LLM)
    → semantic_agent          (TF-IDF clustering + DAG 'extends' edges)
    → consensus_agent         (majority / minority synthesis)
    → critic                  (quality gate — may retry)
    ↑_________________________|   back-edge on retry
    → response_builder
    → END
"""
import json
from typing import TypedDict, Annotated
import operator

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.json_parser import RobustJsonOutputParser
from utils.retry import invoke_with_retry
from langgraph.graph import StateGraph, END

from agents.llm_factory import get_llm
from agents.nodes.arxiv_agent import ArxivAgent
from agents.nodes.web_agent import WebAgent
from agents.nodes.semantic_agent import SemanticAgent
from agents.nodes.consensus_agent import ConsensusAgent
from agents.nodes.critic_agent import CriticAgent
from graph.citation_dag import CitationDAG
from graph.evidence_normalizer import EvidenceNormalizer
from graph.conflict_resolver import ConflictResolver
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class ArguMindState(TypedDict):
    # Input
    query: str
    user_id: str

    # Planning
    domains: list
    complexity: str
    execution_plan: dict

    # Decomposition
    sub_questions: list          # [{question, domain, dependencies}]

    # Agent outputs (accumulates across retries via reducer)
    agent_outputs: Annotated[list, operator.add]

    # Evidence processing
    evidence_set: list
    citation_graph: dict         # serialized CitationDAG

    # Conflict resolution
    conflict_report: dict

    # Semantic clustering
    semantic_clusters: list      # list of cluster records from SemanticAgent

    # Consensus synthesis
    consensus_report: dict       # majority view, key agreements, gaps

    # Quality control
    critic_result: dict
    iteration_count: int

    # Output
    final_response: str
    confidence: float
    reasoning_trace: list


# ── Prompts ───────────────────────────────────────────────────────────────────

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research planning assistant.
Analyze the query and return JSON only:
{{
  "domains": ["relevant domains e.g. machine_learning, nlp, neuroscience"],
  "complexity": "simple|moderate|complex",
  "requires_web": true,
  "estimated_agents": 2
}}"""),
    ("human", "Query: {query}"),
])

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Break this research query into 2-4 specific, answerable sub-questions.
Each should be resolvable by searching academic papers or trusted web sources.
Return JSON only:
{{
  "sub_questions": [
    {{
      "question": "specific sub-question",
      "domain": "primary domain",
      "dependencies": []
    }}
  ]
}}"""),
    ("human", "Query: {query}\nDomains: {domains}"),
])

RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research synthesizer. Write a clear, evidence-based answer.

Structure:
1. Direct answer to the query
2. Key supporting evidence (with paper/source IDs where available)
3. Contradictions or caveats
4. Confidence statement

If evidence is inconclusive, say so clearly. Do not fabricate facts."""),
    ("human", """Query: {query}

Consensus (strength={consensus_strength}):
{majority_view}

Key agreements:
{key_agreements}

Key disagreements:
{key_disagreements}

Weighted claims:
{claims}

Minority reports:
{minority_reports}

Overall confidence: {confidence:.0%}
Citation graph: {graph_summary}

Write the final response."""),
])


# ── Orchestrator ──────────────────────────────────────────────────────────────

class ArguMindOrchestrator:

    def __init__(self):
        self.llm = get_llm()
        self.arxiv_agent = ArxivAgent()
        self.web_agent = WebAgent()
        self.semantic_agent = SemanticAgent()
        self.consensus_agent = ConsensusAgent()
        self.critic = CriticAgent()
        self.normalizer = EvidenceNormalizer()
        self.resolver = ConflictResolver()

        self._planner_chain = PLANNER_PROMPT | self.llm | RobustJsonOutputParser()
        self._decompose_chain = DECOMPOSE_PROMPT | self.llm | RobustJsonOutputParser()
        self._response_chain = RESPONSE_PROMPT | self.llm | StrOutputParser()

        self._graph = self._build_graph()
        logger.info("ArguMind orchestrator initialized")

    # ── Graph definition ───────────────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        g = StateGraph(ArguMindState)

        g.add_node("planner",           self._plan)
        g.add_node("decomposer",        self._decompose)
        g.add_node("agent_executor",    self._execute_agents)
        g.add_node("normalizer",        self._normalize)
        g.add_node("conflict_resolver", self._resolve_conflicts)
        g.add_node("semantic_agent",    self._semantic_cluster)
        g.add_node("consensus_agent",   self._build_consensus)
        g.add_node("critic",            self._critique)
        g.add_node("response_builder",  self._build_response)

        g.set_entry_point("planner")
        g.add_edge("planner",           "decomposer")
        g.add_edge("decomposer",        "agent_executor")
        g.add_edge("agent_executor",    "normalizer")
        g.add_edge("normalizer",        "conflict_resolver")
        g.add_edge("conflict_resolver", "semantic_agent")
        g.add_edge("semantic_agent",    "consensus_agent")
        g.add_edge("consensus_agent",   "critic")

        # Back-edge: critic decides retry vs finish
        g.add_conditional_edges("critic", self._should_retry, {
            "retry":  "agent_executor",
            "finish": "response_builder",
        })
        g.add_edge("response_builder", END)

        return g.compile()

    # ── Node implementations ──────────────────────────────────────────────────

    def _plan(self, state: ArguMindState) -> dict:
        logger.info(f"Planner: {state['query'][:60]}")
        try:
            plan = invoke_with_retry(self._planner_chain, {"query": state["query"]})
        except Exception as e:
            logger.warning(f"Planner failed: {e}")
            plan = {
                "domains": ["general"],
                "complexity": "moderate",
                "requires_web": True,
                "estimated_agents": 2,
            }
        return {
            "domains": plan.get("domains", ["general"]),
            "complexity": plan.get("complexity", "moderate"),
            "execution_plan": plan,
            "iteration_count": 0,
            "agent_outputs": [],
            "semantic_clusters": [],
            "consensus_report": {},
            "reasoning_trace": [
                f"Planner: domains={plan.get('domains')}, "
                f"complexity={plan.get('complexity')}, "
                f"web={'yes' if plan.get('requires_web') else 'no'}"
            ],
        }

    def _decompose(self, state: ArguMindState) -> dict:
        logger.info("Decomposer: breaking query")
        try:
            result = invoke_with_retry(self._decompose_chain, {
                "query": state["query"],
                "domains": ", ".join(state.get("domains", [])),
            })
            sub_questions = result.get("sub_questions", [])
        except Exception as e:
            logger.warning(f"Decomposer failed: {e}")
            sub_questions = [{
                "question": state["query"],
                "domain": "general",
                "dependencies": [],
            }]

        logger.info(f"Decomposer: {len(sub_questions)} sub-questions")
        return {
            "sub_questions": sub_questions,
            "reasoning_trace": [f"Decomposer: {len(sub_questions)} sub-questions"],
        }

    def _execute_agents(self, state: ArguMindState) -> dict:
        """
        Run arXiv + Web agents for each sub-question.
        On retry iteration, broaden queries with 'survey review overview'.
        """
        iteration = state.get("iteration_count", 0)
        sub_questions = state.get("sub_questions", [])
        requires_web = state.get("execution_plan", {}).get("requires_web", True)

        if iteration > 0:
            sub_questions = [
                {**sq, "question": sq["question"] + " survey review overview"}
                for sq in sub_questions
            ]

        all_outputs = []
        for sq in sub_questions:
            # Always search arXiv
            arxiv_outs = self.arxiv_agent.run(sq)
            all_outputs.extend(arxiv_outs)

            # Supplement with web (except on retry — avoid duplicating noise)
            if requires_web and iteration == 0:
                web_outs = self.web_agent.run(sq)
                all_outputs.extend(web_outs)

        logger.info(
            f"AgentExecutor (iter {iteration}): {len(all_outputs)} raw outputs"
        )
        return {
            "agent_outputs": all_outputs,
            "reasoning_trace": [
                f"AgentExecutor (iter {iteration}): "
                f"{len(all_outputs)} outputs "
                f"(arXiv + {'web' if requires_web and iteration == 0 else 'arxiv-only retry'})"
            ],
        }

    def _normalize(self, state: ArguMindState) -> dict:
        logger.info("Normalizer: extracting claims")
        evidence_set = self.normalizer.normalize(state.get("agent_outputs", []))

        dag = CitationDAG()
        for output in state.get("agent_outputs", []):
            pid = output.get("paper_id", "")
            if pid:
                dag.add_paper(
                    paper_id=pid,
                    title=output.get("title", pid),
                    abstract=output.get("summary", ""),
                    source=output.get("source", ""),
                    year=output.get("year", 0),
                    agent=output.get("agent", ""),
                    confidence=output.get("confidence", 0.7),
                )
        for claim in evidence_set:
            claim_id = claim.get("claim_id", "")
            paper_id = claim.get("paper_id", "")
            if claim_id:
                dag.add_claim(
                    claim_id=claim_id,
                    claim_text=claim.get("text", ""),
                    source_paper_id=paper_id,
                    agent=claim.get("agent", ""),
                    confidence=claim.get("confidence", 0.7),
                )

        summary = dag.summary()
        logger.info(f"Normalizer: {len(evidence_set)} claims, DAG={summary}")
        return {
            "evidence_set": evidence_set,
            "citation_graph": dag.to_dict(),
            "reasoning_trace": [
                f"Normalizer: {len(evidence_set)} claims, "
                f"DAG nodes={summary['total_nodes']}"
            ],
        }

    def _resolve_conflicts(self, state: ArguMindState) -> dict:
        logger.info("ConflictResolver: detecting conflicts")
        dag = CitationDAG.from_dict(
            state.get("citation_graph", {"nodes": [], "edges": []})
        )
        conflict_report = self.resolver.resolve(state.get("evidence_set", []), dag)
        logger.info(
            f"ConflictResolver: {conflict_report.get('conflict_count', 0)} conflicts"
        )
        return {
            "conflict_report": conflict_report,
            "reasoning_trace": [
                f"ConflictResolver: "
                f"{conflict_report.get('conflict_count')} conflicts, "
                f"{conflict_report.get('resolution_count')} resolved"
            ],
        }

    def _semantic_cluster(self, state: ArguMindState) -> dict:
        logger.info("SemanticAgent: clustering claims")
        return self.semantic_agent.run(state)

    def _build_consensus(self, state: ArguMindState) -> dict:
        logger.info("ConsensusAgent: synthesizing")
        return self.consensus_agent.run(state)

    def _critique(self, state: ArguMindState) -> dict:
        logger.info("Critic: evaluating quality")
        result = self.critic.evaluate(state)
        new_iter = state.get("iteration_count", 0) + 1
        logger.info(
            f"Critic: passed={result.get('passed')}, "
            f"recommendation={result.get('recommendation')}"
        )
        return {
            "critic_result": result,
            "iteration_count": new_iter,
            "reasoning_trace": [
                f"Critic (iter {new_iter}): "
                f"passed={result.get('passed')}, "
                f"conf={result.get('overall_confidence', 0):.2f}, "
                f"rec={result.get('recommendation')}"
            ],
        }

    def _build_response(self, state: ArguMindState) -> dict:
        logger.info("ResponseBuilder: generating answer")
        conflict_report = state.get("conflict_report", {})
        critic_result = state.get("critic_result", {})
        consensus_report = state.get("consensus_report", {})
        dag = CitationDAG.from_dict(
            state.get("citation_graph", {"nodes": [], "edges": []})
        )

        confidence = critic_result.get(
            "overall_confidence",
            consensus_report.get("confidence", 0.5),
        )

        # "I don't know" gate
        if critic_result.get("recommendation") == "inconclusive":
            issues = "; ".join(critic_result.get("issues", ["insufficient evidence"]))
            response = (
                f"The evidence on this question is currently inconclusive. "
                f"I gathered {len(state.get('evidence_set', []))} relevant claims "
                f"but could not reach a confident conclusion "
                f"(confidence: {confidence:.0%}). "
                f"Key issues: {issues}"
            )
            gaps = consensus_report.get("research_gaps", [])
            if gaps:
                response += f"\n\nResearch gaps: {'; '.join(gaps[:3])}"
            return {
                "final_response": response,
                "confidence": confidence,
                "reasoning_trace": [
                    f"ResponseBuilder: inconclusive (conf={confidence:.2f})"
                ],
            }

        # Format inputs for response prompt
        weighted_claims = conflict_report.get("weighted_claims", [])
        minority_reports = conflict_report.get("minority_reports", [])

        claims_text = "\n".join(
            f"- [{c.get('claim_id', '')}] "
            f"{c.get('reasoning', 'Uncontested')} "
            f"(conf={c.get('confidence', 0):.2f})"
            for c in weighted_claims[:10]
        ) or "No structured claims extracted."

        minority_text = "\n".join(
            f"- Minority view: {m.get('minority_view', '')}"
            for m in minority_reports[:5]
        ) or "No significant contradictions found."

        majority_view = consensus_report.get(
            "majority_view",
            consensus_report.get("overall_consensus", ""),
        )
        key_agreements = "; ".join(
            consensus_report.get("key_agreements", [])[:4]
        ) or "Not available."
        key_disagreements = "; ".join(
            consensus_report.get("key_disagreements", [])[:3]
        ) or "None detected."

        try:
            response = invoke_with_retry(self._response_chain, {
                "query": state["query"],
                "consensus_strength": consensus_report.get("consensus_strength", "unknown"),
                "majority_view": majority_view,
                "key_agreements": key_agreements,
                "key_disagreements": key_disagreements,
                "claims": claims_text,
                "minority_reports": minority_text,
                "confidence": confidence,
                "graph_summary": json.dumps(dag.summary()),
            })
        except Exception as e:
            logger.warning(f"ResponseBuilder LLM failed: {e} — using majority view")
            response = majority_view or claims_text

        return {
            "final_response": response,
            "confidence": confidence,
            "reasoning_trace": [
                f"ResponseBuilder: conf={confidence:.2f}, "
                f"{len(state.get('evidence_set', []))} evidence items"
            ],
        }

    # ── Routing ────────────────────────────────────────────────────────────────

    def _should_retry(self, state: ArguMindState) -> str:
        critic = state.get("critic_result", {})
        iteration = state.get("iteration_count", 0)
        if (
            critic.get("recommendation") == "retry"
            and iteration <= settings.max_iterations
        ):
            logger.info(f"Critic: retry (iter {iteration})")
            return "retry"
        return "finish"

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, query: str, user_id: str = "anonymous") -> dict:
        """Run the full ArguMind pipeline synchronously."""
        initial_state: ArguMindState = {
            "query": query,
            "user_id": user_id,
            "domains": [],
            "complexity": "",
            "execution_plan": {},
            "sub_questions": [],
            "agent_outputs": [],
            "evidence_set": [],
            "citation_graph": {"nodes": [], "edges": []},
            "conflict_report": {},
            "semantic_clusters": [],
            "consensus_report": {},
            "critic_result": {},
            "iteration_count": 0,
            "final_response": "",
            "confidence": 0.0,
            "reasoning_trace": [],
        }
        result = self._graph.invoke(initial_state)
        return result
