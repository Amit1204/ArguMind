"""
Citation DAG — core differentiator of ArguMind.

Directed graph where:
  Nodes: papers, claims, synthesized conclusions
  Edges: supports | refutes | supersedes | extends

Built incrementally as agents return evidence.
"""
import json
from dataclasses import dataclass, field
from typing import Literal
import networkx as nx
from utils.logger import get_logger

logger = get_logger(__name__)

EdgeType = Literal["supports", "refutes", "supersedes", "extends"]


@dataclass
class CitationNode:
    node_id: str
    node_type: Literal["paper", "claim", "conclusion"]
    title: str
    content: str
    source: str = ""          # arxiv ID, URL, etc.
    year: int = 0
    confidence: float = 1.0
    agent: str = ""           # which agent added this


@dataclass
class CitationEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0       # strength of relationship
    explanation: str = ""


class CitationDAG:
    """Incrementally-built directed citation graph."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._node_count = 0

    def add_node(self, node: CitationNode) -> str:
        self.graph.add_node(
            node.node_id,
            node_type=node.node_type,
            title=node.title,
            content=node.content,
            source=node.source,
            year=node.year,
            confidence=node.confidence,
            agent=node.agent,
        )
        logger.debug(f"DAG node added: {node.node_id} ({node.node_type})")
        return node.node_id

    def add_edge(self, edge: CitationEdge):
        if not self.graph.has_node(edge.source_id):
            logger.warning(f"Source node {edge.source_id} not in graph")
            return
        if not self.graph.has_node(edge.target_id):
            logger.warning(f"Target node {edge.target_id} not in graph")
            return
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type,
            weight=edge.weight,
            explanation=edge.explanation,
        )
        logger.debug(f"DAG edge: {edge.source_id} --{edge.edge_type}--> {edge.target_id}")

    def add_paper(self, paper_id: str, title: str, abstract: str,
                  source: str = "", year: int = 0, agent: str = "",
                  confidence: float = 1.0) -> str:
        node = CitationNode(
            node_id=paper_id,
            node_type="paper",
            title=title,
            content=abstract,
            source=source,
            year=year,
            confidence=confidence,
            agent=agent,
        )
        return self.add_node(node)

    def add_claim(self, claim_id: str, claim_text: str,
                  source_paper_id: str = "", agent: str = "",
                  confidence: float = 0.8) -> str:
        node = CitationNode(
            node_id=claim_id,
            node_type="claim",
            title=claim_text[:80],
            content=claim_text,
            agent=agent,
            confidence=confidence,
        )
        self.add_node(node)
        if source_paper_id and self.graph.has_node(source_paper_id):
            self.add_edge(CitationEdge(
                source_id=source_paper_id,
                target_id=claim_id,
                edge_type="supports",
                explanation="Paper is the source of this claim",
            ))
        return claim_id

    def link(self, source_id: str, target_id: str,
             edge_type: EdgeType, explanation: str = "", weight: float = 1.0):
        self.add_edge(CitationEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            explanation=explanation,
        ))

    # ── Analysis ──────────────────────────────────────────────────────────────

    def get_supporting_papers(self, claim_id: str) -> list[str]:
        return [
            n for n in self.graph.predecessors(claim_id)
            if self.graph[n][claim_id].get("edge_type") == "supports"
        ]

    def get_refuting_papers(self, claim_id: str) -> list[str]:
        return [
            n for n in self.graph.predecessors(claim_id)
            if self.graph[n][claim_id].get("edge_type") == "refutes"
        ]

    def get_conflicts(self) -> list[dict]:
        """Find all claim nodes that have both supporting and refuting edges."""
        conflicts = []
        for node in self.graph.nodes:
            if self.graph.nodes[node].get("node_type") == "claim":
                supporters = self.get_supporting_papers(node)
                refuters = self.get_refuting_papers(node)
                if supporters and refuters:
                    conflicts.append({
                        "claim_id": node,
                        "claim": self.graph.nodes[node].get("title", ""),
                        "supporting": supporters,
                        "refuting": refuters,
                    })
        return conflicts

    def get_reasoning_path(self, conclusion_id: str) -> list[dict]:
        """Trace the reasoning path leading to a conclusion node."""
        if not self.graph.has_node(conclusion_id):
            return []
        path = []
        for ancestor in nx.ancestors(self.graph, conclusion_id):
            edge_data = {}
            if self.graph.has_edge(ancestor, conclusion_id):
                edge_data = self.graph[ancestor][conclusion_id]
            path.append({
                "node": ancestor,
                "title": self.graph.nodes[ancestor].get("title", ""),
                "type": self.graph.nodes[ancestor].get("node_type", ""),
                "edge_type": edge_data.get("edge_type", ""),
            })
        return path

    def summary(self) -> dict:
        nodes = list(self.graph.nodes(data=True))
        edges = list(self.graph.edges(data=True))
        return {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "papers": sum(1 for _, d in nodes if d.get("node_type") == "paper"),
            "claims": sum(1 for _, d in nodes if d.get("node_type") == "claim"),
            "conclusions": sum(1 for _, d in nodes if d.get("node_type") == "conclusion"),
            "conflicts": len(self.get_conflicts()),
            "supports_edges": sum(1 for _, _, d in edges if d.get("edge_type") == "supports"),
            "refutes_edges": sum(1 for _, _, d in edges if d.get("edge_type") == "refutes"),
        }

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {"id": n, **data}
                for n, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **data}
                for u, v, data in self.graph.edges(data=True)
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "CitationDAG":
        dag = cls()
        for node in data.get("nodes", []):
            node = dict(node)  # don't mutate original
            node_id = node.pop("id", None)
            if node_id is None:
                continue  # skip malformed nodes
            dag.graph.add_node(node_id, **node)
        for edge in data.get("edges", []):
            src, tgt = edge.get("source"), edge.get("target")
            if src is None or tgt is None:
                continue
            dag.graph.add_edge(src, tgt,
                               **{k: v for k, v in edge.items()
                                  if k not in ("source", "target")})
        return dag
