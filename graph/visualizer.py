"""
Citation DAG visualizer using Pyvis.
Produces an interactive HTML graph embeddable in Streamlit.
"""
from graph.citation_dag import CitationDAG
from utils.logger import get_logger

logger = get_logger(__name__)

NODE_COLORS = {
    "paper":      "#4A90D9",   # blue
    "claim":      "#F5A623",   # orange
    "conclusion": "#7ED321",   # green
}

EDGE_COLORS = {
    "supports":   "#27AE60",   # green
    "refutes":    "#E74C3C",   # red
    "supersedes": "#9B59B6",   # purple
    "extends":    "#F39C12",   # amber
}


def render_graph(dag: CitationDAG, height: str = "500px") -> str:
    """
    Render the citation DAG as an interactive Pyvis HTML string.
    Returns HTML that can be embedded with streamlit.components.v1.html().
    """
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p>Install pyvis to view graph: pip install pyvis</p>"

    net = Network(
        height=height,
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True,
    )
    net.set_options("""
    {
      "nodes": {"font": {"size": 12}},
      "edges": {"arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
                "smooth": {"type": "dynamic"}},
      "physics": {"stabilization": {"iterations": 100}}
    }
    """)

    for node_id, data in dag.graph.nodes(data=True):
        node_type = data.get("node_type", "claim")
        title = data.get("title", node_id)
        label = title[:30] + "..." if len(title) > 30 else title
        tooltip = f"<b>{title}</b><br>Type: {node_type}<br>Confidence: {data.get('confidence', '?')}"
        net.add_node(
            node_id,
            label=label,
            title=tooltip,
            color=NODE_COLORS.get(node_type, "#888"),
            size=20 if node_type == "conclusion" else 15,
        )

    for u, v, data in dag.graph.edges(data=True):
        edge_type = data.get("edge_type", "supports")
        net.add_edge(
            u, v,
            title=data.get("explanation", edge_type),
            color=EDGE_COLORS.get(edge_type, "#888"),
            label=edge_type,
            width=2,
        )

    # Return HTML string
    html = net.generate_html()
    return html


def render_legend() -> str:
    """Return a small HTML legend for the graph."""
    items = []
    for edge_type, color in EDGE_COLORS.items():
        items.append(f'<span style="color:{color}">━━</span> {edge_type}')
    for node_type, color in NODE_COLORS.items():
        items.append(f'<span style="color:{color}">●</span> {node_type}')
    return "  |  ".join(items)
