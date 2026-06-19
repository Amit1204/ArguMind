"""
ArguMind Streamlit Frontend
Multi-agent evidence reasoning UI with interactive citation graph.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import json

st.set_page_config(
    page_title="ArguMind",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 ArguMind")
    st.caption("Adaptive Multi-Agent Evidence Reasoning")
    st.divider()

    st.markdown("**Pipeline**")
    pipeline_steps = [
        "🗺️ Planner — domain + complexity",
        "✂️ Decomposer — sub-questions",
        "📄 arXiv Agent — academic papers",
        "🌐 Web Agent — Wikipedia + DDG",
        "🔬 Normalizer — claim extraction",
        "⚖️ Conflict Resolver — DAG arbitration",
        "🧩 Semantic Agent — TF-IDF clustering",
        "🤝 Consensus Agent — majority synthesis",
        "🎯 Critic — quality gate + retry loop",
        "📝 Response Builder — final answer",
    ]
    for step in pipeline_steps:
        st.caption(step)

    st.divider()
    st.markdown("**Stack**")
    st.caption("LangGraph StateGraph · Groq / Gemini LLM · arXiv API · Wikipedia API · NetworkX Citation DAG · Pyvis")
    st.divider()

    if st.button("🗑️ Clear results"):
        for key in ["result", "graph_html", "graph_legend"]:
            st.session_state.pop(key, None)
        st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔍 ArguMind — Evidence Reasoning Agent")
st.caption(
    "Ask a research question. ArguMind decomposes it, searches arXiv + web, "
    "resolves conflicts, clusters evidence, synthesizes consensus, and builds a citation graph."
)

col_input, col_examples = st.columns([3, 1])

with col_input:
    query = st.text_area(
        "Research question",
        placeholder="e.g. Do large language models truly understand language or just pattern match?",
        height=100,
    )

with col_examples:
    st.markdown("**Quick examples**")
    examples = [
        "Do large language models truly understand language or just pattern match?",
        "What is the most effective method for few-shot learning?",
        "Are retrieval-augmented generation systems better than fine-tuning for QA?",
        "Does coffee improve cognitive performance?",
    ]
    for ex in examples:
        if st.button(ex[:44] + "…", key=ex):
            st.session_state["_example"] = ex

if "_example" in st.session_state:
    query = st.session_state.pop("_example")

run_btn = st.button("🚀 Run ArguMind", type="primary", disabled=not query)

# ── Pipeline ───────────────────────────────────────────────────────────────────
if run_btn and query:
    progress = st.progress(0, text="Starting pipeline…")
    status = st.empty()

    def update(pct, msg):
        progress.progress(pct, text=msg)
        status.caption(msg)

    try:
        update(5,  "🗺️ Planning…")
        from agents.orchestrator import ArguMindOrchestrator
        orchestrator = ArguMindOrchestrator()

        update(10, "🚀 Running pipeline (30-90 seconds)…")
        result = orchestrator.run(query)
        st.session_state["result"] = result

        update(90, "🕸️ Rendering citation graph…")
        from graph.citation_dag import CitationDAG
        from graph.visualizer import render_graph, render_legend
        dag = CitationDAG.from_dict(result.get("citation_graph", {"nodes": [], "edges": []}))
        st.session_state["graph_html"] = render_graph(dag)
        st.session_state["graph_legend"] = render_legend()

        update(100, "✅ Done!")
        status.empty()
        progress.empty()

    except Exception as e:
        progress.empty()
        status.empty()
        st.error(f"Pipeline error: {e}")
        import traceback
        st.code(traceback.format_exc())

# ── Results ────────────────────────────────────────────────────────────────────
if "result" in st.session_state:
    result = st.session_state["result"]
    critic = result.get("critic_result", {})
    conflict = result.get("conflict_report", {})
    consensus = result.get("consensus_report", {})
    semantic_clusters = result.get("semantic_clusters", [])

    from graph.citation_dag import CitationDAG
    dag = CitationDAG.from_dict(result.get("citation_graph", {"nodes": [], "edges": []}))
    dag_summary = dag.summary()

    # ── Metrics ────────────────────────────────────────────────────────────────
    st.divider()
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Confidence",      f"{result.get('confidence', 0):.0%}")
    m2.metric("Evidence items",  len(result.get("evidence_set", [])))
    m3.metric("Conflicts",       conflict.get("conflict_count", 0))
    m4.metric("Clusters",        len(semantic_clusters))
    m5.metric("DAG nodes",       dag_summary.get("total_nodes", 0))
    m6.metric("Iterations",      result.get("iteration_count", 1))

    # ── Answer ─────────────────────────────────────────────────────────────────
    st.divider()
    strength = consensus.get("consensus_strength", "unknown")
    strength_emoji = {"strong": "🟢", "moderate": "🟡", "weak": "🟠", "absent": "🔴"}.get(strength, "⚪")
    st.subheader(f"📋 Answer  {strength_emoji} Consensus: {strength}")
    st.write(result.get("final_response", "No response generated."))

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🕸️ Citation Graph",
        "🤝 Consensus",
        "🧩 Clusters",
        "⚖️ Conflicts",
        "🔎 Evidence",
        "🧠 Reasoning",
    ])

    # Citation graph
    with tab1:
        legend = st.session_state.get("graph_legend", "")
        if legend:
            st.markdown(legend, unsafe_allow_html=True)
        graph_html = st.session_state.get("graph_html", "")
        if graph_html and "<p>" not in graph_html[:50]:
            st.iframe(graph_html, height=520)
        else:
            st.info("Install pyvis to view graph: `pip install pyvis`")
        col_a, col_b = st.columns(2)
        col_a.json(dag_summary)
        raw_conflicts = dag.get_conflicts()
        if raw_conflicts:
            col_b.markdown("**Conflicting claims:**")
            for c in raw_conflicts:
                col_b.warning(
                    f"`{c['claim_id']}`: "
                    f"{len(c['supporting'])} support, {len(c['refuting'])} refute"
                )

    # Consensus
    with tab2:
        if consensus:
            st.markdown(f"**Overall consensus:** {consensus.get('overall_consensus', '—')}")
            st.markdown(f"**Strength:** {strength_emoji} {strength}")
            st.markdown(f"**Confidence:** {consensus.get('confidence', 0):.0%}")

            if consensus.get("majority_view"):
                st.divider()
                st.markdown("**Majority view:**")
                st.info(consensus["majority_view"])

            col_a, col_b = st.columns(2)
            with col_a:
                agreements = consensus.get("key_agreements", [])
                if agreements:
                    st.markdown("**Key agreements:**")
                    for a in agreements:
                        st.success(f"✓ {a}")

            with col_b:
                disagreements = consensus.get("key_disagreements", [])
                if disagreements:
                    st.markdown("**Key disagreements:**")
                    for d in disagreements:
                        st.warning(f"✗ {d}")

            gaps = consensus.get("research_gaps", [])
            if gaps:
                st.markdown("**Research gaps:**")
                for g in gaps:
                    st.caption(f"❓ {g}")
        else:
            st.info("No consensus report generated.")

    # Semantic clusters
    with tab3:
        if semantic_clusters:
            st.markdown(f"**{len(semantic_clusters)} semantic clusters found**")
            direction_color = {
                "positive": "green", "negative": "red",
                "neutral": "blue", "mixed": "orange",
            }
            for cl in semantic_clusters:
                direction = cl.get("consensus_direction", "mixed")
                color = direction_color.get(direction, "gray")
                with st.expander(
                    f"🧩 [{cl.get('cluster_id', '')}] {cl.get('topic', 'Untitled')} "
                    f"— {cl.get('size', 0)} claims"
                ):
                    st.markdown(
                        f"**Direction:** :{color}[{direction}]  |  "
                        f"**Agreement:** {cl.get('agreement_score', 0):.0%}"
                    )
                    st.write(cl.get("summary", ""))
                    if cl.get("claim_ids"):
                        st.caption("Claims: " + ", ".join(cl["claim_ids"][:8]))
        else:
            st.info("No multi-claim clusters found (may need more evidence).")

    # Conflicts
    with tab4:
        weighted = conflict.get("weighted_claims", [])
        minority = conflict.get("minority_reports", [])
        if weighted:
            st.markdown("**Weighted claims (after resolution):**")
            for c in weighted:
                conf = c.get("confidence", 0)
                color = "green" if conf > 0.7 else "orange" if conf > 0.4 else "red"
                st.markdown(
                    f":{color}[●] **{c.get('claim_id', '')}** "
                    f"conf={conf:.2f}  \n_{c.get('reasoning', '')}_"
                )
        if minority:
            st.markdown("**Minority reports:**")
            for m in minority:
                st.info(m.get("minority_view", ""))
        if not weighted and not minority:
            st.success("No conflicts — all agents agreed.")

    # Evidence
    with tab5:
        evidence = result.get("evidence_set", [])
        if evidence:
            st.markdown(f"**{len(evidence)} normalized claims:**")
            for i, ev in enumerate(evidence[:25]):
                label = f"[{ev.get('claim_id', i)}] {ev.get('text', '')[:80]}…"
                with st.expander(label):
                    st.json(ev)
        else:
            st.info("No structured evidence extracted.")

    # Reasoning trace
    with tab6:
        trace = result.get("reasoning_trace", [])
        for step in trace:
            st.markdown(f"→ {step}")
