---
title: ArguMind
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: frontend/app.py
pinned: false
---

# ArguMind — Adaptive Multi-Agent Evidence Reasoning System

> LangGraph StateGraph · arXiv API · Wikipedia API · NetworkX Citation DAG · Groq / Gemini LLM

ArguMind is a multi-agent system that answers research questions by gathering evidence from arXiv and the web, resolving contradictions, clustering claims semantically, synthesizing consensus, and producing a transparent citation graph.

---

## Architecture

```
Query
  │
  ▼
Planner ──────────── identifies domains + complexity
  │
  ▼
Decomposer ─────────── breaks query into 2-4 sub-questions
  │
  ▼
Agent Executor ─────── per sub-question:
  ├── arXiv Agent ─── academic paper search + LLM summarization
  └── Web Agent ───── Wikipedia + DuckDuckGo (no API key)
  │
  ▼
Normalizer ─────────── claim extraction + deduplication → evidence_set
  │                    builds Citation DAG (papers → claims)
  ▼
Conflict Resolver ───── finds DAG conflicts (supports ↔ refutes)
  │                    resolves with recency + authority + LLM
  ▼
Semantic Agent ──────── TF-IDF cosine clustering (no sklearn needed)
  │                    labels clusters via LLM
  │                    adds 'extends' edges to Citation DAG
  ▼
Consensus Agent ──────── majority view + key agreements/disagreements
  │                     research gaps
  ▼
Critic ──────────────── quality gate: confidence + evidence sufficiency
  │                    "I don't know" gate for inconclusive evidence
  │                    back-edge retry if evidence is insufficient
  ▼
Response Builder ─────── final answer using consensus + weighted claims
  │
  ▼
Answer + Citation DAG + Reasoning Trace
```

### Citation DAG Edge Types

| Edge | Meaning |
|------|---------|
| `supports` | Paper/claim provides positive evidence |
| `refutes` | Paper/claim contradicts a claim |
| `supersedes` | Newer work invalidates older claim |
| `extends` | Semantically related claims across papers (added by Semantic Agent) |

---

## Agents

| Agent | Role | Source |
|-------|------|--------|
| **Planner** | Domain classification, complexity assessment | LLM |
| **Decomposer** | Query → 2-4 sub-questions | LLM |
| **arXiv Agent** | Searches arXiv API, LLM-summarizes papers | arXiv (free, no key) |
| **Web Agent** | Wikipedia + DuckDuckGo Instant Answers | Wikipedia API (free) |
| **Normalizer** | LLM claim extraction, deduplication | LLM |
| **Conflict Resolver** | Recency + authority + LLM arbitration | LLM + heuristic |
| **Semantic Agent** | TF-IDF clustering, DAG enrichment | Pure Python |
| **Consensus Agent** | Majority view, agreement/gap synthesis | LLM |
| **Critic** | Quality gate, I-don't-know gate, retry trigger | LLM + heuristic |
| **Response Builder** | Final synthesis from all upstream outputs | LLM |

---

## Setup

### Prerequisites

- Python 3.10+
- A Groq API key (free, 100k tokens/day) **or** Google Gemini API key (1M tokens/day)
- No GPU needed — fully CPU, no training

### Install

```bash
cd "Project 2/ArguMind"
pip install -r requirements-local.txt
```

### Configure

```bash
# .env (already created — edit keys as needed)
GROQ_API_KEY=your_groq_key      # https://console.groq.com
GEMINI_API_KEY=your_gemini_key  # https://aistudio.google.com
```

LLM priority: **Groq → Gemini → OpenAI**. Set whichever key you have; leave others blank.

---

## Running

### Streamlit UI (recommended)

```bash
streamlit run frontend/app.py
```

Open http://localhost:8501

### CLI

```bash
# Simple output
python main.py "Do large language models truly understand language?"

# Full JSON output
python main.py "What is few-shot learning?" --json
```

### FastAPI

```bash
uvicorn api.main:app --reload
```

Swagger UI at http://localhost:8000/docs

POST `/api/v1/query` with `{"query": "your research question"}`.

---

## Key Design Decisions

**No training.** ArguMind is fully runtime — it reasons over live API data rather than training a model. No datasets, no model files, no GPU.

**Citation DAG as first-class artifact.** Evidence isn't just a list of facts — it's a directed graph where every claim traces back to its sources, and edges encode the *type* of relationship (supports, refutes, extends, supersedes).

**Conflict resolution with minority reports.** When papers disagree, ArguMind doesn't pick a winner silently. It records the losing view as a "minority report" and surfaces it to the user.

**"I don't know" gate.** If the Critic determines evidence is genuinely inconclusive, ArguMind says so explicitly rather than fabricating a confident-sounding answer.

**Back-edge retry loop.** If evidence is insufficient on the first pass, the Critic sends execution back to the Agent Executor with broadened queries (appends "survey review overview").

---

## Project Structure

```
ArguMind/
├── agents/
│   ├── llm_factory.py          # Groq → Gemini → OpenAI
│   ├── orchestrator.py         # LangGraph StateGraph (full pipeline)
│   └── nodes/
│       ├── arxiv_agent.py      # arXiv search + LLM summarization
│       ├── web_agent.py        # Wikipedia + DuckDuckGo
│       ├── semantic_agent.py   # TF-IDF clustering + DAG enrichment
│       ├── consensus_agent.py  # Majority/minority synthesis
│       └── critic_agent.py     # Quality gate + retry logic
├── graph/
│   ├── citation_dag.py         # NetworkX directed citation graph
│   ├── evidence_normalizer.py  # Claim extraction + deduplication
│   ├── conflict_resolver.py    # Recency + authority + LLM arbitration
│   └── visualizer.py           # Pyvis interactive HTML graph
├── tools/
│   ├── arxiv_search.py         # arXiv API (stdlib urllib, no key)
│   └── web_search.py           # Wikipedia + DuckDuckGo (no key)
├── api/
│   ├── main.py                 # FastAPI app
│   └── routes.py               # POST /query, GET /health
├── frontend/
│   └── app.py                  # Streamlit UI (6-tab results view)
├── config/settings.py          # Pydantic settings + .env loading
├── main.py                     # CLI entry point
└── requirements-local.txt
```

---

## LLM Token Usage

| Provider | Free Limit | Model |
|----------|-----------|-------|
| Groq | 100k tokens/day | llama-3.3-70b-versatile |
| Gemini | 1M tokens/day | gemini-1.5-flash |
| OpenAI | Paid | gpt-4o-mini |

A single ArguMind query uses ~5k–15k tokens depending on complexity.
