"""
Web Search Tool — Wikipedia API + DuckDuckGo Instant Answers.
No API key required. Supplements arXiv with general-knowledge evidence.
"""
import json
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from utils.logger import get_logger

logger = get_logger(__name__)

HEADERS = {"User-Agent": "ArguMind/1.0 (research assistant; contact: opensource)"}


@dataclass
class WebResult:
    title: str
    snippet: str
    url: str
    source: str = "web"
    year: int = 0
    reliability: float = 0.5   # 0–1, web < arxiv


# ── Wikipedia ─────────────────────────────────────────────────────────────────

def search_wikipedia(query: str, max_results: int = 3) -> list[WebResult]:
    """Search Wikipedia and return page summaries via the MediaWiki API."""
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": max_results,
        "format": "json",
        "utf8": 1,
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        hits = data.get("query", {}).get("search", [])
    except Exception as e:
        logger.warning(f"Wikipedia search failed: {e}")
        return []

    results = []
    for hit in hits[:max_results]:
        title = hit.get("title", "")
        snippet = (
            hit.get("snippet", "")
            .replace('<span class="searchmatch">', "")
            .replace("</span>", "")
        )
        page_url = (
            "https://en.wikipedia.org/wiki/"
            + urllib.parse.quote(title.replace(" ", "_"))
        )
        results.append(WebResult(
            title=title,
            snippet=snippet[:400],
            url=page_url,
            source="wikipedia",
            reliability=0.6,
        ))

    logger.info(f"Wikipedia: {len(results)} results for '{query[:50]}'")
    return results


# ── DuckDuckGo Instant Answers ────────────────────────────────────────────────

def search_duckduckgo(query: str) -> list[WebResult]:
    """Query DuckDuckGo Instant Answer API (no auth, no scraping)."""
    params = urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1,
    })
    url = f"https://api.duckduckgo.com/?{params}"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        logger.warning(f"DuckDuckGo failed: {e}")
        return []

    results = []

    # Main abstract
    abstract = data.get("AbstractText", "")
    if abstract:
        results.append(WebResult(
            title=data.get("Heading", query),
            snippet=abstract[:500],
            url=data.get("AbstractURL", ""),
            source="duckduckgo_abstract",
            reliability=0.55,
        ))

    # Related topic snippets
    for topic in data.get("RelatedTopics", [])[:3]:
        if isinstance(topic, dict) and topic.get("Text"):
            results.append(WebResult(
                title=topic.get("Text", "")[:60],
                snippet=topic.get("Text", "")[:300],
                url=topic.get("FirstURL", ""),
                source="duckduckgo_related",
                reliability=0.45,
            ))

    logger.info(f"DuckDuckGo: {len(results)} instant answers for '{query[:50]}'")
    return results


# ── Combined ──────────────────────────────────────────────────────────────────

def search_web(query: str, max_results: int = 5) -> list[WebResult]:
    """Wikipedia + DuckDuckGo combined, deduplicated by title."""
    results: list[WebResult] = []
    results.extend(search_wikipedia(query, max_results=3))
    results.extend(search_duckduckgo(query))

    # Deduplicate by title
    seen_titles: set[str] = set()
    unique = []
    for r in results:
        key = r.title.lower().strip()[:50]
        if key and key not in seen_titles:
            seen_titles.add(key)
            unique.append(r)

    logger.info(f"WebSearch total: {len(unique)} unique results")
    return unique[:max_results]


def web_result_to_agent_output(result: WebResult, agent_name: str = "web_agent") -> dict:
    """Convert WebResult to the standard agent output format."""
    return {
        "paper_id": f"web_{abs(hash(result.url)) % 100_000}",
        "title": result.title,
        "summary": result.snippet,
        "authors": [],
        "year": result.year,
        "source": result.url,
        "categories": ["web", result.source],
        "agent": agent_name,
        "confidence": result.reliability,
        "citations": [result.url],
        "claims": [],
    }
