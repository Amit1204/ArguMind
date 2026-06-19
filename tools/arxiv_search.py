"""
arXiv search tool — fetches papers via the arXiv API.
Returns structured paper metadata + abstracts.
"""
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom",
      "arxiv": "http://arxiv.org/schemas/atom"}


@dataclass
class ArxivPaper:
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    year: int
    url: str
    categories: list[str]


def search_arxiv(query: str, max_results: int = None) -> list[ArxivPaper]:
    """Search arXiv and return structured paper objects."""
    max_results = max_results or settings.arxiv_max_results
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"
    logger.info(f"arXiv query: {query} (max={max_results})")

    try:
        with urllib.request.urlopen(url, timeout=settings.arxiv_timeout) as resp:
            xml_data = resp.read()
    except Exception as e:
        logger.error(f"arXiv request failed: {e}")
        return []

    root = ET.fromstring(xml_data)
    papers = []

    for entry in root.findall("atom:entry", NS):
        try:
            raw_id = entry.find("atom:id", NS).text.strip()
            paper_id = raw_id.split("/abs/")[-1].replace("/", "_")
            title = entry.find("atom:title", NS).text.strip().replace("\n", " ")
            abstract = entry.find("atom:summary", NS).text.strip().replace("\n", " ")
            authors = [
                a.find("atom:name", NS).text.strip()
                for a in entry.findall("atom:author", NS)
            ]
            published = entry.find("atom:published", NS).text
            year = int(published[:4]) if published else 0
            categories = [
                c.attrib.get("term", "")
                for c in entry.findall("arxiv:primary_category", NS)
            ]
            papers.append(ArxivPaper(
                paper_id=paper_id,
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                url=raw_id,
                categories=categories,
            ))
        except Exception as e:
            logger.warning(f"Failed to parse arXiv entry: {e}")
            continue

    logger.info(f"arXiv returned {len(papers)} papers")
    return papers


def paper_to_agent_output(paper: ArxivPaper, agent_name: str = "arxiv_agent") -> dict:
    """Convert ArxivPaper to the standard agent output format."""
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "summary": paper.abstract,
        "authors": paper.authors,
        "year": paper.year,
        "source": paper.url,
        "categories": paper.categories,
        "agent": agent_name,
        "confidence": 0.75,  # default for preprints
        "citations": [paper.url],
        "claims": [],         # filled by EvidenceNormalizer
    }
