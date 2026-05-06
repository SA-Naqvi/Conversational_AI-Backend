"""
PubMed API Tool — FREE, no API key required (higher rate limits with NCBI API key).

Searches the PubMed database of 35+ million biomedical literature citations.
Returns article titles, abstracts, and links for evidence-based medical responses.

Uses NCBI E-utilities API: https://www.ncbi.nlm.nih.gov/books/NBK25497/

Optional: Set NCBI_API_KEY env var for 10 req/s instead of 3 req/s rate limit.
"""
import os
import logging
import asyncio
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
import httpx

from tools.base import BaseTool

logger = logging.getLogger("medical_bot")

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")     # Optional; improves rate limits
REQUEST_TIMEOUT = 15.0


def _ncbi_params(extra: Dict) -> Dict:
    """Add common NCBI E-utilities parameters."""
    params = {"retmode": "json", "tool": "NurseGPT-E", "email": "medical-bot@example.com"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    params.update(extra)
    return params


async def _esearch(query: str, max_results: int = 5, date_range: str = "10") -> List[str]:
    """
    Search PubMed and return a list of PMIDs.
    date_range: number of years to look back (default 10).
    """
    params = _ncbi_params({
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "sort": "relevance",
        "usehistory": "n",
        "reldate": int(date_range) * 365,
        "datetype": "pdat",
    })
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            resp = await client.get(f"{EUTILS_BASE}/esearch.fcgi", params=params)
            resp.raise_for_status()
            data = resp.json()
            return data.get("esearchresult", {}).get("idlist", [])
        except httpx.TimeoutException:
            logger.warning("PubMed esearch timeout")
            return []
        except Exception as e:
            logger.error(f"PubMed esearch error: {e}")
            return []


async def _efetch_abstracts(pmids: List[str]) -> List[Dict[str, str]]:
    """
    Fetch article titles and abstracts for given PMIDs.
    Returns list of dicts: {pmid, title, abstract, authors, journal, year}.
    """
    if not pmids:
        return []

    params = _ncbi_params({
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    })
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            resp = await client.get(f"{EUTILS_BASE}/efetch.fcgi", params=params)
            resp.raise_for_status()
            return _parse_pubmed_xml(resp.text)
        except httpx.TimeoutException:
            logger.warning("PubMed efetch timeout")
            return []
        except Exception as e:
            logger.error(f"PubMed efetch error: {e}")
            return []


def _parse_pubmed_xml(xml_text: str) -> List[Dict[str, str]]:
    """Parse PubMed XML response and extract key fields."""
    articles = []
    try:
        root = ET.fromstring(xml_text)
        for article_set in root.findall(".//PubmedArticle"):
            pmid_el = article_set.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else "N/A"

            # Title
            title_el = article_set.find(".//ArticleTitle")
            title = "".join(title_el.itertext()) if title_el is not None else ""

            # Abstract (may have multiple AbstractText elements)
            abstract_parts = []
            for ab in article_set.findall(".//AbstractText"):
                label = ab.get("Label", "")
                text = "".join(ab.itertext())
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)

            # Authors
            authors = []
            for author in article_set.findall(".//Author")[:3]:
                last = author.findtext("LastName", "")
                fore = author.findtext("ForeName", "")
                if last:
                    authors.append(f"{last} {fore}".strip())
            author_str = ", ".join(authors) + (" et al." if len(authors) >= 3 else "")

            # Journal & year
            journal = article_set.findtext(".//Journal/Title", "")
            year = (
                article_set.findtext(".//PubDate/Year")
                or article_set.findtext(".//PubDate/MedlineDate", "")[:4]
            )

            if title:
                articles.append({
                    "pmid": pmid,
                    "title": title.strip(),
                    "abstract": abstract[:600].strip() if abstract else "",
                    "authors": author_str,
                    "journal": journal,
                    "year": year,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })
    except ET.ParseError as e:
        logger.error(f"PubMed XML parse error: {e}")
    return articles


class PubMedSearchTool(BaseTool):
    name = "search_pubmed"
    description = (
        "Search PubMed for peer-reviewed medical research articles relevant to the patient's question. "
        "Use this when the patient wants evidence-based information such as: "
        "'what does research say about knee replacement recovery?', "
        "'how long does swelling last after surgery?', "
        "'is this treatment effective?', or any time clinical evidence would strengthen the response. "
        "Returns article titles, abstracts, and PubMed links."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Medical search query. Be specific. Include relevant terms. "
                    "Examples: 'post-operative pain management after knee replacement', "
                    "'wound healing stages after abdominal surgery', "
                    "'deep vein thrombosis prevention after hip surgery'."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Number of articles to return (1–5, default 3).",
                "default": 3,
            },
            "years_back": {
                "type": "integer",
                "description": "How many years back to search (default 10, max 20).",
                "default": 10,
            },
        },
        "required": ["query"],
    }

    async def execute(
        self,
        query: str,
        max_results: int = 3,
        years_back: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        max_results = min(max(max_results, 1), 5)
        years_back = min(max(years_back, 1), 20)

        # Add clinical/medical filter to improve relevance
        enriched_query = f"({query})[Title/Abstract] AND (humans[MeSH Terms])"

        logger.info(f"PubMed search: {enriched_query}")
        pmids = await _esearch(enriched_query, max_results=max_results, date_range=str(years_back))

        if not pmids:
            # Retry with simpler query (no MeSH filter)
            pmids = await _esearch(query, max_results=max_results, date_range=str(years_back))

        if not pmids:
            return {
                "found": False,
                "query": query,
                "message": (
                    f"No PubMed articles found for '{query}'. "
                    "Try different search terms or ask about a specific surgical procedure or medication."
                ),
            }

        articles = await _efetch_abstracts(pmids)

        if not articles:
            return {
                "found": False,
                "query": query,
                "pmids_found": pmids,
                "message": "Found PMIDs but could not retrieve abstracts. Try again.",
            }

        return {
            "found": True,
            "query": query,
            "result_count": len(articles),
            "articles": articles,
            "source": "PubMed — National Library of Medicine",
            "search_tip": "Full articles available at the provided PubMed URLs.",
        }


class PubMedGetAbstractTool(BaseTool):
    name = "get_pubmed_abstract"
    description = (
        "Retrieve the full abstract of a specific PubMed article by its PMID. "
        "Use this after search_pubmed when the patient or clinician wants more "
        "detail about a specific research article."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pmid": {
                "type": "string",
                "description": "The PubMed ID (PMID) of the article. E.g., '38041234'.",
            },
        },
        "required": ["pmid"],
    }

    async def execute(self, pmid: str, **kwargs) -> Dict[str, Any]:
        articles = await _efetch_abstracts([pmid.strip()])
        if not articles:
            return {
                "found": False,
                "pmid": pmid,
                "message": f"Could not retrieve article PMID {pmid}. Verify the PMID at https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        article = articles[0]
        article["found"] = True
        return article
