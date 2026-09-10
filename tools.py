"""Public API wrapper functions used only by Agent 2 (research_agent.py).

Each function:
  - never raises: network/timeout/parse errors are caught and returned as
    {"error": "<message>"} alongside an empty result list
  - returns a plain dict shaped {"query": str, "source": str, "raw_count": int,
    "results": [...], "error": str | None} so the caller can log it verbatim
  - each result item is normalized to: id_or_url, title, snippet, source_type

Notes on source selection (see README for the full explanation):
  - search_papers() uses the Semantic Scholar Graph API — public, no key.
  - search_pubmed() uses NCBI E-utilities — public, no key. Included as the
    optional second literature source since it's free and reliable.
  - search_patents() targets the USPTO Open Data Portal Patent Search API,
    the official successor to the old PatentsView "api.patentsview.org"
    endpoint (which has been retired and now redirects to a marketing page —
    verified during development). It requires a free API key from
    https://data.uspto.gov (env var USPTO_ODP_API_KEY). Without a key it
    degrades gracefully to an empty result + error flag, per the guardrail
    that no tool call may crash the agentic loop.
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET

import requests

TIMEOUT_SECONDS = 12

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
USPTO_ODP_PATENT_SEARCH_URL = "https://api.uspto.gov/api/v1/patent/applications/search"


def _empty(query: str, source: str, error: str | None = None) -> dict:
    return {"query": query, "source": source, "raw_count": 0, "results": [], "error": error}


def search_papers(query: str, limit: int = 5) -> dict:
    """Search academic literature via the Semantic Scholar Graph API.

    An unauthenticated shared IP (e.g. a sandbox/CI network) can hit Semantic
    Scholar's public rate limit quickly; SEMANTIC_SCHOLAR_API_KEY (free,
    https://www.semanticscholar.org/product/api#api-key-form) raises that
    limit substantially. One short backoff retry is attempted on a 429
    before giving up gracefully.
    """
    headers = {}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    try:
        resp = requests.get(
            SEMANTIC_SCHOLAR_URL,
            params={"query": query, "limit": limit, "fields": "title,abstract,url,year,externalIds"},
            headers=headers,
            timeout=TIMEOUT_SECONDS,
        )
        if resp.status_code == 429:
            time.sleep(2)
            resp = requests.get(
                SEMANTIC_SCHOLAR_URL,
                params={"query": query, "limit": limit, "fields": "title,abstract,url,year,externalIds"},
                headers=headers,
                timeout=TIMEOUT_SECONDS,
            )
        if resp.status_code == 429:
            return _empty(query, "semantic_scholar", "rate_limited (429) after retry")
        resp.raise_for_status()
        data = resp.json()
        papers = data.get("data", [])
        results = []
        for p in papers:
            url = p.get("url") or (
                f"https://doi.org/{p['externalIds']['DOI']}" if p.get("externalIds", {}).get("DOI") else None
            )
            results.append(
                {
                    "id_or_url": url or f"semanticscholar:{p.get('paperId', 'unknown')}",
                    "title": p.get("title") or "(untitled)",
                    "snippet": (p.get("abstract") or "")[:800],
                    "source_type": "paper",
                    "year": p.get("year"),
                }
            )
        return {"query": query, "source": "semantic_scholar", "raw_count": len(results), "results": results, "error": None}
    except requests.exceptions.Timeout:
        return _empty(query, "semantic_scholar", "timeout")
    except Exception as e:  # noqa: BLE001 - tool layer must never crash the loop
        return _empty(query, "semantic_scholar", f"{type(e).__name__}: {e}")


def search_pubmed(query: str, limit: int = 5) -> dict:
    """Search biomedical literature via NCBI PubMed E-utilities (esearch + efetch)."""
    try:
        search_resp = requests.get(
            PUBMED_ESEARCH_URL,
            params={"db": "pubmed", "term": query, "retmode": "json", "retmax": limit},
            timeout=TIMEOUT_SECONDS,
        )
        search_resp.raise_for_status()
        ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"query": query, "source": "pubmed", "raw_count": 0, "results": [], "error": None}

        fetch_resp = requests.get(
            PUBMED_EFETCH_URL,
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "rettype": "abstract"},
            timeout=TIMEOUT_SECONDS,
        )
        fetch_resp.raise_for_status()
        root = ET.fromstring(fetch_resp.content)

        results = []
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            title_el = article.find(".//ArticleTitle")
            abstract_parts = [t.text or "" for t in article.findall(".//AbstractText")]
            pmid = pmid_el.text if pmid_el is not None else None
            if pmid is None:
                continue
            results.append(
                {
                    "id_or_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "title": (title_el.text if title_el is not None else "(untitled)") or "(untitled)",
                    "snippet": " ".join(abstract_parts)[:800],
                    "source_type": "paper",
                }
            )
        return {"query": query, "source": "pubmed", "raw_count": len(results), "results": results, "error": None}
    except requests.exceptions.Timeout:
        return _empty(query, "pubmed", "timeout")
    except Exception as e:  # noqa: BLE001
        return _empty(query, "pubmed", f"{type(e).__name__}: {e}")


def search_patents(query: str, limit: int = 5) -> dict:
    """Search patent applications via the USPTO Open Data Portal Patent Search API.

    Requires env var USPTO_ODP_API_KEY (free, self-service key from
    https://data.uspto.gov). Degrades gracefully — never raises — if the key
    is missing, invalid, or the API is unreachable.
    """
    api_key = os.environ.get("USPTO_ODP_API_KEY")
    if not api_key:
        return _empty(query, "uspto_odp_patents", "missing USPTO_ODP_API_KEY env var")
    try:
        resp = requests.get(
            USPTO_ODP_PATENT_SEARCH_URL,
            params={"q": query, "limit": limit},
            headers={"X-Api-Key": api_key, "Accept": "application/json"},
            timeout=TIMEOUT_SECONDS,
        )
        if resp.status_code in (401, 403):
            return _empty(query, "uspto_odp_patents", f"auth failed ({resp.status_code}) - check USPTO_ODP_API_KEY")
        resp.raise_for_status()
        data = resp.json()

        # Response shape isn't pinned down against a live key at build time;
        # probe the plausible container keys defensively rather than crash.
        records = None
        for key in ("patentApplicationBag", "patentFileWrapperDataBag", "results", "items"):
            if isinstance(data.get(key), list):
                records = data[key]
                break
        if records is None:
            return _empty(query, "uspto_odp_patents", "unrecognized response shape")

        results = []
        for r in records[:limit]:
            title = (
                r.get("inventionTitle")
                or r.get("patentTitle")
                or r.get("title")
                or "(untitled)"
            )
            app_num = (
                r.get("applicationNumberText")
                or r.get("patentNumber")
                or r.get("applicationNumber")
                or r.get("id")
            )
            snippet = r.get("abstractText") or r.get("abstract") or ""
            results.append(
                {
                    "id_or_url": f"https://ppubs.uspto.gov/pubwebapp/#/search/{app_num}" if app_num else "(no id)",
                    "title": title,
                    "snippet": (snippet or "")[:800],
                    "source_type": "patent",
                }
            )
        return {"query": query, "source": "uspto_odp_patents", "raw_count": len(results), "results": results, "error": None}
    except requests.exceptions.Timeout:
        return _empty(query, "uspto_odp_patents", "timeout")
    except Exception as e:  # noqa: BLE001
        return _empty(query, "uspto_odp_patents", f"{type(e).__name__}: {e}")


TOOL_REGISTRY = {
    "search_papers": search_papers,
    "search_pubmed": search_pubmed,
    "search_patents": search_patents,
}
