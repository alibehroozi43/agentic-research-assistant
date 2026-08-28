"""Research retrieval tools used by the agentic research assistant.

Provenance
----------
The initial version of these retrieval helpers was adapted from starter/reference
code supplied for a course exercise. For this portfolio repository, the module
has been simplified and refactored to keep only the tools actually used by the
agent (arXiv and Tavily), while the agent orchestration, tool loop, reflection
workflow, and report-generation pipeline live in ``main.py``.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any

import requests
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

_ARXIV_API_URL = "https://export.arxiv.org/api/query"
_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": os.getenv(
            "ARXIV_USER_AGENT",
            "agentic-research-assistant/1.0",
        )
    }
)


def arxiv_search_tool(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search arXiv for papers matching a research query."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }

    try:
        response = _SESSION.get(_ARXIV_API_URL, params=params, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        return [{"error": f"arXiv request failed: {exc}"}]

    try:
        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        results: list[dict[str, Any]] = []

        for entry in root.findall("atom:entry", ns):
            title_node = entry.find("atom:title", ns)
            published_node = entry.find("atom:published", ns)
            id_node = entry.find("atom:id", ns)
            summary_node = entry.find("atom:summary", ns)

            authors = [
                author.find("atom:name", ns).text.strip()
                for author in entry.findall("atom:author", ns)
                if author.find("atom:name", ns) is not None
                and author.find("atom:name", ns).text
            ]

            pdf_url = None
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href")
                    break

            results.append(
                {
                    "title": title_node.text.strip() if title_node is not None and title_node.text else "",
                    "authors": authors,
                    "published": published_node.text[:10] if published_node is not None and published_node.text else "",
                    "url": id_node.text.strip() if id_node is not None and id_node.text else "",
                    "summary": summary_node.text.strip() if summary_node is not None and summary_node.text else "",
                    "pdf_url": pdf_url,
                }
            )

        return results
    except (ET.ParseError, AttributeError) as exc:
        return [{"error": f"arXiv response parsing failed: {exc}"}]


def tavily_search_tool(
    query: str,
    max_results: int = 5,
    include_images: bool = False,
) -> list[dict[str, Any]]:
    """Search the web with Tavily and return compact LLM-friendly results."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return [{"error": "TAVILY_API_KEY is not configured."}]

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            include_images=include_images,
        )

        results = [
            {
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "url": item.get("url", ""),
            }
            for item in response.get("results", [])
        ]

        if include_images:
            results.extend(
                {"image_url": image_url}
                for image_url in response.get("images", [])
            )

        return results
    except Exception as exc:  # third-party client errors vary by version
        return [{"error": f"Tavily search failed: {exc}"}]


TOOL_MAPPING = {
    "tavily_search_tool": tavily_search_tool,
    "arxiv_search_tool": arxiv_search_tool,
}
