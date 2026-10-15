import os
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()


def _get_tavily_client():
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key:
        tavily_key = tavily_key.strip().strip('"\'')
    if not tavily_key:
        return None
    return TavilyClient(api_key=tavily_key)


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns title, URLs, and snippets."""
    try:
        client = _get_tavily_client()
        if not client:
            return "Error: TAVILY_API_KEY is not configured. Please set TAVILY_API_KEY in your .env file."

        clean_query = query.strip()
        results = client.search(query=clean_query, max_results=5)

        items = results.get("results", [])
        if not items:
            return f"No search results found for: {clean_query}"

        out = []
        for r in items:
            title = r.get("title", "No Title")
            url = r.get("url", "")
            snippet = r.get("content", "")[:350].strip()
            out.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}")

        return "\n\n-----\n\n".join(out)
    except Exception as e:
        return f"Error performing web search: {str(e)}"


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        # Extract plain URL if model passed markdown link or quotes
        match = re.search(r"https?://[^\s\)\]\"'>]+", url.strip())
        target_url = match.group(0) if match else url.strip().strip('"\'')

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        response = requests.get(target_url, timeout=12, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "svg"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        # Collapse repeated whitespace
        clean_text = re.sub(r"\s+", " ", text).strip()

        if not clean_text:
            return f"No readable text content found on URL: {target_url}"

        return clean_text[:3500]
    except Exception as e:
        return f"Could not scrape URL ({url}): {str(e)}"