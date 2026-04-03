from typing import Dict, List

import requests


def search_web_with_serpapi(api_key: str, query: str, top_k: int = 5) -> List[Dict[str, str]]:
    """Execute a SerpAPI search and normalize result fields.

    This function calls the Google engine through SerpAPI and returns a
    lightweight list with title, link, and snippet for prompt grounding and
    citation rendering in fallback mode.

    Args:
        api_key (str): SerpAPI authentication key.
        query (str): User question forwarded to search.
        top_k (int): Maximum number of normalized results to return.

    Returns:
        List[Dict[str, str]]: Search results with `title`, `link`, and `snippet`.

    Raises:
        requests.HTTPError: Raised when SerpAPI returns a non-2xx response.
    """
    response = requests.get(
        "https://serpapi.com/search.json",
        params={"engine": "google", "q": query, "api_key": api_key, "num": top_k},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    normalized_results: List[Dict[str, str]] = []
    for item in data.get("organic_results", [])[:top_k]:
        normalized_results.append(
            {
                "title": item.get("title", "Untitled"),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
        )
    return normalized_results

