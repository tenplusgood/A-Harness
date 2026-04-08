"""
web-search skill () — text-only

This skill implements a web search tool that:
1. Uses LLM to analyze tasks and generate search queries
2. Performs actual web searches (DuckDuckGo, ConceptNet, Google Scholar, Wikipedia)
3. Crawls and extracts text content from web pages
4. Uses LLM to analyze crawled text and extract key affordance information
5. Saves search results and analysis to skills/web-search/reference/

Search Engine Priority (all free, no API key required):
  1. DuckDuckGo            (via duckduckgo_search library, no key needed)
  2. ConceptNet             (REST API, free knowledge graph for affordance reasoning)
  3. Google Scholar         (web scraping, free academic search)
  4. Wikipedia via Proxy    (AllOrigins proxy, bypasses HPC firewall)
  5. Wikipedia direct       (API, may be blocked by HPC firewall)

Firewall bypass:
  - Wikipedia: uses api.allorigins.win as CORS proxy to bypass firewall blocking

If any step fails, the skill returns an `error` field instead of raising.
"""

from __future__ import annotations

import json
import os
import re
import time
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, quote

try:
    import requests
except ImportError as e:
    raise ImportError(
        " requests，：pip install requests"
    ) from e

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
    print("Warning: BeautifulSoup not installed. Web content extraction will be limited.")
    print("Install with: pip install beautifulsoup4")

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None
        print("Warning: ddgs not installed. DuckDuckGo search will be unavailable.")
        print("Install with: pip install -U ddgs")


# Reference directory: skills/web-search/reference/
REFERENCE_DIR = Path(__file__).parent.parent / "reference"

# AllOrigins proxy URL for bypassing firewall
ALLORIGINS_PROXY = "https://api.allorigins.win/get"


def _load_api_config() -> Dict[str, Any]:
    """Load API configuration from config.py or environment variables.

    Only LLM API config is needed — all search engines are free and keyless.
    """
    config = {}
    try:
        import sys
        from pathlib import Path as _P
        config_path = _P(__file__).resolve().parent.parent.parent / "config.py"
        if config_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", config_path)
            if spec and spec.loader:
                cfg = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cfg)
                config["API_KEY"] = getattr(cfg, "API_KEY", None)
                config["API_BASE_URL"] = getattr(cfg, "API_BASE_URL", None)
                config["DEFAULT_MODEL"] = getattr(cfg, "DEFAULT_MODEL", "gpt-4o")
    except Exception:
        pass

    # Override with environment variables if present
    config["API_KEY"] = os.getenv("API_KEY", config.get("API_KEY"))
    config["API_BASE_URL"] = os.getenv("API_BASE_URL", config.get("API_BASE_URL"))
    config["DEFAULT_MODEL"] = os.getenv("DEFAULT_MODEL", config.get("DEFAULT_MODEL", "gpt-4o"))

    return config


# LLM API Helpers

def _call_llm_api(
    messages: list,
    model: str,
    api_key: Optional[str],
    api_base_url: Optional[str],
    temperature: float = 0.3,
    max_tokens: int = 1000,
) -> Dict[str, Any]:
    """Call LLM API for intelligent analysis."""
    if not api_base_url:
        return {"error": "API_BASE_URL not configured"}
    if not api_key:
        return {"error": "API_KEY not configured"}

    url = f"{api_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "API call timed out after 60 seconds"}
    except requests.exceptions.ConnectionError as e:
        return {"error": f"Connection error: {str(e)[:200]}. Please check your network connection and API_BASE_URL."}
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP error {e.response.status_code}: {str(e)}"
        try:
            error_body = e.response.json()
            error_msg += f"\nResponse: {json.dumps(error_body, indent=2)}"
        except Exception:
            error_msg += f"\nResponse text: {e.response.text[:500]}"
        return {"error": error_msg}
    except requests.exceptions.RequestException as e:
        return {"error": f"API call failed: {str(e)}"}


def _generate_search_queries(
    question: str,
    task: Optional[str],
    api_key: Optional[str],
    api_base_url: Optional[str],
    model: str,
    dynamic_params: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Use LLM to generate search queries from the task, guided by the
    decision model's search strategy (dynamic_params).

    dynamic_params （ task ）：
        - search_focus: （ "cup handle grasping ergonomics"）
        - knowledge_type: （ "structural", "functional", "spatial"）
        - search_depth: （ "quick", "thorough"）
        - target_part: （ "handle"）
        - interaction_type: （ "power grasp", "pinch grip"）
        - 
    """
    direction_block = ""
    if dynamic_params:
        strategy_lines = []
        if "search_focus" in dynamic_params:
            strategy_lines.append(f"- SEARCH FOCUS: {dynamic_params['search_focus']}")
        if "interaction_type" in dynamic_params:
            strategy_lines.append(f"- INTERACTION TYPE: {dynamic_params['interaction_type']}")
        if "target_part" in dynamic_params:
            strategy_lines.append(f"- TARGET PART HYPOTHESIS: {dynamic_params['target_part']}")
        if "knowledge_type" in dynamic_params:
            strategy_lines.append(f"- KNOWLEDGE NEEDED: {dynamic_params['knowledge_type']}")
        for key, value in dynamic_params.items():
            if key not in ("search_focus", "interaction_type", "target_part",
                           "knowledge_type", "search_depth"):
                readable_key = key.replace("_", " ").upper()
                strategy_lines.append(f"- {readable_key}: {value}")

        if strategy_lines:
            direction_block = (
                "\n\nThe decision model has analyzed the task and provided this SEARCH STRATEGY:\n"
                + "\n".join(strategy_lines)
                + "\n\nGenerate queries that are tightly aligned with this strategy. "
                "Each query should target a different aspect of the strategy.\n"
            )

    prompt = f"""Generate 2-3 effective web search queries for the following affordance task.

Task/Question: {task or question}
Specific question: {question}
{direction_block}
IMPORTANT: Generate queries that a person would actually type into Google to find useful information about this specific object interaction. Include the object name and the type of interaction.

Output ONLY a JSON array of search query strings, no additional text:
["query 1", "query 2", "query 3"]"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a search query generation expert specializing in object affordance "
                "and human-object interaction. Generate specific, targeted web search queries "
                "based on the task and the search strategy provided by the decision model. "
                "Queries should help find: (1) how humans interact with the object, "
                "(2) which part is used for the described action, "
                "(3) reference images of the interaction."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    result = _call_llm_api(messages, model, api_key, api_base_url, temperature=0.5, max_tokens=200)

    if "error" in result:
        # Fallback: use search_focus if available, else raw task
        fallback = dynamic_params.get("search_focus") if dynamic_params else None
        return [fallback or f"{task or question}"]

    try:
        response_text = result["choices"][0]["message"]["content"]
        json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if json_match:
            queries = json.loads(json_match.group(0))
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries
    except Exception:
        pass

    fallback = dynamic_params.get("search_focus") if dynamic_params else None
    return [fallback or f"{task or question}"]



# Search Engine Implementations

def _search_duckduckgo(
    query: str,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """Search using DuckDuckGo via duckduckgo_search library (no API key needed)."""
    results: List[Dict[str, Any]] = []

    if not DDGS:
        print("     ⚠️  ddgs (duckduckgo_search) not installed. Install with: pip install -U ddgs")
        return results

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                    "source": "duckduckgo",
                })

        if results:
            print(f"     ✅ DuckDuckGo: {len(results)} results")
    except Exception as e:
        print(f"     ⚠️  DuckDuckGo failed: {type(e).__name__}: {str(e)[:80]}")

    return results



def _search_conceptnet(
    query: str,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search ConceptNet knowledge graph for affordance-relevant relationships.

    ConceptNet provides structured knowledge about objects and their affordances
    via relations like UsedFor, CapableOf, HasProperty, AtLocation, etc.
    Free API, no key needed: https://api.conceptnet.io/
    """
    results: List[Dict[str, Any]] = []

    try:
        # Extract key terms from query for ConceptNet lookup
        terms = query.lower().strip().split()
        stop_words = {
            "the", "a", "an", "is", "are", "how", "to", "do", "you",
            "what", "which", "part", "of", "for", "with", "can", "that",
            "this", "find", "identify", "where", "should", "i", "be",
            "and", "or", "in", "on", "at", "by", "from", "it", "its",
        }
        key_terms = [t for t in terms if t not in stop_words and len(t) > 1]

        # Build concept phrases to try
        concepts_to_try = []
        if len(key_terms) >= 2:
            concepts_to_try.append("_".join(key_terms[:3]))
        for t in key_terms[:4]:
            concepts_to_try.append(t)

        if not concepts_to_try:
            return results

        seen_edges: set = set()

        # Relevant relations for affordance tasks
        affordance_relations = {
            "/r/UsedFor", "/r/CapableOf", "/r/HasProperty",
            "/r/AtLocation", "/r/PartOf", "/r/HasA",
            "/r/MadeOf", "/r/IsA", "/r/MannerOf",
            "/r/Causes", "/r/HasPrerequisite", "/r/ReceivesAction",
        }

        for concept in concepts_to_try:
            if len(results) >= max_results:
                break

            # Try direct API first, then AllOrigins proxy as fallback
            data = None
            api_url = f"https://api.conceptnet.io/c/en/{quote(concept)}?limit=20"

            try:
                response = requests.get(api_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
            except Exception:
                pass

            # Fallback: AllOrigins proxy (bypasses firewall / handles 502)
            if data is None:
                try:
                    proxy_url = f"{ALLORIGINS_PROXY}?url={quote(api_url, safe='')}"
                    proxy_resp = requests.get(proxy_url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
                    if proxy_resp.status_code == 200:
                        proxy_data = proxy_resp.json()
                        contents_str = proxy_data.get("contents", "")
                        if contents_str:
                            data = json.loads(contents_str)
                except Exception:
                    pass

            if data is None:
                continue
            for edge in data.get("edges", []):
                rel = edge.get("rel", {}).get("@id", "")

                # Filter for affordance-relevant relations
                if rel not in affordance_relations:
                    continue

                # Only keep English-language edges
                start_lang = edge.get("start", {}).get("language", "en")
                end_lang = edge.get("end", {}).get("language", "en")
                if start_lang != "en" or end_lang != "en":
                    continue

                start_label = edge.get("start", {}).get("label", "")
                end_label = edge.get("end", {}).get("label", "")
                rel_label = edge.get("rel", {}).get("label", "")
                weight = edge.get("weight", 0)
                surface_text = edge.get("surfaceText", "")

                # Deduplicate
                edge_key = f"{start_label}-{rel_label}-{end_label}"
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                # Build a readable snippet
                snippet = surface_text if surface_text else f"{start_label} {rel_label} {end_label}"

                results.append({
                    "title": f"[ConceptNet] {start_label} → {rel_label} → {end_label}",
                    "url": f"https://conceptnet.io{edge.get('@id', '')}",
                    "snippet": f"{snippet} (weight: {weight:.1f})",
                    "source": "conceptnet",
                    "weight": weight,
                    "relation": rel_label,
                    "start": start_label,
                    "end": end_label,
                })

                if len(results) >= max_results:
                    break

        # Sort by weight (higher = more confident)
        results.sort(key=lambda x: x.get("weight", 0), reverse=True)
        results = results[:max_results]

        if results:
            print(f"     ✅ ConceptNet: {len(results)} affordance-relevant edges")
    except requests.exceptions.Timeout:
        print(f"     ⚠️  ConceptNet API timed out")
    except requests.exceptions.ConnectionError as e:
        print(f"     ⚠️  ConceptNet connection error: {str(e)[:80]}")
    except Exception as e:
        print(f"     ⚠️  ConceptNet failed: {type(e).__name__}: {str(e)[:80]}")

    return results


def _search_google_web(
    query: str,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """Search using Google web search (free scraping, no API key needed).

    Scrapes Google search results directly. Uses multiple parsing strategies
    to handle different Google HTML structures.

    NOTE: Google may return JS-only pages or CAPTCHAs in some environments.
    This function degrades gracefully and returns empty results in such cases.
    """
    results: List[Dict[str, Any]] = []

    try:
        search_url = f"https://www.google.com/search?q={quote(query)}&hl=en&num={max_results + 5}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()

        if not BeautifulSoup:
            print("     ⚠️  BeautifulSoup not installed, cannot parse Google results")
            return results

        soup = BeautifulSoup(response.text, 'html.parser')

        # Strategy 1: Google organic results in <div class="g"> blocks
        for div in soup.find_all('div', class_='g', limit=max_results * 2):
            a_tag = div.find('a', href=True)
            if not a_tag:
                continue
            url = a_tag.get('href', '')
            if not url.startswith('http'):
                continue

            h3 = div.find('h3')
            title = h3.get_text(strip=True) if h3 else ""
            if not title:
                continue

            # Snippet extraction with multiple fallbacks
            snippet = ""
            for span in div.find_all('span'):
                text = span.get_text(strip=True)
                if len(text) > 40:
                    snippet = text
                    break
            if not snippet:
                all_text = div.get_text(separator=' ', strip=True)
                if title in all_text:
                    snippet = all_text[all_text.index(title) + len(title):].strip()[:300]

            results.append({
                "title": title, "url": url,
                "snippet": snippet[:300], "source": "google_web",
            })
            if len(results) >= max_results:
                break

        # Strategy 2: find <a> tags containing <h3> (fallback for different layouts)
        if not results:
            for a_tag in soup.find_all('a', href=True):
                h3 = a_tag.find('h3')
                if not h3:
                    continue
                href = a_tag.get('href', '')
                # Handle Google redirect URLs (/url?q=...)
                if href.startswith('/url?'):
                    import urllib.parse as _up
                    parsed = _up.urlparse(href)
                    params = _up.parse_qs(parsed.query)
                    href = params.get('q', [href])[0]
                if not href.startswith('http') or 'google.' in href:
                    continue

                title = h3.get_text(strip=True)
                if not title:
                    continue

                results.append({
                    "title": title, "url": href,
                    "snippet": "", "source": "google_web",
                })
                if len(results) >= max_results:
                    break

        if results:
            print(f"     ✅ Google Web: {len(results)} results")
        else:
            # JS-only page or CAPTCHA — silent degradation
            pass
    except requests.exceptions.Timeout:
        print(f"     ⚠️  Google Web timed out")
    except requests.exceptions.ConnectionError as e:
        print(f"     ⚠️  Google Web connection error: {str(e)[:80]}")
    except Exception as e:
        print(f"     ⚠️  Google Web failed: {type(e).__name__}: {str(e)[:80]}")

    return results


def _search_google_scholar(
    query: str,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """Search using Google Scholar (web scraping, accessible from HPC)."""
    results: List[Dict[str, Any]] = []

    try:
        search_url = f"https://scholar.google.com/scholar?q={quote(query)}&hl=en"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()

        if not BeautifulSoup:
            return results

        soup = BeautifulSoup(response.text, 'html.parser')

        # Google Scholar: div.gs_ri contains each result
        for item in soup.find_all('div', class_='gs_ri', limit=max_results):
            h3 = item.find('h3')
            a_tag = h3.find('a') if h3 else None
            snippet_div = item.find('div', class_='gs_rs')

            title = h3.get_text(strip=True) if h3 else ""
            url = a_tag.get('href', '') if a_tag else ""
            snippet = snippet_div.get_text(strip=True) if snippet_div else ""

            # Clean title (remove [PDF], [HTML] etc. prefixes)
            title = re.sub(r'^\[.*?\]\s*', '', title)

            if title and url and url.startswith('http'):
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet[:300],
                    "source": "google_scholar",
                })

        if results:
            print(f"     ✅ Google Scholar: {len(results)} results")
    except requests.exceptions.Timeout:
        print(f"     ⚠️  Google Scholar timed out")
    except requests.exceptions.ConnectionError as e:
        print(f"     ⚠️  Google Scholar connection error: {str(e)[:80]}")
    except Exception as e:
        print(f"     ⚠️  Google Scholar failed: {type(e).__name__}: {str(e)[:80]}")

    return results


def _search_wikipedia_via_proxy(
    query: str,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search Wikipedia via AllOrigins proxy to bypass HPC firewall.

    AllOrigins (api.allorigins.win) acts as a CORS/HTTP proxy, fetching
    the Wikipedia API response on our behalf and returning it.
    """
    results: List[Dict[str, Any]] = []

    try:
        # Build the Wikipedia API URL
        wiki_api_url = (
            f"https://en.wikipedia.org/w/api.php?"
            f"action=query&list=search&srsearch={quote(query)}"
            f"&srlimit={max_results}&format=json"
        )
        proxy_url = f"{ALLORIGINS_PROXY}?url={quote(wiki_api_url, safe='')}"

        response = requests.get(proxy_url, timeout=8, headers={
            "User-Agent": "Mozilla/5.0",
        })
        response.raise_for_status()

        proxy_data = response.json()
        contents_str = proxy_data.get("contents", "")
        if not contents_str:
            print(f"     ⚠️  Wikipedia proxy returned empty contents")
            return results

        wiki_data = json.loads(contents_str)

        if "query" in wiki_data and "search" in wiki_data["query"]:
            for item in wiki_data["query"]["search"]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                # Clean HTML tags from snippet
                if BeautifulSoup:
                    snippet = BeautifulSoup(snippet, 'html.parser').get_text()
                else:
                    snippet = re.sub(r'<[^>]+>', '', snippet)

                url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                results.append({
                    "title": f"[Wikipedia] {title}",
                    "url": url,
                    "snippet": snippet,
                    "source": "wikipedia_proxy",
                })

            if results:
                print(f"     ✅ Wikipedia (via proxy): {len(results)} results")

    except requests.exceptions.Timeout:
        print(f"     ⚠️  Wikipedia proxy timed out")
    except requests.exceptions.ConnectionError as e:
        print(f"     ⚠️  Wikipedia proxy connection error: {str(e)[:80]}")
    except json.JSONDecodeError as e:
        print(f"     ⚠️  Wikipedia proxy JSON parse error: {str(e)[:80]}")
    except Exception as e:
        print(f"     ⚠️  Wikipedia proxy failed: {type(e).__name__}: {str(e)[:80]}")

    return results


def _fetch_wikipedia_content_via_proxy(title: str, max_length: int = 2000) -> str:
    """
    Fetch Wikipedia page content (intro) via AllOrigins proxy.
    Returns the plain-text extract of the page intro.
    """
    try:
        wiki_api_url = (
            f"https://en.wikipedia.org/w/api.php?"
            f"action=query&titles={quote(title)}&prop=extracts"
            f"&exintro=1&explaintext=1&format=json"
        )
        proxy_url = f"{ALLORIGINS_PROXY}?url={quote(wiki_api_url, safe='')}"

        response = requests.get(proxy_url, timeout=8, headers={
            "User-Agent": "Mozilla/5.0",
        })
        response.raise_for_status()

        proxy_data = response.json()
        contents_str = proxy_data.get("contents", "")
        wiki_data = json.loads(contents_str)

        pages = wiki_data.get("query", {}).get("pages", {})
        for _pid, page in pages.items():
            extract = page.get("extract", "")
            if extract:
                return extract[:max_length]

    except Exception as e:
        print(f"     ⚠️  Wikipedia content fetch failed: {type(e).__name__}: {str(e)[:60]}")

    return ""


def _search_wikipedia_direct(
    query: str,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """Search using Wikipedia API directly (may be blocked by firewall)."""
    results: List[Dict[str, Any]] = []

    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json",
        }
        response = requests.get(search_url, params=params, timeout=6)
        response.raise_for_status()
        api_result = response.json()

        if "query" in api_result and "search" in api_result["query"]:
            for item in api_result["query"]["search"]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                if BeautifulSoup:
                    snippet = BeautifulSoup(snippet, 'html.parser').get_text()
                url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                results.append({
                    "title": f"[Wikipedia] {title}",
                    "url": url,
                    "snippet": snippet,
                    "source": "wikipedia_direct",
                })
            if results:
                print(f"     ✅ Wikipedia (direct): {len(results)} results")
    except requests.exceptions.Timeout:
        print(f"     ⚠️  Wikipedia direct timed out (firewall blocked)")
    except requests.exceptions.ConnectionError:
        print(f"     ⚠️  Wikipedia direct connection error (firewall blocked)")
    except Exception as e:
        print(f"     ⚠️  Wikipedia direct failed: {type(e).__name__}: {str(e)[:80]}")

    return results




# Image Extraction from Crawled Pages


# Main Search Orchestrator

def _perform_web_search(
    query: str,
    max_results: int = 5,
    search_engines: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Perform web search using multiple free engines in priority order.

    All engines are free and require no API keys.

    Default priority:
      duckduckgo → google_web → conceptnet → google_scholar → wikipedia_proxy → wikipedia_direct
    """
    if search_engines is None:
        search_engines = [
            "duckduckgo", "google_web", "conceptnet", "google_scholar",
            "wikipedia_proxy", "wikipedia_direct",
        ]

    all_results: List[Dict[str, Any]] = []
    seen_urls: set = set()

    def _add_results(new_results: List[Dict[str, Any]]):
        for r in new_results:
            u = r.get("url", "")
            if u and u not in seen_urls:
                seen_urls.add(u)
                all_results.append(r)

    # Engine dispatch table (all free, no API keys)
    engine_funcs = {
        "duckduckgo": lambda: _search_duckduckgo(query, max_results),
        "google_web": lambda: _search_google_web(query, max_results),
        "conceptnet": lambda: _search_conceptnet(query, max_results),
        "google_scholar": lambda: _search_google_scholar(query, max_results),
        "wikipedia_proxy": lambda: _search_wikipedia_via_proxy(query, max_results),
        "wikipedia_direct": lambda: _search_wikipedia_direct(query, max_results),
    }

    # Try engines in priority order; skip slow/unreliable ones once we have enough
    # DuckDuckGo is the fastest/most reliable — try it first
    # Wikipedia/Scholar may timeout behind firewalls — only try if still need results
    for idx, engine_name in enumerate(search_engines):
        # Skip remaining engines if we already have plenty of results
        if len(all_results) >= max_results:
            break
        func = engine_funcs.get(engine_name)
        if func:
            try:
                engine_results = func()
                _add_results(engine_results)
            except Exception as e:
                print(f"     ⚠️  {engine_name} unexpected error: {e}")

    if all_results:
        return all_results[:max_results]

    print(f"     ⚠️  No search results found from any engine")
    return []


# Web Content Extraction

def _extract_web_content(
    url: str,
    max_length: int = 2000,
    return_html: bool = False,
) -> Any:
    """
    Extract text content from a web page.

    Args:
        url: URL to fetch
        max_length: max text length
        return_html: if True, returns (text, raw_html) tuple for image extraction

    Returns:
        str (text only) or Tuple[str, str] (text, raw_html) if return_html=True
    """
    # For Wikipedia URLs, use the proxy + API for cleaner content
    if "wikipedia.org/wiki/" in url:
        title = url.split("/wiki/")[-1]
        content = _fetch_wikipedia_content_via_proxy(title, max_length)
        if content:
            return (content, "") if return_html else content
        # Fall through to normal extraction if proxy fails

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()

        raw_html = response.text

        if BeautifulSoup:
            soup = BeautifulSoup(raw_html, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text = soup.get_text(separator=' ', strip=True)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            text = text[:max_length]
        else:
            text = raw_html[:max_length]

        return (text, raw_html) if return_html else text
    except Exception as e:
        error_msg = f"[Error fetching content: {str(e)[:100]}]"
        return (error_msg, "") if return_html else error_msg


# LLM Content Analysis

def _analyze_search_results_with_llm(
    query: str,
    search_results: List[Dict[str, Any]],
    crawled_content: Dict[str, str],
    question: str,
    task: Optional[str],
    api_key: Optional[str],
    api_base_url: Optional[str],
    model: str,
    dynamic_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Use LLM to analyze search results and crawled text content (text-only, no images).

    dynamic_params ：
        - search_focus / knowledge_type:  LLM 
        - : 
    """

    content_summary = f"Search Query: {query}\n\n"
    content_summary += "Search Results:\n"
    for i, result in enumerate(search_results[:5], 1):
        source = result.get('source', 'unknown')
        content_summary += f"{i}. [{source}] {result.get('title', 'N/A')}\n"
        content_summary += f"   URL: {result.get('url', 'N/A')}\n"
        content_summary += f"   Snippet: {result.get('snippet', 'N/A')[:200]}\n\n"

    if crawled_content:
        content_summary += "\nCrawled Web Content:\n"
        for url, content in list(crawled_content.items())[:3]:
            content_summary += f"From {url}:\n{content[:1000]}...\n\n"

    focus_directive = ""
    if dynamic_params:
        focus_parts = []
        if "search_focus" in dynamic_params:
            focus_parts.append(f"- ANALYSIS FOCUS: {dynamic_params['search_focus']}")
        if "target_part" in dynamic_params:
            focus_parts.append(f"- HYPOTHESIZED TARGET PART: {dynamic_params['target_part']} — verify or refute this hypothesis")
        if "interaction_type" in dynamic_params:
            focus_parts.append(f"- EXPECTED INTERACTION: {dynamic_params['interaction_type']}")
        if "knowledge_type" in dynamic_params:
            kt = dynamic_params["knowledge_type"]
            if kt == "structural":
                focus_parts.append("- KNOWLEDGE FOCUS: structural — describe the physical layout of parts")
            elif kt == "functional":
                focus_parts.append("- KNOWLEDGE FOCUS: functional — explain how the object works")
            elif kt == "spatial":
                focus_parts.append("- KNOWLEDGE FOCUS: spatial — describe where the target part is located")
            else:
                focus_parts.append(f"- KNOWLEDGE FOCUS: {kt}")
        for key, value in dynamic_params.items():
            if key not in ("search_focus", "target_part", "interaction_type",
                           "knowledge_type", "search_depth"):
                readable_key = key.replace("_", " ")
                focus_parts.append(f"- {readable_key}: {value}")

        if focus_parts:
            focus_directive = (
                "\n\nThe decision model has directed the analysis to focus on:\n"
                + "\n".join(focus_parts)
                + "\nAlign your analysis with these directions. If a target part hypothesis "
                "is given, use the evidence to confirm or correct it.\n"
            )

    system_prompt = f"""You are an expert at analyzing web search results to understand how humans physically interact with objects.

Given search results and crawled web content, analyze them to extract:
- affordance_name: the physical action (pressing, turning, gripping, etc.)
- part_name: the specific part a person interacts with (handle, lever, button, etc.)
- object_name: the object involved
- reasoning: concise reasoning combining textual evidence
{focus_directive}
Output JSON:
{{
    "affordance_name": "the physical action",
    "part_name": "the specific interacted part",
    "object_name": "the object name",
    "reasoning": "concise reasoning (2-3 sentences)"
}}"""

    user_prompt = f"""Original Question: {question}
Original Task: {task or question}

{content_summary}

Analyze the above content and extract the key information."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    result = _call_llm_api(messages, model, api_key, api_base_url, temperature=0.3, max_tokens=1500)
    return result


# Reference Saving

def _save_to_reference(
    task: Optional[str],
    question: str,
    search_queries: List[str],
    search_results: List[Dict[str, Any]],
    crawled_content: Dict[str, str],
    crawled_urls: List[str],
    analysis_result: Dict[str, Any],
    final_result: Dict[str, Any],
) -> Optional[str]:
    """
    Save search results, images, and analysis to skills/web-search/reference/.

    Creates a human-readable markdown file + a JSON file for each query.
    Returns the path to the saved markdown file, or None on error.
    """
    try:
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

        # Build a safe filename from the task
        raw_name = task or question or "unknown"
        safe_name = re.sub(r'[^\w\s-]', '', raw_name)[:60].strip().replace(' ', '_')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{safe_name}_{timestamp}"

        md_path = REFERENCE_DIR / f"{base_name}.md"
        json_path = REFERENCE_DIR / f"{base_name}.json"

        # ----- Markdown (human-readable) -----
        lines: List[str] = []
        lines.append(f"# Web Search Reference\n")
        lines.append(f"**Task:** {task or 'N/A'}\n")
        lines.append(f"**Question:** {question}\n")
        lines.append(f"**Time:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("")

        # Search engines used
        sources = {}
        for res in search_results:
            src = res.get('source', 'unknown')
            sources[src] = sources.get(src, 0) + 1
        if sources:
            lines.append("## 🔧 Search Engines Used\n")
            for src, count in sources.items():
                lines.append(f"- **{src}**: {count} result(s)")
            lines.append("")

        # Search queries
        lines.append("## 🔍 Search Queries\n")
        for i, q in enumerate(search_queries, 1):
            lines.append(f"{i}. {q}")
        lines.append("")

        # Search results
        lines.append("## 📊 Search Results\n")
        if search_results:
            for i, res in enumerate(search_results, 1):
                source = res.get('source', 'unknown')
                lines.append(f"### Result {i} [{source}]\n")
                lines.append(f"- **Title:** {res.get('title', 'N/A')}")
                lines.append(f"- **URL:** {res.get('url', 'N/A')}")
                snippet = res.get('snippet', '')
                if snippet:
                    lines.append(f"- **Snippet:** {snippet[:300]}")
                lines.append("")
        else:
            lines.append("*No search results found.*\n")

        # Crawled content
        lines.append("## 📥 Crawled Web Content\n")
        if crawled_content:
            for url, content in crawled_content.items():
                lines.append(f"### {url}\n")
                lines.append("```")
                lines.append(content[:2000])
                lines.append("```\n")
        else:
            lines.append("*No web content crawled.*\n")

        # LLM Analysis
        lines.append("## 🤖 LLM Analysis Result\n")
        lines.append(f"- **affordance_name:** {final_result.get('affordance_name', 'N/A')}")
        lines.append(f"- **part_name:** {final_result.get('part_name', 'N/A')}")
        lines.append(f"- **object_name:** {final_result.get('object_name', 'N/A')}")
        lines.append(f"\n### Reasoning\n")
        lines.append(final_result.get('reasoning', 'N/A'))
        lines.append("")

        # Error info
        if "error" in final_result:
            lines.append(f"\n## ❌ Error\n")
            lines.append(f"{final_result['error']}\n")

        md_path.write_text("\n".join(lines), encoding="utf-8")

        # ----- JSON (machine-readable) -----
        json_data = {
            "task": task,
            "question": question,
            "timestamp": datetime.datetime.now().isoformat(),
            "search_engines_used": sources,
            "search_queries": search_queries,
            "search_results": search_results,
            "crawled_urls": crawled_urls,
            "crawled_content_lengths": {url: len(c) for url, c in crawled_content.items()},
            "analysis": {
                "affordance_name": final_result.get("affordance_name", ""),
                "part_name": final_result.get("part_name", ""),
                "object_name": final_result.get("object_name", ""),
                "reasoning": final_result.get("reasoning", ""),
            },
            "error": final_result.get("error"),
        }
        json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"  📁 Reference saved: {md_path}")
        print(f"  📁 Reference JSON:  {json_path}")
        return str(md_path)

    except Exception as e:
        print(f"  ⚠️  Failed to save reference: {e}")
        return None


# Main Entry Point

def run_web_search_skill(
    question: str,
    task: Optional[str] = None,
    image_hint: Optional[str] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    dynamic_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main entrypoint for the web-search skill.

    Workflow:
    1. Analyzes task and generates search queries using LLM
    2. Performs web searches (Brave → Bing → Google Scholar → Wikipedia proxy →
       Google API → Wikipedia direct → DuckDuckGo)
    3. Crawls and extracts content from web pages
    4. Analyzes crawled content using LLM to extract key information
    5. Saves results to skills/web-search/reference/ for visualization

    Args:
        question: （ task ）
        task: 
        image_hint: /
        api_url: API URL ()
        api_key: API Key ()
        dynamic_params:  task ：
            - search_focus (str): （ "cup handle grasping ergonomics"）
            - target_part (str): （ "handle"）
            - interaction_type (str): （ "power grasp", "pinch grip"）
            - knowledge_type (str): （"structural"/"functional"/"spatial"）
            - search_depth (str): （"quick"/"thorough"）
            -  key-value 
    """
    print("\n" + "=" * 70)
    print("🔍 WEB-SEARCH TOOL: Starting Web Search Workflow")
    print("=" * 70)

    # ： task 
    if dynamic_params:
        print(f"  📋 Decision model's search strategy ({len(dynamic_params)} params):")
        priority_keys = ["search_focus", "target_part",
                         "interaction_type", "knowledge_type", "search_depth"]
        for key in priority_keys:
            if key in dynamic_params:
                print(f"     🎯 {key}: {dynamic_params[key]}")
        for key, value in dynamic_params.items():
            if key not in priority_keys:
                print(f"     📌 {key}: {value}")
    else:
        print("  ℹ️  No search strategy from decision model — using generic search")

    # Step 1: Load Configuration
    print("\n[Step 1] Loading Configuration...")
    config = _load_api_config()
    api_base_url = api_url or config.get("API_BASE_URL")
    api_key = api_key or config.get("API_KEY")
    model = config.get("DEFAULT_MODEL", "gpt-4o")

    if not api_base_url:
        print("  ❌ Warning: API_BASE_URL not configured.")
    if not api_key:
        print("  ❌ Warning: API_KEY not configured.")
    else:
        masked_key = f"{api_key[:10]}..." if len(api_key) > 10 else "***"
        print(f"  ✅ LLM API Base URL: {api_base_url}")
        print(f"  ✅ Model: {model}")
        print(f"  ✅ LLM API Key: {masked_key}")

    # Build search engine priority list (all free, no API keys needed)
    available_engines = [
        "duckduckgo", "google_web", "conceptnet", "google_scholar",
        "wikipedia_proxy", "wikipedia_direct",
    ]
    print(f"  🔧 Search engine priority: {' → '.join(available_engines)}")
    print(f"     All engines are free — no API keys required")
    print(f"     Wikipedia via AllOrigins proxy: enabled (bypasses firewall)")
    if DDGS:
        print(f"  ✅ duckduckgo_search library: available")
    else:
        print(f"  ⚠️  duckduckgo_search library: NOT installed (pip install -U duckduckgo_search)")

    # Step 2: Prepare Input
    print("\n[Step 2] Preparing Input Information...")
    print(f"  📝 Question: {question}")
    if task:
        print(f"  📝 Task: {task}")
    if image_hint:
        print(f"  📝 Image Hint: {image_hint}")

    # Step 3: Generate Search Queries
    print("\n[Step 3] Generating Search Queries with LLM...")
    search_queries = _generate_search_queries(question, task, api_key, api_base_url, model, dynamic_params=dynamic_params)
    print(f"  ✅ Generated {len(search_queries)} search query(ies):")
    for i, q in enumerate(search_queries, 1):
        print(f"     {i}. {q}")

    # Step 4: Perform Web Searches
    print("\n[Step 4] Performing Web Searches...")
    all_search_results: List[Dict[str, Any]] = []
    for q_idx, query in enumerate(search_queries, 1):
        print(f"\n  🔍 [{q_idx}/{len(search_queries)}] Searching: '{query}'")
        results = _perform_web_search(
            query,
            max_results=15,  # Allow diverse results from multiple engines
            search_engines=available_engines,
        )
        all_search_results.extend(results)
        if q_idx < len(search_queries):
            time.sleep(0.5)

    if not all_search_results:
        print("  ⚠️  No search results found from any engine, falling back to LLM-only analysis")
        fallback_result = _fallback_llm_analysis(question, task, image_hint, api_key, api_base_url, model)
        _save_to_reference(task, question, search_queries, [], {}, [], {}, fallback_result)
        return fallback_result

    # Deduplicate
    seen = set()
    deduped = []
    for r in all_search_results:
        u = r.get("url", "")
        if u not in seen:
            seen.add(u)
            deduped.append(r)
    all_search_results = deduped

    print(f"\n  ✅ Total unique search results: {len(all_search_results)}")
    for i, r in enumerate(all_search_results[:10], 1):
        src = r.get('source', '?')
        print(f"     {i}. [{src}] {r.get('title', '')[:60]}")

    # Step 5: Crawl Web Content (Text Only)
    print("\n[Step 5] Crawling Web Content (Text Only)...")

    # Prioritise crawling: wikipedia > google_scholar > google_web > duckduckgo > conceptnet
    source_priority = {
        "wikipedia_proxy": 0, "wikipedia_direct": 0,
        "google_scholar": 1,
        "google_web": 2,
        "duckduckgo": 3,
        "conceptnet": 4,  # ConceptNet results are already structured, low crawl priority
    }
    crawl_order = sorted(
        all_search_results,
        key=lambda r: source_priority.get(r.get("source", ""), 9),
    )

    crawled_content: Dict[str, str] = {}
    crawled_urls: List[str] = []

    # Track which domains are unreachable to skip subsequent attempts
    _blocked_domains: set = set()

    for i, result in enumerate(crawl_order[:8], 1):
        url = result.get("url", "")
        if not url or url in crawled_content:
            continue

        # Skip URLs from domains known to be blocked (saves timeout waits)
        try:
            domain = urlparse(url).netloc
            if domain in _blocked_domains:
                continue
        except Exception:
            pass

        print(f"  📥 Crawling {i}: {url[:80]}...")
        text_and_html = _extract_web_content(url, max_length=2000, return_html=True)
        content, raw_html = text_and_html
        if content and not content.startswith("[Error"):
            crawled_content[url] = content
            crawled_urls.append(url)
            print(f"     ✅ Extracted {len(content)} chars text")
        else:
            print(f"     ⚠️  Failed: {content[:80] if content else 'empty'}")
            # Mark this domain as blocked so we skip future URLs from it
            try:
                fail_domain = urlparse(url).netloc
                if fail_domain:
                    _blocked_domains.add(fail_domain)
            except Exception:
                pass
        time.sleep(0.3)

    print(f"  ✅ Successfully crawled {len(crawled_content)} page(s)")

    # Step 6: Analyze Crawled Text with LLM
    print("\n[Step 6] Analyzing Crawled Web Text with LLM...")
    primary_query = search_queries[0] if search_queries else (task or question)
    analysis_result = _analyze_search_results_with_llm(
        primary_query, all_search_results, crawled_content,
        question, task, api_key, api_base_url, model,
        dynamic_params=dynamic_params,
    )

    if "error" in analysis_result:
        print(f"  ❌ LLM Analysis Failed: {analysis_result['error']}")
        error_result = {
            "question": question, "task": task, "image_hint": image_hint,
            "affordance_name": "", "part_name": "", "object_name": "", "reasoning": "",
            "search_queries": search_queries, "search_results": all_search_results,
            "crawled_urls": crawled_urls,
            "raw_response": analysis_result, "error": analysis_result["error"],
        }
        _save_to_reference(task, question, search_queries, all_search_results,
                           crawled_content, crawled_urls, analysis_result, error_result)
        return error_result

    # Step 7: Extract Structured Fields
    print("\n[Step 7] Extracting Final Results...")
    affordance_name = ""
    part_name = ""
    object_name = ""
    reasoning = ""

    try:
        response_text = analysis_result["choices"][0]["message"]["content"]
        print(f"  ✅ Analysis response received (length: {len(response_text)} characters)")

        brace_count = 0
        start_idx = -1
        found_json = False
        for idx, char in enumerate(response_text):
            if char == '{':
                if start_idx == -1:
                    start_idx = idx
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    json_str = response_text[start_idx:idx + 1]
                    try:
                        parsed = json.loads(json_str)
                        affordance_name = str(parsed.get("affordance_name", "")).strip()
                        part_name = str(parsed.get("part_name", "")).strip()
                        object_name = str(parsed.get("object_name", "")).strip()
                        reasoning = str(parsed.get("reasoning", "")).strip()
                        found_json = True
                        break
                    except json.JSONDecodeError:
                        start_idx = -1

        if not found_json:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    affordance_name = str(parsed.get("affordance_name", "")).strip()
                    part_name = str(parsed.get("part_name", "")).strip()
                    object_name = str(parsed.get("object_name", "")).strip()
                    reasoning = str(parsed.get("reasoning", "")).strip()
                except json.JSONDecodeError:
                    reasoning = response_text
            else:
                reasoning = response_text

    except (KeyError, IndexError) as e:
        print(f"  ❌ Failed to extract response: {e}")
        error_result = {
            "question": question, "task": task, "image_hint": image_hint,
            "affordance_name": "", "part_name": "", "object_name": "", "reasoning": "",
            "search_queries": search_queries, "search_results": all_search_results,
            "crawled_urls": crawled_urls, "raw_response": analysis_result,
            "error": "Could not extract response from API",
        }
        _save_to_reference(task, question, search_queries, all_search_results,
                           crawled_content, crawled_urls, analysis_result, error_result)
        return error_result

    # Step 8: Build Final Result & Save Reference
    print("\n[Step 8] Preparing Final Result & Saving Reference...")
    result = {
        "question": question,
        "task": task,
        "image_hint": image_hint,
        "affordance_name": affordance_name,
        "part_name": part_name,
        "object_name": object_name,
        "reasoning": reasoning,
        "search_queries": search_queries,
        "search_results": all_search_results[:20],
        "crawled_urls": crawled_urls,
        "raw_response": analysis_result,
    }

    print(f"  ✅ Extracted Information:")
    print(f"     - affordance_name: '{affordance_name}'")
    print(f"     - part_name: '{part_name}'")
    print(f"     - object_name: '{object_name}'")
    r_preview = reasoning[:150] + '...' if len(reasoning) > 150 else reasoning
    print(f"     - reasoning: '{r_preview}'")
    print(f"     - search_queries: {len(search_queries)} query(ies)")
    print(f"     - search_results: {len(all_search_results)} result(s)")
    print(f"     - crawled_urls: {len(crawled_urls)} URL(s)")

    ref_path = _save_to_reference(
        task, question, search_queries, all_search_results,
        crawled_content, crawled_urls, analysis_result, result,
    )
    if ref_path:
        result["reference_file"] = ref_path

    if dynamic_params:
        result["dynamic_params_used"] = dynamic_params

    print("\n" + "=" * 70)
    print("✅ WEB-SEARCH TOOL: Web Search Workflow Complete")
    print("=" * 70 + "\n")

    return result


# Fallback LLM-only Analysis

def _fallback_llm_analysis(
    question: str,
    task: Optional[str],
    image_hint: Optional[str],
    api_key: Optional[str],
    api_base_url: Optional[str],
    model: str,
) -> Dict[str, Any]:
    """Fallback to LLM-only analysis when web search fails."""
    print("\n[Fallback] Using LLM-only analysis (no web search)...")

    prompt_parts = []
    if task:
        prompt_parts.append(f"Task description: {task}")
    if question:
        prompt_parts.append(f"Question: {question}")
    if image_hint:
        prompt_parts.append(f"Scene/object context: {image_hint}")

    full_context = "\n".join(prompt_parts) if prompt_parts else question or task or ""

    system_prompt = """You are an expert at analyzing affordance tasks and identifying the key components needed for object interaction.

Given a task description, your job is to:
1. Identify the affordance_name (the action being performed)
2. Identify the part_name (the specific part of the object that should be interacted with)
3. Identify the object_name (the name of the object, if not explicitly mentioned)
4. Provide clear reasoning about your analysis

Output your analysis in JSON format:
{
    "affordance_name": "the action name",
    "part_name": "the object part name",
    "object_name": "the object name (if identifiable)",
    "reasoning": "your reasoning process and key insights"
}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_context}
    ]

    api_result = _call_llm_api(messages, model, api_key, api_base_url)

    if "error" in api_result:
        return {
            "question": question, "task": task, "image_hint": image_hint,
            "affordance_name": "", "part_name": "", "object_name": "", "reasoning": "",
            "search_queries": [], "search_results": [], "crawled_urls": [],
            "raw_response": api_result, "error": api_result["error"],
        }

    try:
        response_text = api_result["choices"][0]["message"]["content"]
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            return {
                "question": question, "task": task, "image_hint": image_hint,
                "affordance_name": str(parsed.get("affordance_name", "")).strip(),
                "part_name": str(parsed.get("part_name", "")).strip(),
                "object_name": str(parsed.get("object_name", "")).strip(),
                "reasoning": str(parsed.get("reasoning", "")).strip(),
                "search_queries": [], "search_results": [], "crawled_urls": [],
        "raw_response": api_result,
    }
    except Exception:
        pass

    return {
        "question": question, "task": task, "image_hint": image_hint,
        "affordance_name": "", "part_name": "", "object_name": "",
        "reasoning": response_text if 'response_text' in locals() else "",
        "search_queries": [], "search_results": [], "crawled_urls": [],
        "raw_response": api_result,
    }
