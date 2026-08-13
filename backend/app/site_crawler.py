"""
Abacus Digital Chatbot - Site Crawler
Crawls the live Framer site (service pages + blog posts) into RAG chunks (PRD 7.1).

Discovery order: sitemap.xml (Framer publishes one), then a shallow link crawl from
the services and blog index pages. Content is chunked with a source URL in metadata so
answers can cite the page they came from.
"""

import asyncio
import hashlib
import logging
import re
from typing import List, Dict, Any, Set, Optional
from urllib.parse import urljoin, urlparse

import httpx

from .config import settings

logger = logging.getLogger(__name__)

# Only these path prefixes are indexed into the public KB
INDEXABLE_PREFIXES = ("/all-services", "/blog", "/services", "/about", "/contact")

# Never index these
EXCLUDED_PATTERNS = (
    "/client", "/portal", "/login", "/admin", "/api/",
    ".pdf", ".jpg", ".png", ".svg", ".zip", ".css", ".js",
)

CHUNK_TARGET_CHARS = 900
CHUNK_OVERLAP_CHARS = 120

_WS = re.compile(r"[ \t ]+")
_BLANKS = re.compile(r"\n{3,}")


class SiteCrawler:
    """Fetches and chunks live website content."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or settings.site_base_url).rstrip("/")
        self.domain = urlparse(self.base_url).netloc

    # --- URL discovery ---

    async def discover_urls(self, client: httpx.AsyncClient) -> List[str]:
        """Find candidate page URLs, sitemap first."""
        urls = await self._from_sitemap(client)
        if urls:
            logger.info(f"Discovered {len(urls)} URLs from sitemap")
            return urls

        logger.info("No usable sitemap, falling back to link crawl")
        return await self._from_link_crawl(client)

    async def _from_sitemap(self, client: httpx.AsyncClient) -> List[str]:
        for path in ("/sitemap.xml", "/sitemap-0.xml"):
            try:
                resp = await client.get(urljoin(self.base_url, path))
                if resp.status_code != 200 or "<urlset" not in resp.text and "<sitemapindex" not in resp.text:
                    continue

                locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", resp.text)

                # Sitemap index: follow the children
                if "<sitemapindex" in resp.text:
                    nested: List[str] = []
                    for child in locs[:10]:
                        try:
                            child_resp = await client.get(child)
                            if child_resp.status_code == 200:
                                nested += re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", child_resp.text)
                        except httpx.HTTPError:
                            continue
                    locs = nested

                return [u for u in locs if self._is_indexable(u)]
            except httpx.HTTPError as e:
                logger.debug(f"Sitemap fetch failed for {path}: {e}")
        return []

    async def _from_link_crawl(self, client: httpx.AsyncClient) -> List[str]:
        """Shallow crawl from the index pages."""
        seeds = [
            self.base_url,
            f"{self.base_url}/all-services",
            f"{self.base_url}/blog",
        ]
        found: Set[str] = set()

        for seed in seeds:
            try:
                resp = await client.get(seed)
                if resp.status_code != 200:
                    continue
                for href in re.findall(r'href=["\']([^"\']+)["\']', resp.text):
                    absolute = urljoin(seed, href).split("#")[0].rstrip("/")
                    if self._is_indexable(absolute):
                        found.add(absolute)
            except httpx.HTTPError as e:
                logger.debug(f"Link crawl failed for {seed}: {e}")

        return sorted(found)

    def _is_indexable(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.netloc and parsed.netloc != self.domain:
            return False
        path = parsed.path or "/"
        lowered = url.lower()
        if any(bad in lowered for bad in EXCLUDED_PATTERNS):
            return False
        return any(path.startswith(prefix) for prefix in INDEXABLE_PREFIXES)

    # --- Fetch + extract ---

    async def fetch_page(self, client: httpx.AsyncClient, url: str) -> Optional[Dict[str, str]]:
        """Fetch one page and extract its readable text."""
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None
            if "text/html" not in resp.headers.get("content-type", ""):
                return None
            title, text = self._extract(resp.text)
            if len(text) < 200:
                return None
            return {"url": url, "title": title, "text": text}
        except httpx.HTTPError as e:
            logger.debug(f"Fetch failed for {url}: {e}")
            return None

    @staticmethod
    def _extract(html: str) -> tuple[str, str]:
        """Extract title and visible text from HTML."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:  # pragma: no cover
            logger.error("beautifulsoup4 is not installed; cannot extract page text")
            return "", ""

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "svg", "iframe", "nav", "footer"]):
            tag.decompose()

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            title = h1.get_text(strip=True)

        text = soup.get_text("\n")
        text = _WS.sub(" ", text)
        text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
        text = _BLANKS.sub("\n\n", text)
        return title, text

    # --- Chunking ---

    @staticmethod
    def chunk_page(page: Dict[str, str]) -> List[Dict[str, Any]]:
        """Split a page into overlapping chunks with citation metadata."""
        url = page["url"]
        title = page["title"] or url
        text = page["text"]

        path = urlparse(url).path
        section = "blog" if "/blog" in path else "service_page"

        chunks: List[Dict[str, Any]] = []
        paragraphs = [p for p in text.split("\n") if p.strip()]

        buf = ""
        for para in paragraphs:
            if len(buf) + len(para) + 1 > CHUNK_TARGET_CHARS and buf:
                chunks.append(buf.strip())
                buf = buf[-CHUNK_OVERLAP_CHARS:] + "\n" + para
            else:
                buf = f"{buf}\n{para}" if buf else para
        if buf.strip():
            chunks.append(buf.strip())

        url_hash = hashlib.sha1(url.encode()).hexdigest()[:10]
        return [
            {
                "id": f"site_{url_hash}_{i}",
                "text": f"{title}\n(Source: {url})\n\n{chunk}",
                "metadata": {
                    "source": url,
                    "source_url": url,
                    "title": title,
                    "section": section,
                    "service": title,
                    "origin": "website",
                },
            }
            for i, chunk in enumerate(chunks)
        ]

    # --- Entry point ---

    async def crawl(self, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
        """Crawl the site and return RAG-ready chunks. Never raises; returns [] on failure."""
        max_pages = max_pages or settings.crawl_max_pages
        chunks: List[Dict[str, Any]] = []

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(settings.crawl_timeout_seconds),
                follow_redirects=True,
                headers={"User-Agent": "AbacusDigitalBot/1.0 (+https://www.abacusdigital.net)"},
            ) as client:
                urls = (await self.discover_urls(client))[:max_pages]
                if not urls:
                    logger.warning("Site crawl found no indexable URLs")
                    return []

                # Modest concurrency: we are a guest on someone else's free-tier host
                semaphore = asyncio.Semaphore(5)

                async def bounded(url: str):
                    async with semaphore:
                        return await self.fetch_page(client, url)

                pages = await asyncio.gather(*(bounded(u) for u in urls), return_exceptions=True)

                for page in pages:
                    if isinstance(page, dict):
                        chunks.extend(self.chunk_page(page))

            logger.info(f"Crawled {len(urls)} URLs into {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Site crawl failed: {e}")
            return []

        return chunks


site_crawler = SiteCrawler()
