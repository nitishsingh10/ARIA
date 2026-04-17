"""
aria/capabilities/external/web_fetch.py — URL fetching capability.

Fetches a URL and optionally extracts text content, title, and links
using BeautifulSoup.

Example:
    >>> from aria.capabilities.registry import GLOBAL_REGISTRY
    >>> r = await GLOBAL_REGISTRY.execute("fetch_url", {
    ...     "url": "https://example.com",
    ...     "extract_text": True,
    ... })
    >>> print(r.data["title"])
    'Example Domain'
"""

from __future__ import annotations

import re
from typing import List, Optional

import httpx
from pydantic import Field

from aria.capabilities.base import (
    Capability,
    CapabilityInput,
    CapabilityOutput,
)
from aria.core.logger import get_logger

log = get_logger("capabilities.web_fetch")


# ═══════════════════════════════════════════════════════════════════════
# fetch_url
# ═══════════════════════════════════════════════════════════════════════


class FetchUrlInput(CapabilityInput):
    """Input schema for the ``fetch_url`` capability.

    Attributes:
        url: The URL to fetch.
        extract_text: Strip HTML and return plain text.
        timeout: Request timeout in seconds.
        max_chars: Maximum characters in the content field.
    """

    url: str = Field(description="URL to fetch.")
    extract_text: bool = Field(
        default=True, description="Extract plain text from HTML."
    )
    timeout: int = Field(default=15, description="Request timeout in seconds.")
    max_chars: int = Field(
        default=50_000, description="Max characters in returned content."
    )


class FetchUrlCapability(Capability):
    """Fetch a URL and optionally extract text, title, and links.

    Uses ``httpx`` for the HTTP request and ``BeautifulSoup`` for HTML
    parsing when ``extract_text=True``.

    Example input::

        {"url": "https://example.com", "extract_text": true}

    Example output::

        {
            "url": "https://example.com",
            "status_code": 200,
            "content": "Example Domain\\nThis domain is for...",
            "title": "Example Domain",
            "links": ["https://www.iana.org/domains/example"]
        }
    """

    name = "fetch_url"
    description = "Fetch a URL and optionally extract text, title, and links from the page."
    input_schema = FetchUrlInput
    tags = ["external", "web", "read"]
    requires_confirmation = False

    async def execute(self, input_data: FetchUrlInput) -> CapabilityOutput:
        """Fetch the URL.

        Args:
            input_data: Validated ``FetchUrlInput``.

        Returns:
            CapabilityOutput with url, status_code, content, title, links.
        """
        try:
            async with httpx.AsyncClient(
                timeout=input_data.timeout, follow_redirects=True
            ) as client:
                resp = await client.get(input_data.url)
        except httpx.HTTPError as exc:
            return CapabilityOutput(
                success=False,
                error=f"HTTP request failed: {exc}",
            )

        raw_content = resp.text
        title: Optional[str] = None
        links: List[str] = []

        if input_data.extract_text:
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(raw_content, "html.parser")

                # Extract title
                title_tag = soup.find("title")
                if title_tag:
                    title = title_tag.get_text(strip=True)

                # Extract links
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"]
                    if href.startswith(("http://", "https://")):
                        links.append(href)

                # Strip HTML to plain text
                # Remove script and style elements
                for tag in soup(["script", "style"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                # Collapse multiple blank lines
                text = re.sub(r"\n{3,}", "\n\n", text)
                content = text
            except ImportError:
                log.warning(
                    "beautifulsoup4 not installed — returning raw HTML"
                )
                content = raw_content
        else:
            content = raw_content

        # Truncate
        if len(content) > input_data.max_chars:
            content = content[: input_data.max_chars] + "\n[TRUNCATED]"

        return CapabilityOutput(
            success=True,
            data={
                "url": str(resp.url),
                "status_code": resp.status_code,
                "content": content,
                "title": title,
                "links": links,
            },
        )


# ═══════════════════════════════════════════════════════════════════════
# Capability instance for registration
# ═══════════════════════════════════════════════════════════════════════

fetch_url = FetchUrlCapability()

ALL_WEB_CAPABILITIES = [fetch_url]
