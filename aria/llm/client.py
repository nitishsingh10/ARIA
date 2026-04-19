"""
aria/llm/client.py — Async Ollama client wrapper for ARIA.

Provides chat completion, streaming chat, embedding, and health-check
methods. Includes automatic retry with exponential back-off and
structured logging of latency/token counts.

Usage:
    from aria.llm.client import OllamaClient
    from aria.config import get_settings

    client = OllamaClient(get_settings().llm)
    response = await client.chat([{"role": "user", "content": "Hello!"}])
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from aria.config import LLMConfig
from aria.core.logger import get_logger

log = get_logger("llm.client")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_RETRIES: int = 3
_BACKOFF_BASE: float = 1.0  # seconds; doubles each retry


class OllamaClient:
    """Async wrapper around the Ollama REST API.

    Args:
        config: An ``LLMConfig`` instance with provider URL, model, etc.
    """

    def __init__(self, config: Any) -> None:
        """Initialise the client with the given LLM configuration.

        Args:
            config: Validated LLM settings (model, base_url, temperature …).
        """
        self.config = config.llm if hasattr(config, "llm") else config
        self._base_url = self.config.base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # HTTP client lifecycle
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        """Return (and lazily create) the shared httpx async client.

        Returns:
            The httpx.AsyncClient instance.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self.config.timeout, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client gracefully."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Retry helper
    # ------------------------------------------------------------------

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> httpx.Response:
        """Execute an HTTP request with retry + exponential back-off.

        Args:
            method: HTTP method (``"POST"``, ``"GET"`` …).
            path: API path relative to base_url (e.g. ``"/api/chat"``).
            json_body: Optional JSON payload.
            stream: If True, return a streaming response.

        Returns:
            The httpx Response object.

        Raises:
            httpx.HTTPStatusError: If the server returns a non-2xx status
                after all retries are exhausted.
            httpx.ConnectError: If Ollama is unreachable after retries.
        """
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                client = await self._get_client()
                if stream:
                    req = client.build_request(method, path, json=json_body)
                    resp = await client.send(req, stream=True)
                    resp.raise_for_status()
                    return resp
                else:
                    resp = await client.request(method, path, json=json_body)
                    resp.raise_for_status()
                    return resp

            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    wait = _BACKOFF_BASE * (2 ** (attempt - 1))
                    log.warning(
                        f"Request failed (attempt {attempt}/{_MAX_RETRIES}), "
                        f"retrying in {wait:.1f}s",
                        data={"error": str(exc), "path": path},
                    )
                    await asyncio.sleep(wait)
                else:
                    log.error(
                        f"Request failed after {_MAX_RETRIES} attempts",
                        data={"error": str(exc), "path": path},
                    )

        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Chat completion
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
    ) -> str:
        """Send a chat completion request to Ollama.

        Args:
            messages: List of ``{"role": …, "content": …}`` dicts.
            system: Optional system prompt prepended to messages.

        Returns:
            The assistant's reply as a plain string.
        """
        payload = self._build_chat_payload(messages, system, stream=False)

        start = time.perf_counter()
        resp = await self._request_with_retry("POST", "/api/chat", json_body=payload)
        duration_ms = (time.perf_counter() - start) * 1_000

        body: dict[str, Any] = resp.json()
        content: str = body.get("message", {}).get("content", "")
        tokens = self._extract_token_counts(body)

        log.info(
            "Chat completion finished",
            data={"duration_ms": round(duration_ms, 1), **tokens},
        )

        return content

    # ------------------------------------------------------------------
    # Streaming chat
    # ------------------------------------------------------------------

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion token-by-token from Ollama.

        Args:
            messages: List of ``{"role": …, "content": …}`` dicts.
            system: Optional system prompt prepended to messages.

        Yields:
            Individual text chunks as they arrive.
        """
        payload = self._build_chat_payload(messages, system, stream=True)

        start = time.perf_counter()
        resp = await self._request_with_retry(
            "POST", "/api/chat", json_body=payload, stream=True
        )

        total_tokens: dict[str, int] = {}

        try:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                token_text = chunk.get("message", {}).get("content", "")
                if token_text:
                    yield token_text

                # The final chunk carries token stats.
                if chunk.get("done"):
                    total_tokens = self._extract_token_counts(chunk)
        finally:
            await resp.aclose()

        duration_ms = (time.perf_counter() - start) * 1_000
        log.info(
            "Stream chat finished",
            data={"duration_ms": round(duration_ms, 1), **total_tokens},
        )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding.
        """
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": text,
        }

        start = time.perf_counter()
        resp = await self._request_with_retry(
            "POST", "/api/embed", json_body=payload
        )
        duration_ms = (time.perf_counter() - start) * 1_000

        body = resp.json()
        embeddings: list[list[float]] = body.get("embeddings", [[]])
        vector = embeddings[0] if embeddings else []

        log.info(
            "Embedding generated",
            data={
                "duration_ms": round(duration_ms, 1),
                "dimensions": len(vector),
            },
        )
        return vector

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Check whether the Ollama server is reachable (synchronous).

        Sends a GET request to the Ollama root endpoint. Returns True
        if the server responds with HTTP 200, False otherwise.

        Returns:
            True if Ollama is running and healthy.
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(self._base_url)
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_chat_payload(
        self,
        messages: list[dict[str, str]],
        system: str | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        """Assemble the JSON payload for the /api/chat endpoint.

        Args:
            messages: Chat messages.
            system: Optional system prompt.
            stream: Whether to request streaming.

        Returns:
            The request body dict.
        """
        all_messages: list[dict[str, str]] = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        return {
            "model": self.config.model,
            "messages": all_messages,
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

    @staticmethod
    def _extract_token_counts(body: dict[str, Any]) -> dict[str, int]:
        """Pull prompt / completion token counts from an Ollama response.

        Args:
            body: The parsed JSON response body.

        Returns:
            A dict with ``prompt_tokens`` and ``completion_tokens`` keys
            (values default to 0 if absent).
        """
        return {
            "prompt_tokens": body.get("prompt_eval_count", 0),
            "completion_tokens": body.get("eval_count", 0),
        }
