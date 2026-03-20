"""Gemini client via Google Generative Language REST API."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ntust_thesis.core.interfaces import LLMClient


class GeminiClient(LLMClient):
    """Minimal Gemini text generation client."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        timeout_seconds: int = 60,
    ) -> None:
        """Initialize Gemini client."""
        self._api_key = api_key
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate text from Gemini."""
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {"temperature": temperature},
        }

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model_name}:generateContent"
        )
        request = Request(  # noqa: S310
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8")
            except Exception:
                error_body = ""
            msg = f"Gemini HTTP error {exc.code}: {exc.reason}. Body: {error_body}"
            raise RuntimeError(msg) from exc
        except URLError as exc:
            msg = f"Gemini connection error: {exc.reason}"
            raise RuntimeError(msg) from exc

        parsed = json.loads(body)
        return _extract_text(parsed)


def _extract_text(parsed: dict[str, Any]) -> str:
    """Extract text from Gemini response JSON."""
    candidates = parsed.get("candidates", [])
    if not candidates:
        msg = f"Gemini response missing candidates: {parsed}"
        raise RuntimeError(msg)

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    merged = "".join(texts).strip()
    if not merged:
        msg = f"Gemini response contains empty text: {parsed}"
        raise RuntimeError(msg)
    return merged
