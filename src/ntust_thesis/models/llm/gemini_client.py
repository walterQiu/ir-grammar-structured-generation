"""Gemini client via Google Generative Language REST API."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ntust_thesis.core.interfaces import LLMClient


class GeminiClient(LLMClient):
    """Minimal Gemini text generation client."""

    def __init__(  # noqa: PLR0913
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        timeout_seconds: int = 60,
        enable_sleep: bool = False,
        sleep_seconds: float = 0.0,
        enable_retry: bool = True,
        max_retries: int = 3,
        backoff_initial_seconds: float = 1.0,
        backoff_multiplier: float = 2.0,
        backoff_max_seconds: float = 16.0,
        retry_http_statuses: tuple[int, ...] = (429, 500, 502, 503, 504),
    ) -> None:
        """Initialize Gemini client."""
        self._api_key = api_key
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds
        self._enable_sleep = enable_sleep
        self._sleep_seconds = sleep_seconds
        self._enable_retry = enable_retry
        self._max_retries = max_retries
        self._backoff_initial_seconds = backoff_initial_seconds
        self._backoff_multiplier = backoff_multiplier
        self._backoff_max_seconds = backoff_max_seconds
        self._retry_http_statuses = retry_http_statuses

    def generate(self, prompt: str, **kwargs: object) -> str:
        """Generate text from Gemini."""
        temperature = _coerce_temperature(kwargs.get("temperature", 0.0))
        allow_empty = bool(kwargs.get("allow_empty", False))
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
        if self._enable_sleep and self._sleep_seconds > 0:
            time.sleep(self._sleep_seconds)

        max_attempts = self._max_retries + 1 if self._enable_retry else 1
        last_error: Exception | None = None

        for attempt_idx in range(max_attempts):
            try:
                with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310
                    body = response.read().decode("utf-8")
                parsed = json.loads(body)
                return _extract_text(parsed, allow_empty=allow_empty)
            except HTTPError as exc:
                error_body = _read_http_error_body(exc)
                last_error = RuntimeError(
                    f"Gemini HTTP error {exc.code}: {exc.reason}. Body: {error_body}"
                )
                if (
                    attempt_idx >= max_attempts - 1
                    or exc.code not in self._retry_http_statuses
                ):
                    raise last_error from exc
            except URLError as exc:
                last_error = RuntimeError(f"Gemini connection error: {exc.reason}")
                if attempt_idx >= max_attempts - 1:
                    raise last_error from exc

            _sleep_backoff(
                attempt_idx=attempt_idx,
                max_attempts=max_attempts,
                initial_seconds=self._backoff_initial_seconds,
                multiplier=self._backoff_multiplier,
                max_seconds=self._backoff_max_seconds,
            )

        if last_error is not None:
            raise last_error
        msg = "Gemini generate failed without explicit error."
        raise RuntimeError(msg)


def _sleep_backoff(
    *,
    attempt_idx: int,
    max_attempts: int,
    initial_seconds: float,
    multiplier: float,
    max_seconds: float,
) -> None:
    """Sleep between retries using capped exponential backoff."""
    if attempt_idx >= max_attempts - 1:
        return
    if initial_seconds <= 0:
        return
    delay = min(initial_seconds * (multiplier**attempt_idx), max_seconds)
    if delay > 0:
        time.sleep(delay)


def _read_http_error_body(exc: HTTPError) -> str:
    """Safely read HTTP error body."""
    try:
        return exc.read().decode("utf-8")
    except Exception:
        return ""


def _extract_text(parsed: dict[str, Any], allow_empty: bool = False) -> str:
    """Extract text from Gemini response JSON."""
    candidates = parsed.get("candidates", [])
    if not candidates:
        msg = f"Gemini response missing candidates: {parsed}"
        raise RuntimeError(msg)

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    merged = "".join(texts).strip()
    if not merged and allow_empty:
        return ""
    if not merged:
        msg = f"Gemini response contains empty text: {parsed}"
        raise RuntimeError(msg)
    return merged


def _coerce_temperature(value: object) -> float:
    """Convert temperature config to float with explicit type narrowing."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    msg = f"Unsupported temperature type: {type(value).__name__}"
    raise TypeError(msg)
