"""vLLM client via OpenAI-compatible Chat Completions API."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ntust_thesis.core.interfaces import LLMClient


class VllmChatCompletionsClient(LLMClient):
    """Minimal vLLM chat completions client."""

    def __init__(  # noqa: PLR0913
        self,
        api_base: str,
        model_name: str,
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
        """Initialize vLLM client."""
        self._api_base = api_base.rstrip("/")
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
        """Generate text from vLLM chat completion."""
        temperature = _coerce_temperature(kwargs.get("temperature", 0.0))
        allow_empty = bool(kwargs.get("allow_empty", False))
        payload = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        url = f"{self._api_base}/chat/completions"
        request = Request(  # noqa: S310
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
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
                    f"vLLM HTTP error {exc.code}: {exc.reason}. Body: {error_body}"
                )
                if (
                    attempt_idx >= max_attempts - 1
                    or exc.code not in self._retry_http_statuses
                ):
                    raise last_error from exc
            except URLError as exc:
                last_error = RuntimeError(f"vLLM connection error: {exc.reason}")
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
        msg = "vLLM generate failed without explicit error."
        raise RuntimeError(msg)


def _extract_text(parsed: dict[str, Any], allow_empty: bool = False) -> str:
    """Extract assistant message text from OpenAI-like response."""
    choices = parsed.get("choices", [])
    if not choices:
        msg = f"vLLM response missing choices: {parsed}"
        raise RuntimeError(msg)

    first = choices[0]
    message = first.get("message", {}) if isinstance(first, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    if not isinstance(content, str):
        msg = f"vLLM response contains invalid content field: {parsed}"
        raise TypeError(msg)
    text = content.strip()
    if not text and allow_empty:
        return ""
    if not text:
        msg = f"vLLM response contains empty text: {parsed}"
        raise RuntimeError(msg)
    return text


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


def _coerce_temperature(value: object) -> float:
    """Convert temperature config to float with explicit type narrowing."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    msg = f"Unsupported temperature type: {type(value).__name__}"
    raise TypeError(msg)
