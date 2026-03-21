"""Stub baseline model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ntust_thesis.core.interfaces import Model
from ntust_thesis.core.registry import MODEL_REGISTRY
from ntust_thesis.core.types import Prediction, Sample
from ntust_thesis.models.llm.gemini_client import GeminiClient
from ntust_thesis.utils.env import get_required_env
from ntust_thesis.utils.json_parser import parse_json_object


class BaselineModel(Model):
    """Baseline model with switchable mock/Gemini generation."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize baseline model from config."""
        cfg = config or {}
        self._backend = str(cfg.get("backend", "mock"))
        self._temperature = float(cfg.get("temperature", 0.0))
        self._llm: GeminiClient | None = None

        if self._backend == "gemini":
            api_key_env = str(cfg.get("api_key_env", "GEMINI_API_KEY"))
            dotenv_path = Path(str(cfg.get("dotenv_path", "dotenv/.env")))
            api_key = get_required_env(api_key_env, fallback_paths=[dotenv_path])
            model_name = str(cfg.get("llm_name", "gemini-2.5-flash-lite"))
            timeout = int(cfg.get("timeout_seconds", 60))
            self._llm = GeminiClient(
                api_key=api_key, model_name=model_name, timeout_seconds=timeout
            )
        elif self._backend != "mock":
            msg = f"Unsupported baseline backend: {self._backend}"
            raise ValueError(msg)

    def name(self) -> str:
        """Return model key."""
        return "baseline"

    def predict(self, sample: Sample) -> Prediction:
        """Generate mock text output then parse it as JSON."""
        raw_output = self._generate(sample.input_text)
        parsed_output = parse_json_object(raw_output)
        return Prediction(
            sample_id=sample.sample_id,
            raw_output=raw_output,
            parsed_output=parsed_output,
            metadata={"model": self.name(), "backend": self._backend},
        )

    def _generate(self, input_text: str) -> str:
        """Dispatch generation by configured backend."""
        if self._backend == "gemini":
            return self._generate_gemini(input_text)
        return self._mock_generate(input_text)

    def _generate_gemini(self, input_text: str) -> str:
        """Generate strict JSON with Gemini."""
        if self._llm is None:
            msg = "Gemini backend is not initialized."
            raise RuntimeError(msg)

        prompt = (
            "Extract event information from the text. "
            "Return only a JSON object with keys: "
            '"event_type" (string), "arguments" (array of objects). '
            'Each argument object has keys: "role", "text", "span".\n'
            f"Text: {input_text}"
        )
        return self._llm.generate(prompt, temperature=self._temperature)

    @staticmethod
    def _mock_generate(input_text: str) -> str:
        """Produce deterministic pseudo-model output from input text only."""
        lower = input_text.lower()
        event_type = "unknown.event"
        if "attack" in lower or "attacked" in lower:
            event_type = "conflict.attack.selfdirectedbattle"
        elif "killed" in lower or "death" in lower or "die" in lower:
            event_type = "life.die.deathcausedbyviolentevents"

        payload = {"event_type": event_type, "arguments": []}
        body = json.dumps(payload, ensure_ascii=False)

        # Add mild formatting noise to emulate real-world model outputs.
        return f"Here is the extracted JSON:\n```json\n{body}\n```"


def register() -> None:
    """Register built-in baseline model."""
    MODEL_REGISTRY.register("baseline", BaselineModel)
