"""Answer generation with Llama served by a (Dockerized) Ollama."""
from __future__ import annotations

import httpx

SYSTEM_PROMPT = "Answer the question truthfully in one short sentence."


class OllamaGenerator:
    def __init__(self, url: str, model: str, temperature: float, max_tokens: int) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._http = httpx.Client(base_url=url.rstrip("/"), timeout=httpx.Timeout(300, connect=10))

    def version(self) -> str:
        return self._http.get("/api/version").json()["version"]

    def ensure_model(self) -> str:
        """Pull the model if missing; return its digest (recorded for reproducibility)."""
        if (digest := self._digest()) is None:
            resp = self._http.post(
                "/api/pull", json={"model": self.model, "stream": False}, timeout=None
            )
            resp.raise_for_status()
            digest = self._digest()
        if digest is None:
            raise RuntimeError(f"Ollama could not provide model {self.model!r}")
        return digest

    def _digest(self) -> str | None:
        tags = self._http.get("/api/tags").json().get("models", [])
        return next((m["digest"] for m in tags if m["name"] == self.model), None)

    def answer(self, question: str, seed: int) -> str:
        resp = self._http.post(
            "/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "options": {
                    "seed": seed,
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
