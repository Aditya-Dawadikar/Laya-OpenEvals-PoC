"""Run settings, read from the environment (.env) with CLI overrides."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Pinned so a run is reproducible; see README "Reproducibility".
TRUTHFULQA_DATASET = "truthfulqa/truthful_qa"
TRUTHFULQA_REVISION = "741b8276f2d1982aa3d5b832d3ee81ed3b896490"


def _pin_caches() -> None:
    """Keep model/dataset downloads inside the repo unless the caller chose a location."""
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "hf"))
    os.environ.setdefault("USE_TF", "0")  # Laya model card: TF can deadlock laya.load()


def _env(name: str, default: str) -> str:
    return os.getenv(name) or default


@dataclass(frozen=True)
class Settings:
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    laya_model: str = "convaiinnovations/laya"
    laya_subfolder: str | None = None
    laya_device: str | None = None
    samples: int = 50
    trials: int = 5
    seed: int = 42
    temperature: float = 0.7
    max_tokens: int = 96
    results_dir: Path = field(default_factory=lambda: REPO_ROOT / "results")

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            ollama_url=_env("OLLAMA_URL", cls.ollama_url),
            ollama_model=_env("OLLAMA_MODEL", cls.ollama_model),
            laya_model=_env("LAYA_MODEL", cls.laya_model),
            laya_subfolder=os.getenv("LAYA_SUBFOLDER") or None,
            laya_device=os.getenv("LAYA_DEVICE") or None,
            samples=int(_env("POC_SAMPLES", str(cls.samples))),
            trials=int(_env("POC_TRIALS", str(cls.trials))),
            seed=int(_env("POC_SEED", str(cls.seed))),
            temperature=float(_env("POC_TEMPERATURE", str(cls.temperature))),
            max_tokens=int(_env("POC_MAX_TOKENS", str(cls.max_tokens))),
        )

    def override(self, **kwargs) -> Settings:
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})

    def to_dict(self) -> dict:
        d = asdict(self)
        d["results_dir"] = str(self.results_dir)
        return d
