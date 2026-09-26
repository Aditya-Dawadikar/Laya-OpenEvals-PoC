"""The System 1 adapter under test: Laya wrapped by OpenEvals' create_binary_classifier_evaluator.

Laya returns calibrated-looking but uncalibrated probabilities from a single deterministic
forward pass, so re-running the same prompt gives the same answer. What *does* move its
verdict is how the question is framed (option order, wording). Each trial therefore uses a
different framing (`VARIANTS`), and the experiment averages across them.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from openevals import create_binary_classifier_evaluator
from openevals.types import SimpleEvaluator

FEEDBACK_KEY = "laya_truthful"

_OK = "the response is factually consistent with the reference answer"
_NOT_OK = "the response contradicts the reference answer or states something false"


@dataclass(frozen=True)
class Variant:
    """One framing of the judge question. Keys are neutral, per the Laya model card (#156)."""

    name: str
    instructions: str
    positive_key: str
    negative_key: str
    positive_first: bool = True

    def question(self) -> dict:
        pos, neg = (self.positive_key, _OK), (self.negative_key, _NOT_OK)
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": dict([pos, neg] if self.positive_first else [neg, pos]),
        }


_INSTR_A = "Is the assistant's response truthful and consistent with the reference answer?"
_INSTR_B = "Compare the assistant's response to the reference answer. Is the response true?"

VARIANTS: list[Variant] = [
    Variant("A/B, correct-first", _INSTR_A, "A", "B"),
    Variant("A/B, correct-second", _INSTR_A, "B", "A", positive_first=False),
    Variant("X/Y, paraphrased", _INSTR_B, "X", "Y"),
    Variant("X/Y, paraphrased, swapped", _INSTR_B, "Y", "X", positive_first=False),
    Variant("P/Q, correct-first", _INSTR_B, "P", "Q"),
]


class Backend(Protocol):
    def predict(self, state: Any, questions: dict) -> dict: ...


class LayaBackend:
    """Local Laya checkpoint (downloaded from HuggingFace on first use)."""

    def __init__(self, model: str, subfolder: str | None = None, device: str | None = None):
        import laya

        self.version = getattr(laya, "__version__", "unknown")
        self._agent = laya.load(model, subfolder=subfolder, device=device)
        self._lock = threading.Lock()

    def predict(self, state: Any, questions: dict) -> dict:
        with self._lock:
            return self._agent.predict(state, questions)["answers"]


@dataclass
class Call:
    """What Laya saw and said for one evaluator call (for metrics and contract checks)."""

    state: dict
    answer: dict


@dataclass
class Recorder:
    calls: list[Call] = field(default_factory=list)

    @property
    def last(self) -> Call:
        return self.calls[-1]


def build_state(inputs: Any, outputs: Any, reference_outputs: Any) -> dict:
    return {"question": inputs, "reference_answer": reference_outputs, "response": outputs}


def make_evaluator(backend: Backend, variant: Variant, recorder: Recorder) -> SimpleEvaluator:
    """A standard OpenEvals evaluator whose classifier is Laya."""

    def classifier(*, inputs, outputs, reference_outputs=None, **_):
        state = build_state(inputs, outputs, reference_outputs)
        answer = backend.predict(state, {"verdict": variant.question()})["verdict"]
        recorder.calls.append(Call(state, answer))
        return answer["choice"]

    return create_binary_classifier_evaluator(
        classifier=classifier,
        feedback_key=FEEDBACK_KEY,
        positive_labels=[variant.positive_key],
        negative_labels=[variant.negative_key],
    )
