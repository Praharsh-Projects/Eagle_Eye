"""Bounded local synthesis providers for canonical RAG responses."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen


_CITATION_GROUP_RE = re.compile(
    r"\[(?P<body>\s*E\d+(?:\s*,\s*E\d+)*\s*)\]"
)
_CITATION_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])E\d+(?![A-Za-z0-9_])")

_RESEARCH_CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence_ids": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                },
                "required": ["text", "evidence_ids"],
                "additionalProperties": False,
            },
        },
        "boundary": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "evidence_ids": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["text", "evidence_ids"],
                    "additionalProperties": False,
                },
            ]
        },
    },
    "required": ["claims", "boundary"],
    "additionalProperties": False,
}


def normalize_evidence_citations(text: str, citation_map: dict[str, str]) -> str:
    """Expand valid local marker groups and reject malformed or unknown markers.

    Qwen may correctly group citations as ``[E1, E2]``. Replacing only
    singleton strings leaves those markers unresolved, which later makes a
    grounded paragraph appear uncited. This parser accepts only singleton or
    comma-separated marker groups, expands each marker to its immutable
    evidence identifier, and fails closed on loose, malformed, or unknown
    marker tokens.
    """

    groups = list(_CITATION_GROUP_RE.finditer(text))
    if not groups:
        raise ValueError("Local research synthesis did not provide valid evidence citations")

    outside_groups: list[str] = []
    cursor = 0
    normalized: list[str] = []
    for group in groups:
        outside_groups.append(text[cursor:group.start()])
        normalized.append(text[cursor:group.start()])
        markers = re.findall(r"E\d+", group.group("body"))
        unknown = [marker for marker in markers if marker not in citation_map]
        if unknown:
            raise ValueError(
                "Local research synthesis cited unknown evidence markers: "
                + ", ".join(unknown)
            )
        # Preserve marker order while avoiding repeated citations in one group.
        identifiers = [citation_map[marker] for marker in dict.fromkeys(markers)]
        normalized.append(" ".join(f"[{identifier}]" for identifier in identifiers))
        cursor = group.end()
    outside_groups.append(text[cursor:])
    normalized.append(text[cursor:])

    if _CITATION_TOKEN_RE.search(" ".join(outside_groups)):
        raise ValueError("Local research synthesis contained malformed evidence citations")
    return "".join(normalized)


@dataclass(frozen=True)
class SynthesisResult:
    text: str
    provider: str
    model: str


class GroundedSynthesizer(Protocol):
    provider: str
    model: str

    def synthesize_research(self, question: str, evidence: Sequence[Any]) -> SynthesisResult: ...

    def answer_general(self, question: str) -> SynthesisResult: ...


class OllamaSynthesizer:
    """Small, injectable Ollama client with evidence-citation enforcement."""

    provider = "ollama"

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen2.5:7b-instruct",
        timeout_seconds: float = 30.0,
        max_output_tokens: int = 600,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Ollama base URL must be an absolute http(s) URL")
        self.base_url = base_url.rstrip("/")
        self.model = str(model).strip()
        if not self.model:
            raise ValueError("Ollama model must not be blank")
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.max_output_tokens = max(64, int(max_output_tokens))

    def _chat(
        self,
        *,
        system: str,
        user: str,
        json_mode: bool = False,
    ) -> SynthesisResult:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": 0,
                "num_predict": self.max_output_tokens,
                "seed": 20260808,
            },
        }
        if json_mode:
            # Ollama accepts a JSON Schema in ``format``.  A plain ``json``
            # request guarantees syntax only and allowed a cold model load to
            # invent extra boundary fields, which correctly failed the local
            # claim validator.  The schema makes the first call obey the same
            # contract as every warm call without retrying or relaxing checks.
            payload["format"] = _RESEARCH_CLAIM_SCHEMA
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        text = str((body.get("message") or {}).get("content") or "").strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response")
        return SynthesisResult(text=text, provider=self.provider, model=self.model)

    def synthesize_research(self, question: str, evidence: Sequence[Any]) -> SynthesisResult:
        bounded = list(evidence)[:8]
        if not bounded:
            raise ValueError("Grounded research synthesis requires at least one evidence item")
        blocks = []
        citation_map: dict[str, str] = {}
        for index, item in enumerate(bounded, start=1):
            marker = f"E{index}"
            citation_map[marker] = str(getattr(item, "id", marker))
            title = str(getattr(item, "title", "Local source"))
            excerpt = str(getattr(item, "excerpt", "") or "")[:1400]
            blocks.append(f"[{marker}] {title}\n{excerpt}")
        result = self._chat(
            system=(
                "You are a source-grounded maritime research assistant. Use only the supplied evidence and return "
                "exactly one JSON object with keys claims and boundary. claims must be an array of zero to four "
                "objects with exactly text and evidence_ids. Each claim text must be one concise, close paraphrase "
                "of what its cited excerpts directly establish; evidence_ids must be a non-empty array using only "
                "the supplied E markers. boundary must be null unless the excerpts support part, but not all, of the "
                "question. A non-null boundary must have exactly text and evidence_ids and state precisely what the "
                "cited excerpts do not establish in exactly one sentence; do not append an explanation or second "
                "sentence inside boundary text. Begin boundary text with 'The supplied excerpts do not establish' "
                "and do not add a reason, inference, or characterization after the unresolved claim. boundary "
                "evidence_ids must cite the supplied excerpts relevant to that evidence boundary. Treat a "
                "directly supported component, input, or capability named "
                "in the question as a supported claim even when the excerpts do not establish the requested "
                "completeness, official status, authority, or conclusion; put that unresolved part in boundary. If "
                "no part is supported, return an empty claims array. Do not put "
                "citations, Markdown, or line breaks inside text. Do not use memory, invent facts, calculate traffic "
                "analytics, add examples or consequences, or use modal speculation to bridge missing evidence. A "
                "general capability does not establish a requested specific feature. Preserve the exact named "
                "instrument and source: never "
                "attribute a statement about 'this code', a code of practice, or guidance to SOLAS, ISPS, or another "
                "instrument unless the same excerpt explicitly makes that attribution. Do not turn 'includes', "
                "'follows', or 'supplements' into a legal requirement."
            ),
            user=f"Question:\n{question}\n\nEvidence:\n" + "\n\n".join(blocks),
            json_mode=True,
        )
        text = self._render_research_claims(result.text, citation_map)
        return SynthesisResult(text=text, provider=result.provider, model=result.model)

    @staticmethod
    def _render_research_claims(text: str, citation_map: dict[str, str]) -> str:
        """Validate a structured local answer and render cited prose deterministically."""

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Local research synthesis did not return valid JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {"claims", "boundary"}:
            raise ValueError("Local research synthesis did not match the claim contract")
        claims = payload["claims"]
        boundary = payload["boundary"]
        if not isinstance(claims, list) or len(claims) > 4:
            raise ValueError("Local research synthesis claims must be an array of at most four items")
        if not claims:
            raise ValueError("Retrieved evidence did not support an answer claim")

        def render_claim(value: Any, *, label: str) -> str:
            if not isinstance(value, dict) or set(value) != {"text", "evidence_ids"}:
                raise ValueError(f"Local research synthesis {label} did not match the claim contract")
            claim_text = value["text"]
            evidence_ids = value["evidence_ids"]
            if (
                not isinstance(claim_text, str)
                or not claim_text.strip()
                or "\n" in claim_text
                or "[" in claim_text
                or "]" in claim_text
                or len(claim_text) > 600
            ):
                raise ValueError(f"Local research synthesis {label} text is invalid")
            if label == "boundary" and not claim_text.strip().casefold().startswith(
                "the supplied excerpts do not establish"
            ):
                raise ValueError("Local research synthesis boundary did not use the evidence-limit form")
            if (
                not isinstance(evidence_ids, list)
                or not evidence_ids
                or any(not isinstance(marker, str) for marker in evidence_ids)
            ):
                raise ValueError(f"Local research synthesis {label} evidence_ids are invalid")
            marker_group = "[" + ", ".join(evidence_ids) + "]"
            citations = normalize_evidence_citations(marker_group, citation_map)
            sentence = claim_text.strip()
            punctuation = sentence[-1] if sentence[-1] in ".?!" else "."
            body = sentence[:-1].rstrip() if sentence[-1] in ".?!" else sentence
            return f"{body} {citations}{punctuation}"

        rendered = [render_claim(claim, label=f"claim_{index}") for index, claim in enumerate(claims, 1)]
        if boundary is not None:
            rendered.append(render_claim(boundary, label="boundary"))
        return " ".join(rendered)

    def answer_general(self, question: str) -> SynthesisResult:
        return self._chat(
            system=(
                "You are Eagle Eye's general assistant. Answer ordinary non-current questions directly and concisely. "
                "Never claim that Eagle Eye's historical maritime datasets contain live facts, and never invent maritime "
                "analytics values. If the request needs current information, say that a current source is required."
            ),
            user=question,
        )


def build_local_synthesizer(config: dict[str, Any]) -> Optional[GroundedSynthesizer]:
    retrieval = config.get("retrieval", {}) if isinstance(config, dict) else {}
    provider = str(
        os.getenv(
            "EAGLE_EYE_LOCAL_SYNTHESIS_PROVIDER",
            retrieval.get("local_synthesis_provider", "auto"),
        )
    ).strip().lower()
    if provider in {"", "none", "disabled", "off"}:
        return None
    if provider not in {"auto", "ollama"}:
        raise ValueError("EAGLE_EYE_LOCAL_SYNTHESIS_PROVIDER must be auto, ollama, or disabled")
    return OllamaSynthesizer(
        base_url=str(
            os.getenv(
                "EAGLE_EYE_OLLAMA_BASE_URL",
                retrieval.get("ollama_base_url", "http://127.0.0.1:11434"),
            )
        ),
        model=str(
            os.getenv(
                "EAGLE_EYE_OLLAMA_MODEL",
                retrieval.get("ollama_model", "qwen2.5:7b-instruct"),
            )
        ),
        timeout_seconds=float(
            os.getenv(
                "EAGLE_EYE_OLLAMA_TIMEOUT_SECONDS",
                retrieval.get("ollama_timeout_seconds", 30),
            )
        ),
        max_output_tokens=int(retrieval.get("ollama_max_output_tokens", 600)),
    )
