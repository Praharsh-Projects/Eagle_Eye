"""Deterministic-first chat orchestration for Eagle Eye."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from openai import OpenAI

from src.carbon.query import (
    CARBON_STATE_COMPUTED,
    CARBON_STATE_COMPUTED_ZERO,
    CARBON_STATE_FORECAST_ONLY,
    CARBON_STATE_NOT_COMPUTABLE,
    CARBON_STATE_RETRIEVAL_ONLY,
    CARBON_STATE_UNSUPPORTED,
    CarbonResult,
)
from src.forecast.forecast import ForecastResult
from src.kpi.query import AnalyticsResult
from src.qa.intent import IntentResult
from src.utils.confidence import extract_confidence_label


CHAT_STATE_NO_DATA = "NO_DATA"
CHAT_STATE_RETRIEVAL_ONLY = "RETRIEVAL_ONLY"
CHAT_STATE_UNSUPPORTED = "UNSUPPORTED"


@dataclass
class ChatTurn:
    conversation_id: str
    turn_id: str
    question: str
    answer: str
    result_state: str
    source_type: str
    confidence: str
    assumptions_used: List[str]
    evidence_lines: List[str]
    tool_trace: Dict[str, Any]
    intent: Dict[str, Any]

def derive_chat_contract(
    result: Union[AnalyticsResult, ForecastResult, CarbonResult],
    evidence_rows: Sequence[Dict[str, Any]],
) -> Tuple[str, str]:
    """Return (result_state, source_type) for chat response envelopes."""
    has_retrieval_rows = bool(evidence_rows)
    if isinstance(result, CarbonResult):
        state = str(result.result_state or CARBON_STATE_NOT_COMPUTABLE)
        if state == CARBON_STATE_UNSUPPORTED:
            return CHAT_STATE_UNSUPPORTED, "Unsupported"
        if state in {CARBON_STATE_NOT_COMPUTABLE, CARBON_STATE_RETRIEVAL_ONLY}:
            return (CHAT_STATE_RETRIEVAL_ONLY if has_retrieval_rows else CHAT_STATE_NO_DATA, "Retrieved")
        if state == CARBON_STATE_FORECAST_ONLY:
            return CARBON_STATE_FORECAST_ONLY, "Estimated"
        if "estimate" in result.source_label.lower():
            return state, "Estimated"
        return state, "Computed"

    status = str(result.status or "").lower()
    if status == "unsupported":
        return CHAT_STATE_UNSUPPORTED, "Unsupported"
    if status != "ok":
        if has_retrieval_rows:
            return CHAT_STATE_RETRIEVAL_ONLY, "Retrieved"
        return CHAT_STATE_NO_DATA, "Computed"
    if isinstance(result, ForecastResult):
        return CARBON_STATE_FORECAST_ONLY, "Estimated"
    return "COMPUTED", "Computed"


class ChatOrchestrator:
    """Wrap deterministic tool outputs into a bounded-memory chat experience."""

    def __init__(
        self,
        model: str,
        max_history_turns: int = 8,
        openai_client: Optional[OpenAI] = None,
    ) -> None:
        self.model = model
        self.max_history_turns = max(2, int(max_history_turns))
        self.openai_client = openai_client

    def _bounded_history(self, history: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not history:
            return []
        return list(history)[-self.max_history_turns :]

    def _history_context(self, history: Sequence[Dict[str, Any]]) -> str:
        bounded = self._bounded_history(history)
        if not bounded:
            return "None."
        lines: List[str] = []
        for turn in bounded:
            q = str(turn.get("question", "")).strip()
            a = str(turn.get("answer", "")).strip()
            citations = list(turn.get("evidence_lines", []) or [])[:2]
            lines.append(f"Q: {q}")
            lines.append(f"A: {a}")
            for c in citations:
                lines.append(f"Citation: {c}")
        return "\n".join(lines)

    @staticmethod
    def _number_tokens(text: str) -> set[str]:
        tokens: set[str] = set()
        for raw in re.findall(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?", text or ""):
            normalized = raw.replace(",", "").lstrip("+")
            try:
                value = float(normalized)
            except ValueError:
                continue
            if value.is_integer():
                tokens.add(str(int(value)))
            else:
                tokens.add(f"{value:.12g}")
        return tokens

    @classmethod
    def _synthesis_preserves_numeric_truth(
        cls,
        *,
        synthesized: str,
        deterministic_answer: str,
        question: str,
        evidence_lines: Sequence[str],
    ) -> bool:
        required = cls._number_tokens(deterministic_answer)
        actual = cls._number_tokens(synthesized)
        allowed = required | cls._number_tokens(question) | cls._number_tokens("\n".join(evidence_lines))
        preserves_computed_result = not required or bool(required & actual)
        return preserves_computed_result and actual.issubset(allowed)

    @staticmethod
    def _natural_fallback(
        deterministic_answer: str,
        result_state: str,
        assumptions: Sequence[str],
    ) -> str:
        reason = next((str(item).strip() for item in assumptions if str(item).strip()), "")
        if result_state == CHAT_STATE_UNSUPPORTED:
            if reason:
                return f"I can't answer that reliably from Eagle Eye's supported data. {reason}"
            return "I can't answer that reliably from Eagle Eye's supported data."
        if result_state in {CHAT_STATE_NO_DATA, CHAT_STATE_RETRIEVAL_ONLY}:
            if reason:
                return f"I can't determine that from the available data. {reason}"
        return deterministic_answer.strip()

    def _llm_compose(
        self,
        question: str,
        deterministic_answer: str,
        source_type: str,
        result_state: str,
        confidence: str,
        assumptions: Sequence[str],
        evidence_lines: Sequence[str],
        history: Sequence[Dict[str, Any]],
    ) -> str:
        fallback = self._natural_fallback(deterministic_answer, result_state, assumptions)
        if self.openai_client is None:
            return fallback

        evidence_block = "\n".join(f"- {line}" for line in list(evidence_lines)[:6]) or "- No evidence lines."
        assumptions_block = "\n".join(f"- {item}" for item in list(assumptions)[:6]) or "- No explicit assumptions."
        history_context = self._history_context(history)
        system_prompt = (
            "You are Eagle Eye, a concise maritime intelligence assistant. Answer the user's question directly in natural prose. "
            "The deterministic answer is the sole authority for every number and factual calculation. Preserve its core computed "
            "result exactly, but secondary diagnostics may be omitted. Introduce no new numbers. Do not duplicate the deterministic answer, expose internal tool names, "
            "or use report headings. Mention limitations only when they materially affect the answer. If the result is unsupported "
            "or unavailable, explain the boundary plainly without guessing."
        )
        user_prompt = (
            f"User question:\n{question}\n\n"
            f"Deterministic answer:\n{deterministic_answer}\n\n"
            f"Source type: {source_type}\nResult state: {result_state}\nConfidence: {confidence}\n\n"
            f"Assumptions:\n{assumptions_block}\n\n"
            f"Evidence lines:\n{evidence_block}\n\n"
            f"Recent conversation context:\n{history_context}\n\n"
            "Return one self-contained conversational answer."
        )
        try:
            response = self.openai_client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=user_prompt,
            )
            text = str(getattr(response, "output_text", "") or "").strip()
            if not text:
                return fallback
            if not self._synthesis_preserves_numeric_truth(
                synthesized=text,
                deterministic_answer=deterministic_answer,
                question=question,
                evidence_lines=evidence_lines,
            ):
                return fallback
            return text
        except Exception:
            return fallback

    def run_turn(
        self,
        *,
        question: str,
        intent: IntentResult,
        result: Union[AnalyticsResult, ForecastResult, CarbonResult],
        evidence_lines: Sequence[str],
        evidence_rows: Sequence[Dict[str, Any]],
        tool_trace: Dict[str, Any],
        conversation_id: Optional[str] = None,
        history: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> ChatTurn:
        cid = str(conversation_id or f"chat_{uuid.uuid4().hex[:12]}")
        tid = f"turn_{uuid.uuid4().hex[:10]}"
        assumptions = list(result.caveats or [])[:8]
        result_state, source_type = derive_chat_contract(result, evidence_rows)
        confidence = extract_confidence_label(result)
        answer = self._llm_compose(
            question=question,
            deterministic_answer=result.answer,
            source_type=source_type,
            result_state=result_state,
            confidence=confidence,
            assumptions=assumptions,
            evidence_lines=evidence_lines,
            history=history or [],
        )
        return ChatTurn(
            conversation_id=cid,
            turn_id=tid,
            question=question,
            answer=answer,
            result_state=result_state,
            source_type=source_type,
            confidence=confidence,
            assumptions_used=assumptions,
            evidence_lines=list(evidence_lines)[:8],
            tool_trace=dict(tool_trace or {}),
            intent={
                "intent": intent.intent,
                "reason": intent.reason,
                "entities": intent.entities,
            },
        )
