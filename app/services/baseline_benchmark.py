from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Optional

from app.services.model_io import (
    OpenAICompatibleModelClient,
    TextGenerationClient,
    build_answer_completion_prompt,
    build_router_completion_prompt,
    extract_json_object,
    normalize_router_prediction,
    token_f1,
)


class BaselineBenchmarkService:
    def __init__(self, client: TextGenerationClient) -> None:
        self._client = client

    def benchmark_router(self, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
        cases = list(rows)
        predictions: List[dict[str, Any]] = []
        parse_failures = 0

        for row in cases:
            raw = self._client.generate(self._router_prompt(row))
            parsed = extract_json_object(raw)
            if parsed is None:
                parse_failures += 1
                predictions.append(
                    {
                        "route": "direct",
                        "intent": "general_direct_answer",
                        "tool_name": "",
                        "tool_arguments": {},
                        "missing_slots": [],
                        "need_clarification": False,
                        "raw_output": raw,
                    }
                )
                continue

            prediction = normalize_router_prediction(parsed)
            prediction["raw_output"] = raw
            predictions.append(prediction)

        gold_routes = [str(row.get("route", "")) for row in cases]
        pred_routes = [pred["route"] for pred in predictions]
        gold_intents = [str(row.get("intent", "")) for row in cases]
        pred_intents = [pred["intent"] for pred in predictions]
        gold_tools = [str(row.get("tool_name", "")) for row in cases]
        pred_tools = [pred["tool_name"] for pred in predictions]
        gold_ask = [bool(row.get("need_clarification", False)) for row in cases]
        pred_ask = [bool(pred.get("need_clarification", False)) for pred in predictions]
        gold_handoff = [str(row.get("route", "")) == "handoff" for row in cases]
        pred_handoff = [pred["route"] == "handoff" for pred in predictions]
        gold_missing = [list(row.get("missing_slots", [])) for row in cases]
        pred_missing = [list(pred.get("missing_slots", [])) for pred in predictions]
        gold_arguments = [row.get("tool_arguments", {}) if isinstance(row.get("tool_arguments"), dict) else {} for row in cases]
        pred_arguments = [pred.get("tool_arguments", {}) if isinstance(pred.get("tool_arguments"), dict) else {} for pred in predictions]

        return {
            "task": "router_baseline",
            "num_cases": len(cases),
            "parse_failure_rate": _ratio(parse_failures, len(cases)),
            "route_accuracy": _accuracy(gold_routes, pred_routes),
            "route_macro_f1": _macro_f1(gold_routes, pred_routes),
            "intent_accuracy": _accuracy(gold_intents, pred_intents),
            "intent_macro_f1": _macro_f1(gold_intents, pred_intents),
            "tool_accuracy": _accuracy(gold_tools, pred_tools),
            "tool_macro_f1": _macro_f1(gold_tools, pred_tools),
            "ask_user_f1": _binary_f1(gold_ask, pred_ask),
            "handoff_f1": _binary_f1(gold_handoff, pred_handoff),
            "missing_slots_exact_match": {
                "correct": sum(int(sorted(g) == sorted(p)) for g, p in zip(gold_missing, pred_missing)),
                "total": len(cases),
                "accuracy": round(
                    sum(int(sorted(g) == sorted(p)) for g, p in zip(gold_missing, pred_missing)) / len(cases), 4
                )
                if cases
                else None,
            },
            "tool_arguments_exact_match": {
                "correct": sum(int(g == p) for g, p in zip(gold_arguments, pred_arguments)),
                "total": len(cases),
                "accuracy": round(sum(int(g == p) for g, p in zip(gold_arguments, pred_arguments)) / len(cases), 4)
                if cases
                else None,
            },
            "details": [
                {
                    "user_query": row.get("user_query"),
                    "gold": {
                        "route": row.get("route"),
                        "intent": row.get("intent"),
                        "tool_name": row.get("tool_name"),
                        "tool_arguments": row.get("tool_arguments"),
                        "missing_slots": row.get("missing_slots", []),
                        "need_clarification": row.get("need_clarification", False),
                    },
                    "predicted": pred,
                }
                for row, pred in zip(cases, predictions)
            ],
        }

    def benchmark_answer(self, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
        cases = list(rows)
        predictions: List[dict[str, Any]] = []
        parse_failures = 0

        for row in cases:
            raw = self._client.generate(self._answer_prompt(row))
            parsed = extract_json_object(raw)
            if parsed is None:
                parse_failures += 1
                predictions.append(
                    {
                        "answer": raw.strip(),
                        "citations": [],
                        "grounded": False,
                        "escalation_required": False,
                        "raw_output": raw,
                    }
                )
                continue

            predictions.append(
                {
                    "answer": str(parsed.get("answer", "")).strip(),
                    "citations": parsed.get("citations", []) if isinstance(parsed.get("citations"), list) else [],
                    "grounded": bool(parsed.get("grounded", False)),
                    "escalation_required": bool(parsed.get("escalation_required", False)),
                    "raw_output": raw,
                }
            )

        gold_answers = [str(row.get("answer", "")) for row in cases]
        pred_answers = [pred.get("answer", "") for pred in predictions]
        gold_citations = [list(row.get("citations", [])) for row in cases]
        pred_citations = [list(pred.get("citations", [])) for pred in predictions]
        gold_grounded = [bool(row.get("grounded", False)) for row in cases]
        pred_grounded = [bool(pred.get("grounded", False)) for pred in predictions]
        gold_escalation = [bool(row.get("escalation_required", False)) for row in cases]
        pred_escalation = [bool(pred.get("escalation_required", False)) for pred in predictions]

        token_f1_scores = [token_f1(gold, pred) for gold, pred in zip(gold_answers, pred_answers)]
        exact_match = [gold == pred for gold, pred in zip(gold_answers, pred_answers)]
        citation_match = [gold == pred for gold, pred in zip(gold_citations, pred_citations)]

        return {
            "task": "answer_baseline",
            "num_cases": len(cases),
            "parse_failure_rate": _ratio(parse_failures, len(cases)),
            "answer_token_f1": {
                "score": round(sum(token_f1_scores) / len(token_f1_scores), 4) if token_f1_scores else None
            },
            "answer_exact_match": _ratio(sum(int(x) for x in exact_match), len(exact_match)),
            "citation_set_accuracy": _ratio(sum(int(x) for x in citation_match), len(citation_match)),
            "grounded_f1": _binary_f1(gold_grounded, pred_grounded),
            "escalation_f1": _binary_f1(gold_escalation, pred_escalation),
            "details": [
                {
                    "query": row.get("query"),
                    "gold": {
                        "answer": row.get("answer"),
                        "citations": row.get("citations", []),
                        "grounded": row.get("grounded", False),
                        "escalation_required": row.get("escalation_required", False),
                    },
                    "predicted": pred,
                }
                for row, pred in zip(cases, predictions)
            ],
        }

    @staticmethod
    def _router_prompt(row: dict[str, Any]) -> str:
        return build_router_completion_prompt(
            user_query=str(row.get("user_query", "")),
            state_before=row.get("state_before", {}) if isinstance(row.get("state_before", {}), dict) else {},
        )

    @staticmethod
    def _answer_prompt(row: dict[str, Any]) -> str:
        return build_answer_completion_prompt(
            query=str(row.get("query", "")),
            route=str(row.get("route", "")),
            intent=str(row.get("intent", "")),
            tool_steps=row.get("tool_steps", []) if isinstance(row.get("tool_steps", []), list) else [],
        )


def load_jsonl(path: Path, limit: Optional[int] = None) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _accuracy(gold: List[str], pred: List[str]) -> dict[str, Any]:
    correct = sum(int(g == p) for g, p in zip(gold, pred))
    return _ratio(correct, len(gold))


def _macro_f1(gold: List[str], pred: List[str]) -> dict[str, Any]:
    labels = sorted(set(gold) | set(pred))
    if not labels:
        return {"labels": 0, "f1": None}

    scores = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        scores.append(_f1_from_counts(tp, fp, fn))
    return {"labels": len(labels), "f1": round(sum(scores) / len(scores), 4)}


def _binary_f1(gold: List[bool], pred: List[bool]) -> dict[str, Any]:
    tp = sum(1 for g, p in zip(gold, pred) if g and p)
    fp = sum(1 for g, p in zip(gold, pred) if not g and p)
    fn = sum(1 for g, p in zip(gold, pred) if g and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = _f1_from_counts(tp, fp, fn)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": sum(int(g) for g in gold),
    }


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
def _ratio(correct: int, total: int) -> dict[str, Any]:
    return {
        "correct": correct,
        "total": total,
        "accuracy": round(correct / total, 4) if total else None,
    }
