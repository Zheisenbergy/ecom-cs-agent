from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol
from urllib import error, request

from app.exporters.training_data import ANSWER_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT

ROUTES = ["direct", "internal_tool", "handoff"]
TOOLS = ["get_product_info", "get_policy", "get_order_status", "get_logistics_status", ""]


class TextGenerationClient(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class OpenAICompatibleModelClient:
    def __init__(
        self,
        model: str,
        timeout_seconds: int = 120,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        max_tokens: int = 256,
    ) -> None:
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0,
            "max_tokens": self._max_tokens,
        }
        req = request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self._timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible HTTP {exc.code}: {details}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"无法连接 OpenAI-compatible 服务 {self._base_url}: {exc.reason}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"服务返回了无法解析的 JSON: {raw[:200]}") from exc

        if parsed.get("error"):
            raise RuntimeError(str(parsed["error"]))
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"响应中缺少 choices: {raw[:300]}")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            return "".join(text_parts).strip()
        return str(content).strip()


class BaselineBenchmarkService:
    def __init__(self, client: TextGenerationClient) -> None:
        self._client = client

    def benchmark_router(self, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
        cases = list(rows)
        predictions: List[dict[str, Any]] = []
        parse_failures = 0

        for row in cases:
            raw = self._client.generate(self._router_prompt(row))
            parsed = _extract_json_object(raw)
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

            prediction = _normalize_router_prediction(parsed)
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
            parsed = _extract_json_object(raw)
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

        token_f1_scores = [_token_f1(gold, pred) for gold, pred in zip(gold_answers, pred_answers)]
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
        return f"""/no_think

系统:
{ROUTER_SYSTEM_PROMPT}

指令:
请根据用户问题和当前状态，输出电商客服 router JSON。
只允许输出一个 JSON 对象，不要输出解释，不要输出 markdown。

route 只能是下面三个之一：
- direct
- internal_tool
- handoff

tool_name 只能是下面五个之一：
- get_product_info
- get_policy
- get_order_status
- get_logistics_status
- ""

输入:
user_query:
{json.dumps(row.get("user_query", ""), ensure_ascii=False)}

state_before:
{json.dumps(row.get("state_before", {}), ensure_ascii=False, indent=2)}

输出字段要求:
- route: string
- intent: string
- tool_name: string
- tool_arguments: object
- missing_slots: string[]
- need_clarification: boolean
- rewrite_query: string

如果缺少关键信息：
- need_clarification 必须为 true
- missing_slots 需要明确写出缺失字段，例如 ["order_id"] 或 ["product_id"]

输出示例:
{{
  "route": "handoff",
  "intent": "complaint_or_manual_support",
  "tool_name": "",
  "tool_arguments": {{}},
  "missing_slots": [],
  "need_clarification": false,
  "rewrite_query": "请转人工客服"
}}

只输出 JSON 对象本身。
"""

    @staticmethod
    def _answer_prompt(row: dict[str, Any]) -> str:
        return f"""/no_think

系统:
{ANSWER_SYSTEM_PROMPT}

指令:
请根据 query、route、intent 和结构化 tool_steps 输出电商客服 answer JSON。
只允许输出一个 JSON 对象，不要输出解释，不要输出 markdown。

输出字段要求：
- answer 必须是用户可见回答
- citations 填写实际依赖的工具名列表
- grounded 表示回答是否基于 observation
- escalation_required 表示是否需要转人工
- waiting_for_user: boolean
- episode_done: boolean

输入:
query:
{json.dumps(row.get("query", ""), ensure_ascii=False)}

route:
{json.dumps(row.get("route", ""), ensure_ascii=False)}

intent:
{json.dumps(row.get("intent", ""), ensure_ascii=False)}

tool_steps:
{json.dumps(row.get("tool_steps", []), ensure_ascii=False, indent=2)}

输出示例:
{{
  "answer": "string",
  "citations": [],
  "grounded": false,
  "escalation_required": false,
  "waiting_for_user": false,
  "episode_done": true
}}

只输出 JSON 对象本身。
"""


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


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    stripped = _strip_thinking_blocks(text).strip()
    candidates = [stripped]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    candidates.extend(fenced)

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _strip_thinking_blocks(text: str) -> str:
    # Qwen reasoning models may emit visible thinking blocks before the final JSON.
    without_tags = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return without_tags.strip()


def _normalize_router_prediction(parsed: dict[str, Any]) -> dict[str, Any]:
    raw_route = str(parsed.get("route", "direct")).strip()
    intent = str(parsed.get("intent", "general_direct_answer")).strip() or "general_direct_answer"
    tool_name = str(parsed.get("tool_name", "")).strip()
    tool_arguments = parsed.get("tool_arguments", {}) if isinstance(parsed.get("tool_arguments"), dict) else {}
    missing_slots = parsed.get("missing_slots", []) if isinstance(parsed.get("missing_slots"), list) else []
    need_clarification = bool(parsed.get("need_clarification", False))

    normalized_route = _coerce_route(
        raw_route=raw_route,
        intent=intent,
        tool_name=tool_name,
        tool_arguments=tool_arguments,
        missing_slots=missing_slots,
        need_clarification=need_clarification,
    )

    if intent in {"none", "direct", "direct_answer"}:
        intent = "general_direct_answer"

    return {
        "route": normalized_route,
        "intent": intent,
        "tool_name": tool_name,
        "tool_arguments": tool_arguments,
        "missing_slots": missing_slots,
        "need_clarification": need_clarification,
    }


def _coerce_route(
    raw_route: str,
    intent: str,
    tool_name: str,
    tool_arguments: dict[str, Any],
    missing_slots: list[Any],
    need_clarification: bool,
) -> str:
    if raw_route in ROUTES:
        return raw_route

    route_hits = [route for route in ROUTES if route in raw_route]
    if len(route_hits) == 1:
        return route_hits[0]

    if intent == "complaint_or_manual_support":
        return "handoff"

    if tool_name in TOOLS and tool_name:
        return "internal_tool"

    if need_clarification or missing_slots or tool_arguments:
        return "internal_tool"

    return "direct"


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


def _token_f1(gold: str, pred: str) -> float:
    gold_tokens = _simple_tokens(gold)
    pred_tokens = _simple_tokens(pred)
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0
    overlap = Counter(gold_tokens) & Counter(pred_tokens)
    num_same = sum(overlap.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _simple_tokens(text: str) -> List[str]:
    return [token for token in re.split(r"[\s，。！？、,:：；（）()]+", text.strip()) if token]


def _ratio(correct: int, total: int) -> dict[str, Any]:
    return {
        "correct": correct,
        "total": total,
        "accuracy": round(correct / total, 4) if total else None,
    }
