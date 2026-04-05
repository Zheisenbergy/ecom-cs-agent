from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, List, Optional, Protocol
from urllib import error, request

from app.exporters.training_data import (
    ANSWER_INSTRUCTION,
    ANSWER_SYSTEM_PROMPT,
    ROUTER_INSTRUCTION,
    ROUTER_SYSTEM_PROMPT,
    build_answer_input,
    build_router_input,
)

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


def build_router_completion_prompt(user_query: str, state_before: dict[str, Any]) -> str:
    return f"""/no_think

系统:
{ROUTER_SYSTEM_PROMPT}

指令:
{ROUTER_INSTRUCTION}

输入:
{build_router_input(user_query=user_query, state_before=state_before)}

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


def build_answer_completion_prompt(query: str, route: str, intent: str, tool_steps: list[dict[str, Any]]) -> str:
    return f"""/no_think

系统:
{ANSWER_SYSTEM_PROMPT}

指令:
{ANSWER_INSTRUCTION}

输入:
{build_answer_input(query=query, route=route, intent=intent, tool_steps=tool_steps)}

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


def extract_json_object(text: str) -> Optional[dict[str, Any]]:
    stripped = strip_thinking_blocks(text).strip()
    # 尽量兼容模型输出的几种常见形态：纯 JSON、代码块 JSON、前面带说明的 JSON。
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


def strip_thinking_blocks(text: str) -> str:
    # Qwen reasoning models may emit visible thinking blocks before the final JSON.
    without_tags = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return without_tags.strip()


def normalize_router_prediction(parsed: dict[str, Any]) -> dict[str, Any]:
    # 运行时和 benchmark 共用同一套轻量归一化，避免模型输出占位文本时完全失效。
    raw_route = str(parsed.get("route", "direct")).strip()
    intent = str(parsed.get("intent", "general_direct_answer")).strip() or "general_direct_answer"
    tool_name = str(parsed.get("tool_name", "")).strip()
    tool_arguments = parsed.get("tool_arguments", {}) if isinstance(parsed.get("tool_arguments"), dict) else {}
    missing_slots = parsed.get("missing_slots", []) if isinstance(parsed.get("missing_slots"), list) else []
    need_clarification = bool(parsed.get("need_clarification", False))

    normalized_route = coerce_route(
        raw_route=raw_route,
        intent=intent,
        tool_name=tool_name,
        tool_arguments=tool_arguments,
        missing_slots=missing_slots,
        need_clarification=need_clarification,
    )

    if intent in {"none", "direct", "direct_answer"}:
        intent = "general_direct_answer"

    rewrite_query = str(parsed.get("rewrite_query", "")).strip()

    return {
        "route": normalized_route,
        "intent": intent,
        "tool_name": tool_name,
        "tool_arguments": tool_arguments,
        "missing_slots": missing_slots,
        "need_clarification": need_clarification,
        "rewrite_query": rewrite_query,
    }


def normalize_answer_prediction(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": str(parsed.get("answer", "")).strip(),
        "citations": parsed.get("citations", []) if isinstance(parsed.get("citations"), list) else [],
        "grounded": bool(parsed.get("grounded", False)),
        "escalation_required": bool(parsed.get("escalation_required", False)),
        "waiting_for_user": bool(parsed.get("waiting_for_user", False)),
        "episode_done": bool(parsed.get("episode_done", False)),
    }


def coerce_route(
    raw_route: str,
    intent: str,
    tool_name: str,
    tool_arguments: dict[str, Any],
    missing_slots: list[Any],
    need_clarification: bool,
) -> str:
    if raw_route in ROUTES:
        return raw_route

    # 有些模型会抄提示里的占位文本；这里优先把“半合法” route 拉回标准标签。
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


def token_f1(gold: str, pred: str) -> float:
    gold_tokens = simple_tokens(gold)
    pred_tokens = simple_tokens(pred)
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


def simple_tokens(text: str) -> List[str]:
    return [token for token in re.split(r"[\s，。！？、,:：；（）()]+", text.strip()) if token]
