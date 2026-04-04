from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from app.exporters.llamafactory import load_episodes


ROUTER_SYSTEM_PROMPT = (
    "你是电商客服场景下的路由与动作决策模型。"
    "你需要根据用户问题和当前任务状态，判断 route、intent、是否需要 ask_user、"
    "是否调用工具，以及工具参数。输出必须是 JSON 对象，不要输出解释。"
)

ANSWER_SYSTEM_PROMPT = (
    "你是电商客服场景下的回答模型。"
    "你需要根据 query、route、intent 和结构化 tool observation 输出 JSON。"
    "输出必须遵守字段要求，不要输出解释，不要输出 markdown。"
)


def export_router_sft_jsonl(
    traces_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    episodes = load_episodes(traces_path)
    rows: List[Dict[str, Any]] = []

    for episode in episodes:
        for turn_index, turn in enumerate(episode.turns, start=1):
            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "turn_index": turn_index,
                    "user_query": turn.request.query,
                    "state_before": _compact_state(turn.state_before.model_dump(mode="json") if turn.state_before else {}),
                    "route": turn.route_decision.route.value,
                    "intent": turn.route_decision.intent,
                    "need_clarification": turn.route_decision.need_clarification,
                    "rewrite_query": turn.route_decision.rewrite_query,
                    "tool_name": turn.route_decision.tool_name or "",
                    "tool_arguments": turn.route_decision.tool_arguments,
                    "missing_slots": turn.route_decision.missing_slots,
                    "confidence": turn.route_decision.confidence,
                    "rationale": turn.route_decision.rationale,
                }
            )

    _write_jsonl(output_path, rows)
    return {
        "num_samples": len(rows),
        "num_episodes": len(episodes),
        "task": "router_sft",
        "output": str(output_path),
    }


def export_answer_sft_jsonl(
    traces_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    episodes = load_episodes(traces_path)
    rows: List[Dict[str, Any]] = []

    for episode in episodes:
        for turn_index, turn in enumerate(episode.turns, start=1):
            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "turn_index": turn_index,
                    "query": turn.request.query,
                    "route": turn.response.route.value,
                    "intent": turn.response.intent,
                    "tool_steps": [
                        {
                            "tool_name": step.result.tool_name,
                            "status": step.result.status,
                            "data": step.result.data,
                            "message": step.result.message,
                        }
                        for step in turn.tool_steps
                    ],
                    "answer": turn.response.answer,
                    "citations": turn.response.citations,
                    "grounded": turn.response.grounded,
                    "escalation_required": turn.response.escalation_required,
                    "waiting_for_user": turn.response.waiting_for_user,
                    "episode_done": turn.response.episode_done,
                }
            )

    _write_jsonl(output_path, rows)
    return {
        "num_samples": len(rows),
        "num_episodes": len(episodes),
        "task": "answer_sft",
        "output": str(output_path),
    }


def export_router_sft_llamafactory(
    traces_path: Path,
    output_path: Path,
    dataset_name: str,
    dataset_info_path: Path,
) -> dict[str, Any]:
    episodes = load_episodes(traces_path)
    rows: List[Dict[str, Any]] = []

    for episode in episodes:
        for turn_index, turn in enumerate(episode.turns, start=1):
            route_decision = turn.route_decision
            rows.append(
                {
                    "instruction": (
                        "请根据用户问题和当前状态，输出电商客服 router JSON。"
                        "只允许输出 JSON 对象，不要输出解释。"
                    ),
                    "input": _router_input(
                        user_query=turn.request.query,
                        state_before=_compact_state(turn.state_before.model_dump(mode="json") if turn.state_before else {}),
                    ),
                    "output": json.dumps(
                        {
                            "route": route_decision.route.value,
                            "intent": route_decision.intent,
                            "tool_name": route_decision.tool_name or "",
                            "tool_arguments": route_decision.tool_arguments,
                            "missing_slots": route_decision.missing_slots,
                            "need_clarification": route_decision.need_clarification,
                            "rewrite_query": route_decision.rewrite_query,
                        },
                        ensure_ascii=False,
                    ),
                    "system": ROUTER_SYSTEM_PROMPT,
                    "metadata": {
                        "episode_id": episode.episode_id,
                        "turn_index": turn_index,
                    },
                }
            )

    _write_json(output_path, rows)
    _write_alpaca_dataset_info(dataset_info_path, dataset_name, output_path.name)
    return {
        "dataset_name": dataset_name,
        "num_samples": len(rows),
        "num_episodes": len(episodes),
        "task": "router_sft_llamafactory",
        "output": str(output_path),
        "dataset_info": str(dataset_info_path),
    }


def export_answer_sft_llamafactory(
    traces_path: Path,
    output_path: Path,
    dataset_name: str,
    dataset_info_path: Path,
) -> dict[str, Any]:
    episodes = load_episodes(traces_path)
    rows: List[Dict[str, Any]] = []

    for episode in episodes:
        for turn_index, turn in enumerate(episode.turns, start=1):
            rows.append(
                {
                    "instruction": (
                        "请根据 query、route、intent 和 tool_steps 输出电商客服 answer JSON。"
                        "只允许输出 JSON 对象，不要输出解释。"
                    ),
                    "input": _answer_input(
                        query=turn.request.query,
                        route=turn.response.route.value,
                        intent=turn.response.intent,
                        tool_steps=[
                            {
                                "tool_name": step.result.tool_name,
                                "status": step.result.status,
                                "data": step.result.data,
                                "message": step.result.message,
                            }
                            for step in turn.tool_steps
                        ],
                    ),
                    "output": json.dumps(
                        {
                            "answer": turn.response.answer,
                            "citations": turn.response.citations,
                            "grounded": turn.response.grounded,
                            "escalation_required": turn.response.escalation_required,
                            "waiting_for_user": turn.response.waiting_for_user,
                            "episode_done": turn.response.episode_done,
                        },
                        ensure_ascii=False,
                    ),
                    "system": ANSWER_SYSTEM_PROMPT,
                    "metadata": {
                        "episode_id": episode.episode_id,
                        "turn_index": turn_index,
                    },
                }
            )

    _write_json(output_path, rows)
    _write_alpaca_dataset_info(dataset_info_path, dataset_name, output_path.name)
    return {
        "dataset_name": dataset_name,
        "num_samples": len(rows),
        "num_episodes": len(episodes),
        "task": "answer_sft_llamafactory",
        "output": str(output_path),
        "dataset_info": str(dataset_info_path),
    }


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def _write_alpaca_dataset_info(path: Path, dataset_name: str, file_name: str) -> None:
    payload = {
        dataset_name: {
            "file_name": file_name,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        }
    }
    existing: Dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            existing = {}
    existing.update(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, ensure_ascii=False, indent=2)


def _router_input(user_query: str, state_before: Dict[str, Any]) -> str:
    return (
        f"user_query:\n{user_query}\n\n"
        f"state_before:\n{json.dumps(state_before, ensure_ascii=False, indent=2)}\n\n"
        "输出字段要求:\n"
        "- route: direct|internal_tool|handoff\n"
        "- intent: string\n"
        "- tool_name: string\n"
        "- tool_arguments: object\n"
        "- missing_slots: string[]\n"
        "- need_clarification: boolean\n"
        "- rewrite_query: string"
    )


def _answer_input(query: str, route: str, intent: str, tool_steps: List[Dict[str, Any]]) -> str:
    return (
        f"query:\n{query}\n\n"
        f"route:\n{route}\n\n"
        f"intent:\n{intent}\n\n"
        f"tool_steps:\n{json.dumps(tool_steps, ensure_ascii=False, indent=2)}\n\n"
        "输出字段要求:\n"
        "- answer: string\n"
        "- citations: string[]\n"
        "- grounded: boolean\n"
        "- escalation_required: boolean\n"
        "- waiting_for_user: boolean\n"
        "- episode_done: boolean"
    )


def _compact_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in state.items() if value not in (None, "", [], {})}
