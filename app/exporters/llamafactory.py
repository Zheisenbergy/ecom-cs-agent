from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from app.models import EpisodeRecord, EpisodeState, TraceRecord
from app.tool_schema import get_tool_schemas_json


SYSTEM_PROMPT = (
    "你是电商客服场景下的工具调用助手。"
    "你需要在一个任务 episode 内判断是否直接回答、是否需要向用户澄清、"
    "是否要调用工具，以及何时结束。"
    "你必须基于工具 observation 生成简洁准确的客服回复。"
)


def export_traces_to_llamafactory(
    traces_path: Path,
    output_path: Path,
    dataset_name: str,
    dataset_info_path: Path,
) -> dict[str, Any]:
    episodes = load_episodes(traces_path)
    samples = [_episode_to_sharegpt(record) for record in episodes]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(samples, handle, ensure_ascii=False, indent=2)

    dataset_info = {
        dataset_name: {
            "file_name": output_path.name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
            },
        }
    }
    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_info_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset_info, handle, ensure_ascii=False, indent=2)

    return {
        "dataset_name": dataset_name,
        "num_samples": len(samples),
        "num_episodes": len(episodes),
        "output": str(output_path),
        "dataset_info": str(dataset_info_path),
    }


def load_episodes(path: Path) -> List[EpisodeRecord]:
    episodes: List[EpisodeRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "turns" in payload and "final_response" in payload:
                episodes.append(EpisodeRecord.model_validate(payload))
                continue
            trace = TraceRecord.model_validate(payload)
            episodes.append(
                EpisodeRecord(
                    episode_id=f"episode-{uuid4().hex[:8]}",
                    turns=[trace],
                    final_response=trace.response,
                    final_state=trace.state_after or trace.state_before or EpisodeState(),
                    waiting_for_user=trace.response.waiting_for_user,
                    completed=trace.response.episode_done,
                )
            )
    return episodes


def _episode_to_sharegpt(record: EpisodeRecord) -> Dict[str, Any]:
    conversations: List[Dict[str, str]] = []
    for turn in record.turns:
        conversations.append({"from": "human", "value": turn.request.query})

        steps = turn.tool_steps
        if not steps and turn.tool_call is not None and turn.tool_result is not None:
            steps = [
                {
                    "call": turn.tool_call,
                    "result": turn.tool_result,
                }
            ]

        for step in steps:
            call = step["call"] if isinstance(step, dict) else step.call
            result = step["result"] if isinstance(step, dict) else step.result
            conversations.append(
                {
                    "from": "function_call",
                    "value": json.dumps(
                        {
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                        ensure_ascii=False,
                    ),
                }
            )
            conversations.append(
                {
                    "from": "observation",
                    "value": json.dumps(
                        {
                            "status": result.status,
                            "data": result.data,
                            "message": result.message,
                        },
                        ensure_ascii=False,
                    ),
                }
            )

        conversations.append({"from": "gpt", "value": turn.response.answer})

    return {
        "conversations": conversations,
        "system": SYSTEM_PROMPT,
        "tools": get_tool_schemas_json(),
    }
