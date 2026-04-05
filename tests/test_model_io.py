from __future__ import annotations

import unittest

from app.services.model_io import extract_json_object, normalize_router_prediction


class ModelIOTests(unittest.TestCase):
    def test_extract_json_object_handles_thinking_and_fenced_json(self) -> None:
        raw = """
<think>
先分析一下。
</think>

```json
{"route": "handoff", "intent": "complaint_or_manual_support"}
```
"""
        parsed = extract_json_object(raw)
        self.assertEqual(
            parsed,
            {"route": "handoff", "intent": "complaint_or_manual_support"},
        )

    def test_normalize_router_prediction_coerces_placeholder_route(self) -> None:
        normalized = normalize_router_prediction(
            {
                "route": "direct|internal_tool|handoff",
                "intent": "complaint_or_manual_support",
                "tool_name": "",
                "tool_arguments": {},
                "missing_slots": [],
                "need_clarification": False,
            }
        )
        self.assertEqual(normalized["route"], "handoff")
        self.assertEqual(normalized["intent"], "complaint_or_manual_support")


if __name__ == "__main__":
    unittest.main()
