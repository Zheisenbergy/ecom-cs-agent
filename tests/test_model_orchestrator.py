from __future__ import annotations

import unittest

from app.models import ChatRequest, EpisodeState
from app.services.model_orchestrator import build_model_orchestrator


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def generate(self, prompt: str) -> str:
        if not self._responses:
            raise RuntimeError(f"no more fake responses for prompt: {prompt[:120]}")
        return self._responses.pop(0)


class ModelOrchestratorTests(unittest.TestCase):
    def test_can_continue_pending_task_and_run_multi_tool_chain(self) -> None:
        router = _FakeClient(
            [
                '{"route": "internal_tool", "intent": "order_product_policy_lookup", "tool_name": "get_order_status", '
                '"tool_arguments": {}, "missing_slots": ["order_id"], "need_clarification": true, '
                '"rewrite_query": "这单里的商品怎么洗，还能退吗"}',
                '{"route": "internal_tool", "intent": "order_product_policy_lookup", "tool_name": "get_order_status", '
                '"tool_arguments": {"order_id": "A1001"}, "missing_slots": [], "need_clarification": false, '
                '"rewrite_query": "这单里的商品怎么洗，还能退吗"}',
            ]
        )
        answer = _FakeClient(
            [
                '{"answer": "请先提供订单号。", "citations": [], "grounded": false, '
                '"escalation_required": false, "waiting_for_user": true, "episode_done": false}',
                '{"answer": "订单 A1001 对应商品是 防风冲锋衣。护理建议见商品页，同时支持 7 天无理由退货。", '
                '"citations": ["get_order_status", "get_product_info", "get_policy"], "grounded": true, '
                '"escalation_required": false, "waiting_for_user": false, "episode_done": true}',
            ]
        )
        orchestrator = build_model_orchestrator(router, answer)

        first_trace = orchestrator.handle_trace(
            ChatRequest(query="这单里的商品怎么洗，还能退吗", shop_id="demo-shop"),
            state=EpisodeState(shop_id="demo-shop"),
        )
        self.assertTrue(first_trace.response.waiting_for_user)
        self.assertEqual(first_trace.route_decision.missing_slots, ["order_id"])

        second_trace = orchestrator.handle_trace(
            ChatRequest(query="A1001", shop_id="demo-shop"),
            state=first_trace.state_after,
        )
        self.assertEqual(
            [step.result.tool_name for step in second_trace.tool_steps],
            ["get_order_status", "get_product_info", "get_policy"],
        )
        self.assertTrue(second_trace.response.grounded)
        self.assertFalse(second_trace.response.waiting_for_user)


if __name__ == "__main__":
    unittest.main()
