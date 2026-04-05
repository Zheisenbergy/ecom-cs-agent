from __future__ import annotations

from typing import Any, Optional

from app.exporters.training_data import compact_state
from app.models import AnswerPayload, ChatRequest, EpisodeState, Route, RouteDecision, ToolResult, ToolStep
from app.services.internal_tools import InternalToolService
from app.services.model_io import (
    TextGenerationClient,
    build_answer_completion_prompt,
    build_router_completion_prompt,
    extract_json_object,
    normalize_answer_prediction,
    normalize_router_prediction,
)
from app.services.orchestrator import QueryOrchestrator


class ModelRouterService:
    """把 OpenAI-compatible router 模型适配成现有 orchestrator 可调用的接口。"""

    def __init__(self, client: TextGenerationClient) -> None:
        self._client = client

    def route(self, request: ChatRequest) -> RouteDecision:
        return self._route_with_state(request)

    def continue_pending(
        self,
        query: str,
        intent: str,
        tool_name: str,
        pending_arguments: dict[str, Any],
        missing_slots: list[str],
        request: ChatRequest,
    ) -> RouteDecision:
        return self._route_with_state(
            request,
            fallback_decision=RouteDecision(
                route=Route.INTERNAL_TOOL,
                intent=intent,
                need_clarification=bool(missing_slots),
                confidence=0.0,
                rewrite_query=query.strip(),
                filters={
                    "shop_id": request.shop_id,
                    "product_id": request.product_id,
                    "order_id": request.order_id,
                },
                tool_name=tool_name,
                tool_arguments=dict(pending_arguments),
                missing_slots=list(missing_slots),
                rationale="router 模型输出解析失败，已沿用上一轮待补槽位计划。",
            ),
        )

    def _route_with_state(
        self,
        request: ChatRequest,
        fallback_decision: Optional[RouteDecision] = None,
    ) -> RouteDecision:
        state_before = self._extract_state_before(request)
        raw = self._client.generate(
            build_router_completion_prompt(
                user_query=request.query,
                state_before=state_before,
            )
        )
        parsed = extract_json_object(raw)
        if parsed is None:
            fallback = fallback_decision or RouteDecision(
                route=Route.DIRECT,
                intent="general_direct_answer",
                need_clarification=False,
                confidence=0.0,
                rewrite_query=request.query.strip(),
                filters={
                    "shop_id": request.shop_id,
                    "product_id": request.product_id,
                    "order_id": request.order_id,
                },
                tool_name="",
                tool_arguments={},
                missing_slots=[],
                rationale=f"router 模型输出解析失败，已回退为 direct。raw={raw[:200]}",
            )
            return fallback

        prediction = normalize_router_prediction(parsed)
        rewrite_query = prediction.get("rewrite_query") or request.query.strip()
        route_value = prediction["route"]

        return RouteDecision(
            route=Route(route_value),
            intent=prediction["intent"],
            time_sensitive=False,
            need_clarification=bool(prediction["need_clarification"]),
            confidence=0.0,
            rewrite_query=rewrite_query,
            filters={
                "shop_id": request.shop_id,
                "product_id": prediction["tool_arguments"].get("product_id") or state_before.get("product_id"),
                "order_id": prediction["tool_arguments"].get("order_id") or state_before.get("order_id"),
            },
            tool_name=prediction["tool_name"] or None,
            tool_arguments=prediction["tool_arguments"],
            missing_slots=list(prediction["missing_slots"]),
            rationale="模型路由决策。",
        )

    @staticmethod
    def _extract_state_before(request: ChatRequest) -> dict[str, Any]:
        raw_state = request.metadata.get("_state_before", {})
        if isinstance(raw_state, dict):
            return compact_state(raw_state)

        state = EpisodeState(
            shop_id=request.shop_id,
            product_id=request.product_id,
            order_id=request.order_id,
        )
        return compact_state(state.model_dump(mode="json"))

    @staticmethod
    def _infer_policy_topic(query: str) -> str:
        mapping = {
            "退货": "return_policy",
            "退款": "refund_policy",
            "换货": "exchange_policy",
            "保修": "warranty_policy",
            "发票": "invoice_policy",
            "发货": "shipping_policy",
        }
        for keyword, topic in mapping.items():
            if keyword in query:
                return topic
        return "return_policy"


class ModelAnswerService:
    """把 OpenAI-compatible answer 模型适配成现有 orchestrator 可调用的接口。"""

    def __init__(self, client: TextGenerationClient) -> None:
        self._client = client

    def generate(
        self,
        query: str,
        decision: RouteDecision,
        evidence: list[Any],
        tool_result: Optional[ToolResult] = None,
        tool_steps: Optional[list[ToolStep]] = None,
    ) -> AnswerPayload:
        del evidence, tool_result

        serialized_steps = [
            {
                "tool_name": step.result.tool_name,
                "status": step.result.status,
                "data": step.result.data,
                "message": step.result.message,
            }
            for step in (tool_steps or [])
        ]
        raw = self._client.generate(
            build_answer_completion_prompt(
                query=query,
                route=decision.route.value,
                intent=decision.intent,
                tool_steps=serialized_steps,
            )
        )
        parsed = extract_json_object(raw)
        if parsed is None:
            return AnswerPayload(
                answer=raw.strip() or "answer 模型输出解析失败。",
                grounded=False,
                escalation_required=decision.route == Route.HANDOFF,
            )

        prediction = normalize_answer_prediction(parsed)
        return AnswerPayload(
            answer=prediction["answer"],
            citations=prediction["citations"],
            grounded=prediction["grounded"],
            escalation_required=prediction["escalation_required"],
        )


def build_model_orchestrator(
    router_client: TextGenerationClient,
    answer_client: TextGenerationClient,
) -> QueryOrchestrator:
    return QueryOrchestrator(
        router=ModelRouterService(router_client),
        internal_tools=InternalToolService(),
        answer_service=ModelAnswerService(answer_client),
    )
