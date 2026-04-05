from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional
from uuid import uuid4

from app.config import get_settings
from app.exporters.training_data import compact_state
from app.models import (
    ChatRequest,
    ChatResponse,
    EpisodeRecord,
    EpisodeState,
    Route,
    TaskState,
    ToolCall,
    ToolStep,
    TraceRecord,
)
from app.services.answer import AnswerService
from app.services.internal_tools import InternalToolService
from app.services.router import RouterService


class QueryOrchestrator:
    """单任务 episode 编排器。"""

    def __init__(
        self,
        router: RouterService,
        internal_tools: InternalToolService,
        answer_service: AnswerService,
    ) -> None:
        self._router = router
        self._internal_tools = internal_tools
        self._answer_service = answer_service

    def handle_query(
        self,
        request: ChatRequest,
        state: Optional[EpisodeState] = None,
    ) -> tuple[ChatResponse, EpisodeState]:
        trace = self.handle_trace(request, state=state)
        return trace.response, trace.state_after or EpisodeState()

    def handle_trace(
        self,
        request: ChatRequest,
        state: Optional[EpisodeState] = None,
    ) -> TraceRecord:
        settings = get_settings()
        state_before = self._clone_state(state, default_shop_id=request.shop_id or settings.default_shop_id)
        effective_request = self._merge_request_with_state(request, state_before)

        if self._should_continue_pending(state_before):
            decision = self._router.continue_pending(
                query=effective_request.query,
                intent=state_before.current_task.intent or "general_direct_answer",
                tool_name=state_before.current_task.planned_tool_name or "",
                pending_arguments=state_before.current_task.planned_arguments,
                missing_slots=state_before.current_task.missing_slots,
                request=effective_request,
            )
        else:
            decision = self._router.route(effective_request)

        shop_id = effective_request.shop_id or settings.default_shop_id
        task_query = (
            state_before.current_task.task_query
            if self._should_continue_pending(state_before) and state_before.current_task.task_query
            else effective_request.query
        )

        evidence = []
        tool_call = None
        tool_result = None
        tool_steps: list[ToolStep] = []
        if decision.route == Route.INTERNAL_TOOL and decision.tool_name and not decision.missing_slots:
            tool_steps = self._execute_tool_chain(
                decision=decision,
                query=task_query,
                shop_id=shop_id,
                max_steps=settings.max_tool_chain_steps,
            )
            if tool_steps:
                tool_call = tool_steps[-1].call
                tool_result = tool_steps[-1].result

        answer = self._answer_service.generate(
            task_query,
            decision,
            evidence,
            tool_result=tool_result,
            tool_steps=tool_steps,
        )

        waiting_for_user = bool(decision.missing_slots)
        episode_done = not waiting_for_user
        state_after = self._update_state(
            state=state_before,
            effective_request=effective_request,
            task_query=task_query,
            decision=decision,
            tool_steps=tool_steps,
        )

        response = ChatResponse(
            route=decision.route,
            intent=decision.intent,
            answer=answer.answer,
            confidence=decision.confidence,
            rewrite_query=decision.rewrite_query,
            evidence=evidence,
            tool_call=tool_call,
            tool_result=tool_result,
            tool_steps=tool_steps,
            citations=answer.citations,
            grounded=answer.grounded,
            escalation_required=answer.escalation_required,
            waiting_for_user=waiting_for_user,
            episode_done=episode_done,
            debug={
                "rationale": decision.rationale,
                "filters": {**decision.filters, "resolved_shop_id": shop_id},
                "evidence_count": len(evidence),
                "missing_slots": decision.missing_slots,
                "task_query": task_query,
                "current_task": state_after.current_task.model_dump(mode="json"),
                "state_before": state_before.model_dump(mode="json"),
                "state_after": state_after.model_dump(mode="json"),
            },
        )
        return TraceRecord(
            request=effective_request,
            route_decision=decision,
            tool_call=tool_call,
            tool_result=tool_result,
            tool_steps=tool_steps,
            response=response,
            state_before=state_before,
            state_after=state_after,
        )

    def run_episode(
        self,
        turns: Iterable[ChatRequest],
        episode_id: Optional[str] = None,
        initial_state: Optional[EpisodeState] = None,
    ) -> EpisodeRecord:
        current_state = self._clone_state(initial_state)
        traces: list[TraceRecord] = []

        for request in turns:
            trace = self.handle_trace(request, state=current_state)
            traces.append(trace)
            current_state = trace.state_after or EpisodeState()

        if not traces:
            raise ValueError("episode 至少需要一轮用户输入。")

        final_response = traces[-1].response
        return EpisodeRecord(
            episode_id=episode_id or f"episode-{uuid4().hex[:8]}",
            turns=traces,
            final_response=final_response,
            final_state=current_state,
            waiting_for_user=final_response.waiting_for_user,
            completed=final_response.episode_done,
        )

    @staticmethod
    def _clone_state(
        state: Optional[EpisodeState],
        default_shop_id: Optional[str] = None,
    ) -> EpisodeState:
        if state is None:
            return EpisodeState(shop_id=default_shop_id)
        payload = state.model_dump(mode="json")
        if payload.get("shop_id") is None:
            payload["shop_id"] = default_shop_id
        return EpisodeState.model_validate(payload)

    @staticmethod
    def _merge_request_with_state(request: ChatRequest, state: EpisodeState) -> ChatRequest:
        metadata = dict(request.metadata)
        # 把当前状态快照带给模型版 router，保证运行时提示和训练/benchmark 使用同一份 state_before。
        metadata["_state_before"] = compact_state(state.model_dump(mode="json"))
        return ChatRequest(
            query=request.query,
            user_id=request.user_id,
            shop_id=request.shop_id or state.shop_id,
            product_id=request.product_id or state.product_id,
            order_id=request.order_id or state.order_id,
            metadata=metadata,
        )

    @staticmethod
    def _should_continue_pending(state: EpisodeState) -> bool:
        task = state.current_task
        return bool(task.planned_tool_name and task.missing_slots)

    def _update_state(
        self,
        state: EpisodeState,
        effective_request: ChatRequest,
        task_query: str,
        decision,
        tool_steps: list[ToolStep],
    ) -> EpisodeState:
        next_state = self._clone_state(state)
        next_state.turn_index += 1
        next_state.shop_id = effective_request.shop_id or next_state.shop_id
        next_state.product_id = effective_request.product_id or next_state.product_id
        next_state.order_id = effective_request.order_id or next_state.order_id
        next_state.recent_queries = [*next_state.recent_queries[-4:], effective_request.query]

        next_state.current_task.resolved_slots = {
            "shop_id": next_state.shop_id,
            "product_id": next_state.product_id,
            "order_id": next_state.order_id,
        }

        if decision.route == Route.INTERNAL_TOOL and decision.tool_name:
            if decision.missing_slots:
                next_state.current_task = TaskState(
                    task_query=task_query,
                    intent=decision.intent,
                    route=decision.route,
                    planned_tool_name=decision.tool_name,
                    planned_arguments=dict(decision.tool_arguments),
                    missing_slots=list(decision.missing_slots),
                    resolved_slots={
                        "shop_id": next_state.shop_id,
                        "product_id": next_state.product_id,
                        "order_id": next_state.order_id,
                    },
                    status="pending_clarification",
                )
            else:
                next_state.current_task = TaskState(
                    task_query=task_query,
                    intent=decision.intent,
                    route=decision.route,
                    planned_tool_name=decision.tool_name,
                    planned_arguments=dict(decision.tool_arguments),
                    missing_slots=[],
                    resolved_slots={
                        "shop_id": next_state.shop_id,
                        "product_id": next_state.product_id,
                        "order_id": next_state.order_id,
                    },
                    status="completed",
                )
        else:
            next_state.current_task = TaskState(
                task_query=task_query,
                intent=decision.intent,
                route=decision.route,
                planned_tool_name=decision.tool_name,
                planned_arguments=dict(decision.tool_arguments),
                missing_slots=list(decision.missing_slots),
                resolved_slots={
                    "shop_id": next_state.shop_id,
                    "product_id": next_state.product_id,
                    "order_id": next_state.order_id,
                },
                status="completed",
            )

        for step in tool_steps:
            tool_result = step.result
            if tool_result.status != "ok":
                continue
            if tool_result.tool_name == "get_product_info":
                next_state.product_id = tool_result.data.get("product_id") or next_state.product_id
            if tool_result.tool_name in {"get_order_status", "get_logistics_status"}:
                next_state.order_id = tool_result.data.get("order_id") or next_state.order_id
            next_state.current_task.resolved_slots = {
                "shop_id": next_state.shop_id,
                "product_id": next_state.product_id,
                "order_id": next_state.order_id,
            }

        return next_state

    def _execute_tool_chain(
        self,
        decision,
        query: str,
        shop_id: str,
        max_steps: int,
    ) -> list[ToolStep]:
        tool_steps: list[ToolStep] = []
        pending_call = ToolCall(name=decision.tool_name, arguments=decision.tool_arguments)
        seen_calls: set[tuple[str, tuple[tuple[str, str], ...]]] = set()

        while pending_call is not None and len(tool_steps) < max_steps:
            call_signature = (
                pending_call.name,
                tuple(sorted((str(key), str(value)) for key, value in pending_call.arguments.items())),
            )
            if call_signature in seen_calls:
                break
            seen_calls.add(call_signature)

            result = self._internal_tools.execute(
                tool_name=pending_call.name,
                arguments=pending_call.arguments,
                shop_id=shop_id,
            )
            tool_steps.append(ToolStep(call=pending_call, result=result))

            if result.status != "ok":
                break

            pending_call = self._plan_next_tool_call(
                intent=decision.intent,
                query=query,
                tool_steps=tool_steps,
            )

        return tool_steps

    def _plan_next_tool_call(
        self,
        intent: str,
        query: str,
        tool_steps: list[ToolStep],
    ) -> ToolCall | None:
        results_by_tool = {step.result.tool_name: step.result for step in tool_steps}
        if intent not in {"order_product_lookup", "order_policy_lookup", "order_product_policy_lookup"}:
            return None

        order_result = results_by_tool.get("get_order_status")
        if order_result is None or order_result.status != "ok":
            return None

        if intent in {"order_product_lookup", "order_product_policy_lookup"} and "get_product_info" not in results_by_tool:
            product_id = order_result.data.get("product_id")
            if product_id:
                return ToolCall(name="get_product_info", arguments={"product_id": product_id})

        if intent in {"order_policy_lookup", "order_product_policy_lookup"} and "get_policy" not in results_by_tool:
            topic = self._router._infer_policy_topic(query)
            return ToolCall(name="get_policy", arguments={"topic": topic})

        return None


@lru_cache(maxsize=1)
def get_orchestrator() -> QueryOrchestrator:
    return QueryOrchestrator(
        router=RouterService(),
        internal_tools=InternalToolService(),
        answer_service=AnswerService(),
    )
