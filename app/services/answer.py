from __future__ import annotations

from typing import List, Optional

from app.models import AnswerPayload, Evidence, Route, RouteDecision, ToolResult, ToolStep


class AnswerService:
    """MVP 阶段的规则式回答渲染器。"""

    def generate(
        self,
        query: str,
        decision: RouteDecision,
        evidence: List[Evidence],
        tool_result: Optional[ToolResult] = None,
        tool_steps: Optional[List[ToolStep]] = None,
    ) -> AnswerPayload:
        if decision.route == Route.HANDOFF:
            return AnswerPayload(
                answer=(
                    "这个问题建议转接人工客服处理，因为它可能涉及投诉、赔偿、规则风险，"
                    "或者需要人工审核后才能继续处理。"
                ),
                grounded=False,
                escalation_required=True,
            )

        if decision.need_clarification and decision.missing_slots:
            return AnswerPayload(
                answer=self._clarification_prompt(decision.missing_slots, decision.intent),
                grounded=False,
                escalation_required=False,
            )

        if decision.route == Route.DIRECT:
            return AnswerPayload(
                answer=(
                    "我可以帮助处理商品信息、店铺规则、订单状态、物流进度和人工转接相关问题。"
                    "如果你告诉我商品 ID、订单号，或者具体问题主题，我可以回答得更准确。"
                ),
                grounded=False,
                escalation_required=False,
            )

        if decision.route == Route.INTERNAL_TOOL:
            return self._generate_tool_answer(query, decision, tool_result, tool_steps or [])

        return AnswerPayload(
            answer="当前路由没有可执行的处理分支，请补充信息或调整路由策略。",
            grounded=False,
            escalation_required=False,
        )

    @staticmethod
    def _clarification_prompt(missing_slots: List[str], intent: str) -> str:
        if "order_id" in missing_slots:
            return "这个问题需要订单号才能继续查询，请先提供订单号。"
        if "product_id" in missing_slots and intent == "product_lookup":
            return "这个问题需要具体商品才能查询，请提供商品 ID 或商品名称。"
        return "为了更准确地回答，我还需要你补充一些关键信息。"

    def _generate_tool_answer(
        self,
        query: str,
        decision: RouteDecision,
        tool_result: Optional[ToolResult],
        tool_steps: List[ToolStep],
    ) -> AnswerPayload:
        by_tool = {step.result.tool_name: step.result for step in tool_steps}
        if decision.intent in {"order_product_lookup", "order_policy_lookup", "order_product_policy_lookup"}:
            return self._generate_multi_tool_answer(query, decision, by_tool)

        if tool_result is None:
            return AnswerPayload(
                answer="当前没有可用的工具结果，请稍后重试。",
                grounded=False,
                escalation_required=False,
            )

        if tool_result.status == "not_found":
            return AnswerPayload(
                answer=f"{tool_result.message}。请检查你提供的商品 ID、订单号或查询主题。",
                grounded=False,
                escalation_required=False,
            )

        if tool_result.status == "missing_args":
            return AnswerPayload(
                answer=tool_result.message,
                grounded=False,
                escalation_required=False,
            )

        if tool_result.status != "ok":
            return AnswerPayload(
                answer="工具调用失败，请稍后重试或转人工处理。",
                grounded=False,
                escalation_required=False,
            )

        data = tool_result.data
        if decision.intent == "product_lookup":
            lowered = query.lower()
            if "材质" in query or "material" in lowered:
                answer = f"{data['name']} 的材质是 {data['material']}。"
            elif "颜色" in query or "color" in lowered:
                answer = f"{data['name']} 当前颜色有 {', '.join(data['colors'])}。"
            elif "尺码" in query or "size" in lowered:
                answer = f"{data['name']} 当前尺码有 {', '.join(data['sizes'])}。"
            elif "怎么洗" in query or "洗" in query or "care" in lowered:
                answer = f"{data['name']} 的护理建议是：{data['care_instructions']}"
            else:
                answer = (
                    f"{data['name']} 的材质是 {data['material']}，"
                    f"当前颜色有 {', '.join(data['colors'])}，尺码有 {', '.join(data['sizes'])}。"
                )
        elif decision.intent == "policy_lookup":
            answer = data["summary"]
        elif decision.intent == "order_status":
            lowered = query.lower()
            if "退款" in query or "refund" in lowered:
                answer = f"订单 {data['order_id']} 当前退款状态为 {data['refund_status']}。"
            elif "支付" in query:
                answer = f"订单 {data['order_id']} 当前支付状态为 {data['payment_status']}。"
            else:
                answer = (
                    f"订单 {data['order_id']} 当前支付状态为 {data['payment_status']}，"
                    f"履约状态为 {data['fulfillment_status']}，退款状态为 {data['refund_status']}。"
                )
        elif decision.intent == "logistics_status":
            lowered = query.lower()
            if "运单" in query or "tracking" in lowered:
                answer = (
                    f"订单 {data['order_id']} 的运单号是 {data['tracking_no']}，"
                    f"承运商是 {data['carrier']}。"
                )
            else:
                answer = (
                    f"订单 {data['order_id']} 当前物流状态为 {data['status']}，"
                    f"承运商是 {data['carrier']}，最新物流信息是：{data['latest_event']}。"
                )
        else:
            answer = "工具已返回结果，但当前没有定义对应的回答模板。"

        return AnswerPayload(
            answer=answer,
            citations=[tool_result.tool_name],
            grounded=True,
            escalation_required=False,
        )

    def _generate_multi_tool_answer(
        self,
        query: str,
        decision: RouteDecision,
        by_tool: dict[str, ToolResult],
    ) -> AnswerPayload:
        for result in by_tool.values():
            if result.status != "ok":
                return AnswerPayload(
                    answer="多工具查询未能成功完成，请检查订单号或稍后重试。",
                    grounded=False,
                    escalation_required=False,
                )

        order = by_tool.get("get_order_status")
        product = by_tool.get("get_product_info")
        policy = by_tool.get("get_policy")
        if order is None:
            return AnswerPayload(
                answer="当前缺少订单查询结果，暂时无法完成多工具回答。",
                grounded=False,
                escalation_required=False,
            )

        order_data = order.data
        citations = [name for name in ("get_order_status", "get_product_info", "get_policy") if name in by_tool]
        product_name = order_data.get("product_name", "该商品")

        if decision.intent == "order_product_lookup":
            if product is None:
                return AnswerPayload(
                    answer="当前缺少商品查询结果，暂时无法完成回答。",
                    grounded=False,
                    escalation_required=False,
                )
            answer = (
                f"订单 {order_data['order_id']} 对应商品是 {product_name}。"
                f"{self._render_product_answer(query, product.data, fallback_name=product_name)}"
            )
            return AnswerPayload(answer=answer, citations=citations, grounded=True, escalation_required=False)

        if decision.intent == "order_policy_lookup":
            if policy is None:
                return AnswerPayload(
                    answer="当前缺少规则查询结果，暂时无法完成回答。",
                    grounded=False,
                    escalation_required=False,
                )
            answer = (
                f"订单 {order_data['order_id']} 对应商品是 {product_name}。"
                f"{policy.data['summary']}"
            )
            return AnswerPayload(answer=answer, citations=citations, grounded=True, escalation_required=False)

        if product is None or policy is None:
            return AnswerPayload(
                answer="当前多工具链缺少必要结果，暂时无法完成回答。",
                grounded=False,
                escalation_required=False,
            )

        product_answer = self._render_product_answer(query, product.data, fallback_name=product_name, short=True)
        answer = (
            f"订单 {order_data['order_id']} 对应商品是 {product_name}。"
            f"{product_answer}"
            f"{policy.data['summary']}"
        )
        return AnswerPayload(answer=answer, citations=citations, grounded=True, escalation_required=False)

    @staticmethod
    def _render_product_answer(
        query: str,
        product_data: dict,
        fallback_name: str,
        short: bool = False,
    ) -> str:
        lowered = query.lower()
        name = product_data.get("name", fallback_name)
        if "材质" in query or "material" in lowered:
            return f"{name} 的材质是 {product_data['material']}。"
        if "颜色" in query or "color" in lowered:
            return f"{name} 当前颜色有 {', '.join(product_data['colors'])}。"
        if "尺码" in query or "size" in lowered:
            return f"{name} 当前尺码有 {', '.join(product_data['sizes'])}。"
        if "怎么洗" in query or "洗" in query or "care" in lowered:
            return f"{name} 的护理建议是：{product_data['care_instructions']}"
        if short:
            return f"{name} 的材质是 {product_data['material']}。"
        return (
            f"{name} 的材质是 {product_data['material']}，"
            f"当前颜色有 {', '.join(product_data['colors'])}，尺码有 {', '.join(product_data['sizes'])}。"
        )
