from __future__ import annotations

import re
from typing import Any, Dict, Optional

from app.models import ChatRequest, Route, RouteDecision


class RouterService:
    """当前使用确定性规则，后续可替换为 Qwen 路由模型。"""

    _HANDOFF_KEYWORDS = ("complaint", "manual support", "refund dispute", "赔偿", "投诉", "人工")
    _LOGISTICS_KEYWORDS = ("shipping", "delivery", "物流", "发货", "快递", "到哪", "到哪了", "运单")
    _ORDER_CONTEXT_KEYWORDS = ("order", "订单", "我买的", "我订单", "这笔订单", "这单")
    _ORDER_STATUS_KEYWORDS = (
        "order",
        "订单",
        "订单状态",
        "退款进度",
        "退款状态",
        "到账",
        "改地址",
        "支付状态",
        "履约状态",
    )
    _POLICY_KEYWORDS = (
        "return",
        "refund",
        "exchange",
        "warranty",
        "policy",
        "rule",
        "退货",
        "退款",
        "换货",
        "保修",
        "售后",
        "规则",
        "政策",
        "能退",
        "退吗",
        "发票",
        "开票",
        "质保",
    )
    _PRODUCT_KEYWORDS = (
        "size",
        "material",
        "color",
        "feature",
        "care",
        "尺码",
        "材质",
        "颜色",
        "参数",
        "功能",
        "特点",
        "商品",
        "衣服",
        "鞋",
        "怎么洗",
        "洗",
        "清洁",
        "清洗",
    )
    _PRODUCT_ATTRIBUTE_KEYWORDS = (
        "material",
        "color",
        "size",
        "feature",
        "care",
        "材质",
        "颜色",
        "尺码",
        "参数",
        "功能",
        "特点",
        "怎么洗",
        "洗",
        "清洁",
        "清洗",
    )
    _RATIONALES = {
        "complaint_or_manual_support": "涉及投诉、赔偿或人工处理的请求应优先转人工。",
        "order_product_lookup": "订单上下文下的商品属性问题应先查订单，再查商品信息。",
        "order_policy_lookup": "订单上下文下的售后规则问题应先查订单，再查政策工具。",
        "order_product_policy_lookup": "同时涉及订单商品属性与售后规则的问题需要串联多个内部工具。",
        "policy_lookup": "售后与店铺规则问题应走内部政策工具。",
        "product_lookup": "商品属性问题应走商品信息工具。",
        "logistics_status": "物流与配送进度问题应走物流状态工具。",
        "order_status": "订单与售后进度问题应走订单状态工具。",
        "general_direct_answer": "没有检测到明显的商品、政策或时效性信号，先按直接回答处理。",
    }

    def route(self, request: ChatRequest) -> RouteDecision:
        query = request.query.strip()
        lowered = query.lower()
        extracted_slots = self.extract_slots(query)
        extracted_order_id = extracted_slots.get("order_id") or request.order_id
        extracted_product_id = extracted_slots.get("product_id") or request.product_id

        intent, route = self._resolve_route(
            query=query,
            lowered=lowered,
            extracted_order_id=extracted_order_id,
            extracted_product_id=extracted_product_id,
        )
        if route != Route.DIRECT:
            tool_name, tool_arguments, missing_slots = self._tool_plan(
                intent=intent,
                request=request,
                extracted_order_id=extracted_order_id,
                extracted_product_id=extracted_product_id,
            )
            return RouteDecision(
                route=route,
                intent=intent,
                time_sensitive=False,
                need_clarification=bool(missing_slots)
                or self._needs_clarification(
                    query,
                    has_context=bool(extracted_order_id or extracted_product_id),
                ),
                confidence=0.82 if route != Route.DIRECT else 0.65,
                rewrite_query=self._rewrite_query(query),
                filters={"shop_id": request.shop_id, "product_id": extracted_product_id, "order_id": extracted_order_id},
                tool_name=tool_name,
                tool_arguments=tool_arguments,
                missing_slots=missing_slots,
                rationale=self._RATIONALES[intent],
            )

        return RouteDecision(
            route=Route.DIRECT,
            intent="general_direct_answer",
            time_sensitive=False,
            need_clarification=self._needs_clarification(
                query,
                has_context=bool(extracted_order_id or extracted_product_id),
            ),
            confidence=0.58,
            rewrite_query=self._rewrite_query(query),
            filters={"shop_id": request.shop_id, "product_id": extracted_product_id, "order_id": extracted_order_id},
            rationale=self._RATIONALES["general_direct_answer"],
        )

    def _resolve_route(
        self,
        query: str,
        lowered: str,
        extracted_order_id: Optional[str],
        extracted_product_id: Optional[str],
    ) -> tuple[str, Route]:
        if self._contains_any(query, lowered, self._HANDOFF_KEYWORDS):
            return "complaint_or_manual_support", Route.HANDOFF

        if self._contains_any(query, lowered, self._LOGISTICS_KEYWORDS):
            return "logistics_status", Route.INTERNAL_TOOL

        if self._is_order_context_query(query, lowered, has_order_context=bool(extracted_order_id)):
            asks_product_attr = self._is_product_attribute_query(query, lowered)
            asks_policy = self._is_policy_query(query, lowered)
            if asks_product_attr and asks_policy:
                return "order_product_policy_lookup", Route.INTERNAL_TOOL
            if asks_product_attr:
                return "order_product_lookup", Route.INTERNAL_TOOL
            if asks_policy:
                return "order_policy_lookup", Route.INTERNAL_TOOL

        if self._is_order_status_query(query, lowered, has_order_context=bool(extracted_order_id)):
            return "order_status", Route.INTERNAL_TOOL

        if self._is_policy_query(query, lowered):
            return "policy_lookup", Route.INTERNAL_TOOL

        if self._is_product_query(query, lowered, has_product_context=bool(extracted_product_id)):
            return "product_lookup", Route.INTERNAL_TOOL

        return "general_direct_answer", Route.DIRECT

    @staticmethod
    def _needs_clarification(query: str, has_context: bool = False) -> bool:
        if has_context:
            return False
        return len(query.strip()) < 5

    @staticmethod
    def _rewrite_query(query: str) -> str:
        return " ".join(query.replace("?", " ").replace("？", " ").split())

    @staticmethod
    def _contains_any(query: str, lowered: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in lowered or keyword in query for keyword in keywords)

    @staticmethod
    def _extract_order_id(query: str) -> Optional[str]:
        match = re.search(r"\b([A-Z]{1,3}\d{3,})\b", query.upper())
        if match:
            return match.group(1)
        return None

    def _tool_plan(
        self,
        intent: str,
        request: ChatRequest,
        extracted_order_id: Optional[str],
        extracted_product_id: Optional[str],
    ) -> tuple[Optional[str], Dict[str, str], list[str]]:
        if intent == "policy_lookup":
            topic = self._infer_policy_topic(request.query)
            return "get_policy", {"topic": topic}, []

        if intent in {"order_product_lookup", "order_policy_lookup", "order_product_policy_lookup"}:
            if not extracted_order_id:
                return "get_order_status", {}, ["order_id"]
            return "get_order_status", {"order_id": extracted_order_id}, []

        if intent == "product_lookup":
            product_id = extracted_product_id
            if not product_id:
                return "get_product_info", {}, ["product_id"]
            return "get_product_info", {"product_id": product_id}, []

        if intent == "order_status":
            if not extracted_order_id:
                return "get_order_status", {}, ["order_id"]
            return "get_order_status", {"order_id": extracted_order_id}, []

        if intent == "logistics_status":
            if not extracted_order_id:
                return "get_logistics_status", {}, ["order_id"]
            return "get_logistics_status", {"order_id": extracted_order_id}, []

        return None, {}, []

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

    def _is_policy_query(self, query: str, lowered: str) -> bool:
        if any(phrase in query or phrase in lowered for phrase in ("退款状态", "退款进度", "订单状态", "支付状态", "履约状态")):
            return False
        return self._contains_any(query, lowered, self._POLICY_KEYWORDS)

    def _is_order_context_query(self, query: str, lowered: str, has_order_context: bool) -> bool:
        return has_order_context or self._contains_any(query, lowered, self._ORDER_CONTEXT_KEYWORDS)

    def _is_order_status_query(self, query: str, lowered: str, has_order_context: bool) -> bool:
        if self._contains_any(query, lowered, self._ORDER_STATUS_KEYWORDS):
            return True
        if has_order_context and any(keyword in query or keyword in lowered for keyword in ("退款", "状态", "进度", "到账", "改地址")):
            return True
        return False

    def _is_product_attribute_query(self, query: str, lowered: str) -> bool:
        return self._contains_any(query, lowered, self._PRODUCT_ATTRIBUTE_KEYWORDS)

    def _is_product_query(self, query: str, lowered: str, has_product_context: bool) -> bool:
        if self._contains_any(query, lowered, self._PRODUCT_KEYWORDS):
            return True
        if has_product_context and any(keyword in query or keyword in lowered for keyword in ("还有", "颜色", "尺码", "材质", "参数", "功能", "怎么洗", "洗", "清洁", "清洗")):
            return True
        return False

    @staticmethod
    def _extract_product_id(query: str) -> Optional[str]:
        mapping = {
            "stormshell": "sku_stormshell_001",
            "冲锋衣": "sku_stormshell_001",
            "urbanstep": "sku_urbanstep_002",
            "休闲鞋": "sku_urbanstep_002",
            "thermomug": "sku_thermomug_003",
            "保温杯": "sku_thermomug_003",
            "cloudrest": "sku_cloudrest_004",
            "记忆枕": "sku_cloudrest_004",
            "flexfit": "sku_flexfit_005",
            "运动裤": "sku_flexfit_005",
            "aeropack": "sku_aeropack_006",
            "双肩包": "sku_aeropack_006",
        }
        lowered = query.lower()
        for keyword, product_id in mapping.items():
            if keyword in lowered or keyword in query:
                return product_id
        return None

    def extract_slots(self, query: str) -> Dict[str, Optional[str]]:
        return {
            "order_id": self._extract_order_id(query),
            "product_id": self._extract_product_id(query),
        }

    def continue_pending(
        self,
        query: str,
        intent: str,
        tool_name: str,
        pending_arguments: Dict[str, Any],
        missing_slots: list[str],
        request: ChatRequest,
    ) -> RouteDecision:
        extracted_slots = self.extract_slots(query)
        merged_arguments = dict(pending_arguments)
        remaining_slots: list[str] = []

        for slot in missing_slots:
            if slot == "order_id":
                value = request.order_id or extracted_slots.get("order_id")
            elif slot == "product_id":
                value = request.product_id or extracted_slots.get("product_id")
            else:
                value = None

            if value:
                merged_arguments[slot] = value
            else:
                remaining_slots.append(slot)

        return RouteDecision(
            route=Route.INTERNAL_TOOL,
            intent=intent,
            time_sensitive=False,
            need_clarification=bool(remaining_slots),
            confidence=0.88,
            rewrite_query=self._rewrite_query(query),
            filters={
                "shop_id": request.shop_id,
                "product_id": merged_arguments.get("product_id"),
                "order_id": merged_arguments.get("order_id"),
            },
            tool_name=tool_name,
            tool_arguments=merged_arguments,
            missing_slots=remaining_slots,
            rationale="延续上一轮未完成的工具查询。",
        )
