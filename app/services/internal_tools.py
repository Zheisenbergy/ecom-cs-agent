from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.models import ToolResult


class InternalToolService:
    """基于本地 mock 数据的内部工具层。"""

    def __init__(self, kb_root: Optional[Path] = None) -> None:
        settings = get_settings()
        self._kb_root = kb_root or settings.resolved_kb_path.parent

    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        shop_id: Optional[str] = None,
    ) -> ToolResult:
        handlers = {
            "get_product_info": self._get_product_info,
            "get_policy": self._get_policy,
            "get_order_status": self._get_order_status,
            "get_logistics_status": self._get_logistics_status,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return ToolResult(
                tool_name=tool_name,
                status="error",
                data={},
                message=f"未知工具: {tool_name}",
            )
        return handler(arguments, shop_id=shop_id)

    def _get_product_info(self, arguments: Dict[str, Any], shop_id: Optional[str]) -> ToolResult:
        product_id = arguments.get("product_id")
        if not product_id:
            return ToolResult(tool_name="get_product_info", status="missing_args", message="缺少 product_id")

        for item in self._load_json("products.json"):
            if item["product_id"] == product_id and item["shop_id"] == shop_id:
                return ToolResult(tool_name="get_product_info", status="ok", data=item)
        return ToolResult(tool_name="get_product_info", status="not_found", message="未找到对应商品")

    def _get_policy(self, arguments: Dict[str, Any], shop_id: Optional[str]) -> ToolResult:
        topic = arguments.get("topic", "return_policy")
        for item in self._load_json("policies.json"):
            if item["topic"] == topic and item["shop_id"] == shop_id:
                return ToolResult(tool_name="get_policy", status="ok", data=item)
        return ToolResult(tool_name="get_policy", status="not_found", message="未找到对应规则")

    def _get_order_status(self, arguments: Dict[str, Any], shop_id: Optional[str]) -> ToolResult:
        order_id = arguments.get("order_id")
        if not order_id:
            return ToolResult(tool_name="get_order_status", status="missing_args", message="缺少 order_id")

        for item in self._load_json("orders.json"):
            if item["order_id"] == order_id and item["shop_id"] == shop_id:
                return ToolResult(tool_name="get_order_status", status="ok", data=item)
        return ToolResult(tool_name="get_order_status", status="not_found", message="未找到对应订单")

    def _get_logistics_status(self, arguments: Dict[str, Any], shop_id: Optional[str]) -> ToolResult:
        order_id = arguments.get("order_id")
        if not order_id:
            return ToolResult(tool_name="get_logistics_status", status="missing_args", message="缺少 order_id")

        for item in self._load_json("logistics.json"):
            if item["order_id"] == order_id and item["shop_id"] == shop_id:
                return ToolResult(tool_name="get_logistics_status", status="ok", data=item)
        return ToolResult(tool_name="get_logistics_status", status="not_found", message="未找到对应物流信息")

    @lru_cache(maxsize=8)
    def _load_json(self, file_name: str) -> List[Dict[str, Any]]:
        path = self._kb_root / file_name
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
