from __future__ import annotations

import json
from typing import Any, Dict, List


def get_tool_schemas() -> List[Dict[str, Any]]:
    return [
        {
            "name": "get_product_info",
            "description": "查询店铺中某个商品的属性信息，例如材质、颜色、尺码和卖点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "商品 ID，例如 sku_stormshell_001。",
                    }
                },
                "required": ["product_id"],
            },
        },
        {
            "name": "get_policy",
            "description": "查询店铺规则，例如退货、退款、换货、保修和发货规则。",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "规则主题，例如 return_policy、refund_policy、shipping_policy。",
                    }
                },
                "required": ["topic"],
            },
        },
        {
            "name": "get_order_status",
            "description": "查询订单状态，例如支付状态、履约状态和退款状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，例如 A1001。",
                    }
                },
                "required": ["order_id"],
            },
        },
        {
            "name": "get_logistics_status",
            "description": "查询订单物流信息，例如承运商、运单号和最新物流节点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，例如 A1001。",
                    }
                },
                "required": ["order_id"],
            },
        },
    ]


def get_tool_schemas_json() -> str:
    return json.dumps(get_tool_schemas(), ensure_ascii=False)
