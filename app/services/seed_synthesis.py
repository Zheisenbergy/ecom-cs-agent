from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def synthesize_episode_seeds(
    config_path: Path,
    output_train_path: Path,
    output_dev_path: Path,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    kb = _load_knowledge_base()
    rng = random.Random(seed if seed is not None else int(config.get("seed", 20260404)))

    shop_id = str(config.get("shop_id", "demo-shop"))
    prefixes = config.get("episode_prefixes", {"train": "syn-train", "dev": "syn-dev"})
    outputs: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    summaries: list[dict[str, Any]] = []

    for split in ("train", "dev"):
        seen: set[str] = set()
        episodes: list[dict[str, Any]] = []
        for scenario in config.get("scenarios", []):
            count = int(scenario.get("counts", {}).get(split, 0))
            if count <= 0:
                continue

            generated = _generate_for_scenario(
                scenario=scenario,
                count=count,
                shop_id=shop_id,
                kb=kb,
                seen=seen,
                rng=rng,
            )
            episodes.extend(generated)
            summaries.append(
                {
                    "split": split,
                    "scenario": str(scenario.get("name", "unnamed")),
                    "generated": len(generated),
                }
            )

        rng.shuffle(episodes)
        for index, episode in enumerate(episodes, start=1):
            episode["episode_id"] = f"{prefixes.get(split, split)}-{index:03d}"
        outputs[split] = episodes

    _write_jsonl(output_train_path, outputs["train"])
    _write_jsonl(output_dev_path, outputs["dev"])

    return {
        "config": str(config_path),
        "seed": seed if seed is not None else int(config.get("seed", 20260404)),
        "train_output": str(output_train_path),
        "dev_output": str(output_dev_path),
        "train_episodes": len(outputs["train"]),
        "dev_episodes": len(outputs["dev"]),
        "scenario_summaries": summaries,
    }


def _generate_for_scenario(
    scenario: dict[str, Any],
    count: int,
    shop_id: str,
    kb: dict[str, list[dict[str, Any]]],
    seen: set[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    entity_pool = _resolve_entity_pool(scenario=scenario, kb=kb)
    query_templates = scenario.get("query_templates", [])
    turn_templates = scenario.get("turn_templates", [])

    if not query_templates and not turn_templates:
        raise ValueError(f"scenario {scenario.get('name', 'unnamed')} 缺少 query_templates 或 turn_templates")

    generated: list[dict[str, Any]] = []
    max_attempts = max(50, count * 50)
    attempts = 0

    while len(generated) < count and attempts < max_attempts:
        attempts += 1
        entity = dict(rng.choice(entity_pool)) if entity_pool else {}
        context = _build_context(entity)
        episode: dict[str, Any] = {"shop_id": shop_id}

        for field in scenario.get("top_level_fields", []):
            value = context.get(field)
            if value is not None:
                episode[field] = value

        if turn_templates:
            selected_turns = rng.choice(turn_templates)
            episode["turns"] = [{"query": _format_template(text, context)} for text in selected_turns]
        else:
            episode["query"] = _format_template(rng.choice(query_templates), context)

        dedupe_key = json.dumps(episode, ensure_ascii=False, sort_keys=True)
        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        generated.append(episode)

    if len(generated) < count:
        raise RuntimeError(
            f"scenario {scenario.get('name', 'unnamed')} 只生成了 {len(generated)} 条，低于目标 {count} 条。"
        )
    return generated


def _resolve_entity_pool(scenario: dict[str, Any], kb: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    source = str(scenario.get("entity_source", "none"))
    if source == "none":
        return [{}]
    if source not in kb:
        raise ValueError(f"不支持的 entity_source: {source}")

    pool = [dict(item) for item in kb[source]]
    filters = scenario.get("entity_filters", {})
    if not filters:
        return pool

    filtered = []
    for item in pool:
        keep = True
        for key, expected in filters.items():
            actual = item.get(key)
            if isinstance(expected, list):
                if actual not in expected:
                    keep = False
                    break
            elif actual != expected:
                keep = False
                break
        if keep:
            filtered.append(item)

    if not filtered:
        raise ValueError(f"scenario {scenario.get('name', 'unnamed')} 经过 entity_filters 过滤后为空。")
    return filtered


def _build_context(entity: dict[str, Any]) -> dict[str, Any]:
    context = dict(entity)
    product_name = str(context.get("product_name") or context.get("name") or "").strip()
    if product_name:
        context["product_name"] = product_name
        context["product_short_name"] = product_name.split()[-1]
    if context.get("order_id"):
        context["order_ref"] = f"订单 {context['order_id']}"
    return context


def _format_template(template: str, context: dict[str, Any]) -> str:
    try:
        return template.format(**context)
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(f"模板缺少字段 `{missing}`: {template}") from exc


def _load_knowledge_base() -> dict[str, list[dict[str, Any]]]:
    base_dir = Path(__file__).resolve().parents[1] / "knowledge_base"
    products = json.loads((base_dir / "products.json").read_text(encoding="utf-8"))
    orders = json.loads((base_dir / "orders.json").read_text(encoding="utf-8"))
    policies = json.loads((base_dir / "policies.json").read_text(encoding="utf-8"))
    logistics = json.loads((base_dir / "logistics.json").read_text(encoding="utf-8"))

    product_by_id = {item["product_id"]: item for item in products}
    order_rows: list[dict[str, Any]] = []
    for order in orders:
        product = product_by_id.get(order["product_id"], {})
        order_rows.append(
            {
                **order,
                "product_name": order.get("product_name") or product.get("name"),
                "product_short_name": str(order.get("product_name") or product.get("name") or "").split()[-1],
                "material": product.get("material"),
                "colors": product.get("colors", []),
                "sizes": product.get("sizes", []),
            }
        )

    return {
        "none": [{}],
        "product": products,
        "order": order_rows,
        "policy": policies,
        "logistics": logistics,
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
