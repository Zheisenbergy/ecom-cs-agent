from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from app.models import EpisodeRecord


@dataclass
class EvalCaseResult:
    episode_id: str
    passed: bool
    checks: Dict[str, bool]
    expected: Dict[str, Any]
    predicted: Dict[str, Any]


class EpisodeEvaluator:
    """基于标注 episode 样本的轻量离线评测器。"""

    def evaluate(
        self,
        labeled_cases: Iterable[dict[str, Any]],
        predictions: Iterable[EpisodeRecord],
    ) -> dict[str, Any]:
        case_results: List[EvalCaseResult] = []
        metrics = {
            "route_accuracy": {"correct": 0, "total": 0},
            "intent_accuracy": {"correct": 0, "total": 0},
            "completed_accuracy": {"correct": 0, "total": 0},
            "any_ask_user_accuracy": {"correct": 0, "total": 0},
            "handoff_accuracy": {"correct": 0, "total": 0},
            "tool_chain_exact_match_accuracy": {"correct": 0, "total": 0},
            "turn_count_accuracy": {"correct": 0, "total": 0},
            "citation_set_accuracy": {"correct": 0, "total": 0},
            "answer_contains_accuracy": {"correct": 0, "total": 0},
            "answer_not_contains_accuracy": {"correct": 0, "total": 0},
            "auto_groundedness_accuracy": {"correct": 0, "total": 0},
        }
        failure_buckets: Dict[str, List[str]] = {name: [] for name in metrics}
        passed_episodes = 0

        for payload, episode in zip(labeled_cases, predictions):
            expected = self._extract_expected(payload)
            predicted = self._extract_predicted(episode)
            predicted["auto_groundedness_accuracy"] = self._auto_grounded_check(episode)
            checks: Dict[str, bool] = {}

            for metric_name, expected_value in expected.items():
                if expected_value is None:
                    continue
                predicted_value = predicted.get(metric_name)
                passed = self._check(metric_name, expected_value, predicted_value)
                checks[metric_name] = passed
                metrics[metric_name]["total"] += 1
                if passed:
                    metrics[metric_name]["correct"] += 1
                else:
                    failure_buckets[metric_name].append(episode.episode_id)

            metrics["auto_groundedness_accuracy"]["total"] += 1
            if predicted["auto_groundedness_accuracy"]:
                metrics["auto_groundedness_accuracy"]["correct"] += 1
            else:
                failure_buckets["auto_groundedness_accuracy"].append(episode.episode_id)

            case_passed = all(checks.values()) if checks else True
            if case_passed and predicted["auto_groundedness_accuracy"]:
                passed_episodes += 1

            case_results.append(
                EvalCaseResult(
                    episode_id=episode.episode_id,
                    passed=case_passed,
                    checks=checks,
                    expected=expected,
                    predicted=predicted,
                )
            )

        summary = {
            name: self._ratio(stat["correct"], stat["total"])
            for name, stat in metrics.items()
        }
        return {
            "num_cases": len(case_results),
            "summary": summary,
            "episode_pass_rate": self._ratio(passed_episodes, len(case_results)),
            "failure_summary": {
                name: {"count": len(episode_ids), "episode_ids": episode_ids}
                for name, episode_ids in failure_buckets.items()
                if episode_ids
            },
            "details": [
                {
                    "episode_id": case.episode_id,
                    "passed": case.passed,
                    "failed_metrics": [metric for metric, passed in case.checks.items() if not passed],
                    "checks": case.checks,
                    "expected": case.expected,
                    "predicted": case.predicted,
                }
                for case in case_results
            ],
        }

    @staticmethod
    def _extract_expected(payload: dict[str, Any]) -> Dict[str, Any]:
        return {
            "route_accuracy": payload.get("expected_final_route"),
            "intent_accuracy": payload.get("expected_final_intent"),
            "completed_accuracy": payload.get("expected_completed"),
            "any_ask_user_accuracy": payload.get("expected_any_ask_user"),
            "handoff_accuracy": payload.get("expected_handoff"),
            "tool_chain_exact_match_accuracy": payload.get("expected_tool_chain"),
            "turn_count_accuracy": payload.get("expected_turn_count"),
            "citation_set_accuracy": payload.get("expected_final_citations"),
            "answer_contains_accuracy": payload.get("expected_answer_contains"),
            "answer_not_contains_accuracy": payload.get("expected_answer_not_contains"),
        }

    @staticmethod
    def _extract_predicted(episode: EpisodeRecord) -> Dict[str, Any]:
        tool_chain: list[str] = []
        for turn in episode.turns:
            for step in turn.tool_steps:
                tool_chain.append(step.call.name)
        any_ask_user = any(turn.response.waiting_for_user for turn in episode.turns)
        return {
            "route_accuracy": episode.final_response.route.value,
            "intent_accuracy": episode.final_response.intent,
            "completed_accuracy": episode.completed,
            "any_ask_user_accuracy": any_ask_user,
            "handoff_accuracy": episode.final_response.escalation_required,
            "tool_chain_exact_match_accuracy": tool_chain,
            "turn_count_accuracy": len(episode.turns),
            "citation_set_accuracy": episode.final_response.citations,
            "answer_contains_accuracy": episode.final_response.answer,
            "answer_not_contains_accuracy": episode.final_response.answer,
        }

    @staticmethod
    def _ratio(correct: int, total: int) -> dict[str, Any]:
        return {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total, 4) if total else None,
        }

    @staticmethod
    def _check(metric_name: str, expected: Any, predicted: Any) -> bool:
        if metric_name == "citation_set_accuracy":
            return list(predicted or []) == list(expected or [])
        if metric_name == "answer_contains_accuracy":
            if not isinstance(predicted, str):
                return False
            return all(fragment in predicted for fragment in expected)
        if metric_name == "answer_not_contains_accuracy":
            if not isinstance(predicted, str):
                return False
            return all(fragment not in predicted for fragment in expected)
        return predicted == expected

    def _auto_grounded_check(self, episode: EpisodeRecord) -> bool:
        response = episode.final_response
        answer = response.answer
        successful_steps = []
        for turn in episode.turns:
            for step in turn.tool_steps:
                if step.result.status == "ok":
                    successful_steps.append(step)

        if response.escalation_required:
            return not response.citations and "转接人工客服" in answer

        if not response.grounded:
            return not response.citations

        cited_successful = [step for step in successful_steps if step.result.tool_name in response.citations]
        if not cited_successful:
            return False

        required_fragments = self._required_fragments(episode, cited_successful)
        return all(fragment in answer for fragment in required_fragments if fragment)

    def _required_fragments(self, episode: EpisodeRecord, successful_steps: List[Any]) -> List[str]:
        response = episode.final_response
        query = ""
        if episode.turns:
            last_turn = episode.turns[-1]
            query = (
                str(last_turn.response.debug.get("task_query", "")).strip()
                if isinstance(last_turn.response.debug, dict)
                else ""
            ) or last_turn.request.query
        by_tool = {step.result.tool_name: step.result for step in successful_steps}
        fragments: List[str] = []

        if response.intent == "logistics_status":
            data = by_tool["get_logistics_status"].data
            fragments.extend([data["order_id"], data["carrier"]])
            if "运单" in query or "tracking" in query.lower():
                fragments.append(data["tracking_no"])
            else:
                fragments.append(data["latest_event"])
            return fragments

        if response.intent == "product_lookup":
            data = by_tool["get_product_info"].data
            fragments.append(data["name"])
            fragments.extend(self._product_fragments(query, data))
            return fragments

        if response.intent == "policy_lookup":
            data = by_tool["get_policy"].data
            fragments.append(data["summary"])
            return fragments

        if response.intent == "order_status":
            data = by_tool["get_order_status"].data
            fragments.append(data["order_id"])
            if "退款" in query or "refund" in query.lower():
                fragments.append(data["refund_status"])
            elif "支付" in query:
                fragments.append(data["payment_status"])
            else:
                fragments.extend([data["payment_status"], data["fulfillment_status"], data["refund_status"]])
            return fragments

        if response.intent in {"order_product_lookup", "order_policy_lookup", "order_product_policy_lookup"}:
            order = by_tool.get("get_order_status")
            if order is None:
                return fragments
            order_data = order.data
            fragments.extend([order_data["order_id"], order_data["product_name"]])
            if response.intent in {"order_product_lookup", "order_product_policy_lookup"}:
                product = by_tool.get("get_product_info")
                if product is not None:
                    fragments.extend(self._product_fragments(query, product.data))
            if response.intent in {"order_policy_lookup", "order_product_policy_lookup"}:
                policy = by_tool.get("get_policy")
                if policy is not None:
                    fragments.append(policy.data["summary"])
            return fragments

        return fragments

    @staticmethod
    def _product_fragments(query: str, product_data: Dict[str, Any]) -> List[str]:
        lowered = query.lower()
        if "材质" in query or "material" in lowered:
            return [product_data["material"]]
        if "颜色" in query or "color" in lowered:
            return list(product_data["colors"])
        if "尺码" in query or "size" in lowered:
            return list(product_data["sizes"])
        if "怎么洗" in query or "洗" in query or "care" in lowered:
            return [product_data["care_instructions"]]
        return [product_data["material"]]
