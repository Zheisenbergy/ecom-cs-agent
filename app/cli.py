from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional
from uuid import uuid4

from app.config import get_settings
from app.exporters.llamafactory import export_traces_to_llamafactory
from app.exporters.training_data import (
    export_answer_sft_jsonl,
    export_answer_sft_llamafactory,
    export_router_sft_jsonl,
    export_router_sft_llamafactory,
)
from app.models import ChatRequest, ChatResponse, EpisodeRecord, EpisodeState, TraceRecord
from app.services.baseline_benchmark import BaselineBenchmarkService, load_jsonl
from app.services.model_io import OpenAICompatibleModelClient
from app.services.model_orchestrator import build_model_orchestrator
from app.services.evaluator import EpisodeEvaluator
from app.services.orchestrator import get_orchestrator
from app.services.seed_synthesis import synthesize_episode_seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecom-cs-agent",
        description="终端式电商客服问答系统",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat_parser = subparsers.add_parser("chat", help="进入交互式会话")
    chat_parser.add_argument("--shop-id", default=None, help="覆盖默认店铺 ID")
    chat_parser.add_argument("--product-id", default=None, help="会话级商品 ID 上下文")
    chat_parser.add_argument("--order-id", default=None, help="会话级订单号上下文")
    chat_parser.add_argument("--show-debug", action="store_true", help="显示调试信息")

    ask_parser = subparsers.add_parser("ask", help="执行单轮提问")
    ask_parser.add_argument("query", help="用户问题")
    ask_parser.add_argument("--shop-id", default=None, help="覆盖默认店铺 ID")
    ask_parser.add_argument("--product-id", default=None, help="显式提供商品 ID")
    ask_parser.add_argument("--order-id", default=None, help="显式提供订单号")
    ask_parser.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    ask_parser.add_argument("--show-debug", action="store_true", help="显示调试信息")

    trace_parser = subparsers.add_parser("trace", help="输出单条问题的完整工具调用轨迹")
    trace_parser.add_argument("query", help="用户问题")
    trace_parser.add_argument("--shop-id", default=None, help="覆盖默认店铺 ID")
    trace_parser.add_argument("--product-id", default=None, help="显式提供商品 ID")
    trace_parser.add_argument("--order-id", default=None, help="显式提供订单号")

    chat_model_parser = subparsers.add_parser("chat-model", help="进入基于 router/answer 模型的交互式会话")
    chat_model_parser.add_argument("--shop-id", default=None, help="覆盖默认店铺 ID")
    chat_model_parser.add_argument("--product-id", default=None, help="会话级商品 ID 上下文")
    chat_model_parser.add_argument("--order-id", default=None, help="会话级订单号上下文")
    chat_model_parser.add_argument("--show-debug", action="store_true", help="显示调试信息")
    _add_model_runtime_args(chat_model_parser)

    ask_model_parser = subparsers.add_parser("ask-model", help="使用 router/answer 模型执行单轮提问")
    ask_model_parser.add_argument("query", help="用户问题")
    ask_model_parser.add_argument("--shop-id", default=None, help="覆盖默认店铺 ID")
    ask_model_parser.add_argument("--product-id", default=None, help="显式提供商品 ID")
    ask_model_parser.add_argument("--order-id", default=None, help="显式提供订单号")
    ask_model_parser.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    ask_model_parser.add_argument("--show-debug", action="store_true", help="显示调试信息")
    _add_model_runtime_args(ask_model_parser)

    trace_model_parser = subparsers.add_parser("trace-model", help="输出模型版单条问题的完整工具调用轨迹")
    trace_model_parser.add_argument("query", help="用户问题")
    trace_model_parser.add_argument("--shop-id", default=None, help="覆盖默认店铺 ID")
    trace_model_parser.add_argument("--product-id", default=None, help="显式提供商品 ID")
    trace_model_parser.add_argument("--order-id", default=None, help="显式提供订单号")
    _add_model_runtime_args(trace_model_parser)

    run_parser = subparsers.add_parser("run", help="批量运行 episode JSONL 样本并导出 episode trace")
    run_parser.add_argument("--input", required=True, help="输入 JSONL 文件路径")
    run_parser.add_argument("--output", required=True, help="输出 episode trace JSONL 文件路径")
    run_parser.add_argument("--limit", type=int, default=None, help="最多处理多少条样本")

    eval_parser = subparsers.add_parser("eval", help="基于带标注的 episode JSONL 做离线评测")
    eval_parser.add_argument("--input", required=True, help="输入带标注的 JSONL 文件路径")
    eval_parser.add_argument("--output", default=None, help="可选，输出评测结果 JSON 路径")
    eval_parser.add_argument("--limit", type=int, default=None, help="最多处理多少条样本")

    export_parser = subparsers.add_parser("export-sft", help="将 trace JSONL 导出为 LLaMA-Factory 数据集")
    export_parser.add_argument("--input", required=True, help="trace JSONL 输入路径")
    export_parser.add_argument("--output", required=True, help="导出的 ShareGPT JSON 路径")
    export_parser.add_argument("--dataset-name", default="ecom_cs_episode_toolcall", help="dataset_info 中的数据集名称")
    export_parser.add_argument(
        "--dataset-info",
        default="training/datasets/dataset_info.ecom_cs_episode_toolcall.json",
        help="导出的 dataset_info JSON 路径",
    )

    export_router_parser = subparsers.add_parser("export-router-sft", help="导出 router 训练 JSONL")
    export_router_parser.add_argument("--input", required=True, help="episode trace JSONL 输入路径")
    export_router_parser.add_argument("--output", required=True, help="导出的 router 训练 JSONL 路径")

    export_answer_parser = subparsers.add_parser("export-answer-sft", help="导出 answer 训练 JSONL")
    export_answer_parser.add_argument("--input", required=True, help="episode trace JSONL 输入路径")
    export_answer_parser.add_argument("--output", required=True, help="导出的 answer 训练 JSONL 路径")

    export_router_lf_parser = subparsers.add_parser(
        "export-router-lf",
        help="导出 LLaMA-Factory 可用的 router SFT 数据",
    )
    export_router_lf_parser.add_argument("--input", required=True, help="episode trace JSONL 输入路径")
    export_router_lf_parser.add_argument("--output", required=True, help="导出的 LLaMA-Factory router JSON 路径")
    export_router_lf_parser.add_argument(
        "--dataset-name",
        default="ecom_cs_router_sft",
        help="dataset_info 中的数据集名称",
    )
    export_router_lf_parser.add_argument(
        "--dataset-info",
        default="training/datasets/dataset_info.ecom_cs_router_sft.json",
        help="导出的 dataset_info JSON 路径",
    )

    export_answer_lf_parser = subparsers.add_parser(
        "export-answer-lf",
        help="导出 LLaMA-Factory 可用的 answer SFT 数据",
    )
    export_answer_lf_parser.add_argument("--input", required=True, help="episode trace JSONL 输入路径")
    export_answer_lf_parser.add_argument("--output", required=True, help="导出的 LLaMA-Factory answer JSON 路径")
    export_answer_lf_parser.add_argument(
        "--dataset-name",
        default="ecom_cs_answer_sft",
        help="dataset_info 中的数据集名称",
    )
    export_answer_lf_parser.add_argument(
        "--dataset-info",
        default="training/datasets/dataset_info.ecom_cs_answer_sft.json",
        help="导出的 dataset_info JSON 路径",
    )

    benchmark_router_parser = subparsers.add_parser("benchmark-router", help="评测未后训练模型的 router 基线")
    benchmark_router_parser.add_argument("--input", required=True, help="router 基线 JSONL 输入路径")
    benchmark_router_parser.add_argument("--model", required=True, help="OpenAI-compatible 模型名")
    benchmark_router_parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible 服务地址，例如 http://127.0.0.1:8000/v1",
    )
    benchmark_router_parser.add_argument("--api-key", default="EMPTY", help="OpenAI-compatible API key，vLLM 常用 EMPTY")
    benchmark_router_parser.add_argument("--timeout-seconds", type=int, default=120, help="单条请求超时秒数")
    benchmark_router_parser.add_argument("--max-tokens", type=int, default=256, help="模型最大输出 token 数")
    benchmark_router_parser.add_argument("--output", default=None, help="可选，输出评测结果 JSON 路径")
    benchmark_router_parser.add_argument("--limit", type=int, default=None, help="最多评测多少条样本")

    benchmark_answer_parser = subparsers.add_parser("benchmark-answer", help="评测未后训练模型的 answer 基线")
    benchmark_answer_parser.add_argument("--input", required=True, help="answer 基线 JSONL 输入路径")
    benchmark_answer_parser.add_argument("--model", required=True, help="OpenAI-compatible 模型名")
    benchmark_answer_parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible 服务地址，例如 http://127.0.0.1:8000/v1",
    )
    benchmark_answer_parser.add_argument("--api-key", default="EMPTY", help="OpenAI-compatible API key，vLLM 常用 EMPTY")
    benchmark_answer_parser.add_argument("--timeout-seconds", type=int, default=120, help="单条请求超时秒数")
    benchmark_answer_parser.add_argument("--max-tokens", type=int, default=256, help="模型最大输出 token 数")
    benchmark_answer_parser.add_argument("--output", default=None, help="可选，输出评测结果 JSON 路径")
    benchmark_answer_parser.add_argument("--limit", type=int, default=None, help="最多评测多少条样本")

    synthesize_parser = subparsers.add_parser("synthesize-episodes", help="按模板与槽位配置批量生成 episode seed")
    synthesize_parser.add_argument(
        "--config",
        default="training/datasets/synthesis_templates.default.json",
        help="合成配置 JSON 路径",
    )
    synthesize_parser.add_argument("--output-train", required=True, help="导出的 train seed JSONL 路径")
    synthesize_parser.add_argument("--output-dev", required=True, help="导出的 dev seed JSONL 路径")
    synthesize_parser.add_argument("--seed", type=int, default=None, help="可选，覆盖配置中的随机种子")

    subparsers.add_parser("meta", help="查看当前系统配置")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "meta":
        print(_meta_output())
        return

    if args.command == "ask":
        response = _run_query(
            args.query,
            shop_id=args.shop_id,
            product_id=args.product_id,
            order_id=args.order_id,
        )
        if args.json:
            print(json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2))
        else:
            print(_format_response(response, show_debug=args.show_debug))
        return

    if args.command == "trace":
        trace = _run_trace(
            args.query,
            shop_id=args.shop_id,
            product_id=args.product_id,
            order_id=args.order_id,
        )
        print(json.dumps(trace.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    if args.command == "ask-model":
        response = _run_model_query(
            args.query,
            shop_id=args.shop_id,
            product_id=args.product_id,
            order_id=args.order_id,
            router_model=args.router_model,
            answer_model=args.answer_model,
            router_base_url=args.router_base_url,
            answer_base_url=args.answer_base_url,
            router_api_key=args.router_api_key,
            answer_api_key=args.answer_api_key,
            timeout_seconds=args.timeout_seconds,
            router_max_tokens=args.router_max_tokens,
            answer_max_tokens=args.answer_max_tokens,
        )
        if args.json:
            print(json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2))
        else:
            print(_format_response(response, show_debug=args.show_debug))
        return

    if args.command == "trace-model":
        trace = _run_model_trace(
            args.query,
            shop_id=args.shop_id,
            product_id=args.product_id,
            order_id=args.order_id,
            router_model=args.router_model,
            answer_model=args.answer_model,
            router_base_url=args.router_base_url,
            answer_base_url=args.answer_base_url,
            router_api_key=args.router_api_key,
            answer_api_key=args.answer_api_key,
            timeout_seconds=args.timeout_seconds,
            router_max_tokens=args.router_max_tokens,
            answer_max_tokens=args.answer_max_tokens,
        )
        print(json.dumps(trace.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    if args.command == "run":
        summary = _run_batch(
            input_path=Path(args.input),
            output_path=Path(args.output),
            limit=args.limit,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "eval":
        summary = _run_eval(
            input_path=Path(args.input),
            output_path=Path(args.output) if args.output else None,
            limit=args.limit,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "export-sft":
        summary = export_traces_to_llamafactory(
            traces_path=Path(args.input),
            output_path=Path(args.output),
            dataset_name=args.dataset_name,
            dataset_info_path=Path(args.dataset_info),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "export-router-sft":
        summary = export_router_sft_jsonl(
            traces_path=Path(args.input),
            output_path=Path(args.output),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "export-answer-sft":
        summary = export_answer_sft_jsonl(
            traces_path=Path(args.input),
            output_path=Path(args.output),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "export-router-lf":
        summary = export_router_sft_llamafactory(
            traces_path=Path(args.input),
            output_path=Path(args.output),
            dataset_name=args.dataset_name,
            dataset_info_path=Path(args.dataset_info),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "export-answer-lf":
        summary = export_answer_sft_llamafactory(
            traces_path=Path(args.input),
            output_path=Path(args.output),
            dataset_name=args.dataset_name,
            dataset_info_path=Path(args.dataset_info),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "benchmark-router":
        summary = _run_router_benchmark(
            input_path=Path(args.input),
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_seconds=args.timeout_seconds,
            max_tokens=args.max_tokens,
            output_path=Path(args.output) if args.output else None,
            limit=args.limit,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "benchmark-answer":
        summary = _run_answer_benchmark(
            input_path=Path(args.input),
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_seconds=args.timeout_seconds,
            max_tokens=args.max_tokens,
            output_path=Path(args.output) if args.output else None,
            limit=args.limit,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "synthesize-episodes":
        summary = synthesize_episode_seeds(
            config_path=Path(args.config),
            output_train_path=Path(args.output_train),
            output_dev_path=Path(args.output_dev),
            seed=args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "chat":
        _interactive_chat(
            shop_id=args.shop_id,
            product_id=args.product_id,
            order_id=args.order_id,
            show_debug=args.show_debug,
        )
        return

    if args.command == "chat-model":
        _interactive_model_chat(
            shop_id=args.shop_id,
            product_id=args.product_id,
            order_id=args.order_id,
            show_debug=args.show_debug,
            router_model=args.router_model,
            answer_model=args.answer_model,
            router_base_url=args.router_base_url,
            answer_base_url=args.answer_base_url,
            router_api_key=args.router_api_key,
            answer_api_key=args.answer_api_key,
            timeout_seconds=args.timeout_seconds,
            router_max_tokens=args.router_max_tokens,
            answer_max_tokens=args.answer_max_tokens,
        )
        return

    parser.error("未知命令")


def _interactive_chat(
    shop_id: Optional[str],
    product_id: Optional[str],
    order_id: Optional[str],
    show_debug: bool,
) -> None:
    settings = get_settings()
    resolved_shop_id = shop_id or settings.default_shop_id
    current_state = EpisodeState(
        shop_id=resolved_shop_id,
        product_id=product_id,
        order_id=order_id,
    )
    print("电商客服问答 CLI")
    print(f"店铺 ID: {resolved_shop_id}")
    if product_id:
        print(f"商品 ID: {product_id}")
    if order_id:
        print(f"订单号: {order_id}")
    print("输入 /meta 查看当前配置，/state 查看当前 episode 状态，/reset 清空当前任务，/exit 退出。")

    while True:
        try:
            query = input("\n你> ").strip()
        except EOFError:
            print("\n会话结束。")
            return
        except KeyboardInterrupt:
            print("\n会话中断。")
            return

        if not query:
            continue
        if query in {"/exit", "exit", "quit"}:
            print("会话结束。")
            return
        if query == "/meta":
            print(_meta_output())
            continue
        if query == "/state":
            print(json.dumps(current_state.model_dump(mode="json"), ensure_ascii=False, indent=2))
            continue
        if query == "/reset":
            current_state = EpisodeState(
                shop_id=resolved_shop_id,
                product_id=product_id,
                order_id=order_id,
            )
            print("当前任务状态已清空。")
            continue

        trace = _run_trace(
            query=query,
            shop_id=resolved_shop_id,
            product_id=product_id,
            order_id=order_id,
            state=current_state,
        )
        print(_format_response(trace.response, show_debug=show_debug))
        current_state = trace.state_after or EpisodeState(shop_id=resolved_shop_id)
        if trace.response.episode_done:
            current_state = EpisodeState(
                shop_id=resolved_shop_id,
                product_id=product_id,
                order_id=order_id,
            )


def _interactive_model_chat(
    shop_id: Optional[str],
    product_id: Optional[str],
    order_id: Optional[str],
    show_debug: bool,
    router_model: str,
    answer_model: str,
    router_base_url: str,
    answer_base_url: str,
    router_api_key: str,
    answer_api_key: str,
    timeout_seconds: int,
    router_max_tokens: int,
    answer_max_tokens: int,
) -> None:
    settings = get_settings()
    resolved_shop_id = shop_id or settings.default_shop_id
    orchestrator = _build_model_runtime_orchestrator(
        router_model=router_model,
        answer_model=answer_model,
        router_base_url=router_base_url,
        answer_base_url=answer_base_url,
        router_api_key=router_api_key,
        answer_api_key=answer_api_key,
        timeout_seconds=timeout_seconds,
        router_max_tokens=router_max_tokens,
        answer_max_tokens=answer_max_tokens,
    )
    current_state = EpisodeState(
        shop_id=resolved_shop_id,
        product_id=product_id,
        order_id=order_id,
    )
    print("电商客服模型联调 CLI")
    print(f"店铺 ID: {resolved_shop_id}")
    print(f"router model: {router_model}")
    print(f"answer model: {answer_model}")
    print("输入 /meta 查看当前配置，/state 查看当前 episode 状态，/reset 清空当前任务，/exit 退出。")

    while True:
        try:
            query = input("\n你> ").strip()
        except EOFError:
            print("\n会话结束。")
            return
        except KeyboardInterrupt:
            print("\n会话中断。")
            return

        if not query:
            continue
        if query in {"/exit", "exit", "quit"}:
            print("会话结束。")
            return
        if query == "/meta":
            print(_meta_output())
            continue
        if query == "/state":
            print(json.dumps(current_state.model_dump(mode="json"), ensure_ascii=False, indent=2))
            continue
        if query == "/reset":
            current_state = EpisodeState(
                shop_id=resolved_shop_id,
                product_id=product_id,
                order_id=order_id,
            )
            print("当前任务状态已清空。")
            continue

        trace = _run_model_trace_with_orchestrator(
            orchestrator=orchestrator,
            query=query,
            shop_id=resolved_shop_id,
            product_id=product_id,
            order_id=order_id,
            state=current_state,
        )
        print(_format_response(trace.response, show_debug=show_debug))
        current_state = trace.state_after or EpisodeState(shop_id=resolved_shop_id)
        if trace.response.episode_done:
            current_state = EpisodeState(
                shop_id=resolved_shop_id,
                product_id=product_id,
                order_id=order_id,
            )


def _run_query(
    query: str,
    shop_id: Optional[str] = None,
    product_id: Optional[str] = None,
    order_id: Optional[str] = None,
) -> ChatResponse:
    response, _ = get_orchestrator().handle_query(
        ChatRequest(
            query=query,
            shop_id=shop_id,
            product_id=product_id,
            order_id=order_id,
        )
    )
    return response


def _run_trace(
    query: str,
    shop_id: Optional[str] = None,
    product_id: Optional[str] = None,
    order_id: Optional[str] = None,
    state: Optional[EpisodeState] = None,
) -> TraceRecord:
    request = ChatRequest(
        query=query,
        shop_id=shop_id,
        product_id=product_id,
        order_id=order_id,
    )
    return get_orchestrator().handle_trace(request, state=state)


def _run_model_query(
    query: str,
    shop_id: Optional[str],
    product_id: Optional[str],
    order_id: Optional[str],
    router_model: str,
    answer_model: str,
    router_base_url: str,
    answer_base_url: str,
    router_api_key: str,
    answer_api_key: str,
    timeout_seconds: int,
    router_max_tokens: int,
    answer_max_tokens: int,
) -> ChatResponse:
    response, _ = _build_model_runtime_orchestrator(
        router_model=router_model,
        answer_model=answer_model,
        router_base_url=router_base_url,
        answer_base_url=answer_base_url,
        router_api_key=router_api_key,
        answer_api_key=answer_api_key,
        timeout_seconds=timeout_seconds,
        router_max_tokens=router_max_tokens,
        answer_max_tokens=answer_max_tokens,
    ).handle_query(
        ChatRequest(
            query=query,
            shop_id=shop_id,
            product_id=product_id,
            order_id=order_id,
        )
    )
    return response


def _run_model_trace(
    query: str,
    shop_id: Optional[str],
    product_id: Optional[str],
    order_id: Optional[str],
    router_model: str,
    answer_model: str,
    router_base_url: str,
    answer_base_url: str,
    router_api_key: str,
    answer_api_key: str,
    timeout_seconds: int,
    router_max_tokens: int,
    answer_max_tokens: int,
    state: Optional[EpisodeState] = None,
) -> TraceRecord:
    orchestrator = _build_model_runtime_orchestrator(
        router_model=router_model,
        answer_model=answer_model,
        router_base_url=router_base_url,
        answer_base_url=answer_base_url,
        router_api_key=router_api_key,
        answer_api_key=answer_api_key,
        timeout_seconds=timeout_seconds,
        router_max_tokens=router_max_tokens,
        answer_max_tokens=answer_max_tokens,
    )
    return _run_model_trace_with_orchestrator(
        orchestrator=orchestrator,
        query=query,
        shop_id=shop_id,
        product_id=product_id,
        order_id=order_id,
        state=state,
    )


def _run_model_trace_with_orchestrator(
    orchestrator,
    query: str,
    shop_id: Optional[str],
    product_id: Optional[str],
    order_id: Optional[str],
    state: Optional[EpisodeState] = None,
) -> TraceRecord:
    request = ChatRequest(
        query=query,
        shop_id=shop_id,
        product_id=product_id,
        order_id=order_id,
    )
    return orchestrator.handle_trace(request, state=state)


def _run_batch(input_path: Path, output_path: Path, limit: Optional[int]) -> dict[str, object]:
    processed = 0
    completed = 0
    waiting_for_user = 0
    route_counts: dict[str, int] = {}

    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as sink:
        for raw_line in source:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            episode = _episode_from_payload(payload)
            sink.write(json.dumps(episode.model_dump(mode="json"), ensure_ascii=False) + "\n")

            processed += 1
            route = episode.final_response.route.value
            route_counts[route] = route_counts.get(route, 0) + 1
            if episode.completed:
                completed += 1
            if episode.waiting_for_user:
                waiting_for_user += 1

            if limit is not None and processed >= limit:
                break

    return {
        "processed": processed,
        "completed": completed,
        "waiting_for_user": waiting_for_user,
        "output": str(output_path),
        "route_counts": route_counts,
    }


def _run_eval(
    input_path: Path,
    output_path: Optional[Path],
    limit: Optional[int],
) -> dict[str, object]:
    payloads = _load_jsonl(input_path, limit=limit)
    episodes = [_episode_from_payload(payload) for payload in payloads]
    summary = EpisodeEvaluator().evaluate(payloads, episodes)
    summary["input"] = str(input_path)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["output"] = str(output_path)
    return summary


def _run_router_benchmark(
    input_path: Path,
    model: str,
    base_url: str,
    api_key: str,
    timeout_seconds: int,
    max_tokens: int,
    output_path: Optional[Path],
    limit: Optional[int],
) -> dict[str, object]:
    rows = load_jsonl(input_path, limit=limit)
    service = BaselineBenchmarkService(
        OpenAICompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
        )
    )
    summary = service.benchmark_router(rows)
    summary["input"] = str(input_path)
    summary["model"] = model
    summary["base_url"] = base_url
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["output"] = str(output_path)
    return summary


def _run_answer_benchmark(
    input_path: Path,
    model: str,
    base_url: str,
    api_key: str,
    timeout_seconds: int,
    max_tokens: int,
    output_path: Optional[Path],
    limit: Optional[int],
) -> dict[str, object]:
    rows = load_jsonl(input_path, limit=limit)
    service = BaselineBenchmarkService(
        OpenAICompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
        )
    )
    summary = service.benchmark_answer(rows)
    summary["input"] = str(input_path)
    summary["model"] = model
    summary["base_url"] = base_url
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["output"] = str(output_path)
    return summary


def _episode_from_payload(payload: dict[str, object]) -> EpisodeRecord:
    episode_id = _optional_str(payload.get("episode_id")) or f"episode-{uuid4().hex[:8]}"
    default_shop_id = _optional_str(payload.get("shop_id"))
    default_product_id = _optional_str(payload.get("product_id"))
    default_order_id = _optional_str(payload.get("order_id"))

    turns_payload = payload.get("turns")
    if turns_payload is None:
        query = str(payload.get("query") or payload.get("user_query") or "").strip()
        if not query:
            raise ValueError("输入样本缺少 `query`、`user_query` 或 `turns` 字段。")
        turns_payload = [{"query": query}]

    if not isinstance(turns_payload, list) or not turns_payload:
        raise ValueError("`turns` 必须是非空列表。")

    turns: list[ChatRequest] = []
    for turn in turns_payload:
        if not isinstance(turn, dict):
            raise ValueError("`turns` 中的每一项都必须是对象。")
        query = str(turn.get("query") or turn.get("user_query") or "").strip()
        if not query:
            raise ValueError("episode turn 缺少 `query` 或 `user_query`。")
        turns.append(
            ChatRequest(
                query=query,
                shop_id=_optional_str(turn.get("shop_id")) or default_shop_id,
                product_id=_optional_str(turn.get("product_id")) or default_product_id,
                order_id=_optional_str(turn.get("order_id")) or default_order_id,
            )
        )

    initial_state = EpisodeState(
        shop_id=default_shop_id,
        product_id=default_product_id,
        order_id=default_order_id,
    )
    return get_orchestrator().run_episode(
        turns=turns,
        episode_id=episode_id,
        initial_state=initial_state,
    )


def _load_jsonl(path: Path, limit: Optional[int]) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as source:
        for raw_line in source:
            line = raw_line.strip()
            if not line:
                continue
            payloads.append(json.loads(line))
            if limit is not None and len(payloads) >= limit:
                break
    return payloads


def _optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    string_value = str(value).strip()
    return string_value or None


def _meta_output() -> str:
    settings = get_settings()
    payload = {
        "app_name": settings.app_name,
        "app_env": settings.app_env,
        "default_shop_id": settings.default_shop_id,
        "models": {
            "router": settings.router_model_name,
            "answer": settings.answer_model_name,
            "embedding": settings.embedding_model_name,
            "reranker": settings.reranker_model_name,
        },
        "available_routes": ["direct", "internal_tool", "handoff"],
        "available_tools": [
            "get_product_info",
            "get_policy",
            "get_order_status",
            "get_logistics_status",
        ],
        "trace_commands": ["trace", "trace-model", "run"],
        "runtime_commands": ["ask", "ask-model", "chat", "chat-model"],
        "export_commands": [
            "export-sft",
            "export-router-sft",
            "export-answer-sft",
            "export-router-lf",
            "export-answer-lf",
        ],
        "eval_commands": ["eval", "benchmark-router", "benchmark-answer"],
        "data_commands": ["synthesize-episodes"],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _add_model_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--router-model", required=True, help="router 的 OpenAI-compatible 模型名，例如 router-lora")
    parser.add_argument("--answer-model", required=True, help="answer 的 OpenAI-compatible 模型名，例如 answer-lora")
    parser.add_argument(
        "--router-base-url",
        default="http://127.0.0.1:8000/v1",
        help="router OpenAI-compatible 服务地址",
    )
    parser.add_argument(
        "--answer-base-url",
        default="http://127.0.0.1:8000/v1",
        help="answer OpenAI-compatible 服务地址",
    )
    parser.add_argument("--router-api-key", default="EMPTY", help="router OpenAI-compatible API key")
    parser.add_argument("--answer-api-key", default="EMPTY", help="answer OpenAI-compatible API key")
    parser.add_argument("--timeout-seconds", type=int, default=120, help="单条模型请求超时秒数")
    parser.add_argument("--router-max-tokens", type=int, default=256, help="router 最大输出 token 数")
    parser.add_argument("--answer-max-tokens", type=int, default=256, help="answer 最大输出 token 数")


def _build_model_runtime_orchestrator(
    router_model: str,
    answer_model: str,
    router_base_url: str,
    answer_base_url: str,
    router_api_key: str,
    answer_api_key: str,
    timeout_seconds: int,
    router_max_tokens: int,
    answer_max_tokens: int,
):
    return build_model_orchestrator(
        router_client=OpenAICompatibleModelClient(
            model=router_model,
            base_url=router_base_url,
            api_key=router_api_key,
            timeout_seconds=timeout_seconds,
            max_tokens=router_max_tokens,
        ),
        answer_client=OpenAICompatibleModelClient(
            model=answer_model,
            base_url=answer_base_url,
            api_key=answer_api_key,
            timeout_seconds=timeout_seconds,
            max_tokens=answer_max_tokens,
        ),
    )


def _format_response(response: ChatResponse, show_debug: bool) -> str:
    lines = [
        f"路由: {response.route.value}",
        f"意图: {response.intent}",
        f"置信度: {response.confidence:.2f}",
        f"改写查询: {response.rewrite_query}",
        "",
        "回答:",
        response.answer,
    ]

    if response.citations:
        lines.extend(
            [
                "",
                f"引用: {', '.join(response.citations)}",
            ]
        )

    if len(response.tool_steps) > 1:
        lines.extend(["", "工具链:"])
        for index, step in enumerate(response.tool_steps, start=1):
            lines.append(
                f"{index}. {step.call.name} {json.dumps(step.call.arguments, ensure_ascii=False)} -> {step.result.status}"
            )
    elif response.tool_call:
        lines.extend(
            [
                "",
                f"工具: {response.tool_call.name}",
                f"参数: {json.dumps(response.tool_call.arguments, ensure_ascii=False)}",
            ]
        )

    if response.tool_result and response.tool_result.status == "ok":
        lines.extend(
            [
                f"工具状态: {response.tool_result.status}",
            ]
        )

    if response.waiting_for_user:
        lines.extend(
            [
                "",
                "状态: 等待用户补充信息",
            ]
        )

    if show_debug:
        lines.extend(
            [
                "",
                "调试信息:",
                json.dumps(response.debug, ensure_ascii=False, indent=2),
            ]
        )

    return "\n".join(lines)


if __name__ == "__main__":
    main()
