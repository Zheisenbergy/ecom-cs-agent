# 电商客服 Agentic 后训练原型

这是一个面向电商客服场景的终端式原型项目，目标不是做完整客服产品，而是做一个适合学习和实验的 agentic 后训练项目。

当前主设计采用 `episode` 思路：

- 一次用户任务就是一个 `episode`
- 一个 `episode` 内部有受控循环：判断、澄清、调工具、观察结果、结束
- 状态只在当前进程内保存，不做文件级会话持久化
- 导出的训练数据以“完整任务”为单位，而不是零散单轮

## 项目目标

这个项目优先训练模型学会：

- 判断能否直接回答
- 识别是否需要转人工
- 选择正确工具
- 填充工具参数
- 在缺关键槽位时向用户澄清
- 基于结构化 observation 给出 grounded answer

不优先做的事情：

- 把商品事实训练进模型
- 长期会话记忆
- 前后端产品工程
- Web 和 RAG

## 当前路由

- `direct`
- `internal_tool`
- `handoff`

内部工具包括：

- `get_product_info`
- `get_policy`
- `get_order_status`
- `get_logistics_status`

当前已经支持受控多工具链，例如：

- 先查订单，拿到 `product_id`
- 再查商品属性
- 再查退货规则
- 最后合成 grounded answer

## 为什么用 Episode

这个项目更适合做成“单任务闭环”而不是“长期会话系统”。

更合理的一条完整数据是：

1. 用户发起任务
2. agent 判断动作
3. 必要时 ask user / tool call
4. 收到 observation
5. 给出 final answer 或 handoff

这比把很多零散单轮拼成伪多轮，更接近你真正想训练的能力。

## 快速开始

```bash
cd /Users/zheisenbergy/code/agent/ecom-cs-agent
uv python install 3.11
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

## CLI

交互模式：

```bash
ecom-cs-agent chat
```

单条提问：

```bash
ecom-cs-agent ask "帮我查订单 A1001 到哪了"
ecom-cs-agent ask "这件衣服是什么材质？"
ecom-cs-agent ask "我订单 A1001 买的那件衣服是什么材质，能退吗？"
ecom-cs-agent ask "我想投诉商家并申请赔偿"
```

单步 trace：

```bash
ecom-cs-agent trace "帮我查订单 A1001 到哪了"
```

批量运行 episode：

```bash
ecom-cs-agent run \
  --input training/datasets/episode_cases.sample.jsonl \
  --output training/datasets/episode_traces.generated.jsonl
```

导出 LLaMA-Factory 数据：

```bash
ecom-cs-agent export-sft \
  --input training/datasets/episode_traces.generated.jsonl \
  --output training/datasets/llamafactory_episode_toolcall.json \
  --dataset-name ecom_cs_episode_toolcall \
  --dataset-info training/datasets/dataset_info.ecom_cs_episode_toolcall.json
```

导出 router / answer 训练集：

```bash
ecom-cs-agent export-router-sft \
  --input training/datasets/episode_traces.generated.jsonl \
  --output training/datasets/router_sft.generated.jsonl

ecom-cs-agent export-answer-sft \
  --input training/datasets/episode_traces.generated.jsonl \
  --output training/datasets/answer_sft.generated.jsonl
```

导出 LLaMA-Factory 可直接训练的 router / answer 数据：

```bash
ecom-cs-agent export-router-lf \
  --input training/datasets/episode_traces.train.seed.generated.jsonl \
  --output training/datasets/router_sft.train.lf.json \
  --dataset-name ecom_cs_router_sft_train \
  --dataset-info training/datasets/dataset_info.ecom_cs_router_sft.json

ecom-cs-agent export-answer-lf \
  --input training/datasets/episode_traces.train.seed.generated.jsonl \
  --output training/datasets/answer_sft.train.lf.json \
  --dataset-name ecom_cs_answer_sft_train \
  --dataset-info training/datasets/dataset_info.ecom_cs_answer_sft.json
```

建议的第一轮训练顺序：

1. 先导出 `router_sft.generated.jsonl`
2. 再导出 `answer_sft.generated.jsonl`
3. 先做 router LoRA SFT
4. 再做 answer LoRA SFT
5. 最后再联调运行时

离线评测：

```bash
ecom-cs-agent eval \
  --input training/datasets/episode_eval.sample.jsonl \
  --output training/datasets/episode_eval.report.json
```

当前评测报告除了基础准确率外，还会输出：

- `episode_pass_rate`
- `failure_summary`
- `auto_groundedness_accuracy`

在开始 Qwen LoRA SFT 之前，建议先测一版未后训练基线：

```bash
ecom-cs-agent export-router-sft \
  --input training/datasets/episode_traces.dev.seed.generated.jsonl \
  --output training/datasets/router_sft.dev.generated.jsonl

ecom-cs-agent export-answer-sft \
  --input training/datasets/episode_traces.dev.seed.generated.jsonl \
  --output training/datasets/answer_sft.dev.generated.jsonl

ecom-cs-agent benchmark-router \
  --input training/datasets/router_sft.dev.generated.jsonl \
  --model Qwen/Qwen3-1.7B \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --output training/datasets/router_benchmark.base.dev.json

ecom-cs-agent benchmark-answer \
  --input training/datasets/answer_sft.dev.generated.jsonl \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --output training/datasets/answer_benchmark.base.dev.json
```

这些 benchmark 结果适合在后续 LoRA 训练后做同集对比。当前支持的核心指标包括：

- `route_macro_f1`
- `intent_macro_f1`
- `tool_macro_f1`
- `ask_user_f1`
- `handoff_f1`
- `answer_token_f1`
- `grounded_f1`
- `escalation_f1`

如果你已经把 router 训到较高分，不想只看同分布 dev，也可以直接评测更难的 holdout 集：

```bash
ecom-cs-agent benchmark-router \
  --input training/datasets/router_sft.dev.holdout.jsonl \
  --model router-lora \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --output training/datasets/router_benchmark.holdout.json
```

这里的 benchmark 现在默认走 OpenAI-compatible 接口，例如 `vLLM`。
也就是说：

- 训练可以在 `AutoDL + LLaMA-Factory`
- 推理评测可以在 `AutoDL + vLLM`
- 不再依赖本地 `Ollama`

当前推荐模型分工：

- `router`: `Qwen/Qwen3-1.7B`
- `answer`: `Qwen/Qwen3-4B-Instruct-2507`

如果你想系统性扩数据，项目现在也支持一条最小可用的数据合成流水线：

```bash
ecom-cs-agent synthesize-episodes \
  --config training/datasets/synthesis_templates.default.json \
  --output-train training/datasets/episode_cases.synthetic.train.jsonl \
  --output-dev training/datasets/episode_cases.synthetic.dev.jsonl
```

## 关键文档

- [系统架构](/Users/zheisenbergy/code/agent/ecom-cs-agent/docs/系统架构.md)
- [CLI 使用说明](/Users/zheisenbergy/code/agent/ecom-cs-agent/docs/CLI说明.md)
- [训练计划](/Users/zheisenbergy/code/agent/ecom-cs-agent/docs/训练计划.md)
- [新手训练指南](/Users/zheisenbergy/code/agent/ecom-cs-agent/docs/新手训练指南.md)
- [数据规范](/Users/zheisenbergy/code/agent/ecom-cs-agent/docs/数据规范.md)
- [数据合成流水线](/Users/zheisenbergy/code/agent/ecom-cs-agent/docs/数据合成流水线.md)
- [数据构建策略](/Users/zheisenbergy/code/agent/ecom-cs-agent/docs/数据策略.md)
- [数据蓝图](/Users/zheisenbergy/code/agent/ecom-cs-agent/docs/数据集蓝图.md)
- [Trace 工作流](/Users/zheisenbergy/code/agent/ecom-cs-agent/docs/Trace工作流.md)
- [LLaMA-Factory 导出说明](/Users/zheisenbergy/code/agent/ecom-cs-agent/docs/LLaMA-Factory导出说明.md)
- [AutoDL 训练与评测流程](/Users/zheisenbergy/code/agent/ecom-cs-agent/docs/AutoDL工作流.md)
- [实验记录与踩坑总结](/Users/zheisenbergy/code/agent/ecom-cs-agent/docs/实验记录.md)
- [详细学习文档](/Users/zheisenbergy/code/agent/ecom-cs-agent/docs/项目导读.md)
