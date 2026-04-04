# CLI 使用说明

本项目只提供终端交互，不做前端和后端服务。

当前 CLI 以 `episode` 为中心，而不是以文件持久化 session 为中心。

## 安装

```bash
cd /Users/zheisenbergy/code/agent/ecom-cs-agent
uv python install 3.11
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

## 命令

### `ecom-cs-agent chat`

进入交互式终端。

特点：

- 在当前进程内维护未完成 episode 的状态
- 如果本轮需要用户补订单号或商品名，下一轮会继续当前任务
- 任务完成后，状态自动清空，开始新任务

示例：

```bash
ecom-cs-agent chat
ecom-cs-agent chat --shop-id demo-shop
ecom-cs-agent chat --order-id A1001
```

会话内指令：

- `/meta`：查看配置
- `/state`：查看当前 episode 状态
- `/reset`：清空当前任务状态
- `/exit`：退出

### `ecom-cs-agent ask "<query>"`

执行单个 turn。

它会：

- 使用 fresh episode state
- 执行当前这一步的 route / tool / answer
- 如果缺信息，会返回 ask_user 风格回答
- 不做跨进程状态延续

示例：

```bash
ecom-cs-agent ask "帮我查订单 A1001 到哪了"
ecom-cs-agent ask "这件衣服是什么材质？"
ecom-cs-agent ask "我订单 A1001 买的那件衣服是什么材质，能退吗？"
ecom-cs-agent ask "我想投诉商家并申请赔偿"
```

参数：

- `--shop-id`
- `--product-id`
- `--order-id`
- `--json`
- `--show-debug`

### `ecom-cs-agent trace "<query>"`

输出当前 turn 的结构化 trace。

示例：

```bash
ecom-cs-agent trace "帮我查订单 A1001 到哪了"
ecom-cs-agent trace "这件衣服是什么材质？"
```

### `ecom-cs-agent run`

批量运行 episode 样本并导出 episode trace。

输入 JSONL 每一行可以是：

1. 单轮 episode

```json
{"episode_id":"ep-001","query":"帮我查订单 A1001 到哪了","shop_id":"demo-shop"}
```

2. 多轮 episode

```json
{
  "episode_id":"ep-002",
  "shop_id":"demo-shop",
  "turns":[
    {"query":"我的订单到哪了？"},
    {"query":"A1001"}
  ]
}
```

运行示例：

```bash
ecom-cs-agent run \
  --input training/datasets/episode_cases.sample.jsonl \
  --output training/datasets/episode_traces.generated.jsonl
```

### `ecom-cs-agent export-sft`

将 episode trace 导出为 LLaMA-Factory 可用 ShareGPT 数据。

```bash
ecom-cs-agent export-sft \
  --input training/datasets/episode_traces.generated.jsonl \
  --output training/datasets/llamafactory_episode_toolcall.json \
  --dataset-info training/datasets/dataset_info.ecom_cs_episode_toolcall.json
```

### `ecom-cs-agent export-router-sft`

把 episode trace 导出成 router 训练 JSONL。

输出重点字段：

- `user_query`
- `state_before`
- `route`
- `intent`
- `tool_name`
- `tool_arguments`
- `missing_slots`

示例：

```bash
ecom-cs-agent export-router-sft \
  --input training/datasets/episode_traces.generated.jsonl \
  --output training/datasets/router_sft.generated.jsonl
```

### `ecom-cs-agent export-answer-sft`

把 episode trace 导出成 answer 训练 JSONL。

输出重点字段：

- `query`
- `route`
- `intent`
- `tool_steps`
- `answer`
- `citations`
- `grounded`
- `waiting_for_user`

示例：

```bash
ecom-cs-agent export-answer-sft \
  --input training/datasets/episode_traces.generated.jsonl \
  --output training/datasets/answer_sft.generated.jsonl
```

### `ecom-cs-agent export-router-lf`

把 episode trace 导出成 LLaMA-Factory 可直接训练的 router 数据。

```bash
ecom-cs-agent export-router-lf \
  --input training/datasets/episode_traces.train.seed.generated.jsonl \
  --output training/datasets/router_sft.train.lf.json \
  --dataset-name ecom_cs_router_sft_train \
  --dataset-info training/datasets/dataset_info.ecom_cs_router_sft.json
```

### `ecom-cs-agent export-answer-lf`

把 episode trace 导出成 LLaMA-Factory 可直接训练的 answer 数据。

```bash
ecom-cs-agent export-answer-lf \
  --input training/datasets/episode_traces.train.seed.generated.jsonl \
  --output training/datasets/answer_sft.train.lf.json \
  --dataset-name ecom_cs_answer_sft_train \
  --dataset-info training/datasets/dataset_info.ecom_cs_answer_sft.json
```

### `ecom-cs-agent eval`

基于带 gold 标签的 episode JSONL 做离线评测。

```bash
ecom-cs-agent eval \
  --input training/datasets/episode_eval.sample.jsonl
```

可选地也可以落盘结果：

```bash
ecom-cs-agent eval \
  --input training/datasets/episode_eval.sample.jsonl \
  --output training/datasets/episode_eval.report.json
```

当前评测结果会包含：

- 各项准确率 summary
- `episode_pass_rate`：整条 episode 是否整体通过
- `failure_summary`：每个指标失败了哪些 `episode_id`
- `details`：逐条样本的 expected / predicted / failed_metrics
- `auto_groundedness_accuracy`：不依赖人工标注的 groundedness 自动诊断

### `ecom-cs-agent benchmark-router`

评测未后训练模型在 router 任务上的基线表现。

适合在开始 LoRA SFT 前，先拿一版 `route / intent / tool / ask_user / handoff` 指标。

当前命令默认对接 OpenAI-compatible 服务，例如 `vLLM`。

```bash
ecom-cs-agent benchmark-router \
  --input training/datasets/router_sft.dev.generated.jsonl \
  --model Qwen/Qwen3-1.7B \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --output training/datasets/router_benchmark.base.dev.json
```

当前输出重点包括：

- `route_accuracy`
- `route_macro_f1`
- `intent_accuracy`
- `intent_macro_f1`
- `tool_accuracy`
- `tool_macro_f1`
- `ask_user_f1`
- `handoff_f1`

### `ecom-cs-agent benchmark-answer`

评测未后训练模型在 answer 任务上的基线表现。

```bash
ecom-cs-agent benchmark-answer \
  --input training/datasets/answer_sft.dev.generated.jsonl \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --output training/datasets/answer_benchmark.base.dev.json
```

当前输出重点包括：

- `answer_token_f1`
- `answer_exact_match`
- `citation_set_accuracy`
- `grounded_f1`
- `escalation_f1`

常用参数：

- `--base-url`：OpenAI-compatible 服务地址，`vLLM` 常见为 `http://127.0.0.1:8000/v1`
- `--api-key`：如果服务不校验，可直接传 `EMPTY`
- `--timeout-seconds`：单条 benchmark 请求超时
- `--max-tokens`：限制回答长度，减少 benchmark 波动

## 输出语义

默认文本输出包含：

- 路由
- 意图
- 置信度
- 改写查询
- 工具信息
- 回答
- 是否等待用户补充信息

如果一个 turn 内触发了多工具串联，CLI 会额外输出：

- `工具链`
- 每一步 `tool_call -> status`

如果返回“等待用户补充信息”，说明当前 episode 尚未完成。
