# LLaMA-Factory 导出说明

本项目现在按 episode 导出，而不是按散落单轮导出。

## 1. 先生成 episode trace

```bash
cd /Users/zheisenbergy/code/agent/ecom-cs-agent
./.venv/bin/ecom-cs-agent run \
  --input training/datasets/episode_cases.sample.jsonl \
  --output training/datasets/episode_traces.generated.jsonl
```

## 2. 导出为 ShareGPT

```bash
cd /Users/zheisenbergy/code/agent/ecom-cs-agent
./.venv/bin/ecom-cs-agent export-sft \
  --input training/datasets/episode_traces.generated.jsonl \
  --output training/datasets/llamafactory_episode_toolcall.json \
  --dataset-name ecom_cs_episode_toolcall \
  --dataset-info training/datasets/dataset_info.ecom_cs_episode_toolcall.json
```

## 3. 样本结构

导出后，每条样本对应一个完整 episode，例如：

```json
{
  "conversations": [
    {"from": "human", "value": "我的订单到哪了？"},
    {"from": "gpt", "value": "这个问题需要订单号才能继续查询，请先提供订单号。"},
    {"from": "human", "value": "A1001"},
    {"from": "function_call", "value": "{\"name\":\"get_logistics_status\",\"arguments\":{\"order_id\":\"A1001\"}}"},
    {"from": "observation", "value": "{\"status\":\"ok\",\"data\":{...},\"message\":\"\"}"},
    {"from": "gpt", "value": "订单 A1001 当前在运输中，承运商是中通，最新节点为已到达杭州转运中心。"}
  ],
  "system": "...",
  "tools": "..."
}
```

如果某个 turn 内包含多工具链，导出结果会连续写出多组：

- `function_call`
- `observation`

例如：

1. `get_order_status`
2. `get_product_info`
3. `get_policy`

## 4. 为什么按 Episode 导出更合理

因为这样模型学到的是：

- 何时 ask_user
- 何时调工具
- 多工具情况下何时继续调下一个工具
- observation 之后如何结束

而不是只学孤立的一步。

## 5. 为第一轮训练准备更细的监督数据

如果你不想一开始就直接做整条 episode 的端到端 SFT，也可以先从 trace 导出更细粒度数据：

### 5.1 导出 router 数据

```bash
cd /Users/zheisenbergy/code/agent/ecom-cs-agent
./.venv/bin/ecom-cs-agent export-router-sft \
  --input training/datasets/episode_traces.generated.jsonl \
  --output training/datasets/router_sft.generated.jsonl
```

### 5.2 导出 answer 数据

```bash
cd /Users/zheisenbergy/code/agent/ecom-cs-agent
./.venv/bin/ecom-cs-agent export-answer-sft \
  --input training/datasets/episode_traces.generated.jsonl \
  --output training/datasets/answer_sft.generated.jsonl
```

这样更适合分阶段训练：

1. 先训练 router
2. 再训练 answer
3. 最后再做 episode 级联调

## 6. 导出成 LLaMA-Factory 可直接训练的 router / answer 数据

如果你不想自己再写一层格式转换，现在项目已经支持直接导出：

### 6.1 Router

```bash
cd /Users/zheisenbergy/code/agent/ecom-cs-agent
./.venv/bin/ecom-cs-agent export-router-lf \
  --input training/datasets/episode_traces.train.seed.generated.jsonl \
  --output training/datasets/router_sft.train.lf.json \
  --dataset-name ecom_cs_router_sft_train \
  --dataset-info training/datasets/dataset_info.ecom_cs_router_sft.json
```

### 6.2 Answer

```bash
cd /Users/zheisenbergy/code/agent/ecom-cs-agent
./.venv/bin/ecom-cs-agent export-answer-lf \
  --input training/datasets/episode_traces.train.seed.generated.jsonl \
  --output training/datasets/answer_sft.train.lf.json \
  --dataset-name ecom_cs_answer_sft_train \
  --dataset-info training/datasets/dataset_info.ecom_cs_answer_sft.json
```

如果你要在 AutoDL 上直接跑训练和 benchmark，可以继续看：

- [AutoDL 训练与评测流程](/Users/zheisenbergy/code/agent/ecom-cs-agent/docs/AutoDL工作流.md)
