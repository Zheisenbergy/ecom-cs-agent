# 数据规范

## 1. 数据单位

当前项目建议把“完整 episode”作为数据单位。

不要把训练数据只做成散落的单轮 query，因为那样很难学到：

- ask_user
- 参数补齐
- pending tool continuation
- grounded final answer

## 2. Episode 输入样本

推荐用 JSONL，每行一个 episode。

### 单轮 episode

```json
{"episode_id":"ep-001","query":"帮我查订单 A1001 到哪了","shop_id":"demo-shop"}
```

### 多轮 episode

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

## 3. Episode Trace 输出

`run` 命令会输出 episode trace JSONL，每行一个 `EpisodeRecord`。

它至少包含：

- `episode_id`
- `turns`
- `final_response`
- `final_state`
- `waiting_for_user`
- `completed`

其中每个 `turn` 里包含：

- 原始 request
- route decision
- tool call
- tool result
- tool steps
- response
- state_before
- state_after

## 4. 工具数据

第一阶段仍然用本地 JSON mock 数据：

- `products.json`
- `policies.json`
- `orders.json`
- `logistics.json`

这些数据不需要训练进模型，只需要作为环境工具返回 observation。

## 5. SFT 数据

导出到 LLaMA-Factory 后，一条样本就是一个完整 episode 的 ShareGPT 对话。

一个典型样本会包含：

- `human`
- `gpt`
- `function_call`
- `observation`

## 6. 评测数据建议

建议单独准备 episode 级评测集，至少标注：

- gold final route
- gold tool sequence
- gold tool arguments
- gold final answer
- gold handoff / ask_user 行为
- gold tool chain

推荐字段：

- `expected_final_route`
- `expected_final_intent`
- `expected_completed`
- `expected_any_ask_user`
- `expected_handoff`
- `expected_tool_chain`
- `expected_turn_count`
- `expected_final_citations`
- `expected_answer_contains`
- `expected_answer_not_contains`

当前 `eval` 输出还会自动补充：

- `auto_groundedness_accuracy`
- `episode_pass_rate`
- `failure_summary`
- `failed_metrics`

## 7. 命名建议

- 输入样本：`episode_cases.train.jsonl`、`episode_cases.dev.jsonl`
- 导出轨迹：`episode_traces.generated.jsonl`
- SFT 数据：`llamafactory_episode_toolcall.json`
- 评测样本：`episode_eval.dev.jsonl`
