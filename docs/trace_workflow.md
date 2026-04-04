# Trace 工作流

## 1. 目标

这个项目当前不是先训练模型自由发挥，而是先产出稳定的 episode trace。

episode trace 是后续训练和评测的中间产物。

## 2. 单步 trace

查看单个 turn 的结构化轨迹：

```bash
cd /Users/zheisenbergy/code/agent/ecom-cs-agent
./.venv/bin/ecom-cs-agent trace "帮我查订单 A1001 到哪了"
```

输出包括：

- request
- route decision
- tool call
- tool result
- tool steps
- response
- state_before
- state_after

如果这个 turn 触发了多工具链，那么 `tool_steps` 会记录完整链路，例如：

1. `get_order_status`
2. `get_product_info`
3. `get_policy`

## 3. 批量 episode trace

先准备输入：

文件：[episode_cases.sample.jsonl](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/episode_cases.sample.jsonl)

然后运行：

```bash
cd /Users/zheisenbergy/code/agent/ecom-cs-agent
./.venv/bin/ecom-cs-agent run \
  --input training/datasets/episode_cases.sample.jsonl \
  --output training/datasets/episode_traces.generated.jsonl
```

输出结果中每一行都是一个完整 episode。

## 4. 这些 trace 用来做什么

可以直接用于：

- episode 级工具调用评测
- 多工具链正确率评测
- ask_user / handoff 行为分析
- ShareGPT SFT 数据导出
- 人工审核和修订

## 5. 更合理的流程

1. 先设计 episode 输入样本
2. 用当前规则系统跑出 episode trace
3. 对 trace 做人工修订
4. 导出成 LLaMA-Factory 数据
5. 再做 Qwen 模型 SFT
