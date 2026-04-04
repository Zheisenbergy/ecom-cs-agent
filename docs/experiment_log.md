# 实验记录与踩坑总结

这份文档记录当前项目到 `2026-04-04` 为止，已经实际做过的训练、benchmark、遇到的问题，以及对应的处理方式。

它的作用不是讲原理，而是回答：

- 这个项目已经试过什么
- 每一轮训练结果怎样
- 哪些改动有帮助
- 哪些坑以后不要再踩

## 1. 当前实验目标

当前阶段优先验证：

1. `router` 能不能学会结构化决策
2. `ask_user` 能不能起来
3. `handoff` 能不能起来
4. benchmark 能不能稳定反映真实能力变化

当前推荐模型分工：

- `router`: `Qwen/Qwen3-1.7B`
- `answer`: `Qwen/Qwen3-4B-Instruct-2507`

## 2. 已完成的主要实验

### 2.1 第一轮：小规模 seed 数据

最早的数据量大致是：

- `router_sft.train.lf.json`：34 条
- `router_sft.dev.lf.json`：14 条
- `answer_sft.train.lf.json`：34 条
- `answer_sft.dev.lf.json`：14 条

这轮的目标主要是：

- 跑通训练链路
- 跑通 benchmark
- 看 LoRA 后是否比 base 稍好

### 2.2 第二轮：mixed synthetic 数据

后面接入了合成数据流水线：

- [build_synthetic_datasets.sh](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/build_synthetic_datasets.sh)
- [synthesis_templates.default.json](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/synthesis_templates.default.json)

第一次 mixed 数据大致产出：

- `router_sft.train.mixed.generated.jsonl`：397 条
- `router_sft.dev.mixed.generated.jsonl`：130 条

再后来为了加强 `handoff` 和 `ask_user`，又扩了一轮模板。

扩模板后的 synthetic 配置总量是：

- train：286 条 synthetic seed
- dev：97 条 synthetic seed

## 3. 关键 benchmark 结果

### 3.1 第一轮 seed 数据：base vs LoRA

以下结果来自同一份小规模 dev 集对比。

| 指标 | base | router LoRA |
|---|---:|---:|
| `route_accuracy` | 0.0714 | 0.2500 |
| `route_macro_f1` | 0.0495 | 0.1580 |
| `intent_accuracy` | 0.1071 | 0.1429 |
| `intent_macro_f1` | 0.0429 | 0.0650 |
| `tool_accuracy` | 0.5357 | 0.5714 |
| `tool_macro_f1` | 0.5098 | 0.5721 |
| `ask_user_f1` | 0.0000 | 0.0000 |
| `handoff_f1` | 0.0000 | 0.0000 |
| `missing_slots_exact_match` | 0.7857 | 0.7857 |
| `tool_arguments_exact_match` | 0.3571 | 0.3214 |

这一轮结论：

- LoRA 确实有提升
- 但 `ask_user` 和 `handoff` 完全没起来
- 说明主要瓶颈不是训练脚本，而是数据覆盖

### 3.2 mixed v1：第一轮 synthetic mixed 数据训练

训练日志：

- `train_loss = 0.2652`
- `eval_loss = 0.0197`
- dev 样本数：130

benchmark：

| 指标 | mixed v1 |
|---|---:|
| `route_accuracy` | 0.2308 |
| `route_macro_f1` | 0.1538 |
| `intent_accuracy` | 0.2923 |
| `intent_macro_f1` | 0.1462 |
| `tool_accuracy` | 0.2462 |
| `tool_macro_f1` | 0.1337 |
| `ask_user_f1` | 0.4324 |
| `handoff_f1` | 0.0000 |
| `missing_slots_exact_match` | 0.8385 |
| `tool_arguments_exact_match` | 0.4923 |

这一轮结论：

- `ask_user` 明显开始学到东西
- `missing_slots` 和参数抽取也更稳
- 但 `handoff` 依然完全没起来
- `route / tool` 仍然偏弱

### 3.3 mixed v2：加强 handoff / ask_user 模板后的训练

训练日志：

- `train_loss = 0.2001`
- `eval_loss = 0.0103`
- dev 样本数：200

第一次 benchmark 结果如下：

| 指标 | mixed v2 首次评测 |
|---|---:|
| `route_accuracy` | 0.2150 |
| `route_macro_f1` | 0.1162 |
| `intent_accuracy` | 0.1850 |
| `intent_macro_f1` | 0.1494 |
| `tool_accuracy` | 0.3200 |
| `tool_macro_f1` | 0.1998 |
| `ask_user_f1` | 0.2182 |
| `handoff_f1` | 0.0000 |
| `missing_slots_exact_match` | 0.7850 |
| `tool_arguments_exact_match` | 0.5700 |

这一轮表面现象是：

- `tool` 和 `tool_arguments` 更好了
- 但 `ask_user` 反而下降
- `handoff` 仍然是 0

但是这轮结果后来被判断为：

- **不完全可信**

原因不是训练失败，而是 benchmark prompt 本身存在不对齐问题，见第 5 节。

### 3.4 mixed v2：校准 benchmark prompt 后重新评测

在 benchmark prompt 改成更接近训练格式之后，同一版 `mixed v2` 的结果变成：

| 指标 | mixed v2 校准后 |
|---|---:|
| `route_accuracy` | 0.5100 |
| `route_macro_f1` | 0.4308 |
| `intent_accuracy` | 0.4350 |
| `intent_macro_f1` | 0.3264 |
| `tool_accuracy` | 0.4900 |
| `tool_macro_f1` | 0.4029 |
| `ask_user_f1` | 0.0784 |
| `handoff_f1` | 0.5000 |
| `missing_slots_exact_match` | 0.7650 |
| `tool_arguments_exact_match` | 0.6900 |

这轮结论：

- `handoff` 终于明显起来了
- `route / tool` 也比第一次评测结果更可信
- 但 `ask_user recall` 反而暴露得更清楚了
- 说明当时真正的主问题已经从 `handoff` 转成了 `ask_user`

### 3.5 mixed v3：继续扩模板后的训练

训练日志：

- `train_loss = 0.1494`
- `eval_loss = 0.0052`
- dev 样本数：288

benchmark：

| 指标 | mixed v3 |
|---|---:|
| `route_accuracy` | 0.6736 |
| `route_macro_f1` | 0.6000 |
| `intent_accuracy` | 0.4062 |
| `intent_macro_f1` | 0.3222 |
| `tool_accuracy` | 0.4722 |
| `tool_macro_f1` | 0.4341 |
| `ask_user_f1` | 0.0652 |
| `handoff_f1` | 0.7429 |
| `missing_slots_exact_match` | 0.7014 |
| `tool_arguments_exact_match` | 0.6493 |

这轮结论：

- `route` 和 `handoff` 明显更强了
- `handoff_f1` 达到目前最好结果
- 但 `ask_user_f1` 依然很差，甚至略低于校准后的 `mixed v2`
- 这说明单纯扩模板数量不够，模型还是没有稳定学会“看状态再决定是否追问”

## 4. 训练 loss 的变化怎么看

目前几轮 router 训练里，loss 变化如下：

| 轮次 | `train_loss` | `eval_loss` | 备注 |
|---|---:|---:|---|
| mixed v1 | 0.2652 | 0.0197 | 第一次大规模 mixed 数据 |
| mixed v2 | 0.2001 | 0.0103 | 加强 handoff / ask_user 模板后 |
| mixed v3 | 0.1494 | 0.0052 | 继续扩模板后 |

怎么理解：

- `loss` 下降说明模型对当前训练分布拟合得更好了
- 但 `loss` 变好不代表业务指标一定变好
- 对 router 这类结构化任务，最后还是要看 benchmark 指标

`mixed v3` 就是一个很典型的例子：

- `loss` 继续下降
- `route / handoff` 继续提升
- 但 `ask_user` 没有同步提升

这说明：

- 训练并没有坏
- 但当前训练分布还没有足够强地约束 ask_user 这个边界行为

所以项目当前的基本原则是：

- `loss` 用来看训练是否正常
- `benchmark` 用来看能力是否真的提升

## 5. 我们实际遇到的问题，以及怎么解决

### 5.1 LLaMA-Factory / 环境依赖错位

遇到过的问题包括：

- `torchaudio` 和 CUDA 运行时不匹配
- `.venv` 里的 `llamafactory-cli` 和外部环境包不一致
- 缺少 `transformers`

解决方法：

- 确保激活项目自己的 `.venv`
- 用 `.venv/bin/llamafactory-cli`
- 在同一个环境里安装 `transformers / datasets / accelerate / peft / trl`

经验：

- 训练前先检查：
  - `which python`
  - `which pip`
  - `which llamafactory-cli`

### 5.2 Hugging Face 在线下载超时

遇到的问题：

- `Qwen/Qwen3-1.7B` 下载时连 `huggingface.co` 超时
- 训练启动前卡在 tokenizer / config 拉取阶段

解决方法：

- 确认本地 cache 目录
- 直接使用本地 snapshot 路径作为 `BASE_MODEL`
- 开启离线模式：

```bash
export HF_HOME=/root/autodl-tmp/cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

经验：

- 在 AutoDL 上，模型一旦下载过，训练时优先走本地 snapshot，不要反复用远程 repo id

### 5.3 synthetic 数据模板容量不够

遇到的问题：

- `RuntimeError: scenario handoff_manual_support 只生成了 29 条，低于目标 40 条`

原因：

- `entity_source = none` 的场景没有实体槽位可以变化
- 唯一可生成数大致就等于模板条数
- 如果 `train/dev` 目标超过模板数，去重后就凑不出来

解决方法：

- 给 `handoff` 模板补充更多等价表达
- 让模板数量至少覆盖目标条数

经验：

- `entity_source = none` 的场景，扩量只能靠扩模板
- 不能只改 `counts`

### 5.4 benchmark prompt 与训练 prompt 不够对齐

这是当前最重要的一次诊断。

当时 benchmark 里看到这类异常输出：

```json
{
  "route": "direct|internal_tool|handoff",
  "intent": "direct_answer"
}
```

以及大量 handoff 样本被预测成：

```json
{
  "route": "direct",
  "intent": "general_direct_answer"
}
```

为什么这提示了 benchmark 可能有问题：

- 训练时模型看到的 `route` 真值永远是：
  - `direct`
  - `internal_tool`
  - `handoff`
- 它从来没有学过：
  - `"direct|internal_tool|handoff"`

旧 benchmark prompt 里却用了这种占位式写法：

```json
"route": "direct|internal_tool|handoff"
```

模型很可能是在“照抄模板”，而不是在真的做 route 决策。

解决方法：

1. benchmark prompt 改成更像训练样本的结构
   - 更接近 `system + instruction + input`
2. 删除会误导模型抄模板的占位式 route 示例
3. 加一层轻量归一化
   - 遇到明显非法 route 时，先做合理映射再打分

对应代码：

- [baseline_benchmark.py](/Users/zheisenbergy/code/agent/ecom-cs-agent/app/services/baseline_benchmark.py)

经验：

- 对结构化任务，训练 prompt 和 benchmark prompt 的对齐非常重要
- 否则你测出来的可能不是“模型能力”，而是“模型是否被考试题模板带偏”

### 5.5 ask_user 的真正问题不是“样本太少”，而是“状态对照不够强”

在 `mixed v3` 的误差分析里，最常见的 ask_user 漏判是：

- `这件衣服有啥颜色`
- `这个有啥颜色`
- `我买的那个有保修吗`
- `帮我看下快递到哪了`

这些样本的共同点是：

- 句子本身很短
- 带明显指代词
- 真正决定要不要追问的，不是句面，而是 `state_before`

例如同一句：

- `帮我看下快递到哪了`

如果 `state_before.order_id = null`：

- 应该 `need_clarification = true`
- `missing_slots = ["order_id"]`

如果 `state_before.order_id = "A1005"`：

- 就不该再追问
- 应该直接进入工具查询

所以这一轮的修正方向不是只补更多 ask_user 模板，而是补：

- **同一句模糊问法**
- **有上下文时不追问**
- **没上下文时必须追问**

这种成对对照样本。

对应改动：

- [training_data.py](/Users/zheisenbergy/code/agent/ecom-cs-agent/app/exporters/training_data.py)
- [baseline_benchmark.py](/Users/zheisenbergy/code/agent/ecom-cs-agent/app/services/baseline_benchmark.py)
- [synthesis_templates.default.json](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/synthesis_templates.default.json)

### 5.6 这次是怎么修 ask_user 的

这轮主要做了两件事：

1. 把 router 训练 prompt 和 benchmark prompt 统一成同一套规则
   - 都强调必须看 `state_before`
   - 都明确写清楚缺 `order_id / product_id` 时要 `need_clarification=true`
   - 都明确禁止编造工具参数
2. 在 synthetic 模板里加入上下文对照样本
   - 新增 `context_only_order_pronoun_queries`
   - 新增 `context_only_product_pronoun_queries`
   - 让同一句模糊问法同时出现在：
     - 无上下文版本
     - 有 `order_id` 或 `product_id` 版本

本地验证时，新的 synthetic-only router 样本里：

- train `ask_user` 占比约 `0.3308`
- dev `ask_user` 占比约 `0.3168`

而且已经能明确看到这种对照：

- `帮我看下快递到哪了`
  - 有 `order_id` 时不追问
  - 无 `order_id` 时追问
- `这个有啥颜色`
  - 有 `product_id` 时不追问
  - 无 `product_id` 时追问

这类样本比单纯扩数量更重要，因为它训练的是：

- “是否要问”的**条件边界**

而不只是“哪些词像是 ask_user”。

### 5.7 `handoff` 不再是唯一最大短板

在前几轮里，`handoff` 一直是最明显的问题。

但到 `mixed v3` 之后，更准确的说法应该是：

- `handoff` 已经明显改善
- 当前第一优先级变成了 `ask_user recall`

`handoff` 之前经常被预测为：

- `route = direct`
- `intent = general_direct_answer`

这说明：

- 模型对人工介入意图仍然不敏感
- 当前 router 的保守默认行为还是太偏 `direct`

这也是为什么项目前一阶段优先扩了：

- `handoff_manual_support`
- `handoff_robot_refusal`

两类 synthetic 模板。

## 6. 当前阶段最可靠的判断

到目前为止，可以比较确定的是：

1. 训练链路已经跑通
2. synthetic 数据流水线已经跑通
3. benchmark prompt 对齐后，评测信号更可信了
4. `handoff` 在 `mixed v2` 校准后和 `mixed v3` 上都明显改善
5. 当前最难补稳的其实是 `ask_user recall`
6. benchmark 本身也需要持续校准，不能只看第一次输出

## 7. 现在最推荐的下一步

当前最推荐的顺序仍然是：

1. 用新的 prompt 对齐 + ask_user 对照样本，继续训下一版 router
2. 重点看 `ask_user_f1` 能不能明显高于 `mixed v3`
3. 如果 ask_user 稳了，再决定是否继续训 answer

换句话说：

- 当前还不建议急着继续训 `answer`
- 更值得优先解决的是：
  - `ask_user recall`
  - benchmark 对齐
  - 上下文条件下的缺槽位决策
