# 新手训练指南

## 1. 这份文档是给谁的

如果你现在对这些词还不熟：

- LoRA
- SFT
- LLaMA-Factory
- vLLM
- train / dev / eval
- router / answer

那这份文档就是给你的。

它的目标不是讲“最先进的方法”，而是让你先把当前这个项目的训练链路看懂、跑通、敢动手。

## 2. 先说结论

对当前项目，最合理的第一阶段路线是：

1. 先用规则系统生成 teacher trace
2. 先做未后训练模型 baseline benchmark
3. 先训练 `router`
4. 再训练 `answer`
5. 最后再做联调

不要一上来就做：

- 端到端大一统训练
- RL
- 长期记忆
- 海量自由生成数据

## 3. 你到底在训练什么

这个项目不是训练一个“什么都能聊的客服聊天机器人”。

你真正要训练的是两块能力。

### 3.1 Router

Router 负责决定“下一步该做什么”。

它要学会：

- 这是 `direct`、`internal_tool` 还是 `handoff`
- 现在是不是缺信息
- 要不要 ask_user
- 该调哪个工具
- 工具参数应该怎么填

简单理解：

- router 更像“决策大脑”

### 3.2 Answer

Answer 负责把 observation 变成用户能看的回答。

它要学会：

- 基于 observation 回答
- 不要脱离工具结果瞎编
- 缺信息时怎么 ask_user
- 高风险时怎么 handoff

简单理解：

- answer 更像“收口大脑”

## 4. 为什么 router 和 answer 要分开训

因为它们学的不是一回事。

如果混在一起训，你后面出了问题很难判断到底是：

- 决策错了
- 工具选错了
- 参数填错了
- 还是回答措辞不对

分开训的好处是：

- 更容易定位问题
- 更容易做 benchmark
- 小数据阶段更稳
- 后面替换运行时更方便

## 5. 一条训练数据到底是什么

这里最重要的单位不是单句问答，而是：

- `episode`

一个 episode 是“一次完整任务”。

例如：

```text
用户：我的订单到哪了？
助手：请提供订单号。
用户：A1001
助手：调用工具并给出最终回答
```

这整段合起来，才是更接近真实训练单位的东西。

但注意：

- 训练 router 时，通常会把 episode 再拆成 turn 级样本
- 训练 answer 时，也通常会把 episode 再拆成 turn 级样本

所以：

- 你维护的是 episode
- 真正喂给模型训练的，常常是由 episode 导出的 turn 样本

## 6. train、dev、hard eval 分别是什么

### 6.1 Train

作用：

- 给模型学习分布

特点：

- 样本最多
- 允许有一定相似表达
- 重点是“覆盖常见任务”

### 6.2 Dev

作用：

- 训练过程中检查模型有没有变好

特点：

- 不能和 train 完全重复
- 类型分布应该和 train 接近
- 数量比 train 少很多

### 6.3 Hard Eval

作用：

- 看模型边界稳不稳
- 做回归测试

重点应该放在：

- 缺槽位
- handoff
- not_found
- 多工具链
- 容易误判的短句

## 7. 训练和测试到底需要多少数据

这是你现在最关心的问题，我直接给你一个最好用的版本。

### 7.1 如果你只是想先把流程跑通

这是“能开训，但不要期待效果太稳”的级别。

- train episode：30 到 60 条
- dev episode：10 到 20 条
- hard eval episode：10 到 20 条

这个量级适合：

- 验证脚本能不能跑
- 验证 LLaMA-Factory 配置是否正确
- 验证 benchmark 能不能出结果

不适合：

- 认真比较模型能力
- 判断模型已经学会了边界

### 7.2 如果你想做第一轮可用验证

这是我更推荐你的起步档。

- train episode：60 到 150 条
- dev episode：20 到 40 条
- hard eval episode：20 到 40 条

通常导出后会变成：

- router train 样本：大约 80 到 250 条
- answer train 样本：大约 80 到 250 条

这个量级适合：

- 第一轮 LoRA SFT
- 比较训练前后 benchmark
- 初步看 router / answer 有没有学到东西

### 7.3 如果你想做比较稳定的第一版

- train episode：150 到 400 条
- dev episode：40 到 80 条
- hard eval episode：40 到 80 条

通常导出后会变成：

- router train 样本：200 到 700 条
- answer train 样本：200 到 700 条

这个量级更适合：

- 稳定比较模型版本
- 做多次迭代
- 更认真看 ask_user / handoff / tool chain

### 7.4 为什么不是一开始就要几千条

因为你现在的瓶颈不是“数据量太少”，而是：

- schema 还在打磨
- benchmark 还在建立
- 你还在学习整条链路

如果一开始就硬堆几千条，很可能只是把错误放大。

更好的顺序是：

1. 先做小而准的数据
2. 先确认 benchmark 有意义
3. 再扩数据

## 8. 当前仓库里的数据够不够

按当前仓库实际情况：

- `episode_cases.train.seed.jsonl`：30 条
- `episode_cases.dev.seed.jsonl`：13 条
- `episode_eval.hard.seed.jsonl`：16 条

导出后的 LLaMA-Factory 样本数目前是：

- `router_sft.train.lf.json`：34 条
- `router_sft.dev.lf.json`：14 条
- `answer_sft.train.lf.json`：34 条
- `answer_sft.dev.lf.json`：14 条

结论很直接：

- 现在这套数据可以跑通训练流程
- 但还不够做一轮有说服力的正式训练

如果你只是想学习：

- 现在够了

如果你想得到一个“明显变好”的模型：

- 还不够，建议先扩到至少 60 到 150 条 train episode

## 9. 最值得优先补哪些数据

如果时间有限，不要平均补所有类型。

优先补下面这些。

### 9.1 Router 最该补

- ask_user
- handoff
- logistics_status
- product_lookup
- policy_lookup
- order_status
- 多工具前置决策

最容易出问题的不是普通 direct，而是：

- 缺槽位
- 短句
- 模糊表达
- 看起来像 direct，实际要调工具

### 9.2 Answer 最该补

- grounded 回答
- ask_user 风格回答
- handoff 风格回答
- not_found 风格回答
- 多工具 observation 后的合成回答

## 10. 一个新手最稳的造数方案

推荐你按下面顺序做。

### 第一步：先补 episode seed

维护：

- `episode_cases.train.seed.jsonl`
- `episode_cases.dev.seed.jsonl`
- `episode_eval.hard.seed.jsonl`

### 第二步：跑 teacher trace

用规则系统跑：

```bash
ecom-cs-agent run \
  --input training/datasets/episode_cases.train.seed.jsonl \
  --output training/datasets/episode_traces.train.seed.generated.jsonl
```

### 第三步：导出 router / answer 训练数据

如果是普通 JSONL：

```bash
ecom-cs-agent export-router-sft \
  --input training/datasets/episode_traces.train.seed.generated.jsonl \
  --output training/datasets/router_sft.generated.jsonl

ecom-cs-agent export-answer-sft \
  --input training/datasets/episode_traces.train.seed.generated.jsonl \
  --output training/datasets/answer_sft.generated.jsonl
```

如果是给 LLaMA-Factory：

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

### 第四步：先记训练前 baseline

不要先训，再看好不好。

正确顺序是：

1. 先跑未后训练模型 benchmark
2. 记录 router / answer 指标
3. 再开始 LoRA

## 11. 你怎么判断“现在该不该继续造数据”

可以用这个简单规则。

### 可以先开始训练

满足这些就可以先开第一轮：

- train episode 至少 60 条左右
- dev 至少 20 条左右
- hard eval 至少 20 条左右
- ask_user / handoff / 多工具 各自都有覆盖

### 先不要急着训练

出现这些情况，先补数据：

- train 只有 20 多条
- dev 只有个位数
- handoff 几乎没有
- ask_user 样本特别少
- 多工具样本极少
- train 和 dev 问法几乎一模一样

## 12. 训练前 benchmark 应该什么时候测

应该测，而且最好在下面这个时点测：

1. 先把 `dev` 和 `hard eval` 补到基本够用
2. 再测未后训练模型 baseline
3. 记录结果
4. 再开始 LoRA

为什么不是更早测？

因为如果 `dev` 只有十来条，结果波动会比较大。

为什么又不能不测？

因为如果不测，你后面就不知道：

- 训练后到底有没有提升
- 是真的学到了，还是只是你感觉更好了

### 12.1 你现在可以用的 benchmark 输入

当前项目里已经更新好了训练前 benchmark 输入：

- [router_sft.dev.generated.jsonl](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/router_sft.dev.generated.jsonl)
- [answer_sft.dev.generated.jsonl](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/answer_sft.dev.generated.jsonl)

当前规模是：

- router dev 样本：28 条
- answer dev 样本：28 条

### 12.2 训练前 benchmark 怎么跑

如果你在 AutoDL 上已经用 `vLLM` 起好了基座模型：

先测 router baseline：

```bash
vllm serve Qwen/Qwen3-1.7B \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key EMPTY
```

那你就可以跑：

```bash
ecom-cs-agent benchmark-router \
  --input training/datasets/router_sft.dev.generated.jsonl \
  --model Qwen/Qwen3-1.7B \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --output training/datasets/router_benchmark.base.dev.json
```

再测 answer baseline：

```bash
vllm serve Qwen/Qwen3-4B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key EMPTY
```

```bash
ecom-cs-agent benchmark-answer \
  --input training/datasets/answer_sft.dev.generated.jsonl \
  --model Qwen/Qwen3-4B-Instruct \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --output training/datasets/answer_benchmark.base.dev.json
```

你需要重点看这些指标：

- router：`route_macro_f1`、`intent_macro_f1`、`tool_macro_f1`、`ask_user_f1`、`handoff_f1`
- answer：`answer_token_f1`、`grounded_f1`、`escalation_f1`

### 12.3 训练后 benchmark 怎么跑

训练后逻辑完全一样，只是把 `--model` 改成你在 `vLLM` 里挂载的 LoRA alias。

例如：

- `router-lora`
- `answer-lora`

也就是说：

- 训练前：测基座模型
- 训练后：测 LoRA alias

这样你才能把前后结果一一对比。

### 12.4 当前推荐的模型分工

当前项目推荐：

- `router`：`Qwen/Qwen3-1.7B`
- `answer`：`Qwen/Qwen3-4B-Instruct`

不要一开始就把两者都设成同一个大模型。

原因：

- router 是结构化决策任务，小模型通常就够
- answer 是生成任务，4B 更稳
- 这样训练和推理成本也更合理

## 13. 一个适合你的现实建议

如果你是小白，我建议不要把目标定成“先训出一个很强的模型”。

更合理的目标是分 3 步。

### 第 1 步：把链路跑通

目标：

- 知道怎么生成数据
- 知道怎么导出数据
- 知道怎么训练
- 知道怎么 benchmark

数据量：

- 30 到 60 条 train episode 就够

### 第 2 步：做第一轮有效训练

目标：

- 让 router 指标开始明显提升
- 让 answer 保持 grounded

数据量：

- 60 到 150 条 train episode
- 20 到 40 条 dev
- 20 到 40 条 hard eval

### 第 3 步：做更稳定的版本

目标：

- 把 ask_user / handoff / 多工具链补稳

数据量：

- 150 到 400 条 train episode

## 14. 你现在最该做什么

如果按当前仓库状态，我建议你下一步做这个：

1. 先把 train episode 从 30 扩到 80 左右
2. 把 dev 从 13 扩到 25 左右
3. 把 hard eval 从 16 扩到 25 左右
4. 重点补 `ask_user`、`handoff`、`order_product_policy_lookup`
5. 再跑第一轮 router LoRA
6. router benchmark 看提升后，再训 answer

## 15. 你可以把这份文档当成最短记忆版

只记住这几句就够了：

- 先训 router，再训 answer
- 先记 baseline，再开 LoRA
- 先做小而准的数据，不要一开始堆几千条
- 第一轮可用验证，目标是 60 到 150 条 train episode
- 现在仓库里的数据够学习，不够正式训练
