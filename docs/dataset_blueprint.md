# 数据蓝图

## 1. 目标

这份文档不是讲抽象原则，而是给当前项目一份可以直接执行的数据蓝图。

当前项目的数据不是一份“普通问答集”，而是三层：

1. 环境事实层
2. agent 行为层
3. grounded 回答层

你后面做训练时，也应该按这三层来组织，而不是把所有样本混成一堆。

## 2. 三层数据分别是什么

## 2.1 环境事实层

对应当前本地知识库：

- `products.json`
- `orders.json`
- `policies.json`
- `logistics.json`

这层数据的作用只有一个：

- 作为工具 observation 的唯一事实源

这层不是直接训练给模型背诵的。

## 2.2 Agent 行为层

这层是当前项目最重要的数据层。

关注的是：

- route
- intent
- ask_user
- tool_call
- tool_arguments
- 多工具是否继续
- 何时结束

这层数据的基本单位是：

- `episode`

也就是：

- 一条完整任务

而不是：

- 一句孤立问答

## 2.3 Grounded 回答层

这层关注的是：

- observation 到 answer 的映射
- not_found 怎么答
- ask_user 怎么答
- handoff 怎么答

这一层不能脱离 observation 单独存在。

否则模型就会学成：

- 不查工具也敢答

## 3. 当前推荐的数据文件布局

建议按下面方式组织：

### 3.1 原始 episode 输入

- `training/datasets/episode_cases.train.seed.jsonl`
- `training/datasets/episode_cases.dev.seed.jsonl`

这两份文件保存“任务输入骨架”。

每行一个 episode。

## 3.2 评测集

- `training/datasets/episode_eval.sample.jsonl`
- `training/datasets/episode_eval.hard.seed.jsonl`

其中：

- `sample` 偏演示
- `hard.seed` 偏边界与回归

## 3.3 运行后 trace

- `training/datasets/episode_traces.generated.jsonl`

这是规则环境跑出来的 teacher trace。

## 3.4 从 trace 导出的训练数据

- `training/datasets/router_sft.generated.jsonl`
- `training/datasets/answer_sft.generated.jsonl`

## 4. 每一层应该有哪些字段

## 4.1 Episode 输入层

建议字段：

- `episode_id`
- `shop_id`
- `query`
- `turns`
- 可选 `product_id`
- 可选 `order_id`

其中：

- 单轮用 `query`
- 多轮用 `turns`

## 4.2 Router SFT 层

建议至少保留：

- `episode_id`
- `turn_index`
- `user_query`
- `state_before`
- `route`
- `intent`
- `need_clarification`
- `tool_name`
- `tool_arguments`
- `missing_slots`

这一层训练的是动作，不训练业务事实记忆。

## 4.3 Answer SFT 层

建议至少保留：

- `episode_id`
- `turn_index`
- `query`
- `route`
- `intent`
- `tool_steps`
- `answer`
- `citations`
- `grounded`
- `waiting_for_user`
- `escalation_required`

这一层训练的是：

- 如何基于 observation 安全收口

## 4.4 Eval 层

建议至少保留：

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

## 5. 当前最适合补的样本类别

当前建议优先补这 8 类：

1. direct
2. logistics_status
3. product_lookup
4. policy_lookup
5. order_status
6. order_product_lookup
7. order_policy_lookup
8. order_product_policy_lookup

同时必须单独补：

- handoff
- not_found
- ask_user

## 6. 训练集、开发集、评测集怎么分工

## 6.1 Train

目的：

- 给模型学分布

特点：

- 量相对大
- 覆盖常见问法
- 允许一定表达重复

## 6.2 Dev

目的：

- 训练过程中做快速验证

特点：

- 和 train 意图分布相似
- 但问法不要完全重复

## 6.3 Hard Eval

目的：

- 边界稳定性和回归测试

重点应该放在：

- 缺槽位
- not_found
- handoff
- 多工具链
- 容易误判的短句

## 7. 一个推荐的起步规模

如果只是第一轮项目验证，建议这样起步：

- `train episode`：60 到 150 条
- `dev episode`：20 到 40 条
- `hard eval episode`：20 到 40 条

从这些 episode 跑出 trace 之后，通常会得到更多 turn 级样本：

- router 样本数 > episode 数
- answer 样本数 > episode 数

## 8. 一条数据怎么从 0 走到训练

推荐流程：

1. 先维护知识库 JSON
2. 新增 episode seed
3. 用 `run` 生成 teacher trace
4. 用 `eval` 检查边界
5. 用 `export-router-sft` 导 router 数据
6. 用 `export-answer-sft` 导 answer 数据
7. 再做 train/dev 划分与人工抽检

## 9. 当前阶段不要做什么

当前阶段不要把重点放在：

- 域外开放问答
- 图片理解数据
- 自由生成商品百科
- 海量但不可验证的 synthetic QA

当前阶段只需要围绕当前受限工具环境，把任务闭环做实。

## 10. 现在仓库里已经有的配套文件

你现在可以直接从这些文件开始：

- [episode_cases.train.seed.jsonl](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/episode_cases.train.seed.jsonl)
- [episode_cases.dev.seed.jsonl](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/episode_cases.dev.seed.jsonl)
- [episode_eval.hard.seed.jsonl](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/episode_eval.hard.seed.jsonl)
- [data_strategy.md](/Users/zheisenbergy/code/agent/ecom-cs-agent/docs/data_strategy.md)

这几份文件的关系是：

- `data_strategy.md` 讲为什么这样做
- `dataset_blueprint.md` 讲具体怎么落
- `seed jsonl` 直接给你起步样本
