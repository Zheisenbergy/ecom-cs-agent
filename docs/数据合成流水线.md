# 数据合成流水线

## 1. 先说结论

对当前项目，最适合的造数流水线不是“让大模型随便写很多电商问答”，而是：

1. 用模板和槽位生成 `episode seed`
2. 用当前规则系统跑出 `teacher trace`
3. 再导出 `router / answer` 训练数据
4. 用 benchmark 结果反推下一轮该补哪些类型

一句话理解：

- 模板生成器负责“出题”
- 规则系统负责“给标准答案”
- 训练脚本负责“把答案教给模型”

## 2. 为什么规则系统能当 teacher

因为你当前项目训练的不是“背百科知识”，而是：

- route
- intent
- tool_name
- tool_arguments
- missing_slots
- ask_user
- handoff
- grounded answer

这些字段都属于：

- 结构化
- 可验证
- 有明确正确答案

所以规则系统很适合当第一阶段 teacher。

当前链路其实已经是 teacher-student 结构：

1. `episode_cases.*.seed.jsonl` 提供用户任务输入
2. `ecom-cs-agent run` 让规则系统跑完整 episode
3. 生成 `episode_traces.*.generated.jsonl`
4. 再导出：
   - `router_sft.*`
   - `answer_sft.*`

## 3. 当前新增的最小可用流水线

项目里已经新增一个命令：

```bash
ecom-cs-agent synthesize-episodes \
  --config training/datasets/synthesis_templates.default.json \
  --output-train training/datasets/episode_cases.synthetic.train.jsonl \
  --output-dev training/datasets/episode_cases.synthetic.dev.jsonl
```

它的作用是：

- 按配置文件批量生成 train/dev seed episode
- 支持：
  - 单轮 query
  - 多轮补槽
  - 订单/商品槽位替换
  - top-level context 注入

默认配置文件在：

- [synthesis_templates.default.json](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/synthesis_templates.default.json)

## 4. 配置文件怎么理解

默认配置里每个 `scenario` 都对应一类你想补的数据。

一个最小例子：

```json
{
  "name": "handoff_manual_support",
  "entity_source": "none",
  "counts": {
    "train": 24,
    "dev": 8
  },
  "query_templates": [
    "这个问题我要人工处理",
    "请直接转人工客服"
  ]
}
```

意思是：

- 这是一个 `handoff` 类别
- 不需要订单或商品槽位
- 生成 24 条 train，8 条 dev
- 问法从这些模板里随机采样

再看一个多轮例子：

```json
{
  "name": "ask_order_id_logistics",
  "entity_source": "order",
  "counts": {
    "train": 16,
    "dev": 6
  },
  "turn_templates": [
    ["帮我看下快递到哪了", "{order_id}"]
  ]
}
```

意思是：

- 第一轮先问一个缺订单号的问题
- 第二轮用户再补 `order_id`
- 这类数据专门用于训练 `ask_user`

### 4.1 单轮和多轮到底是怎么区分的

这个项目离线合成 seed 时，单轮和多轮不是后处理自动扩出来的，而是配置本身决定的。

如果一个 `scenario` 里使用：

- `query_templates`

那么生成器会产出单轮 seed，例如：

```json
{
  "shop_id": "demo-shop",
  "query": "A1004 的记忆枕是什么材质，能退吗"
}
```

如果一个 `scenario` 里使用：

- `turn_templates`

那么生成器会产出多轮 seed，例如：

```json
{
  "shop_id": "demo-shop",
  "turns": [
    {"query": "物流进度帮我查一下"},
    {"query": "A1002"}
  ]
}
```

所以最短记忆法就是：

- `query_templates` -> 单轮
- `turn_templates` -> 多轮
- 单轮输出 `query`
- 多轮输出 `turns`

### 4.2 第二轮“用户补信息”是怎么来的

很多人第一次看到多轮数据时会困惑：

- 这不是离线造数吗
- 为什么会出现“用户第二轮补回答”

答案是：

- 第二轮不是运行时随机编出来的
- 而是模板里一开始就写好了

例如：

```json
["帮我看下快递到哪了", "{order_id}"]
```

这就等于在离线数据里提前定义：

1. 第一轮用户先问一个缺 `order_id` 的问题
2. 第二轮用户再补上真正的 `order_id`

生成器只是在做：

- 从知识库里抽一个真实订单
- 把 `{order_id}` 替换成比如 `A1002`

最终得到：

```json
{
  "shop_id": "demo-shop",
  "turns": [
    {"query": "帮我看下快递到哪了"},
    {"query": "A1002"}
  ]
}
```

### 4.3 `missing_slots` 在 teacher trace 里是怎么起作用的

规则系统在跑 episode 时，如果发现缺关键参数，并不会自己瞎猜。

它会先输出：

- `need_clarification = true`
- `missing_slots = ["order_id"]` 或 `["product_id"]`

然后把未完成任务挂到当前 `EpisodeState.current_task` 上。

等到下一轮用户输入进来后，编排器才会：

1. 检查当前 task 是否还在 `pending_clarification`
2. 从新输入里抽 `order_id` / `product_id`
3. 补回之前挂起的 `planned_arguments`
4. 继续原来的工具调用

所以这里真正发生的不是：

- 系统自动脑补缺失槽位

而是：

- 系统先记住“缺什么”
- 用户下一轮真的提供后再补进去

## 5. `entity_source` 是什么

它表示模板里要从哪种实体池里取槽位。

当前支持：

- `none`
- `order`
- `product`

### 5.1 `none`

不需要任何实体，适合：

- `handoff`
- `direct`

### 5.2 `order`

从订单池里取：

- `order_id`
- `product_name`
- `product_short_name`
- 以及与订单关联的商品信息

适合：

- logistics
- order_status
- order_product_lookup
- order_policy_lookup
- order_product_policy_lookup

### 5.3 `product`

从商品池里取：

- `product_id`
- `product_name`
- `product_short_name`

适合：

- product_lookup
- 商品多轮补槽

## 6. `top_level_fields` 是什么

这个字段用来生成“query 里没写明，但上下文里已经有”的样本。

例如：

```json
{
  "entity_source": "order",
  "top_level_fields": ["order_id"],
  "query_templates": ["帮我查下运单号"]
}
```

生成出来的 episode 会像这样：

```json
{
  "shop_id": "demo-shop",
  "order_id": "A1003",
  "query": "帮我查下运单号"
}
```

这类数据的价值是：

- 训练模型利用会话上下文
- 不只会从 query 文本里生硬抽参数

## 7. 推荐工作流

如果你想把整条链路一键跑完，项目里还提供了一个脚本：

```bash
bash training/build_synthetic_datasets.sh
```

它会自动做这些事：

1. 生成 synthetic train/dev seed
2. 和人工 seed 合并成 mixed seed
3. 跑 teacher trace
4. 导出 router / answer 的 generated JSONL
5. 导出 router / answer 的 LLaMA-Factory JSON

如果你还在学习阶段，我建议你先手动跑下面几步，再用这个脚本。

### 7.1 先生成 synthetic seed

```bash
ecom-cs-agent synthesize-episodes \
  --config training/datasets/synthesis_templates.default.json \
  --output-train training/datasets/episode_cases.synthetic.train.jsonl \
  --output-dev training/datasets/episode_cases.synthetic.dev.jsonl
```

### 7.2 再和人工 seed 合并

```bash
cat training/datasets/episode_cases.train.seed.jsonl \
  training/datasets/episode_cases.synthetic.train.jsonl \
  > training/datasets/episode_cases.train.mixed.jsonl

cat training/datasets/episode_cases.dev.seed.jsonl \
  training/datasets/episode_cases.synthetic.dev.jsonl \
  > training/datasets/episode_cases.dev.mixed.jsonl
```

### 7.3 再跑 teacher trace

```bash
ecom-cs-agent run \
  --input training/datasets/episode_cases.train.mixed.jsonl \
  --output training/datasets/episode_traces.train.mixed.generated.jsonl

ecom-cs-agent run \
  --input training/datasets/episode_cases.dev.mixed.jsonl \
  --output training/datasets/episode_traces.dev.mixed.generated.jsonl
```

### 7.4 导出 router / answer 训练数据

```bash
ecom-cs-agent export-router-lf \
  --input training/datasets/episode_traces.train.mixed.generated.jsonl \
  --output training/datasets/router_sft.train.mixed.lf.json \
  --dataset-name ecom_cs_router_sft_train_mixed \
  --dataset-info training/datasets/dataset_info.ecom_cs_router_sft.json

ecom-cs-agent export-answer-lf \
  --input training/datasets/episode_traces.train.mixed.generated.jsonl \
  --output training/datasets/answer_sft.train.mixed.lf.json \
  --dataset-name ecom_cs_answer_sft_train_mixed \
  --dataset-info training/datasets/dataset_info.ecom_cs_answer_sft.json
```

## 8. 这个流水线最适合补什么

当前最建议优先补这 4 类：

1. `handoff`
2. `ask_user`
3. `order_product_policy_lookup`
4. `order_policy_lookup`

因为这几类在你当前 benchmark 里最容易出问题。

## 9. 工业界通常怎么做

虽然实现方式不同，但大体都类似：

1. 先定义 schema
2. 用模板、规则或大模型生成输入
3. 用规则 teacher、强模型 teacher 或人工标注 teacher 打标签
4. 做过滤、去重、平衡采样
5. 训练
6. benchmark
7. 再按失败类型补数据

你现在这套流水线，属于很典型的：

- programmatic supervision
- rule teacher
- benchmark-driven iteration

这是一条非常正常、也很实用的工程路线。

## 10. 你现在该怎么学

如果你是小白，最推荐按这个顺序理解：

1. 先理解 `episode seed` 是“题目”
2. 再理解 `trace` 是“老师解题过程”
3. 再理解 `router/answer` 是“从老师过程拆出来的训练样本”
4. 最后理解 `synthetic pipeline` 只是“批量生成更多题目”的机制

不要一开始就把注意力放在“大模型会不会自动造神奇数据”上。

对当前项目，更重要的是：

- 先把数据分布补对
- 再让 teacher 稳定产出标签
- 最后再考虑更复杂的 LLM paraphrase 或 self-play
