# 系统架构

## 1. 设计结论

当前项目采用的不是“长期会话持久化客服”，而是“单任务 episode + 内部 agent loop”。

原因很简单：

- 你要训练的是 agentic 能力，不是聊天产品能力
- 你要评测的是工具调用闭环，不是跨天记忆
- 你要的数据单位应该是完整任务，而不是零散单轮

## 2. 核心流程

```text
用户问题
-> 构造 episode 初始状态
-> 判断 direct / internal_tool / handoff
-> 如果缺参数，则 ask_user 并等待
-> 如果参数齐全，则调用工具
-> 读取 tool observation
-> 生成 grounded answer
-> episode 结束
```

如果用户补充了订单号或商品名，则仍然属于同一个 episode 的后续 turn，而不是长期会话记忆。

## 3. 为什么不用文件级会话存储

文件级会话持久化不是这个项目的核心能力，反而会把注意力带偏到产品工程。

这里更合理的设计是：

- `chat` 命令在进程内维护当前未完成 episode 的状态
- `ask` 命令默认只执行当前 turn，不跨进程延续
- `run` 命令直接读取完整 episode 样本

这样更符合训练与评测目标。

## 4. 模块划分

### CLI

- 接收用户输入
- 管理当前进程内的 episode 状态
- 输出单步结果或批量导出结果

### Router

- 判断 `direct / internal_tool / handoff`
- 抽取 `order_id`、`product_id`
- 判断是否缺关键槽位
- 如果当前 episode 有 `current_task`，则继续未完成任务

### Internal Tools

- 商品信息
- 店铺政策
- 订单状态
- 物流状态

### Answer

- 基于 tool result 输出回答
- 缺参时输出 ask_user
- 高风险问题输出 handoff

### Orchestrator

- 合并 request 与 episode state
- 执行 route
- 在需要时执行受控多工具链
- 更新 episode state
- 产出 turn trace 和 episode record

## 5. 当前状态结构

当前 `EpisodeState` 至少包含：

- `shop_id`
- `product_id`
- `order_id`
- `current_task`
- `recent_queries`
- `turn_index`

其中 `current_task` 里会进一步保存：

- `task_query`
- `intent`
- `route`
- `planned_tool_name`
- `planned_arguments`
- `missing_slots`
- `resolved_slots`
- `status`

这不是长期记忆，而是单任务运行时状态。

## 5.1 为什么不用“整段历史直接重推”

当前项目不是把整段 transcript 每轮重新喂给系统，而是：

- 保留必要的最近输入
- 把任务语义压缩进 `current_task`
- 让后续工具链和回答都依赖 `task_query`

这样做的原因是：

- 更容易评测
- 更容易导出训练数据
- 更容易定位是 route、tool 还是 answer 出错

所以当前项目更像：

- 结构化任务状态驱动

而不是：

- 自由聊天历史驱动

## 6. 受控多工具链

当前系统已经支持一类重要的 agentic 闭环：

1. 先查订单
2. 从订单 observation 中拿到 `product_id`
3. 再查商品信息
4. 再查退货或退款规则
5. 合成最终答案

示例问题：

- `我订单 A1001 买的那件衣服是什么材质，能退吗？`

这仍然不是 RAG，而是结构化工具串联。

## 7. 一条完整数据怎么定义

一条完整数据不是一句 query，而是一个完整 episode。

完整 episode 的结束条件通常是：

- 输出 `final answer`
- 输出 `handoff`
- 输出 `ask_user` 并在补齐信息后最终结束

## 8. 为什么这个设计更适合后训练

因为你真正要训练的能力是：

- route decision
- tool selection
- argument filling
- clarification
- grounded answering

这些能力天然发生在一个任务闭环里，而不是一个无限延长的聊天记录里。

## 9. 什么时候开始接入模型

不建议一开始就把 `Router` 和 `Answer` 一次性替换成 Qwen 模型。

更合理的顺序是：

1. 先用规则系统把 episode 数据流、工具流、评测流跑通
2. 再冻结一版 tool schema、trace schema、eval schema
3. 先替换路由与动作决策模块
4. 路由模型稳定后，再替换最终回答模块

原因：

- 路由错误会直接导致工具链全错
- 如果一开始同时替换 route 和 answer，就很难判断问题出在“决策”还是“表达”
- 先单独替换 router，更容易观察模型是否学会 `ask_user / handoff / tool_call`

## 10. 建议的模型插入点

### 10.1 Router 模型

负责输出：

- `route`
- `intent`
- `tool_name`
- `tool_arguments`
- `missing_slots`

推荐先用较小模型：

- `Qwen3-1.7B`
- 或 `Qwen3-4B`

### 10.2 Answer 模型

负责基于 observation 输出：

- grounded final answer
- ask_user 风格回答
- handoff 风格回答

这里建议主力模型直接用：

- `Qwen3-4B`

## 11. 不要一上来训练什么

第一阶段不建议训练：

- 商品知识记忆
- 店铺知识记忆
- 长期会话记忆
- 自由规划几十步工具链

第一阶段只训练：

- 单任务 episode 闭环
- 受控工具调用
- observation 到 grounded answer 的映射
