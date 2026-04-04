# 开发路线

## 阶段 1：已完成

- 项目骨架
- CLI 交互
- route / tool / answer 主链路
- mock 工具环境
- episode 状态与 episode trace
- 受控多工具链
- 中文文档

## 阶段 2：接下来优先做

- 增加更丰富的 episode 样本
- 增加更多多工具链 episode
- 增加更难的歧义与异常 episode 评测集
- 增加人工修订与错误归因工具

阶段 2 结束标志：

- 有一版稳定的 router / answer / chain 训练集
- 有一版稳定的 dev eval 集
- tool schema 与 trace schema 基本冻结
- 可以开始第一轮 Qwen LoRA SFT

当前已经完成到：

- episode trace 稳定
- eval 稳定
- router / answer 训练导出命令已具备

## 阶段 3：模型化替换

- 路由与动作决策替换为 `Qwen3-1.7B` 或 `Qwen3-4B`
- 回答生成替换为 `Qwen3-4B`
- 接入 vLLM

阶段 3 推荐顺序：

1. 先替换 router
2. 再替换 answer
3. 最后做端到端联调

## 阶段 4：更强 agentic

- 更细粒度动作标签
- 更复杂的电商任务 episode

## 阶段 5：后续扩展

- 再评估是否引入 RAG
- 再评估是否引入 Web
- 如果引入，再补对应 route、tool schema 和 episode 数据
