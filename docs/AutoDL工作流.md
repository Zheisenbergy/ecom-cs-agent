# AutoDL Workflow

## 目标

这条链路对应：

- `AutoDL + LLaMA-Factory` 训练
- `AutoDL + vLLM(OpenAI-compatible)` 推理
- 本项目 CLI 跑 benchmark

不依赖 `Ollama`。

## 1. 训练前准备

在 AutoDL 机器上准备：

```bash
cd /root/autodl-tmp
git clone <your-repo> ecom-cs-agent
cd ecom-cs-agent

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

再安装训练和推理依赖：

```bash
uv pip install "llamafactory[torch,metrics]" vllm
```

## 2. 准备 LLaMA-Factory 数据

项目已经提供脚本：

```bash
bash training/autodl/prepare_llamafactory_data.sh
```

这条命令默认准备的是第一轮 `seed` 数据，也就是：

- `episode_traces.train.seed.generated.jsonl`
- `episode_traces.dev.seed.generated.jsonl`

脚本会生成：

- `training/autodl/lf_data/router`
- `training/autodl/lf_data/answer`

每个目录里都有：

- `dataset_info.json`
- `*.train.lf.json`
- `*.dev.lf.json`

如果你已经做了第二轮数据扩充，想直接训练 `mixed` 数据，可以这样：

```bash
bash training/build_synthetic_datasets.sh
DATA_VARIANT=mixed bash training/autodl/prepare_llamafactory_data.sh
```

这时会额外生成：

- `training/autodl/lf_data/router_mixed`
- `training/autodl/lf_data/answer_mixed`

## 3. 训练 Router LoRA

```bash
BASE_MODEL=Qwen/Qwen3-1.7B \
OUTPUT_DIR=/root/autodl-tmp/outputs/router-qwen3-1.7b-lora \
bash training/autodl/train_router_lora.sh
```

如果你要训练第二轮 `mixed` router：

```bash
DATA_VARIANT=mixed \
BASE_MODEL=Qwen/Qwen3-1.7B \
OUTPUT_DIR=/root/autodl-tmp/outputs/router-qwen3-1.7b-lora-mixed \
bash training/autodl/train_router_lora.sh
```

常用可调参数：

- `PER_DEVICE_TRAIN_BATCH_SIZE`
- `GRADIENT_ACCUMULATION_STEPS`
- `LEARNING_RATE`
- `NUM_TRAIN_EPOCHS`
- `CUTOFF_LEN`

## 4. 训练 Answer LoRA

```bash
BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
OUTPUT_DIR=/root/autodl-tmp/outputs/answer-qwen3-4b-lora \
bash training/autodl/train_answer_lora.sh
```

如果你要训练第二轮 `mixed` answer：

```bash
DATA_VARIANT=mixed \
BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
OUTPUT_DIR=/root/autodl-tmp/outputs/answer-qwen3-4b-lora-mixed \
bash training/autodl/train_answer_lora.sh
```

## 5. 用 vLLM 挂载 LoRA

按 vLLM 官方 LoRA 服务方式，`model` 请求参数可以直接写 LoRA alias。

例如先起 router：

```bash
BASE_MODEL=Qwen/Qwen3-1.7B \
ADAPTER_ALIAS=router-lora \
ADAPTER_PATH=/root/autodl-tmp/outputs/router-qwen3-1.7b-lora \
API_KEY=EMPTY \
bash training/autodl/serve_vllm_lora.sh
```

或者起 answer：

```bash
BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
ADAPTER_ALIAS=answer-lora \
ADAPTER_PATH=/root/autodl-tmp/outputs/answer-qwen3-4b-lora \
API_KEY=EMPTY \
bash training/autodl/serve_vllm_lora.sh
```

## 6. 跑 benchmark

Router benchmark：

```bash
MODEL_NAME=router-lora \
BASE_URL=http://127.0.0.1:8000/v1 \
API_KEY=EMPTY \
bash training/autodl/run_router_benchmark.sh
```

如果你要评测第二轮 `mixed` dev：

```bash
BENCHMARK_VARIANT=mixed \
MODEL_NAME=router-lora \
BASE_URL=http://127.0.0.1:8000/v1 \
API_KEY=EMPTY \
bash training/autodl/run_router_benchmark.sh
```

如果你想进一步验证泛化，而不是只看当前 mixed dev，可以直接跑项目里新增的 `holdout` 集：

```bash
BENCHMARK_VARIANT=holdout \
MODEL_NAME=router-lora \
BASE_URL=http://127.0.0.1:8000/v1 \
API_KEY=EMPTY \
bash training/autodl/run_router_benchmark.sh
```

这套 `holdout` 的目标不是复读训练模板，而是专门测：

- 新说法下的 `ask_user`
- 新说法下的 `handoff`
- 同一句模糊问法在不同 `state_before` 下是否还能做对
- `route / intent / tool / missing_slots` 是否还能保持稳定

Answer benchmark：

```bash
MODEL_NAME=answer-lora \
BASE_URL=http://127.0.0.1:8000/v1 \
API_KEY=EMPTY \
bash training/autodl/run_answer_benchmark.sh
```

同理，`answer` 也支持：

```bash
BENCHMARK_VARIANT=mixed \
MODEL_NAME=answer-lora \
BASE_URL=http://127.0.0.1:8000/v1 \
API_KEY=EMPTY \
bash training/autodl/run_answer_benchmark.sh
```

## 7. 推荐顺序

1. 先保留当前规则系统 benchmark 结果
2. 跑未后训练 Qwen baseline
3. 先训 `router-lora`
4. 跑 router benchmark
5. 再训 `answer-lora`
6. 跑 answer benchmark
7. 最后再做运行时联调

当前推荐的模型分工：

- `router`：`Qwen/Qwen3-1.7B`
- `answer`：`Qwen/Qwen3-4B-Instruct-2507`

这样做的原因是：

- router 更偏结构化决策，小模型更省显存和推理成本
- answer 更偏生成和收口，4B 更稳

## 7.1 模型版端到端联调

如果你已经分别把 `router-lora` 和 `answer-lora` 挂成 OpenAI-compatible 服务，可以直接用项目 CLI 跑完整链路：

```bash
ecom-cs-agent ask-model "A1001 到哪了" \
  --router-model router-lora \
  --answer-model answer-lora \
  --router-base-url http://127.0.0.1:8000/v1 \
  --answer-base-url http://127.0.0.1:8001/v1
```

看完整结构化轨迹：

```bash
ecom-cs-agent trace-model "这单里的商品怎么洗，还能退吗" \
  --router-model router-lora \
  --answer-model answer-lora \
  --router-base-url http://127.0.0.1:8000/v1 \
  --answer-base-url http://127.0.0.1:8001/v1
```

做多轮补槽位联调：

```bash
ecom-cs-agent chat-model \
  --router-model router-lora \
  --answer-model answer-lora \
  --router-base-url http://127.0.0.1:8000/v1 \
  --answer-base-url http://127.0.0.1:8001/v1
```

这一套命令复用了训练和 benchmark 的同一套 prompt 规则：

- router 仍然看 `user_query + state_before`
- answer 仍然看 `query + route + intent + tool_steps`
- router 输出仍会经过同一层轻量 JSON 归一化

这样联调时看到的问题，会更接近真实训练分布，而不是另起一套临时 prompt。

## 7.2 单卡注意事项

如果你只有一张 24G 左右的卡，通常不要默认认为可以同时稳定挂：

- `router-lora`
- `answer-lora`

更稳妥的办法是：

1. 把它们挂在不同机器或不同 GPU
2. 或者先单独验证 router，再单独验证 answer
3. 如果必须单卡联调，优先尝试更小的 batch / 更保守的 vLLM 配置，并提前用 `nvidia-smi` 看显存

## 8. 第二轮最推荐怎么跑

如果你现在已经有第一轮训练结果，又想继续提升 router，我建议你在 AutoDL 上按这个顺序做：

```bash
cd /root/autodl-tmp/ecom-cs-agent
git pull
source .venv/bin/activate

bash training/build_synthetic_datasets.sh

DATA_VARIANT=mixed \
BASE_MODEL=Qwen/Qwen3-1.7B \
OUTPUT_DIR=/root/autodl-tmp/outputs/router-qwen3-1.7b-lora-mixed \
bash training/autodl/train_router_lora.sh
```

训练完后：

1. 用 `vLLM` 挂载新的 `router-lora`
2. 跑 `BENCHMARK_VARIANT=mixed` 的 router benchmark
3. 先看 `route_macro_f1 / handoff_f1 / ask_user_f1` 有没有明显提升
4. router 稳了，再决定要不要继续训 answer

## 9. 当前项目里的关键文件

- Router benchmark 输入：[router_sft.dev.generated.jsonl](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/router_sft.dev.generated.jsonl)
- Router holdout benchmark 输入：[router_sft.dev.holdout.jsonl](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/router_sft.dev.holdout.jsonl)
- Answer benchmark 输入：[answer_sft.dev.generated.jsonl](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/answer_sft.dev.generated.jsonl)
- Router LLaMA-Factory train：[router_sft.train.lf.json](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/router_sft.train.lf.json)
- Router LLaMA-Factory dev：[router_sft.dev.lf.json](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/router_sft.dev.lf.json)
- Answer LLaMA-Factory train：[answer_sft.train.lf.json](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/answer_sft.train.lf.json)
- Answer LLaMA-Factory dev：[answer_sft.dev.lf.json](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/answer_sft.dev.lf.json)

如果你想看目前已经实际做过哪些训练、指标怎么变化、踩过哪些坑，可以继续看：

- [实验记录与踩坑总结](/Users/zheisenbergy/code/agent/ecom-cs-agent/docs/实验记录.md)
