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

脚本会生成：

- `training/autodl/lf_data/router`
- `training/autodl/lf_data/answer`

每个目录里都有：

- `dataset_info.json`
- `*.train.lf.json`
- `*.dev.lf.json`

## 3. 训练 Router LoRA

```bash
BASE_MODEL=Qwen/Qwen3-1.7B \
OUTPUT_DIR=/root/autodl-tmp/outputs/router-qwen3-1.7b-lora \
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

Answer benchmark：

```bash
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

## 8. 当前项目里的关键文件

- Router benchmark 输入：[router_sft.dev.generated.jsonl](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/router_sft.dev.generated.jsonl)
- Answer benchmark 输入：[answer_sft.dev.generated.jsonl](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/answer_sft.dev.generated.jsonl)
- Router LLaMA-Factory train：[router_sft.train.lf.json](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/router_sft.train.lf.json)
- Router LLaMA-Factory dev：[router_sft.dev.lf.json](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/router_sft.dev.lf.json)
- Answer LLaMA-Factory train：[answer_sft.train.lf.json](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/answer_sft.train.lf.json)
- Answer LLaMA-Factory dev：[answer_sft.dev.lf.json](/Users/zheisenbergy/code/agent/ecom-cs-agent/training/datasets/answer_sft.dev.lf.json)
