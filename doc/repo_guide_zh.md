# GroundedPRM 仓库 Guide（从数据构建到 Finetune）

## 1. 仓库整体逻辑
这个仓库的主线是：

1. 从数学题中抽取结构化根状态（条件 + 全局目标）。
2. 用 MCTS 生成多步推理轨迹，并对步骤进行外部验证与奖励打分。
3. 从树中抽取正负路径，构造过程监督训练数据。
4. 用 LLaMA-Factory 做 LoRA 微调，得到 PRM 风格模型。
5. 用 ProcessBench 与 reward-guided search 做评测。

一句话：`题目 -> MCTS轨迹+步骤奖励 -> 训练集 -> LoRA微调 -> 推理评测`。

## 2. 各目录分别做什么

- `pipeline/`：数据生成核心（根状态生成 + MCTS 搜索 + 步骤验证）。
- `data_process/`：树轨迹后处理与训练数据构建。
- `evaluation/`：评测脚本（ProcessBench 与奖励引导搜索）。
- `config.example.py`：API、并发和默认参数模板。
- `requirements.txt`：环境依赖。

## 3. 端到端流程（推荐执行顺序）

## 阶段 A：准备环境与配置

### 做什么
安装依赖，配置策略/评审模型 API。

### 怎么做
```bash
conda create -n groundedprm python=3.10
conda activate groundedprm
pip install -r requirements.txt
cp config.example.py config.py
```

然后在 `config.py` 中配置：
- Qwen API（用于步骤生成等）
- DeepSeek API（用于验证与 WA query 生成等）

## 阶段 B：构建根状态数据（Root）

### 做什么
把原始数学题转成结构化任务：
- `conditions`
- `global_objective`
- 原题元信息（level、solution、path）

### 核心脚本
- `pipeline/root_generation.py`

### 怎么做
```bash
cd pipeline
python root_generation.py \
  --task_file_path math/train \
  --output_file_path outputs/root/root.json
```

### 输入/输出
- 输入：原始题目 JSON 目录（如 `math/train`）
- 输出：`pipeline/outputs/root/root.json`

## 阶段 C：MCTS 生成轨迹与奖励

### 做什么
对每个 root task 跑 MCTS：
- 扩展候选步骤
- rollout 模拟
- 用 WA + LLM 评审步骤正确性
- 与 GT 比较最终答案
- 回传价值并保存树

### 核心脚本
- 调度：`pipeline/data_generation.py`
- 搜索：`pipeline/MCTS/mcts.py`
- 状态与验证：`pipeline/MCTS/mcts_utils.py`

### 怎么做
```bash
cd pipeline
python data_generation.py \
  --outputs_dir outputs/state_trace \
  --root_dir outputs/root \
  --task_file root.json \
  --start_index 0 \
  --max_workers 40
```

### 输入/输出
- 输入：`outputs/root/root.json`
- 输出：
  - `outputs/state_trace/mcts_tree/*.json`（完整搜索树）
  - `outputs/state_trace/training_samples/*.json`（正负样本）

## 阶段 D：从树提取训练样本

### 做什么
提取终止路径和可用深度路径，形成 step-level 数据样本。

### 核心脚本
- `data_process/extract_path.py`

### 怎么做
```bash
cd data_process
python extract_path.py \
  --meta_path ../pipeline/outputs/root/root.json \
  --tree_dir ../pipeline/outputs/state_trace/mcts_tree \
  --output_json data/synthetic_data/syn_data.json
```

### 输入/输出
- 输入：`root.json + mcts_tree/*.json`
- 输出：`data/synthetic_data/syn_data.json`

## 阶段 E：清洗并构建 SFT 对话数据

### 做什么
将轨迹数据转换为训练可用格式：
- 一致性过滤（分数与反思匹配）
- 负样本截断
- 二值化标签
- 组装 `<verify> <judge> <output>`
- 转 conversation JSONL

### 核心脚本
- `data_process/fixed_answer.py`
- `data_process/construct.py`

### 怎么做
先按你的数据路径调整脚本中的默认输入输出，再执行：
```bash
cd data_process
python fixed_answer.py
python construct.py
```

常见产物：
- `data/filtered_data/fixed_data.json`
- `data/train/*.json`
- `data/train/training.json`

## 阶段 F：LLaMA-Factory LoRA 微调

### 做什么
用构建好的 conversation 数据进行 SFT（LoRA）。

### 怎么做（示例）
```bash
pip install llamafactory

llamafactory-cli train \
  --stage sft \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset your_dataset \
  --template qwen \
  --finetuning_type lora \
  --output_dir models/GroundedPRM \
  --per_device_train_batch_size 4 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --fp16
```

合并 LoRA：
```bash
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_name_or_path models/GroundedPRM \
  --export_dir models/GroundedPRM-merged
```

## 阶段 G：vLLM 部署（为什么用 + 怎么部署）

### 为什么要用 vLLM

这个仓库大量依赖“OpenAI 兼容 API 调模型”，例如：
- `pipeline/` 阶段要高并发生成步骤与验证结果。
- `evaluation/reward_guided_search/Greedy-Search.py` 每轮都要多次采样候选步骤并打分。

用 vLLM 的主要原因：

1. **吞吐高**：连续批处理和高效 KV Cache 对多请求场景更友好。
2. **接口兼容**：可直接暴露 `/v1/chat/completions`，和 `openai` SDK 对接简单。
3. **部署成本低**：单条命令即可把 HF 模型拉起为 API 服务。
4. **适合集群**：便于在 SLURM 上按 GPU 分别部署 policy/reward 两个服务。

### 部署前准备

1. 确认已安装 `vllm` 并能访问模型权重路径。
2. 确认端口未占用（默认 policy: `8000`，reward: `8001`）。
3. 确认评测脚本里的模型名与服务 `--served-model-name` 一致。

### 方案 1：单模型部署（最小示例）

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/your/model \
  --served-model-name policy-model \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key EMPTY
```

测试连通性（OpenAI 兼容）：
```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer EMPTY"
```

### 方案 2：双模型部署（policy + reward）

仓库已提供脚本：`evaluation/reward_guided_search/deploy_models.sh`。  
它会：
- 在 GPU `POLICY_GPU` 上启动 policy 模型；
- 在 GPU `REWARD_GPU` 上启动 reward 模型；
- 分别监听 `POLICY_PORT`/`REWARD_PORT`。

执行：
```bash
cd evaluation/reward_guided_search
bash deploy_models.sh
```

关键配置（脚本内可改）：
- `POLICY_DIR` / `REWARD_MODEL_DIR`
- `POLICY_MODEL_NAME` / `REWARD_MODEL_NAME`
- `POLICY_GPU` / `REWARD_GPU`
- `POLICY_PORT` / `REWARD_PORT`

### SLURM 场景建议

1. 把部署和评测分成两个作业步骤（先起服务，再跑评测）。
2. 固定节点名与端口，避免 worker 找不到 API。
3. 每个服务单独日志文件，便于定位 OOM/端口冲突。
4. `--gpu-memory-utilization` 保守起步（如 0.85~0.90）。

### 常见问题

1. **请求超时/无响应**：先确认 `curl /v1/models` 能通，再看服务日志是否 OOM。
2. **模型名不匹配**：客户端传的 `model_name` 必须等于 `--served-model-name`。
3. **吞吐低**：先降低 `max_tokens`、并发线程数，再观察 GPU 利用率。
4. **评测脚本连错地址**：检查 `reward_guided_search_eval_api.sh` 的 `node_name/port`。

## 阶段 H：评测

评测细节（原理、原因、命令、输出解释）见：`doc/evaluation_stage_guide_zh.md`

## 1) ProcessBench
```bash
python evaluation/ProcessBench_eval.py \
  --api_url http://<host>:<port>/v1 \
  --model_name <your_model>
```

## 2) Reward-guided Search
先部署策略模型与奖励模型：
```bash
cd evaluation/reward_guided_search
bash deploy_models.sh
```

再运行搜索评测：
```bash
bash reward_guided_search_eval_api.sh
```

## 4. 每个阶段的“你做了什么”可复述模板

- 数据构建：
  - “用 root extraction + MCTS 生成过程监督轨迹，并通过 WA/LLM 双重验证打步骤奖励。”
- 数据处理：
  - “从搜索树提取正负路径，清洗不一致样本，构造成对话式 PRM 训练数据。”
- 微调：
  - “在 LLaMA-Factory 中进行 LoRA SFT，并合并 adapter 用于推理部署。”
- 评测：
  - “在 ProcessBench 与 reward-guided math benchmarks 上评估 step-level 与终局表现。”

## 5. 实操注意事项

1. 仓库当前对外部数据目录有依赖（如 `math/train`、`eval_data/*`），需自行准备。
2. 多处脚本依赖 API 服务先启动（OpenAI 兼容接口）。
3. 并发参数（`max_workers`）与模型吞吐强相关，建议先小规模 smoke test。
4. 若要在 SLURM 上跑，优先将各阶段拆分为独立 job，减少失败重跑成本。
