# GroundedPRM 项目详细解析

## 1. 项目定位与目标
GroundedPRM 是一个面向数学多步推理的过程奖励模型（PRM）数据构建与评测工程。它的核心目标不是直接解题，而是给“中间推理步骤”打分，并把这个打分用于：

1. 构建高质量过程监督数据。
2. 训练能判别步骤正确性的奖励模型。
3. 在推理时通过奖励引导搜索（Greedy Search / Tree Search）提升最终解题正确率。

该仓库强调两个思想：

1. `Tree-Guided`：通过 MCTS 生成多分支推理路径，避免单一路径监督。
2. `Fidelity-Aware`：引入外部工具（Wolfram Alpha）和二次 LLM 评审，给每一步附上“可解释”判定依据。

## 2. 代码目录与职责

### 2.1 训练数据生成主链路
- `pipeline/root_generation.py`
- `pipeline/data_generation.py`
- `pipeline/MCTS/mcts.py`
- `pipeline/MCTS/mcts_utils.py`
- `pipeline/utils/output_utils.py`
- `pipeline/utils/math_utils.py`
- `pipeline/prompts/instructions.py`

职责拆分：

1. `root_generation.py`：从原始数学题 JSON 中抽取 `conditions + global_objective`，形成根状态文件（如 `root.json`）。
2. `data_generation.py`：并发调度每道题的 MCTS 生成。
3. `mcts.py`：实现选择、扩展、模拟、回传、样本保存、树结构保存。
4. `mcts_utils.py`：`State` 数据结构、步骤验证、WA 查询相关操作。
5. `output_utils.py`：模型输出解析、JSON 修复、WA 响应清洗等工具函数。
6. `math_utils.py`：`\boxed{}` 抽取与数学等价校验（`math_verify`）。
7. `instructions.py`：各阶段 prompt 模板（抽取、下一步生成、WA query 生成、步骤校验）。

### 2.2 数据后处理链路
- `data_process/extract_path.py`
- `data_process/fixed_answer.py`
- `data_process/construct.py`
- `data_process/sampling.py`
- `data_process/prm800k/filter_prm800k.py`
- `data_process/training_instruction.py`

职责拆分：

1. `extract_path.py`：从保存的 MCTS 树里抽取终止路径/深度路径并组装样本。
2. `fixed_answer.py`：过滤无 `\boxed` 结果、清理异常 WA 反馈样本。
3. `construct.py`：
   - 反射一致性过滤（step score 与 reflection 一致）；
   - 负样本截断；
   - 分数离散化（±1）；
   - 组装 `<verify>/<judge>/<output>` 响应；
   - 转成 conversation 训练格式。
4. `sampling.py`：随机采样训练文件。
5. `filter_prm800k.py`：PRM800K 数据转换和采样对比。

### 2.3 评测链路
- `evaluation/ProcessBench_eval.py`
- `evaluation/reward_guided_search/GroundedPRM.py`
- `evaluation/reward_guided_search/Greedy-Search.py`
- `evaluation/reward_guided_search/eval_results.py`
- `evaluation/reward_guided_search/deploy_models.sh`
- `evaluation/reward_guided_search/reward_guided_search_eval_api.sh`

职责拆分：

1. `ProcessBench_eval.py`：逐步判别型评测，预测首次错误步骤位置。
2. `GroundedPRM.py`：通过 API `logprobs` 提取 `Right/Wrong` 概率作为 reward。
3. `Greedy-Search.py`：策略模型采样候选步骤 + PRM 评分 + 选最高 reward 步骤。
4. `eval_results.py`：统一读取搜索输出并按数据集算正确率。

## 3. 端到端数据流（从题目到训练样本）

### 阶段 A：根节点构建
输入：原始数学题 JSON（字段通常含 `problem/solution/level/type`）

流程：
1. 对每题调用 `extract_instruction + extract_prompt`。
2. 解析得到 `conditions`、`global_objective`。
3. 与原题元数据一起保存到 `root.json`。

输出：每题一个 root task，供 MCTS 初始化。

### 阶段 B：MCTS 路径搜索与打分
`data_generation.py` 会根据题目难度调 `execute_round/simulation_depth`，并行跑题。

每轮 MCTS 的核心逻辑：
1. `select`：在已扩展节点中选可探索分支。
2. `expand`：调用策略模型生成多个下一步（`step objective + action`）。
3. `simulate`：继续 rollout 若干步。
4. `step_evaluate`：用 `llm_verify` 对当前步骤做真伪判断（依赖 WA 结果与另一个评审模型）。
5. `evaluate_final_state`：最终答案与 ground truth 比对，得到 ±1。
6. `backpropagate`：将 rollout 聚合值回传到祖先。

保存结果：
1. `mcts_tree/<task>.json`：完整树（含真实分支和模拟分支）。
2. `training_samples/<task>.json`：正负路径样本池。

### 阶段 C：从树提取训练样本
`extract_path.py` 会：
1. 找到终止路径（命中 `<end>` 或答案匹配）。
2. 再抽取平均深度附近路径补充多样性。
3. 生成每题多条序列样本：`question + steps + label + reflection + correction`。

### 阶段 D：训练数据构建
`construct.py` 会把样本转成“多轮对话监督”：
1. 每一步都有 user 输入（步骤文本）与 assistant 输出（`<verify>/<judge>/<output>`）。
2. 输出可直接用于 SFT/LoRA 训练的 JSONL conversation 数据。

## 4. 关键算法与实现要点

### 4.1 状态表示（State）
`State` 中包含：
- 全局目标 `global_objective`
- 条件集合 `conditions`
- 历史步骤 `steps`
- 当前步骤目标/答案 `step_objective`, `step_ans`
- 步骤结果 `result`
- WA 查询与历史 `query`, `wa_history`

该结构把“推理文本”和“可验证中间证据”绑定在一个节点上，是该项目可解释监督的基础。

### 4.2 奖励构成
每条 rollout 的奖励由两部分合成：
1. 步骤级评分：`llm_verify` 返回 True/False -> 映射到 ±1。
2. 最终答案评分：与 GT 数学等价验证 -> ±1。

然后在 `backpropagate` 中进行时间衰减累计，更新祖先价值。

### 4.3 终止判定
节点会在以下情况下终止：
1. 步骤文本含 `<end>`。
2. 当前步骤 `\boxed` 结果与 GT 等价。
3. 与父节点答案重复（防止循环）。

### 4.4 奖励引导 Greedy Search
评测中的搜索不是重新跑 MCTS，而是：
1. 策略模型一次采样多个候选步骤。
2. PRM 给每个候选打分。
3. 选 reward 最高者进入下一轮，直到产生终止答案。

## 5. 配置与依赖

### 5.1 Python 依赖
见 `requirements.txt`，核心依赖包括：
- `openai`（兼容 vLLM OpenAI API）
- `datasets`、`transformers`
- `math-verify`
- `camel`（用于 WA 工具查询）

### 5.2 配置文件
`config.example.py` 提供：
- Qwen / DeepSeek API 地址与模型名
- 并发数
- 默认 MCTS 参数
- WA 重试和数学比较容忍度

## 6. 当前仓库可见问题与风险（基于代码现状）

1. 缺失模块风险：仓库内未看到 `pipeline/models/model.py`，但多处 `from models.model import ...`。
2. 缺失 prompt 文件：`Greedy-Search.py` 依赖 `prompts/policy_prompt.py`，仓库未见该文件。
3. 路径拼接疑点：`data_generation.py` 的 `get_project_root()` 可能多回退一级，易导致根路径错位。
4. `mcts.py` 的 `select` 使用 `node.children` 而非 `current.children`，树搜索策略可能异常偏置。
5. `mcts.py` 中 `update_conditions(state, step_ans)` 与函数签名（期望 list）不一致，可能触发类型错误。
6. `extract_path.py` 的正负标签判定使用 `is_math_correct(result, gt_answer)`，其中 `gt_answer` 未规范成 boxed 结果，可能降低标注准确性。
7. `construct.py` 里输出路径拼接 `data/train/ + file_path`，会生成包含上游目录名的非常规文件路径。
8. 多脚本默认依赖外部数据目录（`math/train`、`eval_data/*`、`data/meta/*`），仓库本体不包含这些数据，开箱即跑会失败。

## 7. 建议的最小可运行闭环

1. 补齐并统一 API 适配层：新增 `pipeline/models/model.py`，封装 `request_qwen/request_deepseek`。
2. 补齐 `evaluation/reward_guided_search/prompts/policy_prompt.py`。
3. 修复 `mcts.py` 的 `select/update_conditions` 两处实现问题。
4. 准备一个小型样本集（10~50 题）验证端到端流程：
   - `root_generation.py`
   - `data_generation.py`
   - `extract_path.py`
   - `construct.py`
5. 用 `ProcessBench_eval.py` 和 `Greedy-Search.py` 各做一次 smoke test，确认训练后模型输出格式稳定包含标签。

## 8. 结论
这个仓库体现了“树搜索生成过程监督 + 外部验证增强奖励可信度”的完整思路，设计上有较强研究价值。工程上已经具备从数据构建到评测的主干，但目前存在若干缺失文件和实现细节问题。补齐缺失模块并修复关键函数后，可形成可复现实验管线。