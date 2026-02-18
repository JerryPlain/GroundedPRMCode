# GroundedPRM 端到端示例（单题演示，含打分机制）

这份文档用一道题，从输入到训练再到评测完整走一遍，并明确回答 3 个核心问题：

1. 为什么要抽条件构造 root？
2. WA 是什么，起什么作用？
3. 步骤级标签（True/False、+/-）到底怎么来的？

## 1. 原始输入题目

```json
{
  "problem": "Find the largest value of c such that -2 is in the range of f(x)=x^2+3x+c.",
  "solution": "...最终答案是 \\boxed{\\frac14}",
  "type": "algebra",
  "level": "Level 2"
}
```

注意：`solution` 是题目级 GT（最终答案），不是每一步的人工 GT。

## 2. 为什么先构造 root（重点）

脚本：`pipeline/root_generation.py`

原题是自然语言，MCTS 直接在原题上搜索会很散。先构造 root 是为了把问题转成“可搜索状态”：

1. `conditions`：当前已知条件集合。
2. `global_objective`：最终要求解的目标。

这样后续每一步都能基于统一状态推进与验证，不容易跑偏。

示例 root（`outputs/root/root.json`）：

```json
{
  "question": "Find the largest value of c such that -2 is in the range of f(x)=x^2+3x+c.",
  "global_objective": "Find the largest value of c",
  "conditions": [
    "f(x)=x^2+3x+c",
    "-2 is in the range of f(x)"
  ],
  "path": "algebra_0001.json",
  "level": 2,
  "solution": "...\\boxed{\\frac14}"
}
```

## 3. 阶段B：MCTS 搜索怎么跑

脚本链路：`pipeline/data_generation.py` -> `pipeline/MCTS/mcts.py`

单轮大致是：
1. `Selection`：选一个节点继续探索。
2. `Expansion`：生成候选下一步（`step objective + action`）。
3. `Simulation`：rollout 若干步。
4. `Backpropagation`：把本轮回报回传更新节点价值。

示例节点（`mcts_tree` 局部）：

```json
{
  "value": 0.73,
  "state": {
    "step_objective": "把-2在值域中转化为二次方程有实根条件",
    "step_ans": "令x^2+3x+c=-2，得x^2+3x+(c+2)=0，其判别式为\\boxed{\\Delta=1-4c}"
  },
  "reflection": "[True] ...该步推导与WA结果一致"
}
```

## 4. WA 是什么（重点）

`WA = Wolfram Alpha`。在项目里它是外部数学工具，用来给步骤提供可验证参考，例如：

1. 解方程。
2. 化简表达式。
3. 返回数值/范围结果。

代码里会先为当前步骤生成 WA query，再拿 WA 返回结果参与步骤评审。

## 5. 步骤标签怎么打分（重点）

“每一步有标签”不等于“每一步有人工 GT”。

真实流程是自动打分：

1. 当前步骤会得到四类输入：
   - 当前 step objective
   - 当前 conditions
   - LLM 生成的 step_ans
   - WA 返回结果
2. `llm_verify`（见 `pipeline/MCTS/mcts_utils.py`）会输出 JSON 判定：
   - `result = True` 或 `False`
   - `reason = 解释`
3. 在 `step_evaluate`（`pipeline/MCTS/mcts.py`）映射为分数：
   - `True -> +1.0`
   - `False -> -1.0`
4. 这个分数就是步骤级监督信号，后续在 `construct.py` 再转成训练标签：
   - 正分 -> `+`
   - 负分 -> `-`

所以步骤标签是“验证器自动产生的弱监督标签”，不是人工逐步标注。

## 6. 最终 GT 在哪里用

题目级 GT（`solution`）主要在终局使用：

1. 从最后一步提取 `\boxed{}` 结果。
2. 与 `solution` 的 boxed 结果做数学等价校验（`math_verify`）。
3. 得到终局分数（正/负），并参与回传。

即：
- 步骤标签来自验证器。
- 最终正确性来自题目级 GT 校验。

## 7. 阶段C：从树提取训练样本

脚本：`data_process/extract_path.py`

会抽取终止路径和补充路径，组装为：

```json
{
  "idx": "algebra_0001",
  "question": "Find the largest value of c ...",
  "ground_truth": "1/4",
  "label": "positive",
  "steps": [
    {
      "content": "...\\boxed{\\Delta=1-4c}",
      "step score": 1,
      "reflection": "[True] ...",
      "correction": {"Input": "Solve[...]", "final_answer": "..."}
    }
  ]
}
```

## 8. 阶段D：构造成 SFT 对话数据

脚本：`data_process/construct.py`

处理逻辑：
1. 过滤 score/reflection 不一致样本。
2. 截断部分负样本尾部。
3. 组装 assistant 输出格式 `<verify>/<judge>/<output>`。

示例训练条目：

```json
{
  "conversation": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "题目+当前步骤"},
    {"role": "assistant", "content": "<verify>...</verify><judge>...</judge><output>\\boxed{+}</output>"}
  ]
}
```

## 9. 阶段E：LoRA 微调

```bash
llamafactory-cli train \
  --stage sft \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset your_dataset \
  --template qwen \
  --finetuning_type lora \
  --output_dir models/GroundedPRM
```

训练后可导出 merged 权重用于 vLLM 部署。

## 10. 阶段F：评测

1. `evaluation/ProcessBench_eval.py`：步骤级评测。
2. `evaluation/reward_guided_search/Greedy-Search.py`：
   - 策略模型提候选步骤；
   - PRM 打分；
   - 选最高分继续，直到终止。

## 11. 一句话复述（面试可用）

先把题目结构化成 root，再用 MCTS 生成多路径推理；每一步通过 WA+评审模型自动打标签（非人工 step GT），最终用题目级 GT 做终局校验；再把轨迹转成对话监督数据做 LoRA 微调，并在过程评测与奖励引导搜索中验证收益。

