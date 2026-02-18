# GroundedPRM 端到端示例（单题演示）

这份文档用一题小例子，演示项目从“原始题目”到“可微调数据”和“评测”的完整路径。

## 1. 原始输入题目

假设我们有一题（示意）：

```json
{
  "problem": "Find the largest value of c such that -2 is in the range of f(x)=x^2+3x+c.",
  "solution": "...最终答案是 \\boxed{\\frac14}",
  "type": "algebra",
  "level": "Level 2"
}
```

## 2. 阶段A：Root 构建（抽条件 + 全局目标）

脚本：`pipeline/root_generation.py`

它会调用抽取 prompt，把题目转换成结构化 root state。

示例输出（`outputs/root/root.json` 中的一条）：

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

## 3. 阶段B：MCTS 生成推理树

脚本：`pipeline/data_generation.py` -> `pipeline/MCTS/mcts.py`

核心动作：
1. 生成候选下一步（step objective + action）。
2. rollout 模拟后续步骤。
3. 用 WA + LLM 评审每步正确性。
4. 与 GT 比较最终结果并回传价值。

示例树节点（`outputs/state_trace/mcts_tree/algebra_0001.json` 局部）：

```json
{
  "value": 0.73,
  "state": {
    "step_objective": "把-2在值域中转化为二次方程有实根条件",
    "step_ans": "令x^2+3x+c=-2，得x^2+3x+(c+2)=0，其判别式为\\boxed{\\Delta=1-4c}"
  },
  "reflection": "[True] ...该步推导与WA结果一致",
  "children": [
    {
      "value": 0.92,
      "state": {
        "step_objective": "由判别式非负求c的上界",
        "step_ans": "由1-4c\\ge0得c\\le1/4，所以最大值为\\boxed{c=1/4} <end>"
      }
    }
  ]
}
```

## 4. 阶段C：从树提取训练路径

脚本：`data_process/extract_path.py`

它会抽取：
- 终止路径（含 `<end>` 或命中正确答案）
- 补充的深度路径（增强多样性）

示例样本（`data/synthetic_data/syn_data.json` 局部）：

```json
{
  "idx": "algebra_0001",
  "question": "Find the largest value of c ...",
  "ground_truth": "1/4",
  "label": "positive",
  "steps": [
    {
      "content": "...判别式...\\boxed{\\Delta=1-4c}",
      "step score": 1,
      "reflection": "[True] ...",
      "correction": {"Input": "Solve[...]", "final_answer": "..."}
    },
    {
      "content": "...\\boxed{c=1/4} <end>",
      "step score": 1,
      "reflection": "[True] ...",
      "correction": {"Input": "...", "final_answer": "1/4"}
    }
  ]
}
```

## 5. 阶段D：构造成 SFT 对话数据

脚本：`data_process/construct.py`

它会：
1. 过滤 score/reflection 不一致样本。
2. 负样本截断。
3. 组装每步 assistant 监督格式（`<verify>/<judge>/<output>`）。

示例 conversation（训练条目）：

```json
{
  "conversation": [
    {
      "role": "system",
      "content": "Your task is to analyse and critique the steps..."
    },
    {
      "role": "user",
      "content": "Find the largest value ...\nStep objective...\\boxed{\\Delta=1-4c}"
    },
    {
      "role": "assistant",
      "content": "<verify>...</verify>\n<judge>...</judge>\n<output>...\\boxed{+}</output>"
    }
  ]
}
```

## 6. 阶段E：LoRA 微调

用 LLaMA-Factory 读取上一步产出的 conversation 数据做 SFT：

```bash
llamafactory-cli train \
  --stage sft \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset your_dataset \
  --template qwen \
  --finetuning_type lora \
  --output_dir models/GroundedPRM
```

训练后可导出合并权重给 vLLM 部署。

## 7. 阶段F：评测

### ProcessBench
- 用 `evaluation/ProcessBench_eval.py` 做步骤级评测。

### Reward-guided Search
- 用 `evaluation/reward_guided_search/Greedy-Search.py`：
  - 策略模型生成候选步骤
  - PRM 给每个候选打分
  - 选最高分继续推理

## 8. 一条“你到底做了什么”的直白总结

你做的是：
1. 用 MCTS + 外部验证自动生成带步骤标签的推理轨迹；
2. 把轨迹清洗成 PRM 训练数据；
3. 用 LoRA 微调模型学会步骤判别；
4. 用该奖励模型在推理时引导搜索，提升稳健性。

