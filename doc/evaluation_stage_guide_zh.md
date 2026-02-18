# GroundedPRM 评测阶段详解（原理 + 原因 + 例子）

这份文档专门解释“评测阶段到底怎么做”，对应仓库里的两条评测线：

1. `ProcessBench_eval.py`：步骤判别能力评测（静态过程评测）
2. `reward_guided_search`：用 PRM 引导推理搜索（动态决策评测）

---

## 1. 为什么要分两条评测线

只看最终答案准确率不够，因为 PRM 的核心能力是“判断中间步骤好坏”。

所以项目做两种评测：

1. **静态评测（ProcessBench）**：不让模型自己解题，只测“给定步骤会不会判对错”。
2. **动态评测（Reward-guided Search）**：让模型边生成边被 PRM 打分，测“这个奖励模型能不能真实提高最终解题表现”。

这两者分别回答：
- 你会不会判步骤？
- 你的判分有没有实际决策价值？

---

## 2. ProcessBench 评测：怎么做

对应脚本：`evaluation/ProcessBench_eval.py`

### 2.1 原理

对每道题，脚本按顺序喂入参考解法的每一步，让待评测模型当“步骤裁判”：

1. 输入：题目 + 第1步，模型输出该步对错。
2. 若判错（`-`）则停止，预测“首次错误步骤索引 = 当前步”。
3. 若一直判对，到最后都没判错，则预测 `-1`（表示整条过程无明显错误）。

脚本中关键函数：
- `extract_score(...)`：从模型输出中抽 `+/-` 或 `[Right]/[Wrong]`。
- `single_process(...)`：逐步评审并返回预测错误位置。

### 2.2 为什么这样评

ProcessBench 的标签本质是“错误首次出现在哪一步”。

如果 PRM 真有过程理解能力，它应该能较准确地定位 first error，而不是仅靠最终答案猜测。

### 2.3 运行示例

```bash
python evaluation/ProcessBench_eval.py \
  --api_url http://127.0.0.1:8001/v1 \
  --model_name reward-model \
  --output_prefix groundedprm_v1
```

### 2.4 输出与结果文件

按数据子集输出到：`evaluation/outputs/<dir_name>/`

子集包括：
- `gsm8k`
- `math`
- `olympiadbench`
- `omnimath`

会写两类文件（脚本命名）：
- `error_<config>_<prefix>.jsonl`
- `correct_<config>_<prefix>.jsonl`

每行包含：
- 原始样本
- `prediction`（预测首次错误步）
- `match`（是否命中标签）

### 2.5 一个最小理解例子

假设某题 steps 有 4 步，真实 label=2（第 3 步首次错）：

1. 模型判 step0: 正确
2. 模型判 step1: 正确
3. 模型判 step2: 错误 -> 返回 prediction=2

若 `prediction == 2`，则 `match=True`。

---

## 3. Reward-guided Search：怎么做

对应目录：`evaluation/reward_guided_search/`

关键脚本：
- `Greedy-Search.py`
- `GroundedPRM.py`
- `eval_results.py`

### 3.1 原理

这是“边生成边打分”的闭环：

1. **Policy 模型**一次生成多个候选下一步（默认 `n=8`）。
2. **Reward 模型(PRM)** 对每个候选打分。
3. 选最高分候选作为下一步。
4. 重复迭代直到终止（出现 EOS/最终答案）或到达步数上限。

#### PRM 分数怎么来的（代码级）

在 `GroundedPRM.py`：

1. 奖励模型会输出 `Right/Wrong` 判定文本。
2. 同时请求 `logprobs`。
3. 脚本从 token 概率中取 `Right` 与 `Wrong` 的相对概率：
   - `reward = P(Right) / (P(Right)+P(Wrong))`
4. reward 越高，表示该候选步骤越可能正确。

这不是硬分类，而是概率型分数，便于排序候选步骤。

### 3.2 为什么这样评

这条评测验证的是“PRM 的实战价值”：

- 如果 PRM 只是会给离线样本打标签，但不能指导搜索，它对推理提升有限。
- 让 PRM 参与每步决策，才能验证是否真正提升最终解题成功率。

### 3.3 部署要求

需要两个 OpenAI 兼容 API：

1. policy model（生成步骤）
2. reward model（步骤打分）

仓库脚本：`evaluation/reward_guided_search/deploy_models.sh`

```bash
cd evaluation/reward_guided_search
bash deploy_models.sh
```

### 3.4 运行示例（先小规模）

建议先在一个数据集的小切片上 smoke test：

```bash
python evaluation/reward_guided_search/Greedy-Search.py \
  --policy_api_base http://127.0.0.1:8000/v1 \
  --reward_api_base http://127.0.0.1:8001/v1 \
  --policy_model_name policy-model \
  --reward_model_name reward-model \
  --reward_tokenizer_path /path/to/reward/model \
  --data math \
  --output_dir evaluation/reward_guided_search/outputs/greedy_search/prm/math \
  --data_begin 0 \
  --data_end 50 \
  --temperature 1.0
```

### 3.5 输出文件怎么看

每个样本会记录：
- `iteration_history`：每轮有哪些候选、各自 reward、最后选了谁
- `final_steps`：最终选出来的完整推理链
- `pred_answer`：从最后一步 `\boxed{}` 提取出的预测答案
- `gt_answer`：标准答案

这能直接审计“为什么选了这一步”。

---

## 4. 最终准确率计算（eval_results.py）

脚本：`evaluation/reward_guided_search/eval_results.py`

### 4.1 指标定义

对每个数据集：

1. 读取结果文件 `result-0-None.json`。
2. 过滤有 `pred_answer` 的样本，做数学等价比较（`math_verify`）。
3. 分数按：
   - `score = 正确样本数 / 数据集总样本数`

注意分母用的是总样本数，所以未出答案样本会自然拉低分数。

### 4.2 运行示例

```bash
python evaluation/reward_guided_search/eval_results.py \
  --results_dir evaluation/reward_guided_search/outputs/greedy_search/prm
```

输出：`results.json`（包含各子集与 average）。

---

## 5. 一个完整的小例子（从部署到结果）

1. 启动 policy/reward API（8000/8001）。
2. 跑 50 条 `math`：
   - `Greedy-Search.py --data math --data_begin 0 --data_end 50`
3. 查看单样本 `iteration_history`：
   - 每轮 8 个候选，reward 最高者被选中。
4. 跑 `eval_results.py`：
   - 得到该设置下的 accuracy。
5. 对比 baseline（不用 PRM 或换 PRM）：
   - 观察平均正确率与失败样本类型变化。

这就是“PRM 是否真的有用”的核心验证路径。

---

## 6. 常见误区（你之前问到的点）

1. **误区：每一步都有人工 GT。**
   - 事实：步骤标签由验证器自动生成（WA + 评审模型）。
2. **误区：只看 ProcessBench 就够了。**
   - 事实：ProcessBench 仅证明“会判”；reward-guided search 才证明“会用”。
3. **误区：有 `pred_answer` 才计分。**
   - 事实：脚本分母是总样本数，没答案也算失败，这更接近实战。

