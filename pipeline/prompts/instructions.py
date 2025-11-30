extract_instruction = """You will be given a single mathematics problem. Your task is to extract:

  1. A list of known conditions from the problem statement, and
  2. A single global objective that represents what the problem is asking you to find.

  <instruction>

  Follow these guidelines carefully:
  1. Do not perform any reasoning or explain your interpretations. Only extract and list the given information and the final goal.
  2. You may use mathematical expressions to represent quantities or relationships directly taken from the problem, but do not introduce new content or reformulate the problem beyond a direct translation of the given statements.
  3. Each condition should contain only one distinct piece of given information or one relationship, written as simply as possible.
  4. There must be exactly one global objective, clearly reflecting the ultimate requirement of the original problem.
  5. Do not add any commentary, reasoning steps, or extraneous text.
  6. Format your final answer as a JSON object with the following structure:

  {
    "conditions": "condition_1, condition_2",
    "global_objective": "final_objective"
  }

  Here: 
  - "conditions" is an array, each element capturing a single known fact or given expression from the problem.
  - "global_objective" is a single string stating the final goal of the problem.

  </instruction>

  Here are some examples to help you extract the conditions and global objective from the mathematics problem.

  <examples>
  ### Example 1
  Input:
  Question: "Find the largest value of $c$ such that $-2$ is in the range of  $f(x)=x^2+3x+c$."

  Output:
  {
    "conditions": "condition 1: $f(x)=x^2+3x+c$, condition2: $-2$ is in the range of $f(x)$",
    "global_objective": "Find the largest value of $c$"
  }

  </examples>
  """


generate_step_instruction = """Your task is to propose the next reasoning step for solving a mathematics problem, based on the provided conditions, the global objective, and the previous steps.
  Your output must include two components:

  1. Step Objective: A statement of the immediate local goal, referencing the specific conditions formulations and methods it will use.
  2. Action: A detailed process or reasoning that uses the given conditions to achieve the step objective.

  <instruction>

  Follow these guidelines carefully:

  **Step objective:**

  1. Focus on one reasoning step only. Each step objective must represent the immediate next reasoning goal, not the entire solution.
  2. The step objective must clearly identify which conditions, formulations and methods will be utilized and what intermediate conclusion will be derived.
  3. Ensure the step objective references only relevant conditions and aligns directly with the problem ultimate objective.

  **Action:**
  1. The action should details the logical or computational steps based solely on the referenced conditions.
  2. The action only present the detailed reasoning or calculations necessary to achieve the current step objective. Ensure that the final answer obtained in the action aligns precisely with the step objective, without performing any additional or unnecessary reasoning.
  3. The action should clearly write the step conclusion (the result of the step objective) inside `\\boxed{}`. This ensures the conclusion of this step is highlighted. 
  4. The content inside "\\boxed{}" must explicitly present the relationship between the parameter being solved and its result, using the format "parameter = result" or "objective is expression". Avoid including standalone values or expressions without indicating what they represent.
  5. **END Tag** If and only if the action leads directly to the final solution of the entire problem (the global objective) or steps with the same result are repeated, the action has to end with **<end>** to mark that you have finished the whole task.

  **Output format:**
  Your output must be in JSON format, structured as follows:

  {
    "step objective": "description of the step objective",
    "action": "detailed reasoning or computation process"
  }

  </instruction>

  Here are some examples to help you extract the conditions and global objective from the mathematics problem.

  <examples>
  ### Example 1
  Input:
  {
    "conditions": [
      "$f(x)=x^2+3x+c$",
      "$-2$ is in the range of $f(x)$",
    ],
    "global_objective": "Find the largest value of $c$"
    "previous_steps": []
  }

  Output:
  {
    "step objective": "From the condition 1 and 2 we know if and only if the equation $x^2+3x+c=-2$ has a real root that $-2$ is in the range of $f(x)$. We usually use the discriminant to determine if the quadratic equation has real roots or not, so the step objective is to calculate the discriminant of $x^2+3x+(c+2)=0$",
    "action": "Since the quadratic is $x^2+3x+(c+2)=0$, the discriminant of this quadratic is $3^2 - 4(c + 2)$, so we got the step answer: \\\\boxed{The discriminant of $x^2+3x+(c+2)=0$ is $1 - 4c$}"
  }

  ### Example 2
  Input:
  {
    "conditions": [
      "$f(x)=x^2+3x+c$",
      "$-2$ is in the range of $f(x)$",
      "$The discriminant of $x^2+3x+(c+2)=0$ is $1 - 4c$"
    ],
    "global_objective": "Find the largest value of $c$"
    "previous_steps": [
      {
        "step objective": "From the condition 1 and 2 we know if and only if the equation $x^2+3x+c=-2$ has a real root that $-2$ is in the range of $f(x)$. We usually use the discriminant to determine if the quadratic equation has real roots or not, so the step objective is to calculate the discriminant of $x^2+3x+(c+2)=0$",
        "action": "Since the quadratic is $x^2+3x+(c+2)=0$, the discriminant of this quadratic is $3^2 - 4(c + 2)$, so we got the step answer: \\\\boxed{The discriminant of $x^2+3x+(c+2)=0$ is $1 - 4c$}"
      }
    ]
  }

  Output:
  {
    "step objective": "We see the descriminant of the quadratic in the previous steps, and condition 2 represent that there is at least a real root in the quadratic which means $1 - 4c \\\\ge 0$. So in this step objective, we can calculate the max value of c",
    "action": "From $1 - 4c \\\\ge 0$ we got $1 \\\\ge 4c$, then $c \\\\le 1/4$, we have achieved the global objective: $\\\\boxed{c=\\\\frac{1}{4}}$ <end>"
  }

  </examples>
  """


code_instruction = """You are a WolframAlpha query generation expert. Instead of overthinking the solution to the problem, focus on how to generate the WA query corresponding to the <current_target>
  Please generate a valid query based on the following process:

  ###Generation Process

  1. Query Construction Strategy:
    a. Prioritize using Wolfram built-in functions:
        - Equation solving: Solve / NSolve / DSolve
        - Symbolic computation: Simplify / Expand / Factor
        - Numerical computation: N / Integrate / Sum
        - Unit conversion: UnitConvert
    b. Avoid invalid syntax:
        - Use Solve[{original equation, constraints}, variables] instead of expression substitution (e.g. /. x -> 2)
        - Use Pi/180 for angle units
        - Use natural language to express special computational needs

  2. Safety Check:
    - Ensure the query contains no undefined variables
    - Confirm that the numerical precision matches the problem requirements
    - Avoid mixing symbolic and numerical computations

  ###Input Elements
  <conditions> Full description of known conditions and constraints  
  <global_objective> The final goal to be achieved for the entire problem  
  <current_target> The specific goal in the current step  

  ###Output only the query as JSON  
    - Output only the contents of the json object, do not provide extra explanation or text, 
    - Format the final output strictly as JSON format:
  {
    "query": "Exact WA query statement"
  }

  ###Best Practice Examples
  1. Equation solving: Solve[x^2 + 4x + 6 == 0, x]  
  2. Constrained calculation: Solve[{x + y = 5, 2x - y = 1}, {x, y}]  
  3. Angle conversion: Sin[35*Pi/180]  
  4. Natural language query: convert 1789 base16 to base 2

  """


extract_prompt = """Let us extract the conditions and the global objective from the question.
Input:
Question: {question}
"""


generate_step_prompt = """Let us generate the next step objective and its action.
Input:
{{
  "global_objective": {global_objective},
  "conditions": {conditions},  
  "previous_steps" {previous_steps}
}}
"""


generate_query_prompt = """Let us generate wolfram alpha query for the current step objective based on the known conditions.
Input:
{{
  "global_objective": {global_objective},
  "conditions": {conditions},
  "current_target": {step_objective}
}}
"""

verify_instruction = """You are a mathematical reasoning verification assistant. Please strictly analyze the problem based on the following structure and output the result in JSON format:

  ###Verification Process
  1. Logical Check: Verify whether the LLM reasoning contradicts the known conditions or deviates from the current step goal (e.g., introduces irrelevant variables or incorrect formulas).
  2. WA Relevance Judgment: Analyze whether the Wolfram Alpha (WA) query is directly related to the current reasoning objective, and whether the returned result contains valid verification information.
  3. Result Comparison:
    - When WA returns a precise numerical value: the absolute error between the LLM result and the WA value must be ≤ 0.1
    - When WA returns a numerical range: the LLM result must be completely contained within this range
    - When WA returns a mathematical expression: verify algebraic equivalence with the LLM result (unsimplified forms are acceptable; consistency should be validated by substituting any values)
    - If WA returns an error or its result is invalid or irrelevant: Analyse the LLM answer for correctness based on all provided contexts

  ###Input Elements
  <conditions> Known conditions and constraints of the problem  
  <objective> The specific objective to be achieved in the current step  
  <llm_answer> The reasoning step to be verified  
  <wa_return> The API response of Wolfram Alpha  

  Please output only a standard JSON:
  {
    "reason": "First, ...[conclusion from logical check], then ...[analysis of WA relevance], finally ...[details of expression/numerical comparison]",
    "result": ["True", "False"]
  }
  """


verify_prompt = """Let us verify the llm answer.
Input:
<objective>: {step_objective}
<conditions>: {conditions}
<wa_return>: {wa_answer}
<llm_answer>: {llm_answer}
"""
