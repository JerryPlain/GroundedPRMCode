from concurrent.futures import ThreadPoolExecutor
import os
import argparse
import json
import logging
from typing import List
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from vllm import LLM, SamplingParams
from openai import OpenAI
from prompts.policy_prompt import Qwen_Policy_Prompt, get_qwen_policy_messages
from GroundedPRM import PRM
import math
from typing import List
import regex
import re
import random
import multiprocessing

logger = logging.getLogger(__name__)

allowed_pattern = regex.compile(
    r"[^\p{Latin}\p{Greek}\d\s"
    r"\+\-\*/=\^\_\'\".,:;?!\(\)\{\}\[\]\\\$%<>|&@#"
    r"√∞±×÷°]"
)

EOS_TOKEN="<|im_end|>"

global_scorer = None
SYSTEM_PROMPT = """Your task is to analyse and critique the reasoning step below of a math problem step-by-step.
Please follow the instructions and format below to respond:

1. Judge the logical and calculation correctness of the step by yourself.
2. Provide the judgement conclusion and the label as "Right" or "Wrong" in '<output>\\boxed{}</output>'.
3. Use the following format to respond:
<judge>
[details reasoning]
</judge>
<output>
According to the conclusion of the judgement, the label is: \\boxed{}
</output>

Here is the step:
"""

def extract_boxed(s):
    results = []
    i = 0
    while True:
        start = s.find(r'\boxed{', i)
        if start == -1:
            break
        # advance past the "\boxed{"
        j = start + len(r'\boxed{')
        depth = 1
        while j < len(s) and depth > 0:
            if s[j] == '{':
                depth += 1
            elif s[j] == '}':
                depth -= 1
            j += 1
        if depth == 0:
            # everything from just after the first '{' up to j-1
            content = s[start + len(r'\boxed{') : j - 1]
            results.append(content)
            i = j
        else:
            # unbalanced braces: bail out
            break
    if len(results) == 1:
        return results[0]
    else:
        return None


def find_illegal_chars(text: str) -> list:
    """
     Find characters in the text that are outside the allowed set, returning a list.
    """
    return allowed_pattern.findall(text)


def is_math_answer_valid(answer: str) -> bool:
    """
    Check whether the math answer contains illegal characters:
      - If returns True, it means the text has no disallowed characters
      - If returns False, the text contains illegal characters
    """
    illegal = find_illegal_chars(answer)
    if illegal:
        return False
    return True


def get_next_step(policy_client, policy_model_name, question, previous_steps, policy_prompt, temperature):
    # Use the messages format
    previous_step = "" if len(previous_steps) == 0 else "\n\n".join(previous_steps)
    messages = get_qwen_policy_messages(question, previous_step)

    try:
        response = policy_client.chat.completions.create(
            model=policy_model_name,
            messages=messages,  # Use messages instead of prompt
            n=8,
            temperature=temperature,
            max_tokens=512,
            stop=[EOS_TOKEN, "\n\n"],
        )
        
        new_steps_candidates = []
        str_set = set()
        for choice in response.choices:
            gen_text = choice.message.content
            if not is_math_answer_valid(gen_text):
                continue
            
            # Check the stop_reason to determine how to handle the text
            if hasattr(choice, 'stop_reason'):
                if choice.stop_reason == '\n\n' or extract_boxed(gen_text) == None:
                    # Model stopped at \n\n - this is a regular step
                    gen_text = gen_text.strip()
                else:
                    # Model stopped at <|im_end|> - this is a final step with answer
                    gen_text = gen_text.strip() + EOS_TOKEN
            else:
                # No stop_reason available, check content
                gen_text = gen_text.strip() + EOS_TOKEN

            if gen_text in str_set:
                continue
            new_steps_candidates.append(gen_text)
            str_set.add(gen_text)
        return new_steps_candidates
    except Exception as e:
        logger.error(f"Error in get_next_step: {e}")
        return []


def process_single_data(args_tuple):
    i, data, args_dict, ans_key = args_tuple
    from openai import OpenAI
    # Re-initialize clients in each process
    policy_client = OpenAI(
        base_url=args_dict["policy_api_base"],
        api_key=args_dict["policy_api_key"],
    )
    reward_client = OpenAI(
        base_url=args_dict["reward_api_base"],
        api_key=args_dict["reward_api_key"],
    )
    question = data["problem"]
    previous_steps = []
    previous_steps_reward = []
    used_steps = set()
    max_iteration = 30
    iteration_history = []
    iteration_index = 0
    while max_iteration > 0:
        max_iteration -= 1
        iteration_index += 1
        new_steps_candidates = get_next_step(policy_client, args_dict["policy_model_name"], question, previous_steps, Qwen_Policy_Prompt, args_dict["temperature"])
        logger.debug(f"Question Number: {i}, Iteration: {iteration_index}, New Steps Candidates Number: {len(new_steps_candidates)}")
        
        if len(new_steps_candidates) == 0:
            continue
        new_step_accepted = False
        iteration_data = {
            "iteration_index": iteration_index,
            "candidates_info": [],
            "chosen_step_index": None,
            "chosen_step": None
        }
        for candidate_idx, candidate_step in enumerate(new_steps_candidates):
            if candidate_step in used_steps:
                continue
            if candidate_step.endswith(EOS_TOKEN):
                boxed_answer = extract_boxed(candidate_step)
                if boxed_answer == None:
                    continue
                temp_candidate_step = candidate_step.replace(EOS_TOKEN, "")
            else:
                temp_candidate_step = candidate_step
            
            scorer = PRM(
                reward_client=reward_client,
                reward_model_name=args_dict["reward_model_name"],
                tokenizer_path=args_dict["reward_tokenizer_path"]
            )
            reward_score = scorer.get_reward_score(question, previous_steps, temp_candidate_step, SYSTEM_PROMPT)
            if reward_score == -1:
                continue
            reward = reward_score
            logger.debug(f"Question Number: {i}, Iteration: {iteration_index}, Candidate Step Index: {candidate_idx}, Reward: {reward}")
            candidate_info = {
                "candidate_step": candidate_step,
                "reward": reward
            }
            iteration_data["candidates_info"].append(candidate_info)
        max_reward = -1
        max_reward_indices = []
        for idx, candidate_info in enumerate(iteration_data["candidates_info"]):
            if candidate_info["reward"] > max_reward:
                max_reward = candidate_info["reward"]
                max_reward_indices = [idx]
            elif candidate_info["reward"] == max_reward:
                max_reward_indices.append(idx)
        if len(max_reward_indices) == 0:
            continue
        import random
        max_reward_idx = random.choice(max_reward_indices)
        iteration_data["chosen_step_index"] = max_reward_idx
        iteration_data["chosen_step"] = iteration_data["candidates_info"][max_reward_idx]["candidate_step"]
        previous_steps.append(iteration_data["chosen_step"])
        previous_steps_reward.append(max_reward)
        iteration_history.append(iteration_data)
        if len(previous_steps) > 0 and EOS_TOKEN in previous_steps[-1]:
            logger.debug(f"Question Number: {i}, Early stopping at iteration {iteration_index}")
            break
    return {
        "question": question,
        "iteration_history": iteration_history,
        "final_steps": previous_steps,
        "gt_answer": data[ans_key],
        "pred_answer": extract_boxed(previous_steps[-1]) if previous_steps else None,
    }

def main():
    parser = argparse.ArgumentParser(description="Greedy Search reasoning pipeline with reward model")   
    # API configuration arguments
    parser.add_argument("--policy_api_base", type=str, required=True, help="API base URL for policy model (e.g., http://localhost:8000/v1)")
    parser.add_argument("--reward_api_base", type=str, required=True, help="API base URL for reward model (e.g., http://localhost:8001/v1)")
    parser.add_argument("--policy_api_key", type=str, default="EMPTY", help="API key for policy model")
    parser.add_argument("--reward_api_key", type=str, default="EMPTY", help="API key for reward model")
    parser.add_argument("--policy_model_name", type=str, required=True, help="Model name for policy model API")
    parser.add_argument("--reward_model_name", type=str, required=True, help="Model name for reward model API")
    parser.add_argument("--reward_tokenizer_path", type=str, required=True, help="tokenizer path for reward model")

    parser.add_argument("--data", type=str, required=True, help="Dataset to Evaluate on", 
                        choices = ["math", "amc23", "aime25", "aime24", "college_math", "minerva_math", "olympiadbench"])

    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results.")
    parser.add_argument("--temperature", type=float, default=0.7, help="the temperature of the policy model.")
    parser.add_argument("--data_begin", type=int, default=0, help="Starting index of the dataset to process.")
    parser.add_argument("--data_end", type=int, default=None, help="Ending index of the dataset to process.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    DATA_PATHS = {
        "math": "./eval_data/MATH/Math-OAI.jsonl", 
        "amc23": "./eval_data/AMC23/test.jsonl", 
        "aime25": "./eval_data/AIME25/AIME25_train.jsonl", 
        "aime24": "./eval_data/AIME24/test.jsonl",
        "college_math": "./eval_data/College_Math/college_math_200.jsonl", 
        "minerva_math": "./eval_data/Minerva-MATH/minerva-math.jsonl",
        "olympiadbench" : "./eval_data/OlympiadBench/olympiadbench_200.jsonl"
    }

    if args.data == "minerva_math":
        ans_key = "solution"
    elif args.data == "olympiadbench":
        ans_key = "final_answer"
    else:
        ans_key = "answer"

    dataset = load_dataset("json", data_files=DATA_PATHS[args.data], split="train")
    
    if args.data_begin!=0 or args.data_end != None:
        # Handle dataset slicing properly
        try:
            # Try to use select method if available
            dataset = dataset.select(range(args.data_begin, args.data_end if args.data_end and args.data_end <= len(dataset) else len(dataset)))
        except (AttributeError, TypeError):
            # Fallback: convert to list and slice
            dataset_list = list(dataset)
            dataset = dataset_list[args.data_begin:args.data_end if args.data_end else len(dataset_list)]

    # Prepare arguments for multiprocessing
    args_dict = vars(args)
    args_tuples = [(i, data, args_dict, ans_key) for i, data in enumerate(dataset)]

    with ThreadPoolExecutor(max_workers=10) as executor:
        new_dataset = list(tqdm(executor.map(process_single_data, args_tuples), total=len(args_tuples)))
    output_file = os.path.join(args.output_dir, f"result-{args.data_begin}-{args.data_end}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_dataset, f, ensure_ascii=False, indent=2)
    logger.info(f"Done! Results are saved to {output_file}.")


if __name__ == "__main__":
    main()