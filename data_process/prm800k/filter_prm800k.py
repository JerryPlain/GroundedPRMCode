from datasets import load_dataset
from collections import defaultdict
import re
import json
import os
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from construct import load_json_and_convert

ds = load_dataset("miazhao/prm800k_rating_cls", split="train")

def preprocess_prm_dataset(split="train"):
    ds = load_dataset("miazhao/prm800k_rating_cls", split=split)
    processed = []
    
    for input_str, rating in zip(ds['input'], ds['rating']):
        parts = re.split(r'\n\s*\n\s*\n\s*', input_str, maxsplit=1)
        question = parts[0].strip()
        answer_block = parts[1] if len(parts) > 1 else ""       
        steps = [line.strip() for line in answer_block.split('\n') if line.strip()]
        
        if steps:
            prev_steps = steps[:-1]
            last_step = steps[-1]
        else:
            prev_steps = []
            last_step = ""
        
        record = {
            "question": question,
            "previous steps": prev_steps,
            "steps": [
                {
                    "response": last_step,
                    "step score": rating
                }
            ]
        }
        processed.append(record)
    
    return processed

def convert_to_dialogues(processed_records):
    conversations = []
    for rec in processed_records:
        prev_steps = rec["previous steps"]
        question = rec["question"]
        step = rec["steps"][0]["response"]
        score = rec["steps"][0]["step score"]
        if score >= 0:
            label = "+"
        else:
            label = "-"
        convo = [
            {
                "role": "system",
                "content": "Your task is to analyse and critique the steps in the solution of a math problem step-by-step. '+' means the step is correct, '-' means the step is incorrect."
            }
        ]        

        if prev_steps:
            first_prev = prev_steps[0]
            convo.append(
                {"role": "user", 
                "content": question + "\n" + first_prev}
            )
            convo.append(
                {"role": "assistant", "content": "+"}
            )
            
            for prev in prev_steps[1:]:
                convo.append(
                    {"role": "user", "content": prev}
                )
                convo.append(
                    {"role": "assistant", "content": "+"}
                )
            convo.append(
                {"role": "user", "content": step}
            )
            convo.append(
                {"role": "assistant", "content": label}
            )
        else:
            convo.append(
                {"role": "user", "content": question + "\n" + step}
            )
            convo.append(
                {"role": "assistant", "content": label}
            )
        conversations.append({"conversation": convo})
    return conversations

def filter_processed_by_test(processed_path, test_root, save_path):
    with open(processed_path, "r", encoding="utf-8") as f:
        processed_data = json.load(f)

    test_problems = set()
    for root, dirs, files in os.walk(test_root):
        for file in files:
            if file.endswith(".json"):
                task_path = os.path.join(root, file)
                with open(task_path, "r", encoding="utf-8") as f:
                    task_data = json.load(f)
                    problem = task_data.get("problem", "").strip()
                    if problem:
                        test_problems.add(problem)

    filtered_data = [
        item for item in processed_data
        if item["question"].strip() not in test_problems
    ]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

def save_random_samples(input_path, output_dir, seeds=42):
    random.seed(seeds)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample_sizes = [20000, 30000, 40000, 50000] 
    os.makedirs(output_dir, exist_ok=True)
    for size in sample_sizes:        
        sampled = random.sample(data, size)
        sampled_conv = convert_to_dialogues(sampled)
        save_path = os.path.join(output_dir, f"sample_{size}.json")
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(sampled_conv, f, ensure_ascii=False, indent=2)

        training_dir = "data/train/prm_training"
        llama_path = os.path.join(training_dir, f"PRM_{size}.json")
        training_format_data = load_json_and_convert(sampled_conv, llama_path)
        
def main():
    processed = preprocess_prm_dataset()
    dialogues = convert_to_dialogues(processed)
    orig_path = "prm800k/proceed.json"
    with open(orig_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    processed_path = "prm800k/proceed.json"
    test_root = "math_dataset/test"
    filtered_path = "prm800k/filtered.json"
    samples_output_dir = "prm800k"

    filter_processed_by_test(processed_path, test_root, save_path)
    save_random_samples(filtered_path, samples_output_dir)

if __name__ == "__main__":
    main()
