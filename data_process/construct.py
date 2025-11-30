import json
import os
import re
import logging
from typing import List, Dict
from collections import defaultdict
import pyarrow as pa
import pandas as pd
import sys
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training_instruction import instruction

logger = logging.getLogger(__name__)


def conversation_trans_ref(data):
    all_conversations = []
    for item in data:
        conversations = []

        conversations.append({"role": "system", "content": instruction})
        if item["steps"]:
            first_user_content = item["question"] + "\n" + item["steps"][0]["content"]
            conversations.append({"role": "user", "content": first_user_content})
            first_response = item["steps"][0]["response"]
            conversations.append({"role": "assistant", "content": first_response})
            for step in item["steps"][1:]:
                conversations.append({"role": "user", "content": step["content"]})
                conversations.append({"role": "assistant", "content": step["response"]})
        all_conversations.append(({"conversation": conversations}))
    return all_conversations


def load_json_and_convert(data, output_path):
    # Convert the data to a PyArrow Table
    try:
        df = pd.DataFrame(data)
        if 'history' in df.columns:
            df['history'] = df['history'].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else ("" if pd.isna(x) else str(x))
            )
        table = pa.Table.from_pandas(df)
        logger.info("Data successfully converted to PyArrow Table.")
        
        df = table.to_pandas()
        df.to_json(output_path, orient='records', lines=True)
        logger.info(f"Data successfully saved to {output_path}.")
        
    except Exception as e:
        logger.error(f"An error occurred while converting the data: {e}")


def filter_consistent_samples(data):
    true_pattern = re.compile(r'^\s*\[\[?["\']?True["\']?\]?\]', re.IGNORECASE)
    false_pattern = re.compile(r'^\s*\[\[?["\']?False["\']?\]?\]', re.IGNORECASE)

    filtered = []

    for sample in data:
        all_consistent = True
        for step in sample.get("steps", []):
            score = step.get("step score", 0)
            reflection = step.get("reflection", "")
            
            is_true = bool(true_pattern.search(reflection))
            is_false = bool(false_pattern.search(reflection))

            if (score > 0 and not is_true) or (score < 0 and not is_false) or (score == 0):
                if not true_pattern.search(reflection) or score == 0:
                    all_consistent = False
                    break
            
            marker = ""
            cleaned_reflection = reflection

            match_true = true_pattern.match(reflection)
            match_false = false_pattern.match(reflection)

            if match_true:
                marker = "[Right]"
                cleaned_reflection = true_pattern.sub('', reflection).strip()
            elif match_false:
                marker = "[Wrong]"
                cleaned_reflection = false_pattern.sub('', reflection).strip()

            if marker:
                cleaned_reflection = cleaned_reflection.rstrip('.')  
                cleaned_reflection = f"{cleaned_reflection}. The conclusion is{marker}".strip()

            step["reflection"] = cleaned_reflection
            
        if all_consistent:
            filtered.append(sample)

    return filtered


def truncate_steps(data):
    updated_data = []

    for sample in data:
        if sample.get("label") == "positive":
            updated_data.append(sample)
            continue

        steps = sample.get("steps", [])
        first_negative_idx = None
        truncate_idx = None

        for i, step in enumerate(steps):
            if step.get("step score", 0) <= 0:
                first_negative_idx = i
                break

        if first_negative_idx is not None:
            for j in range(first_negative_idx + 1, len(steps)):
                if steps[j].get("step score", 0) >= 0:
                    truncate_idx = j
                    break

            if truncate_idx is not None:
                sample["steps"] = steps[:truncate_idx]

        updated_data.append(sample)

    return updated_data


def convert_to_binary_label(data): 
    for item in data:
        if "steps" in item and item["steps"]:
            for step in item["steps"]:
                score = step.get("step score", 0)
                step["step score"] = 1 if score > 0 else -1
    return data


def merge_steps(data):
    for sample in data:
        for step in sample.get("steps", []):
            step_score = step.get("step score", "")
            if isinstance(step.get("correction", {}), dict):
                correction = step.get("correction", {})
                correction_input = correction.get("Input", "")
                correction_result = correction.get("Result", None)
                correction_final = correction.get("final_answer", None)
                if correction_result:  
                    correction_value = correction_result                
                    if isinstance(correction_value, list):
                        correction_value = ', '.join(str(x) for x in correction_value)
                    wa_str = f"The expression I needed to verify is: {correction_input}, the result is: {correction_value}"
                elif correction_final:
                    correction_value = correction_final
                    if isinstance(correction_value, list):
                        correction_value = ', '.join(str(x) for x in correction_value)
                    wa_str = f"The expression I needed to verify is: {correction_input}, the result is: {correction_value}"
                else:  
                    wa_str = correction_input
            else:  
                correction_input = "This is a thinking step, no specific calculations are needed"
                correction_result = None
                correction_final = None
                wa_str = correction_input

            verify_str = f"<verify>\n{wa_str}\n</verify>\n"
            reflection = step.get("reflection", "")
            judge_str = f"<judge>\n{reflection}\n</judge>\n"

            if step_score > 0:
                lable = '+'
                output_str = f"<output>\nAccording to the judgement conclusion, this step is: \\boxed{{{lable}}}\n</output>"
            elif step_score < 0:
                lable = '-'
                output_str = f"<output>\nAccording to the judgement conclusion, the label of this step is: \\boxed{{{lable}}}\n</output>"

            response = verify_str + judge_str + output_str

            step["response"] = response
            if "step score" in step:
                del step["step score"]
            if "correction" in step:
                del step["correction"]
            if "reflection" in step:
                del step["reflection"]

    return data


def construct_data(data):
    cleaned_data = filter_consistent_samples(data)                                
    fixed_data = truncate_steps(cleaned_data)               
    binary_data = convert_to_binary_label(fixed_data)          
    merged_data = merge_steps(binary_data)                    
    conversation_data = conversation_trans_ref(merged_data)     
    return conversation_data


def process_file_list(file_list):
    for file_path in file_list:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            construct_data(data)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")


def merge_json_files(file_paths, output_path):
    all_data = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                data = [json.loads(line) for line in f if line.strip()]
                all_data.extend(data)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
    try:
        with open(output_path, 'w') as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
    except Exception as e:
        logger.error(f"Error writing merged file: {str(e)}")


if __name__ == '__main__':
    # construct data
    file_list = [
        "precalculus.json",
        "prealgebra.json",
        "counting_and_probability.json",
        "geometry.json",
        "intermediate_algebra.json",
        "number_theory.json", 
        "algebra.json"
    ]

    file_list = [os.path.join("data/meta", file_name) for file_name in file_list]
    processed_files = []  
    for file_path in file_list:
        with open(file_path, "r") as f:            
            data = json.load(f)
        conversation_data = construct_data(data[:200])


        llama_path = os.path.join("data/train/", file_path)
        training_format_data = load_json_and_convert(conversation_data, llama_path)  
        processed_files.append(llama_path)
        
    merged_output_path = "data/train/training.json"   
    merge_json_files(processed_files, merged_output_path)            


