import json
import os
import re
from typing import List, Dict

def remove_root(data):
    for item in data:
        item['steps'] = item['steps'][1:]
    return data


def check_boxed_answer(content: str) -> bool:
    if "\\boxed{" in content:
        pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
        match = re.findall(pattern, content)
        if match:
            return True
        else:
            pattern2 = r"\$([^$]*)\$"
            match2 = re.findall(pattern2, content)
            if match2:
                return True
    elif "oxed{" in content:
        pattern = r"oxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
        match = re.findall(pattern, content)
        if match:
            return True
    return False


def filter_and_fix_answers(input_file: str, output_file: str):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    filtered_data = []
    
    for item in data:
        if not item.get('steps'):
            continue
        last_step = item['steps'][-1]
        if not check_boxed_answer(last_step.get('content', '')):
            continue
        filtered_data.append(item)
    filtered_data = remove_root(filtered_data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)


def filter_unexpected_answer(data):
    filtered_data = []
    filtered_count = 0
    
    for item in data:
        should_filter = False
        for step in item['steps']:
            reflection = step.get('reflection', '')
            correction = step.get('correction', {})
            _input = ''
            if isinstance(correction, dict):
                _input = correction.get('Input', '')
            elif isinstance(correction, str):
                _input = correction            
            if 'many retrie' in reflection.lower() or 'many retrie' in _input.lower() or isinstance(correction, str):
                should_filter = True
                break
        if not should_filter:
            filtered_data.append(item)
        else:
            filtered_count += 1
    
    return filtered_data, filtered_count


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(current_dir)
    
    input_file = os.path.join(pipeline_dir, "data", "synthetic_data", "syn_data.json")
    output_file = os.path.join(pipeline_dir, "data", "filtered_data", "fixed_data.json")
    
    filter_and_fix_answers(input_file, output_file)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    with open(output_file, 'r', encoding='utf-8') as f:
        filtered_data = json.load(f)


if __name__ == "__main__":
    main()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(current_dir)
    
    input_file = os.path.join(pipeline_dir, "data", "filtered_data", "fixed_data.json")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    filtered_dataset, filtered_count = filter_unexpected_answer(data)

    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_dataset, f, indent=2, ensure_ascii=False)
