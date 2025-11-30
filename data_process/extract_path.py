import json
import os
import sys
import logging
from typing import List, Dict, Tuple
import re
from concurrent.futures import ProcessPoolExecutor, TimeoutError
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipeline'))
from pipeline.utils.math_utils import is_math_correct, parse_math_boxed

logger = logging.getLogger(__name__)

def load_metadata(meta_path: str) -> Dict[str, dict]:
    with open(meta_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    meta_dict = {}
    for task in tasks:
        key = task["path"].replace('.json', '')
        task["ground_truth"] = parse_math_boxed(task.get("solution", ""))
        meta_dict[key] = task
    return meta_dict

def load_tree(json_path: str) -> Dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

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

def extract_terminal_paths(tree_root, ground_truth):
    paths = []
    def dfs(node, current_path):
        current_path.append(node)

        if is_terminal(node, ground_truth):
            paths.append(current_path.copy())
            current_path.pop()
            return

        for child in node.get('children', []):
            child['parent'] = node  
            dfs(child, current_path)

        for sim_child in node.get('simulation_branch', []):
            sim_child['parent'] = node
            dfs(sim_child, current_path)
        current_path.pop()
    dfs(tree_root, [])
    return paths

def extract_paths_with_depth(tree_root, target_depth):
    paths = []
    def dfs(node, current_path):
        current_path.append(node)
        children = node.get('children', [])
        sim_children = node.get('simulation_branch', [])
        if not children and not sim_children:
            if len(current_path) >= target_depth:
                paths.append(current_path.copy())
            current_path.pop()
            return
        for child in children:
            dfs(child, current_path)
        current_path.pop()
    dfs(tree_root, [])
    return paths

def is_terminal(node, ground_truth):
    state = node.get('state', {})
    action = state.get('step_ans', '').strip()
    if not check_boxed_answer(action): 
        return False   
    else:
        llm_answer = parse_math_boxed(action)
        if '<end>' in action:
            return True
        if is_math_correct(llm_answer, ground_truth):
            return True 
    return False

def extract_all_complete_paths(tree_root, ground_truth):
    terminal_paths = extract_terminal_paths(tree_root, ground_truth)
    if not terminal_paths:
        return []
    avg_depth = round(sum(len(p) for p in terminal_paths) / len(terminal_paths))
    depth_paths = extract_paths_with_depth(tree_root, avg_depth)
    all_paths = terminal_paths.copy()
    seen = set(id(p[-1]) for p in terminal_paths)
    for p in depth_paths:
        if id(p[-1]) not in seen:
            all_paths.append(p)
            seen.add(id(p[-1]))
    return all_paths

def build_dataset(paths: List[List[dict]], task: dict) -> List[Dict]:
    dataset = []
    question = task["question"]
    gt_answer = task["solution"]
    idx = task["path"].replace('.json', '')
    gt = parse_math_boxed(gt_answer)
    for path in paths:
        final_state = path[-1]['state']
        result = final_state.get('result', '').strip('.')
        label = 'positive' if is_math_correct(result, gt_answer) else 'negative'
        steps = []
        for node in path:
            state = node['state']
            step_obj = state.get('step_objective', '')
            action = state.get('step_ans', '')
            content = step_obj + '\n' + action
            wa_answers = state.get('wa_history', [])
            correction = wa_answers[-1] if wa_answers else ""
            reflection = node.get('reflection', '')
            steps.append({
                "content": content,
                "step score": node.get('value', 0),
                "reflection": reflection,
                "correction": correction
            })
        dataset.append({
            "idx": idx,
            "question": question,
            "ground_truth": gt,
            "steps": steps,
            "label": label
        })
    return dataset

def process_single_file(meta_data: Dict[str, dict], tree_file: str) -> List[Dict]:
    tree_data = load_tree(tree_file)
    task_id = os.path.splitext(os.path.basename(tree_file))[0]
    task = meta_data.get(task_id)
    if not task:
        logger.warning(f"Cannot find the task {task_id} in meta_data")
        return []
    gt = task["ground_truth"]
    all_paths = extract_all_complete_paths(tree_data, gt)
    return build_dataset(all_paths, task)

def process_directory(meta_data: Dict[str, dict], tree_dir: str, start_idx: int = 0, end_idx: int = None) -> List[Dict]:
    dataset = []
    filenames = [f for f in os.listdir(tree_dir) if f.endswith('.json')]
    filenames.sort()  
    if end_idx is None:
        end_idx = len(filenames)
    filenames = filenames[start_idx:end_idx]
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = []
        for filename in filenames:
            file_path = os.path.join(tree_dir, filename)
            futures.append(
                executor.submit(process_single_file, meta_data, file_path)
            )
        for future, filename in zip(futures, filenames):
            try:
                file_dataset = future.result(timeout=90)
                dataset.extend(file_dataset)
            except TimeoutError:
                logger.warning(f"Process file {filename} timeout, skipped")
            except Exception as e:
                import traceback
                logger.error(f"Process file {filename} failed: {str(e)}")
                logger.debug(traceback.format_exc())
    return dataset

def save_dataset_append(output_json, new_data):
    if os.path.exists(output_json):
        with open(output_json, 'r', encoding='utf-8') as f:
            try:
                old_data = json.load(f)
            except Exception:
                old_data = []
    else:
        old_data = []    
    existing_indices = {item['idx'] for item in old_data}
    unique_new_data = [item for item in new_data if item['idx'] not in existing_indices]
    all_data = old_data + unique_new_data
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract training paths from MCTS tree data")
    parser.add_argument(
        "--meta_path", 
        type=str, 
        default="outputs/root/root.json",
        help="Path to the metadata JSON file containing task information"
    )
    parser.add_argument(
        "--tree_dir", 
        type=str, 
        default="outputs/state_trace/mcts_tree",
        help="Directory containing MCTS tree JSON files"
    )
    parser.add_argument(
        "--output_json", 
        type=str, 
        default="data/synthetic_data/syn_data.json",
        help="Output path for the generated synthetic dataset"
    )
    parser.add_argument(
        "--start_idx", 
        type=int, 
        default=0,
        help="Starting index for processing files"
    )
    parser.add_argument(
        "--end_idx", 
        type=int, 
        default=None,
        help="Ending index for processing files (None for all)"
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    
    meta_data = load_metadata(args.meta_path)
    dataset = process_directory(meta_data, args.tree_dir, start_idx=args.start_idx, end_idx=args.end_idx)
    save_dataset_append(args.output_json, dataset)
    logger.info(f"Successfully processed {len(dataset)} samples and saved to {args.output_json}")


if __name__ == "__main__":
    main()