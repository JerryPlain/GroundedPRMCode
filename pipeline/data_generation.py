import os
import json
from pathlib import Path
from MCTS.mcts import run
from utils.math_utils import parse_math_boxed
from utils.output_utils import save_json_file, parse_conditions
from prompts.instructions import extract_prompt, extract_instruction
from models.model import request_qwen
import concurrent.futures
from typing import List, Dict, Any
import traceback
import argparse


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

def get_relative_path(*path_parts) -> str:
    return str(get_project_root().joinpath(*path_parts))

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_state_as_json(_id, state, file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([], f)    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)    
    data.append({"id": _id, "state": state})    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def process_single_task(task: Dict[str, Any], task_index: str, execute_round: int = 8, 
                       exploration_weight: float = 1.8, child_nums: int = 3, simulation_depth: int = 7,
                       outputs_dir: str = None, root_dir: str = None, task_file: str = None) -> Dict[str, Any]:
    try:
        level = task.get("level", 3)
        if level <= 3:
            execute_round = 8
            child_nums = 3
            simulation_depth = 5
        else:
            execute_round = 12
            child_nums = 3
            simulation_depth = 9
        solution = task["solution"]
        ground_truth = parse_math_boxed(solution)
        run(task_index, execute_round, exploration_weight, 
            child_nums, simulation_depth, ground_truth, 
            outputs_dir, root_dir, task_file)
        
        return {
            "index": task_index,
            "status": "success",
            "path": task["path"].split(".")[0]
        }
    except Exception as e:
        return {
            "index": task_index,
            "status": "failed",
            "path": task["path"].split(".")[0],
            "error": str(e)
        }

def main(task_file: str, start_index: int = 0, end_index: int = None, max_workers: int = 10, 
                        outputs_dir: str = None, root_dir: str = None) -> None:
    if outputs_dir is not None:
        os.makedirs(outputs_dir, exist_ok=True)
    if root_dir is not None:
        os.makedirs(root_dir, exist_ok=True)

    if root_dir is None:
        dir = Path(__file__).resolve().parent / "outputs/root"
        os.makedirs(dir, exist_ok=True)
    else:
        dir = Path(root_dir)
    task_path = os.path.join(dir, task_file)
    tasks = read_json(task_path)
    if end_index is None:
        end_index = len(tasks)
    tasks_to_process = tasks[start_index:end_index]

    tree_dir = Path(outputs_dir) / "mcts_tree"  
    os.makedirs(tree_dir, exist_ok=True)  

    filtered_tasks = []
    skipped_tasks = []
    for task in tasks_to_process:
        output_filename = f"{task['path'].split('.')[0]}.json"
        output_filepath = tree_dir / output_filename
        if output_filepath.exists():
            skipped_tasks.append(output_filename)
            continue
        filtered_tasks.append(task)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(process_single_task, task=task, task_index=task["path"].split(".")[0], outputs_dir=outputs_dir, root_dir=root_dir, task_file=task_file): task 
            for task in filtered_tasks
        }
    
        failed_tasks = []
        for future in concurrent.futures.as_completed(future_to_task):
            result = future.result()
            if result["status"] == "failed":
                failed_tasks.append(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="process math tasks")
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--task_file', type=str, required=True)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=None)
    parser.add_argument('--max_workers', type=int, default=15)
    args = parser.parse_args()
    
    main(
        task_file=args.task_file,
        start_index=args.start_index,
        end_index=args.end_index,
        max_workers=args.max_workers,
        outputs_dir=args.outputs_dir,
        root_dir=args.root_dir
    )

