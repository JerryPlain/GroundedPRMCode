import os
import json
import logging
from prompts.instructions import extract_instruction, extract_prompt
from models.model import request_qwen
from utils.output_utils import save_json_file, parse_conditions

logger = logging.getLogger(__name__)


def generate_root_state(file_path: str, output_path: str, max_files: int = None):
    json_files = []
    for root, _, files in os.walk(file_path): # all task files are in this folder
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    
    if max_files is not None:
        json_files = json_files[:max_files]
    
    processed_files = set()
    existing_data = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
                for item in existing_data:
                    if "path" in item:
                        processed_files.add(item["path"].split('_')[-1])  
            except json.JSONDecodeError:
                logger.warning(f"Existing file {output_path} is not a valid JSON. It will be overwritten.")
    
    total_files = len(json_files)
    
    if existing_data:
        all_data = existing_data
    else:
        all_data = []
    for json_file_path in json_files:
        file_name = os.path.basename(json_file_path)
        # check if tasks already processed
        if file_name in processed_files:
            continue
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                question = data["problem"]
                prompt = extract_prompt.format(question=question)
                json_response = request_qwen(extract_instruction, prompt, 1, 0.1)[0]
                json_response = json.loads(json_response)
                conditions = json_response["conditions"]
                condition_list = parse_conditions(conditions)
                if "global_objective" not in json_response:
                    logger.warning(f"No global_objective found in {json_file_path}")
                    continue
                global_objective = json_response["global_objective"]
                # construct the root state entry
                entry = {
                    "question": question,
                    "global_objective": global_objective,
                    "conditions": condition_list,
                    "path": data["type"].replace(' ', '_').lower() + '_' + file_name,
                    "level": int(data["level"][-1]),
                    "solution": data["solution"]
                }
                all_data.append(entry)
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.error(f"Error processing file {json_file_path}: {e}")
            continue

    if all_data:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate root states from MATH dataset problems"
    )
    parser.add_argument(
        "--task_file_path",
        type=str,
        default="math/train",
        help="Path to the directory containing MATH dataset JSON files"
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        default="outputs/root/root.json",
        help="Output path for the generated root states JSON file"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process (None for all files)"
    )
    args = parser.parse_args()
    
    logger.info(f"Processing tasks from: {args.task_file_path}")
    logger.info(f"Output will be saved to: {args.output_file_path}")
    if args.max_files:
        logger.info(f"Processing first {args.max_files} files only")
    
    generate_root_state(args.task_file_path, args.output_file_path, max_files=args.max_files)
    logger.info("Successfully generated root states!")
