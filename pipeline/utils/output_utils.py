import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import json5
import logging
import regex as re
from pathlib import Path
from typing import Dict, List, Optional
from camel.toolkits import SearchToolkit
from models.model import request_deepseek
from prompts.instructions import generate_query_prompt, code_instruction

logger = logging.getLogger(__name__)


def save_json_file(question, json_response, file_path):
    json_response = fix_latex_symb(json_response)
    json_response = json.loads(json_response)
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump([], file)           
    conditions = json_response["conditions"]
    condition_list = parse_conditions(conditions)
    global_objective = json_response["global_objective"]
    entry = {
        "question": question,
        "global_objective": global_objective,
        "conditions": condition_list
    }
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Append the new entry to the data
    data.append(entry)

    # Write the updated data back to the file
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def read_json(file_path: str) -> dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        logger.error(f"Error: Cannot find the file - {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error: JSON decode failed - {str(e)}")
        raise


def dump_json(source, datas):
    if not os.path.exists(source):
        return "cannot find the json file path"
    with open(source, 'w', encoding='utf-8') as f:
        for item in datas:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')   


def find_task(task_list, task_index):
    for task in task_list:
        if "path" in task.keys():
            path = task["path"].split(".")[0]
            if path == task_index:
                return task
        else:
            _id = task["id"]
            if _id == task_index:
                return task


def parse_conditions(conditions):
    """
    Parse the conditions string into a list of individual conditions.

    Parameters:
        conditions (str or list): The conditions string or list.

    Returns:
        List[str]: A list of parsed conditions.
    """
    condition_list = []
    
    if isinstance(conditions, str):
        conditions = conditions.replace("conditions:", "").strip()
        condition_parts = re.split(r'(?:condition\s*\d+:|,\s*condition\s*\d+:)', conditions)        
        for part in condition_parts:
            part = part.strip()
            if part:
                condition_list.append(part)
                
    elif isinstance(conditions, list):
        for condition in conditions:
            if isinstance(condition, str):
                condition = re.sub(r'condition\s*\d+:', '', condition).strip()
                if condition:
                    condition_list.append(condition)                    
    return condition_list


def validate_json_string(s):
    try:
        json.loads(s) 
        return True
    except json.JSONDecodeError as e:
        return False    


def fix_json(json_str):
    def fix_unclosed_string_lines(s):
        pattern = r'(".*?)(\n)(?=\s*[}\]])'  
        return re.sub(pattern, r'\1"\2', s)

    json_str = fix_unclosed_string_lines(json_str)
    quote_count = json_str.count('"')
    if quote_count % 2 == 1:
        last_brace = json_str.rfind('}')
        if last_brace != -1:
            last_newline = json_str.rfind('\n', 0, last_brace)
            if last_newline != -1 and last_newline < last_brace:
                json_str = json_str[:last_newline+1] + '"' + json_str[last_newline+1:]

    try:
        obj = json5.loads(json_str)
        return True, obj
    except Exception as e:
        logger.debug(f"JSON fix failed: {e}")
        return False, json_str


def extract_json(next_step):
    if isinstance(next_step, str):
        next_step = json.loads(next_step)
    
    if "step objective" in next_step.keys():
        step_obj = next_step["step objective"] 
    elif "step_objective" in next_step.keys():       
        step_obj = next_step["step_objective"]
    if "action" in next_step.keys():
        step_ans = next_step["action"]
    elif "actions" in next_step.keys():
        step_ans = next_step["actions"]
    return step_obj, step_ans


def parse_next_step(next_step):
    if validate_json_string(next_step):
        step_obj, step_ans = extract_json(next_step)
    else:
        res, s = fix_json(next_step)
        if res:
            step_obj, step_ans = extract_json(s)
        else:
            s_clean = s.strip().strip('{}').strip()
            pattern = r'"step objective": "(.*?)",\s*"action": "(.*?)"'
            match = re.search(pattern, s_clean, re.DOTALL)
            
            if match:
                step_obj = match.group(1)   
                step_ans = match.group(2)
            else:
                step_obj, step_ans = extract_string(s_clean, '"step objective": ', '"action": ')
                step_obj = step_obj.strip(':",\' \n\t')
                step_ans = step_ans.strip(':",\' \n\t')
    return step_obj, step_ans


def extract_string(s, k1, k2):
    k1_start = s.find(k1)
    k1_start += len(k1)
    if k2:
        k2_start = s.find(k2, k1_start)
        step_obj = s[k1_start:k2_start].strip()
        step_ans = s[k2_start+len(k2):].strip()
        return step_obj, step_ans         
    else:
        step_obj = s[k1_start:-1].strip()
        return step_obj


def extract_result(step_ans: str):
    if "\\boxed{" in step_ans:
        pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
        match = re.findall(pattern, step_ans)
        if match:
            answer = match[-1] 
            return answer.strip('.')
        else:
            pattern2 = r"\$([^$]*)\$"
            match2 = re.findall(pattern2, step_ans)
            if match2:
                answer = match2[-1]
                return answer.strip('.')
            else:
                return "Cannot find the boxed final result in the answer"
    elif "oxed{" in step_ans:
        pattern = r"oxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
        match = re.findall(pattern, step_ans)
        if match:
            answer = match[-1] 
            return answer.strip('.')
        else:
            pattern2 = r"\$([^$]*)\$"
            match2 = re.findall(pattern2, step_ans)
            if match2:
                answer = match2[-1]
                return answer.strip('.')
            else:
                return "Cannot find the boxed final result in the answer"
    else:
        return step_ans


def get_wa_code(obj, step_objective, conditions):
    wa_response = ""
    prompt = generate_query_prompt.format(global_objective=obj, conditions=conditions, step_objective=step_objective)
    try_count = 5
    repeat = False
    final_answer = None
    while (isinstance(wa_response, str) or not wa_response or not final_answer or repeat) and try_count > 0:
        query_str = request_deepseek(system_msg=code_instruction, prompt=prompt, n=1, temperature=0.6)

        if query_str[0] == "achieved max tokens":
            query = "Cannot generate a valid query"
            try_count -= 1 
            continue
        
        try:
            think, query_json = fix_deepseek_ans(query_str[0])
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse query: {query_str[0][:100]}...")
            try_count -= 1 
            query = "Cannot generate a valid query"
            continue

        try:
            query = extract_query(query_json)
            wa_response = fix_wa_ans(query)
            if isinstance(wa_response, dict):
                final_answer = wa_response["final_answer"]                
        except AttributeError as e:
            logger.warning(f"Error encountered: {e}. Generating a new query...")
        try_count -= 1       
    return query   


def fix_deepseek_ans(s: str):
    think = ""
    if '</think>' in s:
        think = s.split('</think>')[0].strip()
        s = s.split('</think>')[1].strip()

    patterns = [
        r"```json\s*(.*?)\s*```",  
        r"```(?:[a-z]*)\s*({.*?})\s*```",  
        r"({(?:[^{}]|(?R))*})"  
    ]

    match1 = re.search(patterns[0], s, re.DOTALL)
    if match1:
        json_str = match1.group(1).strip()
        return think, json_str
    else:
        match2 = re.search(patterns[1], s, re.DOTALL)
        if match2:
            json_str = match2.group(1).strip()
            return think, json_str
        else:
            match3 = re.search(patterns[2], s, re.DOTALL)
            if match3:
                json_str = match3.group(1).strip()
                return think, json_str
            else:
                return think, s


def extract_query(output: Dict[str, str]):
    if isinstance(output, list):
        output = output[0]
    if validate_json_string(output):
        json_str = json.loads(output)
        query = json_str["query"]
    else:
        res, s = fix_json(output)
        if res:
            query = output["query"]

        else:
            s_clean = s.strip().strip('{}').strip()
            pattern = r'"query": "(.*?)"'            
            match = re.search(pattern, s_clean, re.DOTALL)            
            if match:
                query = match.group(1)   
            else:
                query = extract_string(s_clean, '"query": ', None)
                query = query.strip(':",\' \n\t')
    return query


def fix_wa_ans(query):
    wa_ans = SearchToolkit().query_wolfram_alpha(query, is_detailed=True)
    if isinstance(wa_ans, dict):
        wa_response = {}
        wa_response["Input"] = query
        pod_infos = wa_ans["pod_info"]
        for info in pod_infos:
            if 'title' in info and 'description' in info:               
                wa_response[info['title']] = info['description']
        wa_response["intermediate_steps"] = wa_ans["steps"]       
        wa_response["final_answer"] = wa_ans["final_answer"]

        if "Result" in wa_response.keys() and wa_response["Result"]:
            wa_response["Result"] = wa_response["Result"].split("\n")

        elif "Results" in wa_response.keys() and wa_response["Results"]:
            wa_response["Results"] = wa_response["Results"].split("\n")

    elif isinstance(wa_ans, str):
        wa_response = wa_ans
    return wa_response 


def format_number(s):
    if isinstance(s, str):
        try:
            return f"{float(s):.4f}"  # Convert to float and format
        except ValueError:
            return s
    else:
        return s
    

def extract_wa_ans(wa_ans: dict):
    if isinstance(wa_ans, str):
        return wa_ans
    if not wa_ans["final_answer"]:
        final_answer = "An error occurred, Wolfram Alpha doesn't understand your query"
        return final_answer
    final_answer = {}
    final_answer["Input"] = wa_ans["Input"]
    if "Result" in wa_ans.keys():
        result = wa_ans["Result"]
        result = format_number(result)
        final_answer["Result"] = result
    if "Results" in wa_ans.keys():
        result = wa_ans["Results"]
        result = format_number(result)        
        final_answer["Results"] = result
    if "final_answer" in wa_ans.keys():
        answer = wa_ans["final_answer"]
        answer = format_number(answer)
        final_answer["final_answer"] = answer
    return final_answer


def fix_latex_symb(string: str):
    string = string.strip()
    string = string.replace('\\', '\\\\')

    string = string.replace("\n", "")  
    return string

