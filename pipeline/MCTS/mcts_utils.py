import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import time
import math
import random
import numpy
from functools import partial
import copy
from typing import List, Dict, Optional, Union
import json
from utils.output_utils import extract_result, extract_query, fix_wa_ans, fix_json, extract_string, validate_json_string, fix_deepseek_ans
from prompts.instructions import generate_query_prompt, code_instruction, verify_prompt, verify_instruction
from models.model import request_deepseek


error_list = [
    "doesn't understand your query",
    "wasn't able to answer it",
    "Error",
    "not present in"
]

class State:
    def __init__(
            self, 
            global_objective: str = "",
            conditions: List[str] = None, 
            steps: List[Dict[str, str]] = None, 
            step_objective: str = "",
            step_ans: str = "",
            result: str = "",
            query: str = "",
            wa_history: List[str] = None
        ):
        """
        Initialize the state for a node in MCTS.

        Args:
            global_objective (str): The description of the problem objective.
            conditions (List[str]): The known conditions extracted from the question or the result from the excuted steps
            steps (List[Dict[str, str]]): A set of dictionary of steps executed so far.
            step_objective (str): The local objective of the current step
            step_ans (str): The detailed reasoning and computation process of the current step
            query (str): A single wa query of the current step, prompted by step objective
            wa_history (List[str]): A list of wa returns (intermediate steps)
        """
        self.global_objective = global_objective
        self.conditions = conditions
        self.steps = steps
        self.step_objective = step_objective
        self.step_ans = step_ans
        self.result = result
        self.query = query
        self.wa_history = wa_history


    def __str__(self) -> str:
        global_obj = self.global_objective
        conditions = self.conditions
        step_obj = self.step_objective
        step_ans = self.step_ans
        steps = self.steps
        result = self.result
        query = self.query
        wa_history = self.wa_history
        state = {
            "global_obj": global_obj,
            "conditions": conditions,
            "previous_steps": steps,
            "step_obj": step_obj,
            "actions": step_ans,
            "result": result,
            "query": query,
            "wa_history": wa_history            
        }
        return json.dumps(state, indent=4)


def add_step(state: State, step_obj: str, step_ans: str):
    previous_steps = {
        "step_objective": step_obj,
        "action": step_ans
    }
    updated_steps = state.steps + [previous_steps]
    return updated_steps

def update_conditions(conditions: list, step_ans: str):
    new_condition = extract_result(step_ans)
    updated_conditions = conditions + [new_condition]
    return updated_conditions

def update_wa_history(state: State, query):
    if query == "Cannot generate a valid query":
        intermediate = "An Error occurred, Wolfram Alpha can not solve this problem."
        updated_wa_history = state.wa_history + [intermediate]        
        return updated_wa_history
    
    intermediate = fix_wa_ans(query)
    if intermediate:
        updated_wa_history = state.wa_history + [intermediate]
    else:
        intermediate = "An Error occurred, Wolfram Alpha can not recognize the query."
        updated_wa_history = state.wa_history + [intermediate]
    return updated_wa_history

def save_state_as_json(id: str, state: State, file_path: str):
    """
    In this json file there is only a completed solutions for a specific math problem.
    Each dict represents the state info of a node.
    ID 0.0 means the parent node has id 0, and the current node is the first child of its parent,
    the number of digits represents the depth of the tree.
    """
    global_obj = state.global_objective
    conditions = state.conditions
    previous_steps = state.steps
    step_obj = state.step_objective
    step_ans = state.step_ans
    result = state.result
    query = state.query
    wa_history = state.wa_history
    state_dict = {
        "id": id,
        "global_obj": global_obj,
        "conditions": conditions,
        "previous_steps": previous_steps,
        "step_obj": step_obj,
        "action": step_ans,
        "result": result,
        "query": query,
        "wa_history": wa_history
    }

    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump([], file, indent=4, ensure_ascii=False)
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    data.append(state_dict)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)   

def parse_verification(s):
    if validate_json_string(s):
        if isinstance(s, str):
            s = json.loads(s)
        result = s["result"]
        reason = s["reason"]  

    else:
        res, fixed_str = fix_json(s)
        if res:
            result = fixed_str["result"]
            reason = fixed_str["reason"]  
        else:
            s_clean = fixed_str.strip().strip('{}').strip()
            pattern = r'"reason": "(.*?)",\s*"result": "(.*?)"'
            match = re.search(pattern, s_clean, re.DOTALL)
            if match:
                reason = match.group(1)   
                result = match.group(2)

            else:
                reason, result = extract_string(s_clean, '"reason": ', '"result": ')
                reason = reason.strip(':",\' \n\t')
                result = result.strip(':",\' \n\t') 
    return result, reason            

def llm_verify(step_obj, condition, llm_ans, wa_ans):
    prompt = verify_prompt.format(step_objective=step_obj, conditions=condition, llm_answer=llm_ans, wa_answer=wa_ans)
           
    instruction = verify_instruction
    eval_ans = request_deepseek(instruction, prompt, 1, 0.6)[0]    
    think, json_str = fix_deepseek_ans(eval_ans) 
    result, reason = parse_verification(json_str)
    reason = '<think>\n' + think + '\n</think>\n' + reason

    return result, reason
