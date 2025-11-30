import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import math
import networkx as nx
from typing import List, Dict, Union
from MCTS.mcts_node import MCTSNode
from MCTS.mcts_utils import *
from models.model import request_qwen
from prompts.instructions import *
from utils.output_utils import *
from utils.math_utils import is_math_correct, res_normalize
from pathlib import Path


EXPLORATION_WEIGHT = 1.414 # UCT constant
NODE_LOWER_BOUND = 0.0 # threshold for selecting node 


class MCTS:
    def __init__(self, task_index, execute_round=10, exploration_weight=1.414, 
                child_nums=3, simulation_depth=5, ground_truth=0, 
                outputs_dir=None, root_dir=None, task_file=None):  
        self.outputs_dir = outputs_dir           # output directory
        self.root_dir = root_dir                 # root directory
        self.task_file = task_file               # task file
        self.task_index = task_index             # the json file describes the math task
        self.execute_round = execute_round
        self.exploration_weight = exploration_weight
        self.root = None
        self.graph = nx.Graph()
        self.node_labels = {}
        self.child_nums = child_nums             # number of children
        self.simulation_depth = simulation_depth # max simulate steps of rollout
        self.ground_truth = ground_truth         # groud truth of the problem
        self.max_edge = 0
        self.positive_samples = []  # store complete positive sample path
        self.negative_samples = []  # store complete negative sample path
        self.phase = "collect_positive"  # phase mark
        self.required_pos = 3        # required number of positive samples
        self.required_neg = 3        # required number of negative samples

    def get_file_path(self, _suffix) -> str:
        prefix_path = Path(self.task_file)
        file_path =f"{self.task_index}{_suffix}{prefix_path.suffix}" 
        task_dir = self.outputs_dir / str(self.task_index)
        task_dir.mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(task_dir, file_path)
        return output_file


    def select(self, node: MCTSNode) -> MCTSNode:
        if not node.children:
            return node
        if self.is_terminal(node):
            return node
        current = node
        while current.isFullyExpanded:
            best_child = random.choice(list(node.children.values()))                 
            if not best_child.children or self.is_terminal(best_child):
                return best_child
            current = best_child
            return current    


    def expand(self, node: MCTSNode) -> MCTSNode:   
        actions = self.generate_next_actions(node.state, sample_num=self.child_nums)
        simulate_node = None
        if not node.isFullyExpanded:
            for i, action in enumerate(actions):
                new_state = self.update_state(node.state, action)
                child = MCTSNode(new_state, parent=node)
                child.edge = self.max_edge + 1
                self.max_edge = child.edge
                child.value = 0.0                                                      
                child.depth = node.depth + 1
                node.append_children(child)
                self.graph.add_edge(node.id, child.id)
                self.node_labels[child.id] = f"Visits: {child.visits}\nValue: {child.value:.2f}"
        max_value = float("-inf")
        for child in node.children.values():
            if child.is_simulated:
                continue
            if child.value > max_value and not child.is_simulated:
                max_value = child.value
                simulate_node = child
        if simulate_node is None:
            return node

        simulate_node.is_simulated = True
        if len(node.children) >= 3:
            node.isFullyExpanded = True
        return simulate_node


    def simulate(self, node: MCTSNode) -> None:
        rollout_ans = []
        simulate_path = [node]
        path_value = [node.value]
        previous_node = node
        state = node.state
        new_state = state 
        simulation_branch = []
        new_node = node
        for step in range(self.simulation_depth):
            if self.is_terminal(new_node):
                break             
            next_action = self.generate_next_actions(new_state, sample_num=1)[0]             
            new_state = self.update_state(new_state, next_action)
            new_node = MCTSNode(new_state, parent=previous_node)

            new_node.is_virtual = True  # mark as virtual node
            new_node.edge = f"sim_{step+1}"  # edge in simulation, represents exploration order

            simulate_path.append(new_node)
            new_node.value = self.step_evaluate(previous_node, new_node)
            simulation_branch.append(new_node)   # update simulation branch
            previous_node = new_node
            path_value += [new_node.value]

        node.simulation_branch = simulation_branch
        final_answer_value = self.evaluate_final_state(new_node.state)  
        path_value[-1] = final_answer_value
        if final_answer_value == -1.0:
            path_label = "negative"
        elif final_answer_value == 1.0:
            path_label = "positive"

        return (simulate_path, path_value, path_label)


    def backpropagate(self, path: list[MCTSNode], path_value: list):
        cur_node = path[0]
        final_answer_score = path_value[-1]
        if path_value:
            path_value.pop(-1) 
        if path_value:
            path_value.pop(0)                  
        N = len(path_value)
        V = 0
        delta = ["-inf"] * N
        if N >= 1:
            factor = 1 / N                    
            for i in range(0, N):
                V += factor  * path_value[i] * (N - i) / N                  
        else:
            factor = 1
        final_V = V + factor * final_answer_score
        cur_node.value += final_V
        node = cur_node
        dis = 1
        delay_weight = 0.9

        real_path = []
        while node.parent:            
            node = node.parent
            real_path.append(node)
            node.value += pow(delay_weight, dis) * final_V    
            node.visits += 1
            dis += 1
        real_path = real_path[::-1]
        full_path = real_path + path
        return full_path
            

    def search(self) -> list:
        if not self.root:
            root_file = os.path.join(self.root_dir, self.task_file)
            initial_state = read_json(root_file)
            if self.task_index:
                root_state = find_task(initial_state, self.task_index)
            else:
                root_state = initial_state[0]
            # initialize the root node
            self.root = MCTSNode(root_state)
            self.root.state = self.root.initial_root(root_state)

            # add the root node to the graph
            self.graph.add_node(self.root.id)
            self.node_labels[self.root.id] = f"Root\nVisits: 0\nValue: 0.00"
            if not hasattr(self.root, "edge"):
                self.root.edge = 0
            self.max_edge = self.root.edge
        
        current_node = self.root
        current_round = 0  
        while not self.stop_condition(current_round):
            selected_node = self.select(current_node)
            if self.is_terminal(selected_node):
                real_path = []
                label = 'positive' if self.evaluate_final_state(selected_node.state) == 1.0 else 'negative'
                node = selected_node
                while node is not None:
                    real_path.append(node)
                    node = node.parent
                real_path = real_path[::-1]  

                sample = {
                    "path": [self.node_to_dict(n) for n in real_path],
                    "branch": "real",
                }
                if label == 'positive' and len(self.positive_samples) < self.required_pos:
                    self.positive_samples.append(sample)
                elif label == 'negative' and len(self.negative_samples) < self.required_neg:
                    self.negative_samples.append(sample)
                current_round += 1
                continue

            expanded_node = self.expand(selected_node)
            simulate_path, path_value, path_label = self.simulate(expanded_node)
            full_path = self.backpropagate(simulate_path, path_value)

            if self.is_terminal(simulate_path[-1]):

                sample = {
                    "path": [self.node_to_dict(n) for n in full_path],
                    "branch": "simulation",  
                }
                if path_label == "positive" and len(self.positive_samples) < self.required_pos:
                    self.positive_samples.append(sample)
                elif path_label == "negative" and len(self.negative_samples) < self.required_neg:
                    self.negative_samples.append(sample)

            if self.phase == "collect_positive" and len(self.positive_samples) >= self.required_pos:
                self.phase = "collect_negative"
            current_round += 1
        self.save_samples()

        best_path = self.find_best_path(self.root)
        self.save_tree_as_json()
        return best_path


    def node_to_dict(self, node: MCTSNode) -> dict:
        return {
            "id": node.id,
            "value": node.value,
            "state": node.state.__dict__.copy() if hasattr(node.state, '__dict__') else node.state,
            "reflection": getattr(node, "reflection", None),
            "terminate_node": getattr(node, "terminate_node", False),
            "edge": getattr(node, "edge", None),
            "on_final_route": getattr(node, "on_final_route", False),
            "is_virtual": getattr(node, "is_virtual", False)
        }


    def save_samples(self):
        output_dir = Path(self.outputs_dir) / "training_samples"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_file = output_dir / f"{self.task_index}.json"
        data = {
            "positive": self.positive_samples,
            "negative": self.negative_samples
        }
        with open(all_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


    def getBestChild(self, node: MCTSNode)  -> MCTSNode:
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            if self.phase == "collect_positive":
                exploitation = child.value  
            else:
                exploitation = -child.value
            exploration = self.exploration_weight * math.sqrt(math.log(node.visits + 1) / (child.visits + 1))  
            uct_value = exploitation + exploration
            if uct_value > bestValue:
                bestValue = uct_value
                bestNodes = [child]
            elif uct_value == bestValue:
                bestNodes.append(child)
        if not bestNodes:
            return node                
        return random.choice(bestNodes)


    def generate_next_actions(self, state: State, sample_num: int) -> List[str]: 
        conditions = state.conditions
        global_obj = state.global_objective
        previous_steps = state.steps
        prompt = generate_step_prompt.format(conditions=conditions, global_objective=global_obj, previous_steps=previous_steps)
        next_actions = request_qwen(generate_step_instruction, prompt, n=sample_num, temperature=1.0)
        return next_actions


    def update_state(self, state: State, action) -> State: 
        global_objective = state.global_objective
        step_obj, step_ans = parse_next_step(action)
        context = step_obj + '\n' + step_ans
        obj = state.global_objective
        steps = add_step(state, step_obj, step_ans)
        result = extract_result(step_ans)
        query = get_wa_query(state.conditions, step_obj)                
        content = step_obj + '\n' + step_ans
        conditions = update_conditions(state, step_ans)
        wa_history = update_wa_history(state, query)
        
        return State(
            global_objective=global_objective, 
            conditions=conditions, 
            step_objective=step_obj, 
            step_ans=step_ans, 
            result=result,
            steps=steps,
            query=query,
            wa_history=wa_history
            )


    def is_terminal(self, obj: MCTSNode) -> bool:
        if obj.state is None:
            return False
        action = obj.state.step_ans.strip()
        llm_answer = extract_result(action)
        result = False    
        if '<end>' in action:
            return True
        if is_math_correct(llm_answer, self.ground_truth):
            return True
        if obj.parent:
            last_node = obj.parent
            last_action = last_node.state.step_ans.strip()
            last_llm_answer = extract_result(last_action)
            if is_math_correct(last_llm_answer, llm_answer):
                return True 
        obj.terminate_node = result       
        return False


    def stop_condition(self, current_round):
        return (len(self.positive_samples) >= self.required_pos and
                len(self.negative_samples) >= self.required_neg) or \
            (current_round >= self.execute_round)
    

    def step_evaluate(self, previous_node: MCTSNode, node: MCTSNode):
        state = node.state
        step_obj = state.step_objective
        condition = state.conditions
        llm_answer = state.step_ans
        wa_answer = state.wa_history[-1]  
        res = 0         
        count = 5
        while res == 0 and count > 0:
            result, reason = llm_verify(step_obj, condition, llm_answer, wa_answer)
            if "True" in result or "true" in result:
                res = 1.0
            elif "False" in result or "false" in result:
                res = -1.0
            count -= 1

        node.value = res
        node.reflection = f'[{result}]' + reason       
        return node.value


    def evaluate_final_state(self, state: State) -> float:
        result =  extract_result(state.step_ans)
        final_answer_score = 0.0

        if is_math_correct(result, self.ground_truth): 
            final_answer_score = 1.0

        elif not is_math_correct(result, self.ground_truth): 
            final_answer_score = -1.0
        return final_answer_score
    

    def find_best_path(self, root: State):
        def path_average_value(path):
            return sum(node.value for node in path) / len(path)

        def dfs(node: MCTSNode, path: list):
            path.append(node)
            if not node.children:  
                best_path = path.copy()
            else:
                best_path = None
                best_avg = float('-inf')
                for child in node.children.values():
                    candidate_path = dfs(child, path)
                    candidate_avg = path_average_value(candidate_path)
                    if candidate_avg > best_avg:
                        best_avg = candidate_avg
                        best_path = candidate_path
            path.pop()
            return best_path
        best_path = dfs(root, [])
        for node in best_path:
            node.on_final_route = True
        return best_path    


    def build_tree_dict(self, node: MCTSNode) -> dict:
        node_dict = {
            "id": node.id,
            "value": node.value,
            "visits": node.visits,
            "state": node.state.__dict__,  
            "reflection": node.reflection,
            "terminate_node": getattr(node, "terminate_node", False),
            "edge": getattr(node, "edge", None),
            "on_final_route": getattr(node, "on_final_route", False),
            "is_virtual": getattr(node, "is_virtual", False),
            "children": [],
            "simulation_branch": []
        }
        for child in node.children.values():
            child_dict = self.build_tree_dict(child)
            node_dict["children"].append(child_dict)
        if hasattr(node, "simulation_branch"):
            for sim_node in node.simulation_branch:
                sim_node_dict = self.build_tree_dict(sim_node)
                node_dict["simulation_branch"].append(sim_node_dict)
        return node_dict


    def save_tree_as_json(self):
        output_dir = Path(self.outputs_dir) / "mcts_tree"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(output_dir / f"{self.task_index}.json")
        tree_data = self.build_tree_dict(self.root)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(tree_data, f, ensure_ascii=False, indent=4)


def run(task_index, execute_round, exploration_weight, child_nums, simulation_depth, ground_truth, outputs_dir, root_dir, task_file):
    mcts = MCTS(
        task_index=task_index,
        execute_round=execute_round,
        exploration_weight=exploration_weight,
        child_nums=child_nums,
        simulation_depth=simulation_depth,
        ground_truth=ground_truth,
        outputs_dir=outputs_dir,
        root_dir=root_dir,
        task_file=task_file
    )
    final_solution = mcts.search()
    return final_solution
