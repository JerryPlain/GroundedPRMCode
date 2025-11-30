import copy
import numpy as np
from .mcts_utils import State
from typing import Dict, Union, List
import random
import string


class MCTSNode(object):
    def __init__(self, state: State, parent=None, depth=0):
        self.state = state               # state of the node
        self.parent = parent             # parent node
        self.visits = 0                  # visits of the node
        self.value = 0.0                 # value of the node
        self.children = {}               # children nodes, type: dict{str: MCTSNode}
        self.id = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        self.depth = depth               # depth of the node
        self.isFullyExpanded = False     # whether the node is fully expanded
        self.final_ans_flag = 0
        self.reflection = ''             # explaination of wa verification
        self.isTerminal = False
        self.on_final_route = False      
        self.is_simulated = False        # ensure no error in MCTS
        self.edge = 0                    # edge information for recording exploration order
        self.is_virtual = False          # whether the node is virtual (simulated)
        self.simulation_branch = []      # save virtual nodes generated in rollout
        self.wa_gt = False               # whether the wa is consistent with gt
        self.terminate_node = False
        self.branches = []               # save all branches (including real and simulated)

    def __str__(self):  
        return f"Node {self.id}: Visits={self.visits}, Value={self.value}, \nPossibleActions: \n{self.children.keys()}"

    def append_children(self, child: "MCTSNode"): 
        node_info = child.state.step_objective
        self.children.update({node_info: child})
        return self
    
    def initial_root(self, items: Dict[str, Union[str, List[str]]]) -> State:
        global_obj = items["global_objective"]
        conditions = items["conditions"]
        steps = []
        step_obj = ""
        step_ans = ""
        result = ""
        query = ""
        wa_history = []
        root = State(
            global_objective=global_obj,
            conditions=conditions,
            steps=steps,
            step_objective=step_obj,
            step_ans=step_ans,
            result=result,
            query=query, 
            wa_history=wa_history
            )
        return root  
