
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import json
import logging
from tqdm import tqdm
from multiprocessing import Pool
from openai import OpenAI
from datasets import load_dataset
from openai import AsyncOpenAI
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import traceback
import argparse
import re

logger = logging.getLogger(__name__)

instruction = """Your task is to analyse and critique the reasoning step below of a math problem step-by-step.
Please follow the instructions and format below to respond:

1. Judge the logical and calculation correctness of the step by yourself.
2. Provide the judgement conclusion and the label as "correct" or "incorrect" in '<output>\\boxed{}</output>'.
3. Use the following format to respond:
<judge>
[details reasoning]
</judge>
<output>
According to the conclusion of the judgement, the label is: \\boxed{}
</output>

Here is the step:
"""

def extract_score(s):
    if '[Wrong]' in s:
        return -1.0
    elif '[Right]' in s:
        return 1.0

    boxed_pattern = r'\\boxed\{([^}]*)\}'    
    matches = re.findall(boxed_pattern, s)
    if matches:
        score_str = matches[-1].strip()
        if score_str == '+':
            return 1.0
        elif score_str == '-':
            return -1.0
    return -2.0

def single_process(d):
    from openai import OpenAI
    global api_url, model_name
    client = OpenAI(
        base_url=api_url,
        api_key="EMPTY",
    )
    try:
        steps = d['steps']
        messages = []
        messages.append({'role': 'system', 'content': instruction})

        for sdx, step in enumerate(steps):
            if sdx == 0:
                messages.append({'role': 'user', 'content': d['problem'] + '\n\n' + step})
            else:
                messages.append({'role': 'user', 'content': step})
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                n=1,
                temperature=0.6,
                max_tokens=4096,
            )
            response = completion.choices[0].message.content
            score = extract_score(response)  # extract score from \boxed{
            if float(score) <= 0:
                return sdx           
            if float(score) > 0:
                label = "+"
            else:
                label = "-"
            messages.append({'role': 'assistant', 'content': f"This step is: {label}"})
        return -1
    except Exception as e:
        logger.error(f"Error in single_process: {traceback.format_exc()}")
        return [-2, f"[ERROR] {type(e).__name__}: {e}"]

def main(api_url_arg, output_prefix, desc, model_name_arg):
    global api_url, model_name
    api_url = api_url_arg
    model_name = model_name_arg

    if model_name == 'qwen2.5-7b-instruct' or model_name == 'qwen2.5-math-7b-instruct':
        dir_name = 'base'
    else:
        dir_name = model_name

    os.makedirs(f'outputs/{dir_name}', exist_ok=True)
    configs = ['gsm8k', 'math', 'olympiadbench', 'omnimath'] 
    for config in configs:
        input_data = load_dataset('Qwen/ProcessBench', split=config)
        from functools import partial
        with Pool(8) as p:
            predictions = list(tqdm(p.imap(single_process, input_data), total=len(input_data),
                                desc=f'Processing {config}', dynamic_ncols=True))
        res_data = []
        for idx, d in enumerate(input_data):
            new_d = d.copy()
            new_d['prediction'] = predictions[idx]
            new_d['match'] = predictions[idx] == d['label']
            res_data.append(new_d)
        
        data1 = [e for e in res_data if e['label'] != -1]
        data2 = [e for e in res_data if e['label'] == -1]
        with open(f'outputs/{dir_name}/error_{config}_{output_prefix}.jsonl', 'w') as f:
            for e in data1:
                f.write(json.dumps(e) + '\n')
        with open(f'outputs/{dir_name}/correct_{config}_{output_prefix}.jsonl', 'w') as f:
            for e in data2:
                f.write(json.dumps(e) + '\n')
        count = 0
        for e in data1:
            predict = e['prediction']
            if predict == -1:
                count += 1

        acc1 = np.mean([e['match'] for e in data1]) * 100
        acc2 = np.mean([e['match'] for e in data2]) * 100
        f1 = 2 * acc1 * acc2 / (acc1 + acc2) if (acc1 + acc2) > 0 else 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_url', type=str, default='base url')
    parser.add_argument('--model_name', type=str, default='qwen-7b')
    parser.add_argument('--output_prefix', type=str, default='ref_v1.3_3')
    parser.add_argument('--desc', type=str, default='trained with ref, convasation, binary data, tested on math')
    args = parser.parse_args()

    main(api_url_arg=args.api_url, output_prefix=args.output_prefix, desc=args.desc, model_name_arg=args.model_name)