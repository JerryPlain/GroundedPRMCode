import os
import random
import json

def sample_conversation_data(input_path, output_path, sample_size=20000, seed=42):
    random.seed(seed)
    with open(input_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]
    
    if sample_size > len(all_data):
        sampled = all_data
    else:
        sampled = random.sample(all_data, sample_size)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_path = "data/conversation_data.jsonl"
    output_path = "data/conversation_data_sampled.jsonl"
    sample_conversation_data(
        input_path=input_path,
        output_path=output_path,
        sample_size=20000,
        seed=42
    )