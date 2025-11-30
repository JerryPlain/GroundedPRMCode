#!/bin/bash    

mkdir -p logs
source ~/.bashrc
conda activate GroundedPRM

# NOTE: Update this path if your script location differs
# cd src/evaluation/reward_guided_search/

dataset_list=("amc23" "aime24" "math" "college_math" "minerva_math" "olympiadbench")

echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"

# NOTE: Update these paths to your model directories
model_dir="${MODEL_DIR:-/baseline}"
model_name="Grounded-PRM"
reward_tokenizer_path=$model_dir/$model_name

# API configuration for deployed models
node_name="host"
policy_port=8000
reward_port=8001
policy_api_base="http://$node_name:$policy_port/v1"
reward_api_base="http://$node_name:$reward_port/v1"
policy_api_key="EMPTY"
reward_api_key="EMPTY"
policy_model_name="policy-model"
reward_model_name="reward-model"

echo "Using deployed models via API:"
echo "  Policy API: $policy_api_base"
echo "  Reward API: $reward_api_base"

# Performing Search and Generating Responses
for dataset in "${dataset_list[@]}"; do
    echo "Dataset: $dataset"
    python Greedy-Search.py \
        --policy_api_base "$policy_api_base" \
        --reward_api_base "$reward_api_base" \
        --policy_api_key "$policy_api_key" \
        --reward_api_key "$reward_api_key" \
        --policy_model_name "$policy_model_name" \
        --reward_model_name "$reward_model_name" \
        --reward_tokenizer_path "$reward_tokenizer_path" \
        --data $dataset \
        --output_dir outputs/greedy_search/prm/$dataset \
        --temperature 1.0 
        
done

# Evaluating Responses
python eval_results.py \
    --results_dir outputs/greedy_search/prm