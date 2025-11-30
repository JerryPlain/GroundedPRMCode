#!/bin/bash
#SBATCH --job-name=task
#SBATCH --output=logs/data_gen_output_%j.log
#SBATCH --error=logs/data_gen_error_%j.log
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH --qos=mcml
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00          



source ~/.bashrc
conda activate groundedprm


mkdir -p logs

python ../data_generation.py \
  --outputs_dir ../outputs/state_trace/counting_and_probability \
  --root_dir ../outputs/root \
  --task_file example_algebra.json \
  --start_index 0 \
  --end_index 10 \
  --max_workers 40