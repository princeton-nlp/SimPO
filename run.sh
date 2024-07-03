#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=512G
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --partition=pli-c
#SBATCH --output=/scratch/gpfs/mengzhou/space17/out/slurm/%x-%j.out 
#SBATCH --err=/scratch/gpfs/mengzhou/space17/out/slurm/%x-%j.err

conda activate handbook

cd $n/space17/SimPO

seed=${1:-1}
output_dir=$n/space17/out/simpo_seed${seed}
mkdir -p $output_dir

# 4 gpus
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simpo.py training_configs/llama-3-8b-instruct-simpo.yaml --seed=$seed --output_dir=$output_dir 