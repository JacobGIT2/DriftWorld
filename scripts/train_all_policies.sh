#!/bin/bash
# Train PPO policies for all 3 world models sequentially.
# Uses combined reward: latent cosine similarity + physical EE-to-object distance.
#
# Usage (SSH-safe):
#   nohup bash scripts/train_all_policies.sh > train_policies.log 2>&1 &
#   tail -f train_policies.log

set -e

TASK="push_blue_block_left"
COMMON="--total_iterations 1000 --num_trajectories 256 --chunk_size 8 --task $TASK --reward_alpha 1.0 --reward_beta 1.0"

echo "===== Training policy on VAE world model (task: $TASK) ====="
uv run python train_policy.py \
    --world_model_checkpoint outputs/vae_calvin/checkpoint_step_150000.pt \
    --output_dir outputs/policy_vae \
    $COMMON

echo "===== Training policy on Flow Matching world model (task: $TASK) ====="
uv run python train_policy.py \
    --world_model_checkpoint outputs/flow_calvin/checkpoint_step_170000.pt \
    --output_dir outputs/policy_flow \
    $COMMON

echo "===== Training policy on DDPM world model (task: $TASK) ====="
uv run python train_policy.py \
    --world_model_checkpoint outputs/ddpm_calvin/checkpoint_step_190000.pt \
    --output_dir outputs/policy_ddpm \
    $COMMON

echo "===== All policy training complete ====="
