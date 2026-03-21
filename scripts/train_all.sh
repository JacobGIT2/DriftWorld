#!/bin/bash
# Train VAE, Flow Matching, and DDPM world models on full task_D_D dataset.
# Drift model skipped until mode collapse fix is implemented.
#
# Estimated time on 3090: ~45 hours total (15h × 3 models)
#
# Usage (SSH-safe):
#   nohup bash scripts/train_all.sh > train_all.log 2>&1 &
#   tail -f train_all.log   # to monitor

set -e

COMMON="training.total_steps=200000 training.warmup_steps=1000 training.log_every=100 training.vis_every=5000 training.eval_every=5000 training.save_every=10000 training.num_workers=4"

echo "===== Training VAE ====="
uv run python train.py model.type=vae training.batch_size=64 $COMMON

echo "===== Training Flow Matching ====="
uv run python train.py model.type=flow training.batch_size=64 $COMMON

echo "===== Training DDPM ====="
uv run python train.py model.type=ddpm training.batch_size=64 $COMMON

echo "===== All training complete ====="
