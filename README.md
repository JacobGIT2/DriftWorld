# DriftWorld

A world model framework for robot manipulation that predicts the next visual state given (current image, action). Compares four generative architectures — **VAE**, **DDPM**, **Flow Matching**, and **Drifting Models** — using a shared DiT backbone operating in SD-VAE latent space.

## Architecture

```
Image (256×256) → [Frozen SD-VAE Encoder] → z (4×32×32)
                                                │
                              ┌─────────────────┴─────────────────┐
                              │   World Model  f(z_t, a_t, r_t)   │
                              │                                    │
                              │   Shared DiT-S/2 backbone (~33M)   │
                              │   + AdaLN-Zero conditioning        │
                              │                                    │
                              │   Variants:                        │
                              │   • VAE        (1-step, blurry)    │
                              │   • DDPM       (50-step, sharp)    │
                              │   • Flow Match (50-step, sharp)    │
                              │   • Drifting   (1-step, sharp)     │
                              └─────────────────┬─────────────────┘
                                                │
                                          ẑ_{t+1} + r̂_{t+1}
                                                │
                              [Frozen SD-VAE Decoder] → Image (256×256)
```

**Inputs**: current image latent `z_t`, 7D relative action `a_t`, 15D proprioception `r_t`
**Outputs**: predicted next image latent `ẑ_{t+1}`, predicted next proprioception `r̂_{t+1}`

## Setup

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Data Preparation

### 1. Download CALVIN

```bash
# Debug dataset (~500MB, for pipeline testing)
bash scripts/download_calvin.sh debug

# Full single-environment dataset (~165GB)
bash scripts/download_calvin.sh D
```

The dataset will be saved to the current directory. Move it to `data/calvin/`:

```bash
mkdir -p data/calvin
mv task_D_D data/calvin/       # or calvin_debug_dataset
```

### 2. Precompute SD-VAE Latents

Encodes all images through the frozen SD-VAE once, saving as FP16 `.pt` files for fast training:

```bash
uv run python scripts/precompute_latents.py --data_dir data/calvin/task_D_D/training
uv run python scripts/precompute_latents.py --data_dir data/calvin/task_D_D/validation
```

### 3. Precompute DINOv2 Features (Drifting Model Only)

```bash
uv run python scripts/precompute_dinov2.py --data_dir data/calvin/task_D_D/training
uv run python scripts/precompute_dinov2.py --data_dir data/calvin/task_D_D/validation
```

## Training

### Single Model

```bash
# Flow Matching (default)
uv run python train.py

# Specific model
uv run python train.py model.type=vae
uv run python train.py model.type=ddpm
uv run python train.py model.type=drift dataset.load_dinov2_features=true training.batch_size=32

# Override hyperparameters
uv run python train.py model.type=flow training.lr=5e-5 training.total_steps=100000
```

### All Models (Sequential)

```bash
# SSH-safe (survives disconnects)
nohup bash scripts/train_all.sh > train_all.log 2>&1 &
tail -f train_all.log
```

Trains VAE → Flow → DDPM sequentially. ~45 hours total on an RTX 3090.

### Monitoring

Training logs to [Weights & Biases](https://wandb.ai) under the `driftworld` project. Each run logs:
- **Scalar metrics**: `train/loss`, `val/loss`, `train/lr`, model-specific losses
- **Prediction images**: side-by-side `[Input | Ground Truth | Predicted]` grids
- **Robot obs table**: predicted vs ground truth proprioception (15 dimensions)

```bash
# First-time setup
uv run wandb login
```

### Key Config Options

All configuration lives in `configs/default.yaml` and can be overridden via command line (Hydra):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.type` | `flow` | Architecture: `vae`, `ddpm`, `flow`, `drift` |
| `training.batch_size` | `64` | Batch size (use 32 for drift) |
| `training.lr` | `1e-4` | Learning rate |
| `training.total_steps` | `200000` | Total training steps |
| `training.fp16` | `true` | Mixed precision training |
| `dataset.data_dir` | `data/calvin/task_D_D` | Path to CALVIN dataset |

## Evaluation

```bash
# Single-step metrics (SSIM, LPIPS, MSE)
uv run python evaluate.py --checkpoint outputs/flow_calvin/checkpoint_final.pt

# Multi-step rollout
uv run python evaluate.py --checkpoint outputs/flow_calvin/checkpoint_final.pt --rollout_horizon 30
```

Outputs:
- `metrics.json` — quantitative results
- `comparison_grid.png` — visual comparison
- `rollout.mp4` — autoregressive rollout video

## Model Comparison

| Model | Inference Steps | Training Loss | Key Property |
|-------|:-:|---|---|
| **VAE** | 1 | MSE + KL | Fast, tends to produce blurry outputs |
| **DDPM** | 50 (DDIM) | Noise prediction MSE | Sharp outputs, slow rollouts |
| **Flow Matching** | 50 (Euler) | Velocity MSE | Sharp outputs, simple training |
| **Drifting** | 1 | Kernel-based drift field | Sharp + fast (50× fewer NFEs than diffusion) |

## Project Structure

```
DriftWorld/
├── configs/default.yaml              # Hydra config (model, dataset, training)
├── scripts/
│   ├── download_calvin.sh            # Download CALVIN dataset
│   ├── precompute_latents.py         # Offline SD-VAE encoding
│   ├── precompute_dinov2.py          # Offline DINOv2 features (drift only)
│   └── train_all.sh                  # Batch training script
├── src/driftworld/
│   ├── models/
│   │   ├── dit_backbone.py           # Shared DiT-S/2 + AdaLN-Zero
│   │   ├── sd_vae.py                 # Frozen SD-VAE encoder/decoder
│   │   ├── dinov2.py                 # Frozen DINOv2-S feature extractor
│   │   ├── vae_world_model.py        # VAE world model
│   │   ├── ddpm_world_model.py       # DDPM world model
│   │   ├── flow_world_model.py       # Flow Matching world model
│   │   └── drift_world_model.py      # Drifting world model
│   ├── data/calvin_dataset.py        # CALVIN latent dataset
│   ├── training/
│   │   ├── trainer.py                # Training loop (AdamW, OneCycleLR, W&B)
│   │   └── losses.py                 # Per-model loss computation
│   └── evaluation/
│       ├── metrics.py                # SSIM, LPIPS, MSE
│       └── visualize.py              # Grids and rollout videos
├── train.py                          # Training entry point
└── evaluate.py                       # Evaluation entry point
```

## References

- **Drifting Models**: [Drifting Models: Generative Modeling as Distribution Evolution](https://arxiv.org/abs/2602.04770)
- **DiT**: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- **Flow Matching**: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- **CALVIN**: [CALVIN: A Benchmark for Language-Conditioned Policy Learning](https://arxiv.org/abs/2112.03227)
