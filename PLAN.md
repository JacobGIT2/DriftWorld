# DriftWorld — Project Plan

## Goal

Build a world model training framework comparing 4 generative architectures
(VAE, DDPM, Flow Matching, Drifting Models) for robot manipulation next-frame
prediction, then train an RL policy in the learned world model to demonstrate
the full world-model-to-control pipeline.

**Target audience**: Robotics company building robot hands, interested in world models.

**Key selling points**:
- Clean, modular comparison of 4 generative paradigms
- Drifting models as a novel 1-NFE alternative (50x faster rollouts)
- Full pipeline: world model training -> policy training in imagination -> evaluation in sim
- Real dataset (CALVIN), not toy

---

## Architecture Overview

```
  Image (200x200)
       |
  [Frozen SD-VAE encoder] ──> z (4, 32, 32) latent
       |
  ┌────┴────────────────────────────────────────────┐
  │            World Model: f(z_t, a_t) -> z_{t+1}  │
  │                                                   │
  │  Variants (shared DiT backbone + AdaLN-Zero):     │
  │   1. VAE: encode->bottleneck->decode              │
  │   2. DDPM: denoise epsilon, DDIM sampling         │
  │   3. Flow Matching: velocity field, Euler ODE     │
  │   4. Drifting: 1-step generation, kernel loss     │
  │              (DINOv2 for kernel features)          │
  └────┬────────────────────────────────────────────┘
       |
  [Frozen SD-VAE decoder] ──> Image (256x256) for eval/vis
       |
  ┌────┴────────────────────────────────────────────┐
  │  Phase 2: Policy Training in Imagination (PPO)   │
  │   - Roll out world model K steps                  │
  │   - Action-chunking policy π(a_{t:t+K} | z_t)    │
  │   - CALVIN reward signal                          │
  └──────────────────────────────────────────────────┘
```

### Role of each pretrained model

| Component | Used by | Purpose | When loaded |
|---|---|---|---|
| **SD-VAE** (stabilityai/sd-vae-ft-mse) | All models | Encode images -> 32x32x4 latents; decode latents -> images | Precompute (offline) + eval/vis |
| **DINOv2** (dinov2_vits14, 384-dim) | Drifting model only | Feature encoder phi for drifting field kernel: k(x,y) = exp(-\|\|phi(x)-phi(y)\|\|/tau) | Training time (drift only) |

DINOv2 is NOT a general feature encoder for all models. It is specifically required
by the drifting model's loss function because the kernel-based drifting field needs
a semantic similarity measure. The paper states it "cannot work without a feature encoder."

---

## Dataset

### Primary: CALVIN (debug split for development)

- **Training**: 2770 transitions (1 episode, frames 358482-361252, scene D)
- **Validation**: ~1680 transitions
- **Per-frame data**:
  - `rgb_static`: (200, 200, 3) uint8 — static camera view (our input)
  - `rgb_gripper`: (84, 84, 3) uint8 — gripper camera (not used)
  - `rel_actions`: (7,) float64 — relative actions, range [-0.096, 1.0]
  - `actions`: (7,) float64 — absolute actions
  - `robot_obs`: (15,) float64 — joint positions etc.
  - `scene_obs`: (24,) float64 — object states
- **Action space**: 7D = [x, y, z, rx, ry, rz, gripper]
- **We use**: `rgb_static` (state) + `rel_actions` (action)

### Full dataset (for real training)

- Download `task_D_D` (~165GB) for single-environment, or `task_ABC_D` for multi-env
- Debug dataset is sufficient for pipeline validation and architecture comparison
- For portfolio: results on debug dataset are fine as long as we show the pipeline works

### Potential extension: RoboNet

- Multi-robot video prediction dataset (15M frames, 7 robots)
- Would demonstrate generalization across embodiments
- Stretch goal only — not blocking

---

## Phase 1: World Model Training

### 1.1 Data Pipeline

**Preprocessing (offline, run once):**
1. `scripts/precompute_latents.py` — encode all `rgb_static` images through frozen SD-VAE
   - Input: `episode_XXXXXXX.npz` -> `rgb_static` (200x200x3)
   - Resize to 256x256, normalize to [-1, 1]
   - Encode through SD-VAE -> (4, 32, 32) latent
   - Save as `episode_XXXXXXX_latent.pt` (fp16)
   - Batch size 64, ~1 min for debug dataset on 3090

2. `scripts/precompute_dinov2.py` (for drifting model only) — extract DINOv2 features
   - Decode latent -> image -> resize to 224x224 -> ImageNet normalize -> DINOv2
   - Save as `episode_XXXXXXX_dinov2.pt` (fp16, 384-dim vector)
   - OR: compute DINOv2 features on-the-fly during training (simpler, slower)
   - Decision: precompute. Decoding latents at training time is too expensive.

**Runtime dataloader** (`CalvinLatentDataset`):
- Loads precomputed `_latent.pt` files
- Returns `{z_t, action, z_tp1, [dinov2_tp1]}`
- Respects trajectory boundaries from `ep_start_end_ids.npy`

### 1.2 Model Architectures

All 4 models share the **DiT backbone** (`DiTBackbone`):
- Patch size 2 on 32x32 latents -> 256 tokens (16x16 grid)
- 12 transformer blocks, hidden_dim=384, 6 heads
- AdaLN-Zero conditioning from action (+ timestep for diffusion/flow)
- ~25M trainable params per model

| Model | Input to backbone | Backbone config | Training signal | Inference |
|---|---|---|---|---|
| **VAE** | z_t only (cond_channels=0) | no timestep | recon MSE + KL | 1 forward pass (use mu) |
| **DDPM** | noisy_z_{t+1} + z_t concat | has timestep | epsilon MSE | DDIM 50 steps |
| **Flow** | z_interp + z_t concat | has timestep | velocity MSE | Euler ODE 50 steps |
| **Drift** | noise + z_t concat | no timestep | drifting field MSE | 1 forward pass |

#### VAE World Model
- Encoder: DiT(z_t, action) -> flatten -> project to mu, logvar (latent_dim=256)
- Decoder: MLP + ConvTranspose2d (4x4 -> 8x8 -> 16x16 -> 32x32)
- Loss: MSE(z_pred, z_tp1) + kl_weight * KL(q || N(0,1))
- Known issue: blurry outputs, but trains fastest. Good baseline.

#### DDPM World Model
- Forward: z_noisy = add_noise(z_tp1, eps, t); predict eps = backbone(z_noisy, z_t, action, t)
- Loss: MSE(pred_eps, true_eps)
- Inference: DDIM sampling from pure noise, 50 steps
- Uses diffusers `DDPMScheduler` + `DDIMScheduler`

#### Flow Matching World Model
- Forward: t ~ U[0,1]; z_interp = (1-t)*noise + t*z_tp1; predict v = backbone(z_interp, z_t, action, t)
- Target: v_target = z_tp1 - noise (straight-line OT path)
- Loss: MSE(pred_v, target_v)
- Inference: Euler integration from noise, 50 steps

#### Drifting World Model
- Generator: z_gen = backbone(noise, z_t, action) — single pass
- Drifting field: kernel-weighted attraction toward real data
  - phi_gen = DINOv2(decode(z_gen)), phi_real = DINOv2(decode(z_tp1))
  - k = exp(-||phi_gen - phi_real|| / tau)
  - drift = k * (z_tp1 - z_gen) / sum(k)
  - Loss = MSE(z_gen, stopgrad(z_gen + drift))
- Key: DINOv2 features computed on decoded images, not latents directly
- DINOv2 + SD-VAE decoder are frozen, only generator is trained
- Advantage: 1 NFE at inference (50x faster than diffusion for rollouts)

### 1.3 Training Configuration

| Hyperparameter | Value | Notes |
|---|---|---|
| Batch size | 64 (32 for drift) | Drift needs more VRAM for DINOv2+SD-VAE |
| Learning rate | 1e-4 | AdamW with cosine schedule |
| Warmup | 1000 steps | Linear warmup |
| Total steps | 200K (reduce for debug) | ~72 epochs on debug dataset |
| FP16 | Yes | Mixed precision via torch.amp |
| Gradient clip | 1.0 | Max grad norm |
| Weight decay | 0.01 | AdamW |

### 1.4 Evaluation Metrics

**Single-step prediction quality:**
- Latent MSE: ||z_pred - z_gt||^2 in latent space
- Pixel MSE: ||img_pred - img_gt||^2 after SD-VAE decode
- SSIM: Structural Similarity Index (pixel space)
- LPIPS: Perceptual similarity (AlexNet features)

**Multi-step rollout quality:**
- Same metrics computed at each timestep of autoregressive rollout
- Rollout horizon: 20 steps
- Shows error accumulation over time — key for world model usefulness

**Visualizations:**
- Comparison grid: input | GT | VAE | DDPM | Flow | Drift
- Rollout video: side-by-side predicted vs GT trajectories
- Metrics over rollout horizon plot

---

## Phase 2: Policy Training in Imagination (deferred)

### Design decisions (settled):
- **World model predicts single step**: f(z_t, a_t) -> z_{t+1}
- **Action chunking via rollout**: policy outputs K=8 actions, roll out world model K times
- **Drifting model advantage**: 8 forward passes vs 8*50=400 for diffusion
- **Algorithm**: PPO in imagination
- **Reward**: from CALVIN task completion (requires CALVIN env or a learned reward model)

### Open questions (to resolve when we reach Phase 2):
- Reward function: use CALVIN env API or learn a reward predictor from the dataset?
- Policy architecture: MLP on latent z, or small transformer?
- How to handle stochastic world models (VAE, DDPM, Flow generate different samples each time)?
- Should we compare policies trained on each world model variant?

---

## Known Issues and Fixes Needed

### VAE model bug
The VAE's DiTBackbone is constructed with `cond_channels=0` but also passes
`**backbone_cfg` which includes defaults. Need to verify `backbone_cfg` doesn't
contain conflicting `in_channels`/`cond_channels`/`out_channels` keys.

### Config for debug dataset
The default config points to `data/calvin/task_D_D` but we're using the debug dataset
at `data/calvin/calvin_debug_dataset`. Need to update or add a config override.

### DINOv2 feature precomputation
No `scripts/precompute_dinov2.py` exists yet. Need to create it, or modify the
drifting model to compute features on-the-fly (expensive but simpler).

### hatch build path
`pyproject.toml` has `packages = ["src/driftworld"]` which may not resolve correctly
with hatch. Standard layout should be `packages = ["driftworld"]` with
`[tool.hatch.build.targets.wheel] sources = ["src"]`.

### Missing .gitignore entries
Should add: `data/`, `outputs/`, `wandb/`, `*.pt` (precomputed), `*.mp4`, `*.gif`

---

## File Structure

```
DriftWorld/
├── PLAN.md                          # This file
├── pyproject.toml                   # Project config (uv)
├── .python-version                  # Python 3.10
├── .gitignore
├── train.py                         # Training entry point (hydra)
├── evaluate.py                      # Evaluation entry point
├── configs/
│   ├── default.yaml                 # Base config
│   └── model/
│       ├── flow.yaml
│       ├── ddpm.yaml
│       ├── vae.yaml
│       └── drift.yaml
├── scripts/
│   ├── download_calvin.sh           # Download CALVIN dataset
│   ├── precompute_latents.py        # SD-VAE latent precomputation
│   └── precompute_dinov2.py         # DINOv2 feature precomputation (TODO)
├── src/driftworld/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── calvin_dataset.py        # CALVIN dataloader
│   ├── models/
│   │   ├── __init__.py              # build_world_model() factory
│   │   ├── sd_vae.py                # Frozen SD-VAE wrapper
│   │   ├── dinov2.py                # Frozen DINOv2 wrapper
│   │   ├── dit_backbone.py          # Shared DiT backbone
│   │   ├── vae_world_model.py       # VAE world model
│   │   ├── ddpm_world_model.py      # DDPM world model
│   │   ├── flow_world_model.py      # Flow matching world model
│   │   └── drift_world_model.py     # Drifting world model
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Training loop
│   │   └── losses.py                # Per-model loss dispatch
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               # SSIM, LPIPS, latent MSE
│   │   └── visualize.py             # Comparison grids, rollout videos
│   └── utils/
│       └── __init__.py
├── tests/
│   └── __init__.py
└── data/                            # (gitignored)
    └── calvin/
        └── calvin_debug_dataset/
            ├── training/            # 2777 episodes
            └── validation/          # 1681 episodes
```

---

## Execution Plan

### Day 1: Data pipeline + VAE baseline
- [x] Project skeleton created
- [x] Download CALVIN debug dataset
- [x] Fix pyproject.toml (hatch source path + sources)
- [x] Update default.yaml to point to debug dataset
- [x] Add robot_obs (15-dim) as input conditioning + prediction target
- [x] Run precompute_latents.py on training + validation
- [x] Smoke-test: all 4 models pass (VAE 36M, Flow/DDPM/Drift ~33M params)
- [x] Train VAE world model on debug dataset (fast baseline)

### Day 2: DDPM + Flow Matching
- [ ] Train DDPM world model
- [ ] Train Flow Matching world model
- [ ] Compare single-step predictions across 3 models

### Day 3: Drifting Model
- [ ] Run precompute_dinov2.py
- [ ] Train Drifting world model
- [ ] Debug drifting loss (kernel computation, gradient flow)

### Day 4: Evaluation + Visualization
- [ ] Full evaluation on all 4 models
- [ ] Comparison grids (input | GT | VAE | DDPM | Flow | Drift)
- [ ] Rollout videos for each model
- [ ] Metrics over rollout horizon plots

### Day 5: Policy training (Phase 2)
- [ ] Implement PPO policy with action chunking
- [ ] Train policy in imagination using best world model
- [ ] Evaluate policy in CALVIN sim

### Day 6-7: Polish
- [ ] Clean code, type hints, docstrings
- [ ] README with results
- [ ] Pretty visualizations for portfolio

---

## Status Log

- **2026-03-19**: Project skeleton reviewed. CALVIN debug dataset downloaded (2770 train,
  ~1680 val transitions). All source files written but untested. Next: fix configs,
  precompute latents, smoke test.
- **2026-03-19**: Added robot_obs (15-dim) as conditioning input (concat with action -> 22-dim)
  and auxiliary prediction target. Fixed pyproject.toml, configs, .gitignore. Precomputed
  SD-VAE latents for training and validation splits. All 4 model architectures smoke-tested
  successfully (forward pass + loss + inference). Ready for training.
- **2026-03-19**: training on debug set complete
