"""Evaluation entry point for DriftWorld world models.

Loads a trained checkpoint, runs single-step and rollout evaluation,
generates comparison grids and rollout videos.

Usage:
    uv run python evaluate.py --checkpoint outputs/flow_calvin/checkpoint_final.pt
    uv run python evaluate.py --checkpoint outputs/flow_calvin/checkpoint_final.pt --rollout_horizon 30
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from driftworld.data.calvin_dataset import CalvinLatentDataset
from driftworld.evaluation.metrics import WorldModelEvaluator
from driftworld.evaluation.visualize import make_comparison_grid, make_rollout_video
from driftworld.models import build_world_model
from driftworld.models.sd_vae import FrozenSDVAE


def main():
    parser = argparse.ArgumentParser(description="Evaluate world model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--rollout_horizon", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])

    # Build model
    model = build_world_model(cfg.model)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # SD-VAE for decoding
    sd_vae = FrozenSDVAE().to(device)

    # Evaluator
    evaluator = WorldModelEvaluator(sd_vae=sd_vae, device=args.device)

    # Dataset
    import os
    val_dataset = CalvinLatentDataset(
        data_dir=os.path.join(cfg.dataset.data_dir, cfg.dataset.val_split),
        action_key=cfg.dataset.action_key,
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Output directory
    output_dir = Path(args.output_dir or f"eval_{cfg.model.type}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Single-step evaluation ---
    print("Running single-step evaluation...")
    all_metrics = []
    for i, batch in enumerate(tqdm(val_loader)):
        if i * 16 >= args.num_samples:
            break
        z_t = batch["z_t"].to(device)
        action = batch["action"].to(device)
        z_tp1 = batch["z_tp1"].to(device)

        with torch.no_grad():
            z_pred = model.predict(z_t, action)

        metrics = evaluator.compute_metrics(z_pred, z_tp1)
        all_metrics.append(metrics)

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0]:
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    print("\nSingle-step metrics:")
    for key, val in avg_metrics.items():
        print(f"  {key}: {val:.6f}")

    # --- Comparison grid ---
    print("\nGenerating comparison grid...")
    batch = next(iter(val_loader))
    z_t = batch["z_t"].to(device)
    action = batch["action"].to(device)
    z_tp1 = batch["z_tp1"].to(device)
    with torch.no_grad():
        z_pred = model.predict(z_t, action)
    make_comparison_grid(
        sd_vae, z_t, z_tp1,
        {cfg.model.type: z_pred},
        save_path=output_dir / "comparison_grid.png",
        num_samples=4,
    )
    print(f"Saved to {output_dir / 'comparison_grid.png'}")

    # --- Rollout video ---
    print(f"\nGenerating {args.rollout_horizon}-step rollout video...")
    # Get a trajectory sequence
    sample_0 = val_dataset[0]
    z_init = sample_0["z_t"].unsqueeze(0).to(device)
    actions_seq = []
    gt_seq = []
    for t in range(args.rollout_horizon):
        if t < len(val_dataset):
            s = val_dataset[t]
            actions_seq.append(s["action"])
            gt_seq.append(s["z_tp1"])
    actions_seq = torch.stack(actions_seq)
    gt_seq = torch.stack(gt_seq)

    make_rollout_video(
        sd_vae, model, z_init, actions_seq,
        save_path=output_dir / "rollout.mp4",
        gt_latents=gt_seq,
    )
    print(f"Saved to {output_dir / 'rollout.mp4'}")

    # Save metrics
    import json
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(avg_metrics, f, indent=2)
    print(f"\nMetrics saved to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
