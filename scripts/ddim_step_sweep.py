"""DDIM/Flow step count sweep: measure reconstruction quality vs denoising steps.

Generates:
- metrics.json: raw numbers for all step counts
- metrics_vs_steps.png: graph of SSIM, LPIPS, pixel_mse vs step count
- grid_step_{N}.png: comparison grids (input | GT | prediction) per step count

Usage:
    uv run python scripts/ddim_step_sweep.py
    uv run python scripts/ddim_step_sweep.py --checkpoint outputs/ddpm_calvin/checkpoint_step_190000.pt
    uv run python scripts/ddim_step_sweep.py --steps 1,5,10,25,50,100,250
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from driftworld.data.calvin_dataset import CalvinLatentDataset
from driftworld.evaluation.metrics import WorldModelEvaluator
from driftworld.evaluation.visualize import make_comparison_grid
from driftworld.models import build_world_model
from driftworld.models.sd_vae import FrozenSDVAE


def run_sweep(model, val_loader, evaluator, sd_vae, step_counts, num_samples, output_dir, device):
    results = {s: [] for s in step_counts}

    # Get a fixed batch for visual comparison
    vis_batch = next(iter(val_loader))
    z_t_vis = vis_batch["z_t"].to(device)
    action_vis = vis_batch["action"].to(device)
    z_tp1_vis = vis_batch["z_tp1"].to(device)

    for steps in step_counts:
        print(f"\n--- Evaluating with {steps} steps ---")

        # Generate comparison grid for this step count
        with torch.no_grad():
            out = model.predict(z_t_vis, action_vis, num_steps=steps)
            z_pred_vis = out[0] if isinstance(out, tuple) else out

        make_comparison_grid(
            sd_vae, z_t_vis, z_tp1_vis,
            {f"steps={steps}": z_pred_vis},
            save_path=output_dir / f"grid_step_{steps}.png",
            num_samples=4,
        )

        # Compute metrics over validation set
        metrics_list = []
        for i, batch in enumerate(tqdm(val_loader, desc=f"steps={steps}")):
            if i * val_loader.batch_size >= num_samples:
                break
            z_t = batch["z_t"].to(device)
            action = batch["action"].to(device)
            z_tp1 = batch["z_tp1"].to(device)

            with torch.no_grad():
                out = model.predict(z_t, action, num_steps=steps)
                z_pred = out[0] if isinstance(out, tuple) else out

            metrics = evaluator.compute_metrics(z_pred, z_tp1)
            metrics_list.append(metrics)

        # Average metrics
        avg = {}
        for key in metrics_list[0]:
            avg[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
        results[steps] = avg
        print(f"  SSIM={avg.get('ssim', 0):.4f}  LPIPS={avg.get('lpips', 0):.4f}  "
              f"pixel_mse={avg.get('pixel_mse', 0):.6f}")

    return results


def plot_results(results, output_dir):
    step_counts = sorted(results.keys())
    metrics_to_plot = ["ssim", "lpips", "pixel_mse", "latent_mse"]
    titles = ["SSIM (higher=better)", "LPIPS (lower=better)",
              "Pixel MSE (lower=better)", "Latent MSE (lower=better)"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Reconstruction Quality vs Denoising Steps", fontsize=14)

    for ax, metric, title in zip(axes.flat, metrics_to_plot, titles):
        values = [results[s].get(metric, 0) for s in step_counts]
        ax.plot(step_counts, values, "o-", linewidth=2, markersize=6)
        ax.set_xlabel("Denoising Steps")
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_vs_steps.png", dpi=150)
    plt.close()
    print(f"\nSaved plot to {output_dir / 'metrics_vs_steps.png'}")


def main():
    parser = argparse.ArgumentParser(description="DDIM/Flow step count sweep")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/ddpm_calvin/checkpoint_step_190000.pt")
    parser.add_argument("--steps", type=str, default="1,5,10,25,50,100,250,500,1000")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="outputs/ddim_sweep")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step_counts = [int(s) for s in args.steps.split(",")]

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])

    model = build_world_model(cfg.model)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    model_type = cfg.model.type
    print(f"Model type: {model_type}")

    if model_type not in ("ddpm", "flow"):
        print(f"Warning: model type '{model_type}' may not support variable step counts")

    # SD-VAE and evaluator
    sd_vae = FrozenSDVAE().to(device)
    evaluator = WorldModelEvaluator(sd_vae=sd_vae, device=args.device)

    # Dataset
    val_dataset = CalvinLatentDataset(
        data_dir=os.path.join(cfg.dataset.data_dir, cfg.dataset.val_split),
        action_key=cfg.dataset.action_key,
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Run sweep
    results = run_sweep(model, val_loader, evaluator, sd_vae,
                        step_counts, args.num_samples, output_dir, device)

    # Plot
    plot_results(results, output_dir)

    # Save JSON (convert int keys to strings for JSON)
    json_results = {str(k): v for k, v in results.items()}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
