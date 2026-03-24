"""Render expert trajectory through all 3 world models for comparison.

Takes the first frame of an expert trajectory, then autoregressively rolls out
each world model using the expert actions. Produces:
1. Side-by-side video: GT | VAE | Flow | DDPM
2. Individual model videos

Usage:
    uv run python scripts/render_world_model_rollouts.py
"""

import numpy as np
import torch
import imageio
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf

from driftworld.models import build_world_model
from driftworld.models.sd_vae import FrozenSDVAE


def load_world_model(checkpoint_path, device="cuda"):
    """Load a world model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    model = build_world_model(cfg.model)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg.model.type


def decode_to_image(sd_vae, z, clamp=True):
    """Decode latent to uint8 numpy image."""
    with torch.no_grad():
        img = sd_vae.decode(z)  # (1, 3, 256, 256) in [-1, 1]
    img = ((img + 1) / 2).clamp(0, 1)
    img = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return img


def rollout_world_model(model, z_init, actions_list, robot_obs_init, sd_vae, device="cuda"):
    """Autoregressively roll out a world model using expert actions.

    Returns list of decoded images (numpy uint8).
    """
    frames = []
    z_t = z_init.clone()
    robot_obs = robot_obs_init.clone()

    # Add initial frame
    frames.append(decode_to_image(sd_vae, z_t))

    with torch.no_grad():
        for action_7d in actions_list:
            # Build 22D input: [rel_actions(7), robot_obs(15)]
            action_22d = torch.cat([action_7d.to(device), robot_obs]).unsqueeze(0)

            # World model predict
            out = model.predict(z_t, action_22d)
            if isinstance(out, tuple):
                z_tp1, robot_obs_pred = out
                robot_obs = robot_obs_pred.squeeze(0)
            else:
                z_tp1 = out

            z_t = z_tp1
            frames.append(decode_to_image(sd_vae, z_t))

    return frames


def main():
    device = "cuda"
    output_dir = Path("outputs/world_model_rollouts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find expert trajectory
    ann = np.load(
        "data/calvin/task_D_D/training/lang_annotations/auto_lang_ann.npy",
        allow_pickle=True,
    ).item()
    tasks = ann["language"]["task"]
    indx = ann["info"]["indx"]

    for i, t in enumerate(tasks):
        if t == "push_blue_block_left":
            start, end = indx[i]
            break

    print(f"Expert trajectory: frames {start}-{end} (length={end - start})")

    # Load SD-VAE for decoding
    sd_vae = FrozenSDVAE().to(device)

    # Load initial frame latent
    z_init = torch.load(
        f"data/calvin/task_D_D/training/episode_{start:07d}_latent.pt",
        map_location=device,
    ).float().unsqueeze(0)  # (1, 4, 32, 32)

    start_frame = np.load(f"data/calvin/task_D_D/training/episode_{start:07d}.npz")
    robot_obs_init = torch.tensor(
        start_frame["robot_obs"], dtype=torch.float32, device=device
    )

    # Load expert actions and GT latents
    actions = []
    gt_frames = [decode_to_image(sd_vae, z_init)]  # initial frame

    for frame_idx in range(start, end):
        ep = np.load(f"data/calvin/task_D_D/training/episode_{frame_idx:07d}.npz")
        actions.append(torch.tensor(ep["rel_actions"], dtype=torch.float32))

        # Load GT next frame
        next_idx = frame_idx + 1
        z_gt = torch.load(
            f"data/calvin/task_D_D/training/episode_{next_idx:07d}_latent.pt",
            map_location=device,
        ).float().unsqueeze(0)
        gt_frames.append(decode_to_image(sd_vae, z_gt))

    print(f"Loaded {len(actions)} expert actions and {len(gt_frames)} GT frames")

    # Load world models and roll out
    checkpoints = {
        "VAE": "outputs/vae_calvin/checkpoint_step_150000.pt",
        "Flow": "outputs/flow_calvin/checkpoint_step_170000.pt",
        "DDPM": "outputs/ddpm_calvin/checkpoint_step_190000.pt",
    }

    all_frames = {"GT": gt_frames}

    for name, ckpt_path in checkpoints.items():
        print(f"Rolling out {name}...")
        model, model_type = load_world_model(ckpt_path, device)
        frames = rollout_world_model(
            model, z_init, actions, robot_obs_init, sd_vae, device
        )
        all_frames[name] = frames
        print(f"  {name}: {len(frames)} frames")

        # Save individual video
        imageio.mimwrite(str(output_dir / f"rollout_{name.lower()}.mp4"), frames, fps=12)

    # Save GT video
    imageio.mimwrite(str(output_dir / "rollout_gt.mp4"), gt_frames, fps=12)

    # Build side-by-side comparison: GT | VAE | Flow | DDPM
    model_order = ["GT", "VAE", "Flow", "DDPM"]
    max_len = max(len(all_frames[k]) for k in model_order)

    combined_frames = []
    for i in range(max_len):
        panels = []
        for name in model_order:
            frames = all_frames[name]
            frame = frames[min(i, len(frames) - 1)]
            # Resize to same size (256x256)
            frame = np.array(Image.fromarray(frame).resize((256, 256)))
            # Add label
            import cv2
            frame = frame.copy()
            cv2.putText(frame, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            panels.append(frame)
        combined = np.concatenate(panels, axis=1)
        combined_frames.append(combined)

    imageio.mimwrite(str(output_dir / "comparison_gt_vae_flow_ddpm.mp4"), combined_frames, fps=12)
    print(f"\nSaved comparison video: {output_dir / 'comparison_gt_vae_flow_ddpm.mp4'}")
    print(f"Individual videos: rollout_gt.mp4, rollout_vae.mp4, rollout_flow.mp4, rollout_ddpm.mp4")

    # Save first and last frame as images for quick viewing
    first_combined = combined_frames[0]
    last_combined = combined_frames[-1]
    Image.fromarray(first_combined).save(str(output_dir / "frame_first.png"))
    Image.fromarray(last_combined).save(str(output_dir / "frame_last.png"))
    print(f"Saved frame_first.png and frame_last.png")


if __name__ == "__main__":
    main()
