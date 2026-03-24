"""Render forward and reversed trajectory comparisons.

Columns: GT | CALVIN Sim | VAE | Flow(5) | DDPM(1000)

Key design decisions based on analysis:
1. CALVIN sim reversed is INVALID for physics comparison — reversed actions in a
   forward simulator do NOT produce time-reversal (contact dynamics are irreversible).
   We still render it but label it "Sim(rev actions)" to be transparent.
2. Gripper action (index 6) in reversed trajectory uses the PREVIOUS frame's gripper
   state, not a copy of the forward action.
3. Single env instance reused with reset() to avoid PyBullet cleanup bugs.
4. env.reset(robot_obs, scene_obs) loses velocity/contact state — noted as limitation.

Usage:
    nohup uv run python scripts/render_final_comparison.py > render_final.log 2>&1 &
"""

import os
import sys

import cv2
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image

from driftworld.models import build_world_model
from driftworld.models.sd_vae import FrozenSDVAE

DEVICE = "cuda"
OUTPUT_DIR = Path("outputs/world_model_rollouts/final")
DATA_DIR = "data/calvin/task_D_D/training"
TASK = "push_blue_block_left"
CELL_SIZE = 200
FPS = 12


def decode_uint8(sd_vae, z):
    with torch.no_grad():
        img = sd_vae.decode(z)
    return (
        ((img + 1) / 2).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
    ).astype(np.uint8)


def load_trajectory():
    ann = np.load(
        f"{DATA_DIR}/lang_annotations/auto_lang_ann.npy", allow_pickle=True
    ).item()
    for i, t in enumerate(ann["language"]["task"]):
        if t == TASK:
            return ann["info"]["indx"][i]
    raise ValueError(f"Task {TASK} not found")


def load_actions_forward(start, end):
    actions = []
    for idx in range(start, end):
        ep = np.load(f"{DATA_DIR}/episode_{idx:07d}.npz")
        actions.append(torch.tensor(ep["rel_actions"], dtype=torch.float32))
    return actions


def load_actions_reversed(start, end):
    """Build reversed actions with correct gripper handling.

    For position dims (0-5): negate the action (move in opposite direction).
    For gripper dim (6): use the gripper state from the TARGET frame in the
    reversed sequence (i.e., the previous frame in forward time), not a copy
    of the forward action.
    """
    actions = []
    for idx in range(end - 1, start - 1, -1):
        ep_curr = np.load(f"{DATA_DIR}/episode_{idx:07d}.npz")
        # rel_actions[:3] = TCP pos delta * 50, [3:6] = orn delta * 20
        # Negating gives opposite movement direction — correct for reversal
        neg = -torch.tensor(ep_curr["rel_actions"], dtype=torch.float32)
        # Gripper (dim 6): use the command from the target frame in reversed time.
        # Reversing z_{idx+1} -> z_{idx}: use gripper action from frame idx-1.
        if idx > start:
            ep_prev = np.load(f"{DATA_DIR}/episode_{idx - 1:07d}.npz")
            neg[6] = torch.tensor(ep_prev["rel_actions"][6], dtype=torch.float32)
        else:
            ep_start = np.load(f"{DATA_DIR}/episode_{start:07d}.npz")
            neg[6] = torch.tensor(ep_start["rel_actions"][6], dtype=torch.float32)
        actions.append(neg)
    return actions


def load_gt_frames(sd_vae, start, end):
    frames = []
    for idx in range(start, end + 1):
        z = (
            torch.load(
                f"{DATA_DIR}/episode_{idx:07d}_latent.pt", map_location=DEVICE
            )
            .float()
            .unsqueeze(0)
        )
        frames.append(decode_uint8(sd_vae, z))
    return frames


def run_calvin_sim_both(start_data, end_data, fwd_actions_np, rev_actions_np):
    """Run CALVIN sim for both forward and reversed using a SINGLE env instance."""
    os.environ["CALVIN_ROOT"] = "/tmp/calvin_repo/calvin_env"
    sys.path.insert(0, "/tmp/calvin_repo/calvin_env")
    from calvin_env.envs.play_table_env import get_env

    env = get_env(
        dataset_path=DATA_DIR,
        obs_space={
            "rgb_obs": ["rgb_static"],
            "depth_obs": [],
            "state_obs": ["robot_obs"],
        },
        show_gui=False,
    )

    # Forward
    print("  CALVIN sim forward...", flush=True)
    obs = env.reset(
        robot_obs=start_data["robot_obs"], scene_obs=start_data["scene_obs"]
    )
    sim_fwd = [
        np.array(
            Image.fromarray(obs["rgb_obs"]["rgb_static"]).resize((256, 256))
        )
    ]
    for a in fwd_actions_np:
        obs, _, _, _ = env.step(a)
        sim_fwd.append(
            np.array(
                Image.fromarray(obs["rgb_obs"]["rgb_static"]).resize((256, 256))
            )
        )
    print(f"    {len(sim_fwd)} frames", flush=True)

    # Reversed (note: this is physically invalid — forward sim with negated
    # actions, NOT true time-reversal. Contact dynamics won't reverse.)
    print("  CALVIN sim reversed (negated actions, NOT true reversal)...", flush=True)
    obs = env.reset(
        robot_obs=end_data["robot_obs"], scene_obs=end_data["scene_obs"]
    )
    sim_rev = [
        np.array(
            Image.fromarray(obs["rgb_obs"]["rgb_static"]).resize((256, 256))
        )
    ]
    for a in rev_actions_np:
        obs, _, _, _ = env.step(a)
        sim_rev.append(
            np.array(
                Image.fromarray(obs["rgb_obs"]["rgb_static"]).resize((256, 256))
            )
        )
    print(f"    {len(sim_rev)} frames", flush=True)

    try:
        env.close()
    except Exception:
        pass

    return sim_fwd, sim_rev


def rollout_world_model(sd_vae, ckpt_path, num_steps, z_init, rob_init, actions):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    model = build_world_model(cfg.model)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()

    z_t, rob = z_init.clone(), rob_init.clone()
    frames = [decode_uint8(sd_vae, z_t)]

    with torch.no_grad():
        for a in actions:
            a22 = torch.cat([a.to(DEVICE), rob]).unsqueeze(0)
            if num_steps is not None:
                out = model.predict(z_t, a22, num_steps=num_steps)
            else:
                out = model.predict(z_t, a22)
            if isinstance(out, tuple):
                z_t, rob = out
                rob = rob.squeeze(0)
            else:
                z_t = out
            frames.append(decode_uint8(sd_vae, z_t))

    del model
    torch.cuda.empty_cache()
    return frames


def make_video(sources, col_names, out_path):
    n = max(len(v) for v in sources.values())
    frames = []
    for i in range(n):
        panels = []
        for c in col_names:
            f = sources[c][min(i, len(sources[c]) - 1)]
            f = np.array(Image.fromarray(f).resize((CELL_SIZE, CELL_SIZE))).copy()
            cv2.putText(
                f, c, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2
            )
            cv2.putText(
                f, c, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1
            )
            panels.append(f)
        frames.append(np.concatenate(panels, axis=1))

    imageio.mimwrite(str(out_path), frames, fps=FPS)
    stem = out_path.stem
    parent = out_path.parent
    Image.fromarray(frames[0]).save(str(parent / f"{stem}_first.png"))
    Image.fromarray(frames[len(frames) // 2]).save(str(parent / f"{stem}_mid.png"))
    Image.fromarray(frames[-1]).save(str(parent / f"{stem}_last.png"))
    print(f"Saved {out_path} ({len(frames)} frames)", flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sd_vae = FrozenSDVAE().to(DEVICE)

    start, end = load_trajectory()
    print(f"Trajectory: {start}->{end} (length={end - start})", flush=True)

    start_data = np.load(f"{DATA_DIR}/episode_{start:07d}.npz")
    end_data = np.load(f"{DATA_DIR}/episode_{end:07d}.npz")

    # Load actions
    fwd_actions = load_actions_forward(start, end)
    rev_actions = load_actions_reversed(start, end)

    # GT frames
    print("Loading GT frames...", flush=True)
    gt_fwd = load_gt_frames(sd_vae, start, end)
    gt_rev = list(reversed(gt_fwd))

    # CALVIN sim (single env instance for both directions)
    print("Running CALVIN simulator...", flush=True)
    fwd_actions_np = [a.numpy() for a in fwd_actions]
    rev_actions_np = [a.numpy() for a in rev_actions]
    sim_fwd, sim_rev = run_calvin_sim_both(
        start_data, end_data, fwd_actions_np, rev_actions_np
    )

    # World model rollouts
    models = [
        ("VAE", "outputs/vae_calvin/checkpoint_step_150000.pt", None),
        ("Flow(5)", "outputs/flow_calvin/checkpoint_step_170000.pt", 5),
        ("DDPM(1000)", "outputs/ddpm_calvin/checkpoint_step_190000.pt", 1000),
    ]

    z_fwd = (
        torch.load(
            f"{DATA_DIR}/episode_{start:07d}_latent.pt", map_location=DEVICE
        )
        .float()
        .unsqueeze(0)
    )
    rob_fwd = torch.tensor(
        start_data["robot_obs"], dtype=torch.float32, device=DEVICE
    )
    z_rev = (
        torch.load(
            f"{DATA_DIR}/episode_{end:07d}_latent.pt", map_location=DEVICE
        )
        .float()
        .unsqueeze(0)
    )
    rob_rev = torch.tensor(
        end_data["robot_obs"], dtype=torch.float32, device=DEVICE
    )

    fwd_all = {"GT": gt_fwd, "Sim": sim_fwd}
    rev_all = {"GT(rev)": gt_rev, "Sim(rev)": sim_rev}

    for name, ckpt_path, ns in models:
        print(f"{name} forward...", flush=True)
        fwd_all[name] = rollout_world_model(
            sd_vae, ckpt_path, ns, z_fwd, rob_fwd, fwd_actions
        )
        print(f"  {len(fwd_all[name])} frames", flush=True)

        print(f"{name} reversed...", flush=True)
        rev_all[name] = rollout_world_model(
            sd_vae, ckpt_path, ns, z_rev, rob_rev, rev_actions
        )
        print(f"  {len(rev_all[name])} frames", flush=True)

    # Build videos
    fwd_cols = ["GT", "Sim", "VAE", "Flow(5)", "DDPM(1000)"]
    rev_cols = ["GT(rev)", "Sim(rev)", "VAE", "Flow(5)", "DDPM(1000)"]

    print("\nBuilding forward video...", flush=True)
    make_video(fwd_all, fwd_cols, OUTPUT_DIR / "forward.mp4")

    print("Building reversed video...", flush=True)
    make_video(rev_all, rev_cols, OUTPUT_DIR / "reversed.mp4")

    print("\nAll done!", flush=True)


if __name__ == "__main__":
    main()
