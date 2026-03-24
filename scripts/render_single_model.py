"""Render forward + reversed trajectory for a SINGLE world model.

Usage:
    # Run each model separately (can run in parallel if GPU memory allows):
    uv run python scripts/render_single_model.py --model vae --ckpt outputs/vae_calvin/checkpoint_step_150000.pt
    uv run python scripts/render_single_model.py --model flow5 --ckpt outputs/flow_calvin/checkpoint_step_170000.pt --num_steps 5
    uv run python scripts/render_single_model.py --model ddpm1000 --ckpt outputs/ddpm_calvin/checkpoint_step_190000.pt --num_steps 1000
    uv run python scripts/render_single_model.py --model gt  # GT only, no checkpoint needed
    uv run python scripts/render_single_model.py --model sim  # CALVIN sim only
"""

import argparse
import os
import sys

import cv2
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image

DATA_DIR = "data/calvin/task_D_D/training"
TASK = "push_blue_block_left"
OUTPUT_DIR = Path("outputs/world_model_rollouts/final")
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


def load_actions(start, end):
    fwd, rev = [], []
    for idx in range(start, end):
        ep = np.load(f"{DATA_DIR}/episode_{idx:07d}.npz")
        fwd.append(torch.tensor(ep["rel_actions"], dtype=torch.float32))
    for idx in range(end - 1, start - 1, -1):
        ep_curr = np.load(f"{DATA_DIR}/episode_{idx:07d}.npz")
        # Negate position/orientation deltas (dims 0-5)
        neg = -torch.tensor(ep_curr["rel_actions"], dtype=torch.float32)
        # Gripper (dim 6): use the command from the TARGET frame in reversed time.
        # Reversing z_{idx+1} -> z_{idx}: the gripper state at z_{idx} was produced
        # by the action at frame idx-1->idx, so use rel_actions[6] from frame idx.
        # But since we're iterating idx from end-1 to start, the target is frame idx,
        # and we need the gripper command that reaches frame idx = action from idx-1.
        if idx > start:
            ep_prev = np.load(f"{DATA_DIR}/episode_{idx - 1:07d}.npz")
            neg[6] = torch.tensor(ep_prev["rel_actions"][6], dtype=torch.float32)
        else:
            # First reversed step: use the start frame's gripper state
            ep_start = np.load(f"{DATA_DIR}/episode_{start:07d}.npz")
            neg[6] = torch.tensor(ep_start["rel_actions"][6], dtype=torch.float32)
        rev.append(neg)
    return fwd, rev


def save_video(frames, path, label):
    labeled = []
    for f in frames:
        f = np.array(Image.fromarray(f).resize((256, 256))).copy()
        cv2.putText(f, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(f, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        labeled.append(f)
    imageio.mimwrite(str(path), labeled, fps=FPS)
    # Save key frames
    Image.fromarray(labeled[0]).save(str(path).replace(".mp4", "_first.png"))
    Image.fromarray(labeled[len(labeled) // 2]).save(str(path).replace(".mp4", "_mid.png"))
    Image.fromarray(labeled[-1]).save(str(path).replace(".mp4", "_last.png"))


def render_gt(start, end, device):
    from driftworld.models.sd_vae import FrozenSDVAE
    sd_vae = FrozenSDVAE().to(device)

    fwd = []
    for idx in range(start, end + 1):
        z = torch.load(f"{DATA_DIR}/episode_{idx:07d}_latent.pt", map_location=device).float().unsqueeze(0)
        fwd.append(decode_uint8(sd_vae, z))

    save_video(fwd, OUTPUT_DIR / "gt_forward.mp4", "GT")
    save_video(list(reversed(fwd)), OUTPUT_DIR / "gt_reversed.mp4", "GT(rev)")
    print(f"GT: {len(fwd)} frames saved", flush=True)


def render_sim(start, end, fwd_actions_np, rev_actions_np, start_data, end_data):
    os.environ["CALVIN_ROOT"] = "/tmp/calvin_repo/calvin_env"
    sys.path.insert(0, "/tmp/calvin_repo/calvin_env")
    from calvin_env.envs.play_table_env import get_env

    env = get_env(
        dataset_path=DATA_DIR,
        obs_space={"rgb_obs": ["rgb_static"], "depth_obs": [], "state_obs": ["robot_obs"]},
        show_gui=False,
    )

    # Forward
    obs = env.reset(robot_obs=start_data["robot_obs"], scene_obs=start_data["scene_obs"])
    sim_fwd = [np.array(Image.fromarray(obs["rgb_obs"]["rgb_static"]).resize((256, 256)))]
    for a in fwd_actions_np:
        obs, _, _, _ = env.step(a)
        sim_fwd.append(np.array(Image.fromarray(obs["rgb_obs"]["rgb_static"]).resize((256, 256))))

    # Reversed
    obs = env.reset(robot_obs=end_data["robot_obs"], scene_obs=end_data["scene_obs"])
    sim_rev = [np.array(Image.fromarray(obs["rgb_obs"]["rgb_static"]).resize((256, 256)))]
    for a in rev_actions_np:
        obs, _, _, _ = env.step(a)
        sim_rev.append(np.array(Image.fromarray(obs["rgb_obs"]["rgb_static"]).resize((256, 256))))

    try:
        env.close()
    except Exception:
        pass

    save_video(sim_fwd, OUTPUT_DIR / "sim_forward.mp4", "Sim")
    save_video(sim_rev, OUTPUT_DIR / "sim_reversed.mp4", "Sim(rev)")
    print(f"Sim: {len(sim_fwd)} fwd + {len(sim_rev)} rev frames saved", flush=True)


def render_world_model(model_name, ckpt_path, num_steps, fwd_actions, rev_actions, start, end, device):
    from driftworld.models.sd_vae import FrozenSDVAE
    from driftworld.models import build_world_model

    sd_vae = FrozenSDVAE().to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    model = build_world_model(cfg.model)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    start_data = np.load(f"{DATA_DIR}/episode_{start:07d}.npz")
    end_data = np.load(f"{DATA_DIR}/episode_{end:07d}.npz")

    def rollout(z_init, rob_init, actions):
        z_t, rob = z_init.clone(), rob_init.clone()
        frames = [decode_uint8(sd_vae, z_t)]
        with torch.no_grad():
            for a in actions:
                a22 = torch.cat([a.to(device), rob]).unsqueeze(0)
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
        return frames

    z_fwd = torch.load(f"{DATA_DIR}/episode_{start:07d}_latent.pt", map_location=device).float().unsqueeze(0)
    rob_fwd = torch.tensor(start_data["robot_obs"], dtype=torch.float32, device=device)
    z_rev = torch.load(f"{DATA_DIR}/episode_{end:07d}_latent.pt", map_location=device).float().unsqueeze(0)
    rob_rev = torch.tensor(end_data["robot_obs"], dtype=torch.float32, device=device)

    print(f"{model_name} forward...", flush=True)
    fwd_frames = rollout(z_fwd, rob_fwd, fwd_actions)
    save_video(fwd_frames, OUTPUT_DIR / f"{model_name}_forward.mp4", model_name)
    print(f"  {len(fwd_frames)} frames", flush=True)

    print(f"{model_name} reversed...", flush=True)
    rev_frames = rollout(z_rev, rob_rev, rev_actions)
    save_video(rev_frames, OUTPUT_DIR / f"{model_name}_reversed.mp4", model_name)
    print(f"  {len(rev_frames)} frames", flush=True)

    del model
    torch.cuda.empty_cache()


def main():
    global DATA_DIR, OUTPUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="gt | sim | vae | flow5 | ddpm50")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path (not needed for gt/sim)")
    parser.add_argument("--num_steps", type=int, default=None, help="Inference steps override")
    parser.add_argument("--data_dir", default=DATA_DIR, help="Dataset directory")
    parser.add_argument("--output_dir", default=None, help="Output directory override")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start, end = load_trajectory()
    print(f"Trajectory: {start}->{end} (length={end - start})", flush=True)

    fwd_actions, rev_actions = load_actions(start, end)

    if args.model == "gt":
        render_gt(start, end, args.device)
    elif args.model == "sim":
        start_data = np.load(f"{DATA_DIR}/episode_{start:07d}.npz")
        end_data = np.load(f"{DATA_DIR}/episode_{end:07d}.npz")
        render_sim(start, end, [a.numpy() for a in fwd_actions], [a.numpy() for a in rev_actions], start_data, end_data)
    else:
        if args.ckpt is None:
            raise ValueError("--ckpt required for world model rendering")
        render_world_model(args.model, args.ckpt, args.num_steps, fwd_actions, rev_actions, start, end, args.device)

    print(f"Done: {args.model}", flush=True)


if __name__ == "__main__":
    main()
