"""Evaluate a trained policy in the CALVIN simulator and record videos.

Usage:
    uv run python scripts/eval_in_calvin.py \
        --policy_checkpoint outputs/policy_vae/policy_final.pt \
        --task push_blue_block_left \
        --num_episodes 10 \
        --output_dir outputs/calvin_eval/vae/

    # Compare all 3:
    uv run python scripts/eval_in_calvin.py --policy_checkpoint outputs/policy_vae/policy_final.pt --output_dir outputs/calvin_eval/vae
    uv run python scripts/eval_in_calvin.py --policy_checkpoint outputs/policy_flow/policy_final.pt --output_dir outputs/calvin_eval/flow
    uv run python scripts/eval_in_calvin.py --policy_checkpoint outputs/policy_ddpm/policy_final.pt --output_dir outputs/calvin_eval/ddpm
"""

import argparse
import json
import os
import sys
from pathlib import Path

import imageio
import numpy as np

# CALVIN env needs to be loaded from the cloned repo with URDF assets
CALVIN_REPO = os.environ.get("CALVIN_ROOT", "/tmp/calvin_repo/calvin_env")
if CALVIN_REPO not in sys.path:
    sys.path.insert(0, CALVIN_REPO)

from calvin_env.envs.play_table_env import get_env


def make_calvin_env(
    dataset_path: str = "data/calvin/task_D_D/training",
    show_gui: bool = False,
):
    """Create a CALVIN PlayTable environment using get_env helper."""
    env = get_env(
        dataset_path=dataset_path,
        obs_space={"rgb_obs": ["rgb_static"], "depth_obs": [], "state_obs": ["robot_obs"]},
        show_gui=show_gui,
    )
    return env


def run_episode(
    env: PlayTableSimEnv,
    policy,
    max_steps: int = 360,
    record: bool = True,
) -> dict:
    """Run one episode and optionally record frames.

    Returns:
        dict with keys: success, total_reward, steps, frames (if recorded)
    """
    obs = env.reset()
    policy.reset()

    frames = []
    total_reward = 0.0

    for step in range(max_steps):
        # Get action from policy
        action = policy.act(obs)

        # Step environment
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Record frame
        if record:
            frame = obs["rgb_obs"]["rgb_static"]  # (200, 200, 3) uint8
            frames.append(frame.copy())

        if done:
            break

    result = {
        "steps": step + 1,
        "total_reward": float(total_reward),
        "success": bool(info.get("success", False)),
    }
    if record:
        result["frames"] = frames

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate policy in CALVIN simulator")
    parser.add_argument("--policy_checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="data/calvin/task_D_D")
    parser.add_argument("--task", type=str, default="push_blue_block_left")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=360)
    parser.add_argument("--output_dir", type=str, default="outputs/calvin_eval")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_video", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import wrapper here to avoid loading torch before args are parsed
    from driftworld.policy.calvin_wrapper import CalvinPolicyWrapper

    # Create environment
    print(f"Creating CALVIN environment from {args.dataset_path}")
    env = make_calvin_env(dataset_path=args.dataset_path, show_gui=False)

    # Load policy
    print(f"Loading policy from {args.policy_checkpoint}")
    policy = CalvinPolicyWrapper(
        policy_checkpoint=args.policy_checkpoint,
        device=args.device,
    )

    # Run episodes
    results = []
    for ep in range(args.num_episodes):
        print(f"Episode {ep + 1}/{args.num_episodes}...", end=" ", flush=True)
        result = run_episode(
            env, policy,
            max_steps=args.max_steps,
            record=not args.no_video,
        )
        results.append(result)
        status = "SUCCESS" if result["success"] else "FAIL"
        print(f"{status} (steps={result['steps']}, reward={result['total_reward']:.3f})")

        # Save video
        if not args.no_video and result.get("frames"):
            video_path = output_dir / f"episode_{ep:02d}.mp4"
            imageio.mimwrite(str(video_path), result["frames"], fps=12)

    # Summary
    successes = sum(r["success"] for r in results)
    avg_reward = np.mean([r["total_reward"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])

    summary = {
        "policy_checkpoint": args.policy_checkpoint,
        "task": args.task,
        "num_episodes": args.num_episodes,
        "successes": successes,
        "success_rate": successes / args.num_episodes,
        "avg_reward": float(avg_reward),
        "avg_steps": float(avg_steps),
    }

    print(f"\n=== Results ===")
    print(f"Success rate: {successes}/{args.num_episodes} ({summary['success_rate']:.1%})")
    print(f"Avg reward: {avg_reward:.3f}")
    print(f"Avg steps: {avg_steps:.1f}")

    # Save summary
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {output_dir / 'results.json'}")

    env.close()


if __name__ == "__main__":
    main()
