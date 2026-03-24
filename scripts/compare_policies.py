"""Compare all 3 policies side-by-side in CALVIN simulator.

Produces:
1. Side-by-side comparison videos (3 panels)
2. Success rate comparison table
3. Summary JSON

Usage:
    uv run python scripts/compare_policies.py \
        --vae_checkpoint outputs/policy_vae/policy_final.pt \
        --flow_checkpoint outputs/policy_flow/policy_final.pt \
        --ddpm_checkpoint outputs/policy_ddpm/policy_final.pt \
        --num_episodes 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np

# Add scripts dir to path for eval_in_calvin import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_in_calvin import make_calvin_env, run_episode


def add_text(frame: np.ndarray, text: str, color: tuple = (255, 255, 255)) -> np.ndarray:
    """Add text overlay to frame."""
    frame = frame.copy()
    cv2.putText(frame, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def make_comparison_video(
    frames_dict: dict[str, list[np.ndarray]],
    output_path: str,
    fps: int = 12,
):
    """Create side-by-side comparison video from multiple policies.

    Args:
        frames_dict: {model_name: [frames]} — each frame is (H, W, 3) uint8
        output_path: where to save the mp4
        fps: frames per second
    """
    model_names = list(frames_dict.keys())
    max_len = max(len(f) for f in frames_dict.values())

    combined_frames = []
    for i in range(max_len):
        panels = []
        for name in model_names:
            frames = frames_dict[name]
            if i < len(frames):
                frame = add_text(frames[i], name)
            else:
                # Pad with last frame + "DONE" label
                frame = add_text(frames[-1], f"{name} (DONE)", color=(0, 255, 0))
            panels.append(frame)

        # Concatenate horizontally
        combined = np.concatenate(panels, axis=1)
        combined_frames.append(combined)

    imageio.mimwrite(output_path, combined_frames, fps=fps)
    print(f"Saved comparison video: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare policies in CALVIN")
    parser.add_argument("--vae_checkpoint", type=str, default="outputs/policy_vae/policy_final.pt")
    parser.add_argument("--flow_checkpoint", type=str, default="outputs/policy_flow/policy_final.pt")
    parser.add_argument("--ddpm_checkpoint", type=str, default="outputs/policy_ddpm/policy_final.pt")
    parser.add_argument("--dataset_path", type=str, default="data/calvin/task_D_D")
    parser.add_argument("--task", type=str, default="push_blue_block_left")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=360)
    parser.add_argument("--output_dir", type=str, default="outputs/comparison")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from driftworld.policy.calvin_wrapper import CalvinPolicyWrapper

    # Create environment (shared across all policies)
    print(f"Creating CALVIN environment")
    env = make_calvin_env(dataset_path=args.dataset_path, show_gui=False)

    # Load all policies
    checkpoints = {
        "VAE": args.vae_checkpoint,
        "Flow": args.flow_checkpoint,
        "DDPM": args.ddpm_checkpoint,
    }

    policies = {}
    for name, ckpt_path in checkpoints.items():
        if Path(ckpt_path).exists():
            print(f"Loading {name} policy: {ckpt_path}")
            policies[name] = CalvinPolicyWrapper(
                policy_checkpoint=ckpt_path,
                device=args.device,
            )
        else:
            print(f"WARNING: {name} checkpoint not found at {ckpt_path}, skipping")

    # Run all episodes
    all_results = {name: [] for name in policies}

    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")

        # Collect frames for comparison video
        ep_frames = {}

        for name, policy in policies.items():
            # Reset env to same state for fair comparison
            result = run_episode(env, policy, max_steps=args.max_steps, record=True)
            all_results[name].append(result)

            status = "OK" if result["success"] else "FAIL"
            print(f"  {name}: {status} (steps={result['steps']}, reward={result['total_reward']:.3f})")

            ep_frames[name] = result.get("frames", [])

        # Save comparison video for this episode
        if ep_frames:
            video_path = str(output_dir / f"comparison_ep{ep:02d}.mp4")
            make_comparison_video(ep_frames, video_path)

    # Summary table
    print(f"\n{'='*50}")
    print(f"{'Model':<10} {'Success Rate':<15} {'Avg Reward':<15} {'Avg Steps':<10}")
    print(f"{'-'*50}")

    summary = {}
    for name in policies:
        results = all_results[name]
        successes = sum(r["success"] for r in results)
        avg_reward = np.mean([r["total_reward"] for r in results])
        avg_steps = np.mean([r["steps"] for r in results])

        rate = successes / len(results)
        print(f"{name:<10} {successes}/{len(results)} ({rate:.0%}){'':>5} {avg_reward:<15.3f} {avg_steps:<10.1f}")

        summary[name] = {
            "successes": successes,
            "num_episodes": len(results),
            "success_rate": rate,
            "avg_reward": float(avg_reward),
            "avg_steps": float(avg_steps),
        }

    # Save summary
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_dir / 'comparison_results.json'}")

    env.close()


if __name__ == "__main__":
    main()
