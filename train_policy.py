"""Train PPO policy in imagination using a frozen world model.

Uses combined reward: latent cosine similarity + physical EE-to-object distance.

Usage:
    uv run python train_policy.py \
        --world_model_checkpoint outputs/ddpm_calvin/checkpoint_step_190000.pt \
        --output_dir outputs/policy_ddpm \
        --task slide_block_left

    # Train for all 3 models:
    uv run python train_policy.py --world_model_checkpoint outputs/vae_calvin/checkpoint_step_150000.pt --output_dir outputs/policy_vae
    uv run python train_policy.py --world_model_checkpoint outputs/flow_calvin/checkpoint_step_170000.pt --output_dir outputs/policy_flow
    uv run python train_policy.py --world_model_checkpoint outputs/ddpm_calvin/checkpoint_step_190000.pt --output_dir outputs/policy_ddpm
"""

import argparse
import os

import torch
import wandb
from omegaconf import OmegaConf

from driftworld.data.calvin_dataset import CalvinLatentDataset
from driftworld.models import build_world_model
from driftworld.policy.imagination_env import CombinedReward, GoalSpec, ImaginationEnv
from driftworld.policy.networks import PolicyNetwork, ValueNetwork
from driftworld.policy.ppo import PPOTrainer


def main():
    parser = argparse.ArgumentParser(description="Train PPO policy in imagination")
    parser.add_argument("--world_model_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/policy")
    parser.add_argument("--total_iterations", type=int, default=1000)
    parser.add_argument("--num_trajectories", type=int, default=256)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--task", type=str, default="push_blue_block_left",
                        help="CALVIN task name for goal extraction")
    parser.add_argument("--reward_alpha", type=float, default=1.0,
                        help="Weight for latent cosine similarity reward")
    parser.add_argument("--reward_beta", type=float, default=1.0,
                        help="Weight for physical EE-to-object distance reward")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="Override world model inference steps (default: use model's default)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Load world model
    print(f"Loading world model: {args.world_model_checkpoint}")
    ckpt = torch.load(args.world_model_checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])

    world_model = build_world_model(cfg.model)
    world_model.load_state_dict(ckpt["model_state_dict"])
    world_model.to(device)
    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad = False

    model_type = cfg.model.type
    print(f"World model type: {model_type}")

    # Dataset for initial state sampling
    train_data_dir = os.path.join(cfg.dataset.data_dir, cfg.dataset.split)
    train_dataset = CalvinLatentDataset(
        data_dir=train_data_dir,
        action_key=cfg.dataset.action_key,
    )

    # Extract goal from task annotations
    print(f"Extracting goal for task: {args.task}")
    goal = GoalSpec.from_calvin(
        data_dir=train_data_dir,
        task_name=args.task,
        device=args.device,
    )
    print(f"  Goal object position: {goal.goal_obj_pos}")

    # Combined reward
    reward_fn = CombinedReward(
        goal=goal,
        alpha=args.reward_alpha,
        beta=args.reward_beta,
    )

    # Imagination environment
    env = ImaginationEnv(
        world_model=world_model,
        dataset=train_dataset,
        reward_fn=reward_fn,
        max_steps=args.chunk_size,
        num_inference_steps=args.num_inference_steps,
        device=args.device,
    )

    # Policy and value networks (image-conditioned)
    obs_dim = cfg.model.backbone.get("robot_obs_dim", 15)
    action_dim = 7
    policy = PolicyNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        chunk_size=args.chunk_size,
        latent_channels=4,
        latent_feature_dim=128,
    )
    value_net = ValueNetwork(
        obs_dim=obs_dim,
        latent_channels=4,
        latent_feature_dim=128,
    )

    num_params = sum(p.numel() for p in policy.parameters()) + sum(p.numel() for p in value_net.parameters())
    print(f"Policy + Value params: {num_params:,}")

    # Wandb
    wandb.init(
        project="driftworld",
        name=f"ppo_{model_type}_{args.task}",
        config={
            "world_model": model_type,
            "world_model_checkpoint": args.world_model_checkpoint,
            "total_iterations": args.total_iterations,
            "num_trajectories": args.num_trajectories,
            "chunk_size": args.chunk_size,
            "policy_lr": args.policy_lr,
            "task": args.task,
            "reward_alpha": args.reward_alpha,
            "reward_beta": args.reward_beta,
        },
    )

    # PPO Trainer
    trainer = PPOTrainer(
        policy=policy,
        value_net=value_net,
        env=env,
        policy_lr=args.policy_lr,
        num_trajectories=args.num_trajectories,
        chunk_size=args.chunk_size,
        device=args.device,
    )

    # Train
    trainer.train(
        total_iterations=args.total_iterations,
        log_every=10,
        save_every=200,
        save_dir=args.output_dir,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
