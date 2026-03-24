"""Gym-like environment wrapping a frozen world model for imagination rollouts."""

import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class GoalSpec:
    """Holds goal information extracted from a CALVIN task annotation.

    Contains both the goal latent z_goal (for visual reward) and
    goal object position (for physical reward).
    """

    def __init__(self, z_goal: torch.Tensor, goal_obj_pos: torch.Tensor, task_name: str):
        self.z_goal = z_goal            # (4, 32, 32)
        self.goal_obj_pos = goal_obj_pos  # (3,) target object xyz
        self.task_name = task_name

    @staticmethod
    def from_calvin(
        data_dir: str,
        task_name: str = "slide_block_left",
        device: str = "cuda",
    ) -> "GoalSpec":
        """Extract goal from CALVIN task annotations.

        Loads the end frame of a labeled trajectory for the given task.
        """
        data_path = Path(data_dir)
        ann = np.load(
            data_path / "lang_annotations/auto_lang_ann.npy",
            allow_pickle=True,
        ).item()

        tasks = ann["language"]["task"]
        indx = ann["info"]["indx"]

        # Find first matching task
        for i, t in enumerate(tasks):
            if t == task_name:
                start, end = indx[i]
                end_frame = np.load(data_path / f"episode_{end:07d}.npz")

                # Load goal latent
                latent_path = data_path / f"episode_{end:07d}_latent.pt"
                z_goal = torch.load(latent_path, map_location=device).float()

                # Extract object position from scene_obs
                # scene_obs layout: 3 objects x 6D (xyz + rot3), then 6D slider/drawer
                scene = end_frame["scene_obs"]
                # Use block 2 position (indices 6:9) as default target
                # Different tasks may target different objects
                goal_obj_pos = torch.tensor(scene[6:9], device=device, dtype=torch.float32)

                return GoalSpec(z_goal=z_goal, goal_obj_pos=goal_obj_pos, task_name=task_name)

        raise ValueError(f"Task '{task_name}' not found in annotations")


class CombinedReward:
    """Combined reward: latent cosine similarity + physical EE-to-object distance.

    reward = α * cosine_sim(z_t, z_goal) + β * (-||ee_pos - goal_obj_pos||)
    """

    def __init__(
        self,
        goal: GoalSpec,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        self.goal = goal
        self.alpha = alpha
        self.beta = beta

    def __call__(
        self,
        z_t: torch.Tensor,
        robot_obs: torch.Tensor,
        prev_robot_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined reward.

        Args:
            z_t: (4, 32, 32) current predicted latent
            robot_obs: (15,) current predicted robot observation
            prev_robot_obs: (15,) previous robot observation

        Returns:
            scalar reward
        """
        reward = torch.tensor(0.0, device=z_t.device)

        # Component 1: latent cosine similarity to goal image
        z_flat = z_t.flatten()
        z_goal_flat = self.goal.z_goal.flatten().to(z_t.device)
        cosine_sim = F.cosine_similarity(z_flat.unsqueeze(0), z_goal_flat.unsqueeze(0))
        reward = reward + self.alpha * cosine_sim.squeeze()

        # Component 2: EE distance improvement toward goal object
        ee_pos = robot_obs[:3]
        prev_ee_pos = prev_robot_obs[:3]
        goal_pos = self.goal.goal_obj_pos.to(z_t.device)
        prev_dist = torch.norm(prev_ee_pos - goal_pos)
        curr_dist = torch.norm(ee_pos - goal_pos)
        reward = reward + self.beta * (prev_dist - curr_dist)

        return reward


class ImaginationEnv:
    """Wraps a frozen world model as a step-able environment.

    Returns both z_t (image latent) and robot_obs at each step,
    so the policy can be image-conditioned.
    """

    def __init__(
        self,
        world_model: nn.Module,
        dataset: Dataset,
        reward_fn: CombinedReward,
        max_steps: int = 8,
        num_inference_steps: int | None = None,
        device: str = "cuda",
    ):
        self.world_model = world_model
        self.world_model.eval()
        self.dataset = dataset
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.num_inference_steps = num_inference_steps
        self.device = device

        self._step_count = 0
        self.z_t = None
        self.robot_obs = None

    @torch.no_grad()
    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random initial state from the dataset.

        Returns:
            z_t: (4, 32, 32) initial image latent
            robot_obs: (15,) initial robot observation
        """
        idx = random.randint(0, len(self.dataset) - 1)
        sample = self.dataset[idx]
        self.z_t = sample["z_t"].unsqueeze(0).to(self.device)  # (1, 4, 32, 32)
        action_full = sample["action"]  # (22,) = [rel_actions(7), robot_obs(15)]
        self.robot_obs = action_full[7:].to(self.device)  # (15,)
        self._step_count = 0
        return self.z_t.squeeze(0).clone(), self.robot_obs.clone()

    @torch.no_grad()
    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float, bool, dict]:
        """Take one step in the imagined environment.

        Args:
            action: (7,) action from policy (rel_actions format)

        Returns:
            z_tp1: (4, 32, 32) next image latent
            robot_obs: (15,) next robot observation
            reward: scalar reward
            done: whether episode is over
            info: extra info dict
        """
        prev_robot_obs = self.robot_obs.clone()

        # Build 22D world model input: [action(7), robot_obs(15)]
        action_22d = torch.cat([action.to(self.device), self.robot_obs]).unsqueeze(0)

        # World model forward
        predict_kwargs = {}
        if self.num_inference_steps is not None and hasattr(self.world_model, 'predict'):
            import inspect
            sig = inspect.signature(self.world_model.predict)
            if 'num_steps' in sig.parameters:
                predict_kwargs['num_steps'] = self.num_inference_steps
        out = self.world_model.predict(self.z_t, action_22d, **predict_kwargs)
        if isinstance(out, tuple):
            z_tp1, robot_obs_pred = out
            robot_obs_pred = robot_obs_pred.squeeze(0)
        else:
            z_tp1 = out
            robot_obs_pred = self.robot_obs

        # Compute combined reward
        reward = self.reward_fn(
            z_t=z_tp1.squeeze(0),
            robot_obs=robot_obs_pred,
            prev_robot_obs=prev_robot_obs,
        )

        # Update state
        self.z_t = z_tp1
        self.robot_obs = robot_obs_pred
        self._step_count += 1
        done = self._step_count >= self.max_steps

        return self.z_t.squeeze(0).clone(), self.robot_obs.clone(), float(reward), done, {}
