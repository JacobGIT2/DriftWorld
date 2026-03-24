"""PPO trainer for image-conditioned policy learning in imagination."""

import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from .imagination_env import ImaginationEnv
from .networks import PolicyNetwork, ValueNetwork


class RolloutBuffer:
    """Stores trajectories collected during imagination rollouts."""

    def __init__(self):
        self.z_ts = []
        self.robot_obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, z_t, robot_obs, action, log_prob, reward, value, done):
        self.z_ts.append(z_t)
        self.robot_obs.append(robot_obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(
        self, gamma: float = 0.99, gae_lambda: float = 0.95
    ):
        """Compute GAE advantages and discounted returns."""
        T = len(self.rewards)
        advantages = torch.zeros(T, device=self.rewards[0].device)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
            else:
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t].float()) - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - self.dones[t].float()) * last_gae

        self.advantages = advantages
        self.returns = advantages + torch.stack(self.values)

    def get_batches(self, minibatch_size: int):
        """Yield minibatches for PPO updates."""
        z_ts = torch.stack(self.z_ts)          # (T, 4, 32, 32)
        robot_obs = torch.stack(self.robot_obs)  # (T, 15)
        actions = torch.stack(self.actions)      # (T, K, 7)
        log_probs = torch.stack(self.log_probs)  # (T,)

        T = z_ts.shape[0]
        indices = torch.randperm(T)

        for start in range(0, T, minibatch_size):
            end = min(start + minibatch_size, T)
            idx = indices[start:end]
            yield {
                "z_t": z_ts[idx],
                "robot_obs": robot_obs[idx],
                "actions": actions[idx],
                "old_log_probs": log_probs[idx],
                "advantages": self.advantages[idx],
                "returns": self.returns[idx],
            }

    def clear(self):
        self.__init__()


class PPOTrainer:
    """Proximal Policy Optimization trainer for imagination-based policy learning."""

    def __init__(
        self,
        policy: PolicyNetwork,
        value_net: ValueNetwork,
        env: ImaginationEnv,
        policy_lr: float = 3e-4,
        value_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        epochs_per_update: int = 4,
        minibatch_size: int = 64,
        num_trajectories: int = 256,
        chunk_size: int = 8,
        max_grad_norm: float = 0.5,
        device: str = "cuda",
    ):
        self.policy = policy.to(device)
        self.value_net = value_net.to(device)
        self.env = env
        self.device = device

        self.policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(value_net.parameters(), lr=value_lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.epochs_per_update = epochs_per_update
        self.minibatch_size = minibatch_size
        self.num_trajectories = num_trajectories
        self.chunk_size = chunk_size
        self.max_grad_norm = max_grad_norm

    def collect_trajectories(self) -> tuple[RolloutBuffer, dict]:
        """Collect trajectories by rolling out policy in imagination."""
        self.policy.eval()
        self.value_net.eval()
        buffer = RolloutBuffer()
        total_rewards = []

        for _ in range(self.num_trajectories):
            z_t, robot_obs = self.env.reset()  # (4,32,32), (15,)
            episode_reward = 0.0

            # Get action chunk from policy
            with torch.no_grad():
                z_t_batch = z_t.unsqueeze(0)          # (1, 4, 32, 32)
                obs_batch = robot_obs.unsqueeze(0)     # (1, 15)
                actions, log_probs = self.policy.sample(z_t_batch, obs_batch)
                value = self.value_net(z_t_batch, obs_batch)

            actions = actions.squeeze(0)  # (K, 7)

            # Execute each action in the chunk
            for k in range(self.chunk_size):
                action_k = actions[k]  # (7,)
                next_z_t, next_robot_obs, reward, done, _ = self.env.step(action_k)

                buffer.add(
                    z_t=z_t,
                    robot_obs=robot_obs,
                    action=actions,  # store full chunk for log_prob recomputation
                    log_prob=log_probs.squeeze(0),
                    reward=torch.tensor(reward, device=self.device),
                    value=value.squeeze(0),
                    done=torch.tensor(done, device=self.device),
                )

                episode_reward += reward
                z_t = next_z_t
                robot_obs = next_robot_obs

                if done:
                    break

            total_rewards.append(episode_reward)

        buffer.compute_returns_and_advantages(self.gamma, self.gae_lambda)

        stats = {
            "mean_reward": sum(total_rewards) / len(total_rewards),
            "min_reward": min(total_rewards),
            "max_reward": max(total_rewards),
        }
        return buffer, stats

    def update(self, buffer: RolloutBuffer) -> dict:
        """Run PPO update on collected trajectories."""
        self.policy.train()
        self.value_net.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        num_updates = 0

        for _ in range(self.epochs_per_update):
            for batch in buffer.get_batches(self.minibatch_size):
                z_t = batch["z_t"]
                robot_obs = batch["robot_obs"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Recompute log probs and entropy
                new_log_probs = self.policy.log_prob(z_t, robot_obs, actions)
                entropy = self.policy.entropy(z_t, robot_obs).mean()

                # Policy loss (clipped)
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values = self.value_net(z_t, robot_obs)
                value_loss = nn.functional.mse_loss(values, returns)

                # Combined loss
                loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

                # Update
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl += approx_kl.item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "approx_kl": total_kl / max(num_updates, 1),
        }

    def train(self, total_iterations: int, log_every: int = 10, save_every: int = 100,
              save_dir: str = "outputs/policy"):
        """Main training loop."""
        from pathlib import Path
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for iteration in tqdm(range(1, total_iterations + 1), desc="PPO Training"):
            buffer, collect_stats = self.collect_trajectories()
            update_stats = self.update(buffer)
            buffer.clear()

            if iteration % log_every == 0:
                log_dict = {
                    "policy/mean_reward": collect_stats["mean_reward"],
                    "policy/min_reward": collect_stats["min_reward"],
                    "policy/max_reward": collect_stats["max_reward"],
                    "policy/policy_loss": update_stats["policy_loss"],
                    "policy/value_loss": update_stats["value_loss"],
                    "policy/entropy": update_stats["entropy"],
                    "policy/approx_kl": update_stats["approx_kl"],
                }
                wandb.log(log_dict, step=iteration)
                tqdm.write(
                    f"Iter {iteration}: reward={collect_stats['mean_reward']:.4f} "
                    f"ploss={update_stats['policy_loss']:.4f} "
                    f"vloss={update_stats['value_loss']:.4f}"
                )

            if iteration % save_every == 0:
                self.save(save_path / f"policy_step_{iteration}.pt")

        self.save(save_path / "policy_final.pt")
        print(f"Training complete. Saved to {save_path}")

    def save(self, path):
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value_net.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.value_net.load_state_dict(ckpt["value_state_dict"])
