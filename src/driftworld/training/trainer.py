"""Shared training loop for all world model variants."""

import os
from pathlib import Path

import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.calvin_dataset import CalvinLatentDataset
from ..models import build_world_model
from ..models.sd_vae import FrozenSDVAE
from .losses import compute_loss


class Trainer:
    """Trains a world model on precomputed latent transitions.

    Handles: DataLoader, optimizer, scheduler, mixed precision, logging, checkpoints.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Seed
        torch.manual_seed(cfg.training.seed)

        # Build model
        self.model = build_world_model(cfg.model)
        self.model.to(self.device)
        self.model_type = cfg.model.type

        # Count trainable parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model type: {self.model_type}, trainable params: {num_params:,}")

        # Frozen SD-VAE for visualization (decode latents -> images)
        self.sd_vae = FrozenSDVAE().to(self.device)

        # Datasets
        load_dinov2 = cfg.dataset.get("load_dinov2_features", False)
        robot_obs_key = cfg.dataset.get("robot_obs_key", "robot_obs")
        self.train_dataset = CalvinLatentDataset(
            data_dir=os.path.join(cfg.dataset.data_dir, cfg.dataset.split),
            action_key=cfg.dataset.action_key,
            robot_obs_key=robot_obs_key,
            load_dinov2=load_dinov2,
        )
        self.val_dataset = CalvinLatentDataset(
            data_dir=os.path.join(cfg.dataset.data_dir, cfg.dataset.val_split),
            action_key=cfg.dataset.action_key,
            robot_obs_key=robot_obs_key,
            load_dinov2=load_dinov2,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
        )

        # Cache a fixed validation batch for consistent visualization
        self._vis_batch = None

        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            betas=(0.9, 0.999),
        )

        # LR scheduler: linear warmup + cosine decay
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg.training.lr,
            total_steps=cfg.training.total_steps,
            pct_start=cfg.training.warmup_steps / cfg.training.total_steps,
            anneal_strategy="cos",
        )

        # Mixed precision
        self.use_fp16 = cfg.training.fp16
        self.scaler = torch.amp.GradScaler("cuda") if self.use_fp16 else None

        # Output directory
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # W&B logging
        # Extract dataset dir basename for run naming (e.g., "task_D_D", "calvin_debug_dataset")
        data_dir = Path(cfg.dataset.data_dir)
        dataset_tag = data_dir.name
        wandb.init(
            project="driftworld",
            name=f"{self.model_type}_{cfg.dataset.name}_{dataset_tag}",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

        self.global_step = 0

    def _get_vis_batch(self) -> dict:
        """Get a fixed validation batch for consistent visualization across steps."""
        if self._vis_batch is None:
            self._vis_batch = next(iter(self.val_loader))
        return self._vis_batch

    def train(self):
        """Main training loop."""
        total_steps = self.cfg.training.total_steps
        pbar = tqdm(total=total_steps, desc="Training")

        while self.global_step < total_steps:
            for batch in self.train_loader:
                if self.global_step >= total_steps:
                    break

                loss_dict = self._train_step(batch)
                self.global_step += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss_dict['loss'].item():.4f}")

                # Logging
                if self.global_step % self.cfg.training.log_every == 0:
                    log_dict = {
                        f"train/{k}": v.item() for k, v in loss_dict.items()
                    }
                    log_dict["train/lr"] = self.scheduler.get_last_lr()[0]
                    wandb.log(log_dict, step=self.global_step)

                # Visualization
                if self.global_step % self.cfg.training.vis_every == 0:
                    self._log_visualizations()

                # Validation
                if self.global_step % self.cfg.training.eval_every == 0:
                    val_loss = self._validate()
                    wandb.log({"val/loss": val_loss}, step=self.global_step)
                    print(
                        f"\nStep {self.global_step}: "
                        f"train_loss={loss_dict['loss'].item():.4f}, "
                        f"val_loss={val_loss:.4f}"
                    )

                # Save checkpoint
                if self.global_step % self.cfg.training.save_every == 0:
                    self._save_checkpoint()

        pbar.close()
        self._save_checkpoint(final=True)
        wandb.finish()

    def _train_step(self, batch: dict) -> dict:
        """Single training step."""
        self.model.train()
        z_t = batch["z_t"].to(self.device)
        action = batch["action"].to(self.device)
        z_tp1 = batch["z_tp1"].to(self.device)

        with torch.amp.autocast("cuda", enabled=self.use_fp16):
            loss_dict = compute_loss(
                self.model, z_t, action, z_tp1, batch,
                model_type=self.model_type,
                kl_weight=self.cfg.model.vae.get("kl_weight", 1e-4),
                robot_obs_weight=self.cfg.training.get("robot_obs_weight", 1.0),
            )

        loss = loss_dict["loss"]

        self.optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.training.gradient_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.training.gradient_clip
            )
            self.optimizer.step()

        self.scheduler.step()
        return loss_dict

    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            z_t = batch["z_t"].to(self.device)
            action = batch["action"].to(self.device)
            z_tp1 = batch["z_tp1"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_fp16):
                loss_dict = compute_loss(
                    self.model, z_t, action, z_tp1, batch,
                    model_type=self.model_type,
                    kl_weight=self.cfg.model.vae.get("kl_weight", 1e-4),
                    robot_obs_weight=self.cfg.training.get("robot_obs_weight", 1.0),
                )
            total_loss += loss_dict["loss"].item()
            num_batches += 1

            if num_batches >= 50:  # cap validation batches
                break

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _log_visualizations(self):
        """Log prediction images and robot_obs to wandb."""
        self.model.eval()
        batch = self._get_vis_batch()
        z_t = batch["z_t"].to(self.device)
        action = batch["action"].to(self.device)
        z_tp1 = batch["z_tp1"].to(self.device)
        robot_obs_gt = batch["robot_obs_tp1"].to(self.device)

        # Get predictions
        out = self.model.predict(z_t, action)
        if isinstance(out, tuple):
            z_pred, robot_obs_pred = out
        else:
            z_pred = out
            robot_obs_pred = None

        # Decode latents to images (take first 4 samples)
        n = min(4, z_t.shape[0])
        img_input = self.sd_vae.decode(z_t[:n])     # (n, 3, 256, 256) in [-1,1]
        img_gt = self.sd_vae.decode(z_tp1[:n])
        img_pred = self.sd_vae.decode(z_pred[:n])

        # Build comparison grid: each row is [input, GT, predicted]
        images = []
        for i in range(n):
            row = torch.cat([img_input[i], img_gt[i], img_pred[i]], dim=2)  # concat W
            row = ((row + 1) / 2).clamp(0, 1)  # [-1,1] -> [0,1]
            images.append(row)
        grid = torch.cat(images, dim=1)  # concat H
        grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        wandb.log({
            "vis/predictions": wandb.Image(
                grid_np,
                caption="Left: input(t) | Middle: GT(t+1) | Right: predicted(t+1)",
            ),
        }, step=self.global_step)

        # Log robot_obs comparison as a table
        if robot_obs_pred is not None:
            obs_labels = [
                "x", "y", "z", "rx", "ry", "rz",  # TCP position + orientation
                "gripper_w",  # gripper width
                "j1", "j2", "j3", "j4", "j5", "j6", "j7",  # joint positions
                "gripper_action",
            ]
            # Log per-dimension error for first sample
            pred_np = robot_obs_pred[0].cpu().numpy()
            gt_np = robot_obs_gt[0].cpu().numpy()
            table = wandb.Table(
                columns=["dimension", "predicted", "ground_truth", "abs_error"],
                data=[
                    [obs_labels[j] if j < len(obs_labels) else f"dim_{j}",
                     float(pred_np[j]), float(gt_np[j]),
                     float(abs(pred_np[j] - gt_np[j]))]
                    for j in range(len(pred_np))
                ],
            )
            wandb.log({"vis/robot_obs": table}, step=self.global_step)

    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        suffix = "final" if final else f"step_{self.global_step}"
        path = self.output_dir / f"checkpoint_{suffix}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "config": OmegaConf.to_container(self.cfg),
            },
            path,
        )
        print(f"\nSaved checkpoint to {path}")
