"""
Training utilities for PyTorch implementation.

Converted from TensorFlow implementation.
Original Authors: Ankush Gupta, Tomas Jakab
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from ..utils.colorize import colorize
from ..models.imm_model import IMMModel
import csv
import json


class Trainer:
    """Main trainer class for IMM model."""
    
    def __init__(self, model: nn.Module, config: Any, log_dir: str, 
                 device: torch.device = None, rank: int = 0, world_size: int = 1):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            log_dir: Directory for logging
            device: Training device
            rank: Process rank for distributed training
            world_size: Total number of processes
        """
        self.config = config
        self.log_dir = log_dir
        self.rank = rank
        self.world_size = world_size
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Optimizer/scheduler will be set up AFTER a warm-up forward
        # to ensure all lazily created parameters (e.g., renderer layers)
        # are materialized before optimizer construction.
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
        
        # Setup logging
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

        # Histories for loss plotting
        self.history_train_loss: List[float] = []
        self.history_val_loss: List[float] = []

        # Log files
        self._log_file_path = os.path.join(self.log_dir, 'train.log')
        self._csv_file_path = os.path.join(self.log_dir, 'losses.csv')
        # Ensure logdir exists
        os.makedirs(self.log_dir, exist_ok=True)
        
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        if self.optimizer is not None:
            return
        # Get optimizer parameters
        lr = self.config.training.lr.start_val
        weight_decay = self.config.training.get('weight_decay', 1e-4)
        
        if self.config.training.optim.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif self.config.training.optim.lower() == 'adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=lr, rho=0.95, eps=1e-06)
        elif self.config.training.optim.lower() == 'adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f'Optimizer = {self.config.training.optim} not supported')
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, 
            gamma=self.config.training.lr.decay,
            last_epoch=-1
        )
        
        # Apply LR schedule every N steps
        self.lr_step_interval = self.config.training.lr.step
    
    
    def save_visualization(self, batch, outputs, step):
        """Save visualization images for monitoring training progress."""
        if self.rank != 0:  # Only save on main process
            return
            
        try:
            import matplotlib.pyplot as plt
            import os
            
            # Create visualization directory
            viz_dir = os.path.join(self.log_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # Get first sample from batch
            input_img = batch['image'][0].detach().cpu()  # [C, H, W]
            target_img = batch['future_image'][0].detach().cpu()  # [C, H, W]
            pred_img = outputs['future_im_pred'][0].detach().cpu()  # [C, H, W]
            
            # Convert to numpy and handle range
            def to_numpy_img(img_tensor):
                img = img_tensor.permute(1, 2, 0).numpy()  # [H, W, C]
                # Normalize to [0,1] for display
                if img.max() > 2.0:  # Assume [0,255] range
                    img = img / 255.0
                img = np.clip(img, 0, 1)
                return img
            
            input_np = to_numpy_img(input_img)
            target_np = to_numpy_img(target_img)
            pred_np = to_numpy_img(pred_img)
            
            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(input_np)
            axes[0].set_title(f'Input\nRange: [{input_img.min():.1f}, {input_img.max():.1f}]')
            axes[0].axis('off')
            
            axes[1].imshow(target_np)
            axes[1].set_title(f'Target\nRange: [{target_img.min():.1f}, {target_img.max():.1f}]')
            axes[1].axis('off')
            
            axes[2].imshow(pred_np)
            axes[2].set_title(f'Prediction\nRange: [{pred_img.min():.1f}, {pred_img.max():.1f}]')
            axes[2].axis('off')
            
            plt.suptitle(f'Step {step} - Training Progress', fontsize=16)
            plt.tight_layout()
            
            # Save image
            save_path = os.path.join(viz_dir, f'step_{step:06d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📸 Visualization saved: {save_path}")
            
        except Exception as e:
            print(f"⚠️  Failed to save visualization: {e}")
            
    def _plot_and_save_loss_curve(self):
        """Plot and save the training and validation loss curve."""
        if self.rank != 0 or not self.history_train_loss:
            return
        
        plt.figure(figsize=(10, 6))
        epochs = range(len(self.history_train_loss))
        
        # Plot training loss
        plt.plot(epochs, self.history_train_loss, 'b-o', label='Training Loss')
        
        # Plot validation loss if available
        if self.history_val_loss:
            plt.plot(epochs, self.history_val_loss, 'r-o', label='Validation Loss')
            
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(self.log_dir, 'loss_curve.png')
        plt.savefig(save_path)
        plt.close()
        print(colorize(f'Saved loss curve to {save_path}', 'green', bold=True))

    def log_tensorboard_images(self, batch, outputs, step, phase='train'):    
            """Log images to TensorBoard for monitoring progress."""
            if self.writer is None or self.rank != 0:
                return
                
            try:
                # Get first few samples from batch (max 4 for display)
                batch_size = min(4, batch['image'].shape[0])
                
                # Prepare images for logging
                input_imgs = batch['image'][:batch_size].detach().cpu()  # [B, C, H, W]
                target_imgs = batch['future_image'][:batch_size].detach().cpu()
                pred_imgs = outputs['future_im_pred'][:batch_size].detach().cpu()
                
                # Normalize to [0,1] for TensorBoard
                def normalize_for_tb(imgs):
                    if imgs.max() > 2.0:  # Assume [0,255] range
                        imgs = imgs / 255.0
                    return torch.clamp(imgs, 0, 1)
                
                input_imgs = normalize_for_tb(input_imgs)
                target_imgs = normalize_for_tb(target_imgs)
                pred_imgs = normalize_for_tb(pred_imgs)
                
                # Log individual image sets
                self.writer.add_images(f'{phase}/input_images', input_imgs, step)
                self.writer.add_images(f'{phase}/target_images', target_imgs, step)
                self.writer.add_images(f'{phase}/predicted_images', pred_imgs, step)
                
                # Create comparison grid (side by side)
                comparison_imgs = []
                for i in range(batch_size):
                    # Stack horizontally: input | target | prediction
                    comparison = torch.cat([input_imgs[i], target_imgs[i], pred_imgs[i]], dim=2)  # [C, H, W*3]
                    comparison_imgs.append(comparison)
                
                comparison_grid = torch.stack(comparison_imgs, dim=0)  # [B, C, H, W*3]
                self.writer.add_images(f'{phase}/comparison_grid', comparison_grid, step)
                
                # Log landmarks if available
                if 'gauss_yx' in outputs:
                    self.log_tensorboard_landmarks(batch, outputs, step, phase)
                    
            except Exception as e:
                print(f"⚠️  Failed to log TensorBoard images: {e}")

    def log_tensorboard_landmarks(self, batch, outputs, step, phase='train'):
        """Log landmark visualizations to TensorBoard."""
        if self.writer is None or self.rank != 0:
            return
            
        try:
            from ..utils.utils import colorize_landmark_maps
            
            # Get landmark maps if available
            if 'pose_embeddings' in outputs and len(outputs['pose_embeddings']) > 0:
                batch_size = min(4, batch['image'].shape[0])
                
                # Get largest resolution pose embedding
                pose_maps = outputs['pose_embeddings'][0][:batch_size]  # [B, H, W, N]
                
                # Colorize landmarks
                colored_landmarks = []
                for i in range(batch_size):
                    colored_map = colorize_landmark_maps(pose_maps[i:i+1])  # [1, H, W, 3]
                    colored_map = colored_map.squeeze(0).permute(2, 0, 1)  # [3, H, W]
                    colored_landmarks.append(colored_map)
                
                colored_landmarks = torch.stack(colored_landmarks, dim=0)  # [B, 3, H, W]
                colored_landmarks = torch.clamp(colored_landmarks, 0, 1)
                
                self.writer.add_images(f'{phase}/landmarks', colored_landmarks, step)
                
        except Exception as e:
            print(f"⚠️  Failed to log landmarks: {e}")

    def log_tensorboard_scalars(self, metrics_dict, step, phase='train'):
        """Log scalar metrics to TensorBoard."""
        if self.writer is None or self.rank != 0:
            return
            
        for metric_name, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.writer.add_scalar(f'{phase}/{metric_name}', value, step)
    def _write_text_log(self, text: str):
        """Append a line to the text log file (rank 0 only)."""
        if self.rank != 0:
            return
        try:
            with open(self._log_file_path, 'a', encoding='utf-8') as f:
                f.write(text + '\n')
        except Exception as e:
            print(colorize(f"Failed writing train.log: {e}", 'red', bold=True))

    def _write_csv_row(self, row: Dict[str, Any]):
        """Append a metrics row to losses.csv (rank 0 only)."""
        if self.rank != 0:
            return
        header = ['type', 'epoch', 'step', 'loss', 'val_loss', 'lr', 'batch_time', 'examples_per_sec']
        file_exists = os.path.exists(self._csv_file_path) and os.path.getsize(self._csv_file_path) > 0
        try:
            with open(self._csv_file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                if not file_exists:
                    writer.writeheader()
                # Ensure all keys present
                full_row = {h: row.get(h, '') for h in header}
                writer.writerow(full_row)
        except Exception as e:
            print(colorize(f"Failed writing losses.csv: {e}", 'red', bold=True))

    def _save_loss_histories(self):
        """Persist loss histories to JSON and refresh plot (rank 0 only)."""
        if self.rank != 0:
            return
        # Save JSON with current histories
        try:
            hist_path = os.path.join(self.log_dir, 'loss_history.json')
            with open(hist_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'train_loss': self.history_train_loss,
                    'val_loss': self.history_val_loss
                }, f)
        except Exception as e:
            print(colorize(f"Failed writing loss_history.json: {e}", 'red', bold=True))
        # Update plot
        self._plot_and_save_loss_curve()

    def warmup_with_batch(self, sample_batch: Dict[str, torch.Tensor]):
        """
        Run a warm-up forward pass to ensure any lazily created layers are
        instantiated BEFORE building the optimizer.
        """
        # Move sample to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in sample_batch.items()}
        self.model.eval()
        with torch.no_grad():
            _ = self.model(batch, training=False)
        # After warmup, build optimizer if not set
        self.setup_optimizer()
    
    def train_epoch(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_metrics = defaultdict(list)
        
        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            start_time = time.time()
            
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch, training=True)
            
            # Compute loss (unwrap DDP if needed)
            loss = self._get_model_for_loss().compute_loss(outputs, batch, training=True)
            
            if self.global_step % 10 == 0:  # Print every 10 steps
                print(f"Step {self.global_step}: Loss = {loss.item():.6f}, "
                    f"Pred range = [{outputs['future_im_pred'].min():.2f}, {outputs['future_im_pred'].max():.2f}]")
            
                    # Log metrics to TensorBoard every step
            train_metrics = {
                'loss': loss.item(),
                'pred_min': outputs['future_im_pred'].min().item(),
                'pred_max': outputs['future_im_pred'].max().item(),
                'pred_mean': outputs['future_im_pred'].mean().item(),
            }
            
            # Add learning rate
            if self.optimizer is not None:
                train_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
                
            self.log_tensorboard_scalars(train_metrics, self.global_step, 'train')
            
            # Console logging every 10 steps
            if self.global_step % 10 == 0:
                print(f"Step {self.global_step}: Loss = {loss.item():.6f}, "
                    f"Pred range = [{outputs['future_im_pred'].min():.2f}, {outputs['future_im_pred'].max():.2f}]")
            
            # Image logging every 100 steps (more frequent than file saves)
            if self.global_step % 100 == 0:
                self.log_tensorboard_images(batch, outputs, self.global_step, 'train')
            
            # File visualization every 1000 steps
            if self.global_step % 1000 == 0 and self.global_step > 0:
                self.save_visualization(batch, outputs, self.global_step)
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if hasattr(self.config.training, 'gradclip'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradclip)
            
            self.optimizer.step()
            
            # Update learning rate
            if self.global_step > 0 and self.global_step % self.lr_step_interval == 0:
                self.scheduler.step()
            
            # Metrics
            batch_time = time.time() - start_time
            epoch_metrics['loss'].append(loss.item())
            epoch_metrics['batch_time'].append(batch_time)
            
            # Logging
            if self.rank == 0:
                if self.global_step % 10 == 0:  # Log every 10 steps
                    examples_per_sec = len(batch['image']) / batch_time
                    msg = (f'{datetime.now()}: step {self.global_step}, loss = {loss.item():.4f} '
                           f'({examples_per_sec:.1f} examples/sec) {batch_time:.3f} sec/batch')
                    print(msg)
                    # Mirror to text log and CSV
                    self._write_text_log(msg)
                    self._write_csv_row({
                        'type': 'step',
                        'epoch': self.epoch,
                        'step': self.global_step,
                        'loss': loss.item(),
                        'lr': self.optimizer.param_groups[0]['lr'],
                        'batch_time': batch_time,
                        'examples_per_sec': examples_per_sec
                    })
                
                if self.global_step % 100 == 0:  # Tensorboard logging
                    self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                    
                    # Log images
                    if 'future_im_pred' in outputs:
                        self._log_images(batch, outputs, self.global_step, 'train')
            
            self.global_step += 1
        
        # Validation
        if val_loader is not None and self.rank == 0:
            val_metrics = self.validate(val_loader)
            epoch_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
        
        self.epoch += 1
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch, training=False)
                loss = self._get_model_for_loss().compute_loss(outputs, batch, training=False)
                
                val_metrics['loss'].append(loss.item())
                
                # Log validation images (first batch only)
                if batch_idx == 0 and self.writer is not None:
                    self._log_images(batch, outputs, self.global_step, 'val')
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
        
        # Log to tensorboard
        if self.writer is not None:
            for k, v in avg_metrics.items():
                self.writer.add_scalar(f'val/{k}', v, self.global_step)
        
        return avg_metrics

        def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> float:
            """Validate model for one epoch."""
            if val_loader is None:
                return 0.0
                
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    # Move batch to device
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(batch, training=False)
                    loss = self.model.compute_loss(outputs, batch, training=False)
                    
                    val_losses.append(loss.item())
                    
                    # Log first batch images to TensorBoard
                    if batch_idx == 0:
                        self.log_tensorboard_images(batch, outputs, self.global_step, 'validation')
                    
                    # Only validate on subset for speed
                    if batch_idx >= 50:  # Validate on 50 batches max
                        break
            
            avg_val_loss = np.mean(val_losses)
            
            # Log validation metrics
            val_metrics = {
                'loss': avg_val_loss,
                'num_batches': len(val_losses)
            }
            self.log_tensorboard_scalars(val_metrics, self.global_step, 'validation')
            
            print(f"Validation Loss: {avg_val_loss:.6f}")
            
            self.model.train()
            return avg_val_loss
    def _get_model_for_loss(self) -> nn.Module:
        """Return the underlying model for loss computation (unwrap DDP if present)."""
        return self.model.module if hasattr(self.model, 'module') else self.model
    
    def _log_images(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor],
                   step: int, prefix: str = 'train'):
        """Log images to tensorboard."""
        if self.writer is None:
            return
        
        # Get images (first few from batch)
        n_images = min(4, batch['image'].shape[0])
        
        # Input images
        input_images = batch['image'][:n_images]  # [N, C, H, W]
        future_images = batch['future_image'][:n_images]
        
        # Predicted images
        if 'future_im_pred' in outputs:
            pred_images = outputs['future_im_pred'][:n_images]
            pred_images = torch.clamp(pred_images, 0, 255)
        
        # Normalize to [0, 1] for tensorboard
        input_images = input_images / 255.0
        future_images = future_images / 255.0
        if 'future_im_pred' in outputs:
            pred_images = pred_images / 255.0
        
        # Log images
        self.writer.add_images(f'{prefix}/input_images', input_images, step)
        self.writer.add_images(f'{prefix}/future_images', future_images, step)
        if 'future_im_pred' in outputs:
            self.writer.add_images(f'{prefix}/predicted_images', pred_images, step)
        
        # Log pose embeddings if available
        if 'pose_embeddings' in outputs and len(outputs['pose_embeddings']) > 0:
            pose_maps = outputs['pose_embeddings'][0][:n_images]  # [N, n_maps, H, W]
            
            # Convert to RGB visualization
            pose_vis = []
            for i in range(n_images):
                # Sum across landmarks and normalize
                pose_sum = torch.sum(pose_maps[i], dim=0, keepdim=True)  # [1, H, W]
                pose_norm = pose_sum / (torch.max(pose_sum) + 1e-8)
                pose_rgb = pose_norm.repeat(3, 1, 1)  # [3, H, W]
                pose_vis.append(pose_rgb)
            
            pose_vis = torch.stack(pose_vis, dim=0)  # [N, 3, H, W]
            self.writer.add_images(f'{prefix}/pose_maps', pose_vis, step)
    
    def save_checkpoint(self, filepath: str, **kwargs):
        """Save model checkpoint."""
        if self.rank != 0:
            return
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config,
            **kwargs
        }
        
        torch.save(checkpoint, filepath)
        print(colorize(f'Saved checkpoint to {filepath}', 'green', bold=True))
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = True) -> Dict[str, Any]:
        """Load model checkpoint."""
        print(colorize(f'Loading checkpoint from {filepath}', 'blue', bold=True))
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Load optimizer and scheduler
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if load_optimizer and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        
        return checkpoint
    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """Load checkpoint with architecture change support."""
        print(f"Loading checkpoint from {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Get state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 0)
            step = checkpoint.get('step', 0)
        else:
            state_dict = checkpoint
            epoch = 0
            step = 0
        
        # Remove 'module.' prefix if present (DDP checkpoints)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        
        # CRITICAL: Filter out incompatible weights manually
        model_state = self.model.state_dict()
        compatible_state = {}
        skipped_keys = []
        
        for key, value in state_dict.items():
            if key in model_state:
                if value.shape == model_state[key].shape:
                    compatible_state[key] = value
                else:
                    skipped_keys.append(f"{key}: {value.shape} -> {model_state[key].shape}")
            else:
                skipped_keys.append(f"{key}: missing in current model")
        
        # Load only compatible weights
        result = self.model.load_state_dict(compatible_state, strict=False)
        
        print(f"✅ Loaded {len(compatible_state)} compatible parameters")
        if skipped_keys:
            print(f"⚠️  Skipped {len(skipped_keys)} incompatible parameters:")
            for skip in skipped_keys[:10]:  # Show first 10
                print(f"   {skip}")
            if len(skipped_keys) > 10:
                print(f"   ... and {len(skipped_keys) - 10} more")
        
        # Load optimizer if requested and compatible
        if load_optimizer and isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
            try:
                # Check if optimizer architecture matches
                if hasattr(self, 'optimizer') and self.optimizer is not None:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("✅ Optimizer state loaded")
                else:
                    print("⚠️  Optimizer not initialized, skipping optimizer state")
            except Exception as e:
                print(f"⚠️  Could not load optimizer state: {e}")
                print("   Continuing with fresh optimizer state")
        
        # Update training state
        self.start_epoch = epoch
        self.global_step = step
        
        print(f"🚀 Resumed from epoch {epoch}, step {step}")
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
             num_epochs: int = 100, save_interval: int = 5):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs to train
            save_interval: Save checkpoint every N epochs
        """
        if self.rank == 0:
            print(colorize(f'Starting training for {num_epochs} epochs', 'green', bold=True))
            print(colorize(f'Log directory: {self.log_dir}', 'green', bold=True))
        
        for epoch in range(self.epoch, self.epoch + num_epochs):
            try:
                start_time = time.time()
                
                # Train one epoch
                metrics = self.train_epoch(train_loader, val_loader)
                
                epoch_time = time.time() - start_time
                
                if self.rank == 0:
                    # Update histories
                    train_loss_mean = metrics.get('loss', None)
                    if train_loss_mean is not None:
                        self.history_train_loss.append(train_loss_mean)
                    val_loss_mean = metrics.get('val_loss', None)
                    if val_loss_mean is not None:
                        self.history_val_loss.append(val_loss_mean)
                    
                    # Print and mirror epoch summary
                    msg = f'Epoch {epoch}: loss={train_loss_mean:.4f}, val_loss={val_loss_mean if val_loss_mean is not None else "-"}, time={epoch_time:.2f}s'
                    print(msg)
                    self._write_text_log(msg)
                    self._write_csv_row({
                        'type': 'epoch',
                        'epoch': epoch,
                        'step': self.global_step,
                        'loss': train_loss_mean,
                        'val_loss': val_loss_mean,
                        'lr': self.optimizer.param_groups[0]['lr']
                    })

                                        # Log epoch-level metrics to TensorBoard
                    epoch_metrics = {
                        'train_loss_epoch': train_loss_mean if train_loss_mean is not None else 0.0,
                        'validation_loss_epoch': val_loss_mean if val_loss_mean is not None else 0.0,
                        'epoch': epoch,
                        'epoch_time': epoch_time
                    }
                    self.log_tensorboard_scalars(epoch_metrics, self.global_step, 'epoch')
                    # Save/refresh loss curves and histories every epoch
                    self._save_loss_histories()
                    
                    # Save checkpoint at interval
                    if epoch % save_interval == 0:
                        checkpoint_path = os.path.join(self.log_dir, f'model_epoch_{epoch}.pth')
                        self.save_checkpoint(checkpoint_path)
            except KeyboardInterrupt:
                if self.rank == 0:
                    print(colorize('Training interrupted by user. Saving artifacts...', 'yellow', bold=True))
                    # Persist plots and histories
                    self._save_loss_histories()
                    # Save an interruption checkpoint
                    interrupted_ckpt = os.path.join(self.log_dir, 'model_interrupted.pth')
                    try:
                        self.save_checkpoint(interrupted_ckpt)
                    except Exception as e:
                        print(colorize(f'Failed to save interruption checkpoint: {e}', 'red', bold=True))
                break
        
        if self.rank == 0:
            # Save final checkpoint
            final_checkpoint_path = os.path.join(self.log_dir, 'model_final.pth')
            self.save_checkpoint(final_checkpoint_path)
            print(colorize('Training completed!', 'green', bold=True))
