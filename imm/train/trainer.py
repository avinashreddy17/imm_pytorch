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

from ..utils.colorize import colorize
from ..models.imm_model import IMMModel


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
        
        # Setup optimizer
        self._setup_optimizer()
        
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
        
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Get optimizer parameters
        lr = self.config.training.lr.start_val
        
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
                    print(f'{datetime.now()}: step {self.global_step}, loss = {loss.item():.4f} '
                          f'({examples_per_sec:.1f} examples/sec) {batch_time:.3f} sec/batch')
                
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
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if load_optimizer and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        
        return checkpoint
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
             num_epochs: int = 100, save_interval: int = 10):
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
            start_time = time.time()
            
            # Train one epoch
            metrics = self.train_epoch(train_loader, val_loader)
            
            epoch_time = time.time() - start_time
            
            if self.rank == 0:
                # Print epoch summary
                print(f'Epoch {epoch}: loss={metrics["loss"]:.4f}, time={epoch_time:.2f}s')
                
                # Save checkpoint
                if epoch % save_interval == 0:
                    checkpoint_path = os.path.join(self.log_dir, f'model_epoch_{epoch}.pth')
                    self.save_checkpoint(checkpoint_path)
        
        if self.rank == 0:
            # Save final checkpoint
            final_checkpoint_path = os.path.join(self.log_dir, 'model_final.pth')
            self.save_checkpoint(final_checkpoint_path)
            print(colorize('Training completed!', 'green', bold=True))
