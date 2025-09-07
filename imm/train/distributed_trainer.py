"""
Distributed training utilities for PyTorch implementation.

Supports both multi-GPU single-node and multi-node training.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import os
import socket
from typing import Dict, List, Optional, Any, Callable

from .trainer import Trainer
from ..utils.colorize import colorize


class DistributedTrainer:
    """Distributed training manager for multi-GPU training."""
    
    def __init__(self, world_size: int, backend: str = 'nccl'):
        """
        Initialize distributed trainer.
        
        Args:
            world_size: Total number of processes
            backend: Distributed backend ('nccl', 'gloo')
        """
        self.world_size = world_size
        self.backend = backend
        
    def setup_process_group(self, rank: int, master_addr: str = 'localhost', 
                           master_port: str = '12355'):
        """
        Setup distributed process group.
        
        Args:
            rank: Process rank
            master_addr: Master node address
            master_port: Master node port
        """
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        
        dist.init_process_group(
            backend=self.backend,
            init_method='env://',
            world_size=self.world_size,
            rank=rank
        )
        
        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())
    
    def cleanup(self):
        """Cleanup distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()
    
    def train_worker(self, rank: int, model_fn: Callable, config: Any, 
                    train_dataset, val_dataset, log_dir: str, 
                    num_epochs: int = 100, **kwargs):
        """
        Training worker for distributed training.
        
        Args:
            rank: Process rank
            model_fn: Function that returns model instance
            config: Training configuration
            train_dataset: Training dataset
            val_dataset: Validation dataset
            log_dir: Logging directory
            num_epochs: Number of epochs
            **kwargs: Additional arguments
        """
        try:
            # Setup process group
            self.setup_process_group(rank)
            
            if rank == 0:
                print(colorize(f'Initialized distributed training with {self.world_size} processes', 
                              'green', bold=True))
            
            # Create model
            model = model_fn()
            device = torch.device(f'cuda:{rank % torch.cuda.device_count()}' 
                                 if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Wrap model with DDP
            model = DDP(model, device_ids=[rank % torch.cuda.device_count()] 
                       if torch.cuda.is_available() else None)
            
            # Create distributed samplers
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=self.world_size, rank=rank, shuffle=True
            )
            val_sampler = None
            if val_dataset is not None:
                val_sampler = DistributedSampler(
                    val_dataset, num_replicas=self.world_size, rank=rank, shuffle=False
                )
            
            # Create data loaders
            batch_size = config.training.batch // self.world_size
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=train_sampler,
                num_workers=4, pin_memory=True
            )
            
            val_loader = None
            if val_dataset is not None:
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, sampler=val_sampler,
                    num_workers=4, pin_memory=True
                )
            
            # Create trainer
            trainer = Trainer(
                model, config, log_dir, device=device, 
                rank=rank, world_size=self.world_size
            )
            
            # Training loop
            for epoch in range(num_epochs):
                # Set epoch for sampler (important for shuffling)
                train_sampler.set_epoch(epoch)
                if val_sampler is not None:
                    val_sampler.set_epoch(epoch)
                
                # Train one epoch
                trainer.train_epoch(train_loader, val_loader)
                
                # Save checkpoint (only rank 0)
                if rank == 0 and epoch % 5 == 0:
                    checkpoint_path = os.path.join(log_dir, f'model_epoch_{epoch}.pth')
                    trainer.save_checkpoint(checkpoint_path)
                
                # Synchronize all processes
                dist.barrier()
            
            # Save final checkpoint
            if rank == 0:
                final_checkpoint_path = os.path.join(log_dir, 'model_final.pth')
                trainer.save_checkpoint(final_checkpoint_path)
                print(colorize('Distributed training completed!', 'green', bold=True))
        
        except Exception as e:
            print(colorize(f'Error in rank {rank}: {e}', 'red', bold=True))
            raise
        
        finally:
            self.cleanup()
    
    def spawn_training(self, model_fn: Callable, config: Any, train_dataset, 
                      val_dataset, log_dir: str, num_epochs: int = 100, **kwargs):
        """
        Spawn distributed training processes.
        
        Args:
            model_fn: Function that returns model instance
            config: Training configuration
            train_dataset: Training dataset
            val_dataset: Validation dataset
            log_dir: Logging directory
            num_epochs: Number of epochs
            **kwargs: Additional arguments
        """
        mp.spawn(
            self.train_worker,
            args=(model_fn, config, train_dataset, val_dataset, log_dir, num_epochs),
            nprocs=self.world_size,
            join=True
        )


class SLURMDistributedTrainer(DistributedTrainer):
    """Distributed trainer for SLURM clusters."""
    
    def __init__(self):
        """Initialize SLURM distributed trainer."""
        # Get SLURM environment variables
        self.world_size = int(os.environ.get('SLURM_NTASKS', 1))
        self.rank = int(os.environ.get('SLURM_PROCID', 0))
        self.local_rank = int(os.environ.get('SLURM_LOCALID', 0))
        
        # Get master node info
        node_list = os.environ.get('SLURM_NODELIST', 'localhost')
        self.master_addr = self._get_master_addr(node_list)
        self.master_port = os.environ.get('MASTER_PORT', '12355')
        
        super().__init__(self.world_size, backend='nccl')
    
    def _get_master_addr(self, node_list: str) -> str:
        """Extract master address from SLURM node list."""
        # Simple extraction - assumes first node is master
        if '[' in node_list:
            # Format like "node[01-04]" -> "node01"
            base = node_list.split('[')[0]
            range_part = node_list.split('[')[1].split(']')[0]
            first_node = range_part.split('-')[0] if '-' in range_part else range_part.split(',')[0]
            return base + first_node
        else:
            # Single node or already resolved
            return node_list.split(',')[0]
    
    def setup_process_group(self):
        """Setup process group for SLURM."""
        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = self.master_port
        os.environ['WORLD_SIZE'] = str(self.world_size)
        os.environ['RANK'] = str(self.rank)
        
        if self.rank == 0:
            print(colorize(f'SLURM Setup: master_addr={self.master_addr}, '
                          f'master_port={self.master_port}, '
                          f'world_size={self.world_size}, rank={self.rank}', 
                          'blue', bold=True))
        
        dist.init_process_group(
            backend=self.backend,
            init_method='env://',
            world_size=self.world_size,
            rank=self.rank
        )
        
        # Set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
    
    def train(self, model_fn: Callable, config: Any, train_dataset, 
             val_dataset, log_dir: str, num_epochs: int = 100, **kwargs):
        """
        Train with SLURM (no spawning needed).
        
        Args:
            model_fn: Function that returns model instance
            config: Training configuration
            train_dataset: Training dataset
            val_dataset: Validation dataset
            log_dir: Logging directory
            num_epochs: Number of epochs
            **kwargs: Additional arguments
        """
        try:
            # Setup process group
            self.setup_process_group()
            
            # Create model
            model = model_fn()
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{self.local_rank}')
                model = model.to(device)
                model = DDP(model, device_ids=[self.local_rank])
            else:
                device = torch.device('cpu')
                model = model.to(device)
                model = DDP(model)
            
            # Create distributed samplers
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
            )
            val_sampler = None
            if val_dataset is not None:
                val_sampler = DistributedSampler(
                    val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False
                )
            
            # Create data loaders
            batch_size = config.training.batch // self.world_size
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=train_sampler,
                num_workers=4, pin_memory=True
            )
            
            val_loader = None
            if val_dataset is not None:
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, sampler=val_sampler,
                    num_workers=4, pin_memory=True
                )
            
            # Create trainer
            trainer = Trainer(
                model, config, log_dir, device=device,
                rank=self.rank, world_size=self.world_size
            )
            
            # Training loop
            for epoch in range(num_epochs):
                # Set epoch for samplers
                train_sampler.set_epoch(epoch)
                if val_sampler is not None:
                    val_sampler.set_epoch(epoch)
                
                # Train one epoch
                trainer.train_epoch(train_loader, val_loader)
                
                # Save checkpoint (only rank 0)
                if self.rank == 0 and epoch % 5 == 0:
                    checkpoint_path = os.path.join(log_dir, f'model_epoch_{epoch}.pth')
                    trainer.save_checkpoint(checkpoint_path)
                
                # Synchronize
                dist.barrier()
            
            # Save final checkpoint
            if self.rank == 0:
                final_checkpoint_path = os.path.join(log_dir, 'model_final.pth')
                trainer.save_checkpoint(final_checkpoint_path)
                print(colorize('SLURM training completed!', 'green', bold=True))
        
        except Exception as e:
            print(colorize(f'Error in SLURM rank {self.rank}: {e}', 'red', bold=True))
            raise
        
        finally:
            self.cleanup()
