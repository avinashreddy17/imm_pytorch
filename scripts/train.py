#!/usr/bin/env python
"""
Training script for PyTorch IMM implementation.

Converted from TensorFlow implementation.
Original Authors: Ankush Gupta, Tomas Jakab
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import os.path as osp
import argparse
from typing import Dict, Any

# Add project root to path
import sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from imm.models.imm_model import IMMModel
from imm.train.trainer import Trainer
from imm.train.distributed_trainer import DistributedTrainer, SLURMDistributedTrainer
from imm.utils.box import Box
from imm.utils.colorize import colorize
from imm.utils.dataset_import import import_dataset
import metayaml


def load_configs(file_names: list) -> Box:
    """Load configuration from YAML files."""
    config = Box(metayaml.read(file_names))
    return config


def create_model(config: Box) -> nn.Module:
    """Create IMM model from configuration."""
    model = IMMModel(config.model)
    return model


def main(args):
    """Main training function."""
    # Load configuration
    config = load_configs(args.configs)
    train_config = config.training
    
    # Setup directories
    log_dir = train_config.logdir
    os.makedirs(log_dir, exist_ok=True)
    
    print(colorize(f'Log directory: {log_dir}', 'green', bold=True))
    print(colorize(f'Batch size: {train_config.batch}', 'red', bold=True))
    
    # Import dataset class
    dataset_class = import_dataset(train_config.dset)
    
    # Setup dataset parameters
    train_dset_params = {}
    test_dset_params = {}
    train_subset = 'train'
    test_subset = 'test'
    
    if hasattr(train_config, 'train_dset_params'):
        train_dset_params.update(train_config.train_dset_params)
        if 'subset' in train_dset_params:
            train_subset = train_dset_params['subset']
            del train_dset_params['subset']
    
    if hasattr(train_config, 'test_dset_params'):
        test_dset_params.update(train_config.test_dset_params)
        if 'subset' in test_dset_params:
            test_subset = test_dset_params['subset']
            del test_dset_params['subset']
    
    # Create datasets
    train_dataset = dataset_class(
        train_config.datadir, subset=train_subset, **train_dset_params
    )
    
    test_dataset = None
    if hasattr(train_config, 'test_dset_params'):
        test_dataset = dataset_class(
            train_config.datadir, subset=test_subset, **test_dset_params
        )
    
    # Training function
    def model_fn():
        return create_model(config)
    
    # Check for distributed training
    if args.ngpus > 1:
        # Multi-GPU training
        if 'SLURM_PROCID' in os.environ:
            # SLURM environment
            trainer = SLURMDistributedTrainer()
            trainer.train(
                model_fn, config, train_dataset, test_dataset, 
                log_dir, num_epochs=args.num_epochs
            )
        else:
            # Regular multi-GPU
            trainer = DistributedTrainer(world_size=args.ngpus)
            trainer.spawn_training(
                model_fn, config, train_dataset, test_dataset,
                log_dir, num_epochs=args.num_epochs
            )
    else:
        # Single GPU training
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(colorize(f'Training on device: {device}', 'blue', bold=True))
        
        # Create model
        model = create_model(config)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=train_config.batch,
            shuffle=True, num_workers=4, pin_memory=True
        )
        
        test_loader = None
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset, batch_size=train_config.batch,
                shuffle=False, num_workers=4, pin_memory=True
            )
        
        # Create trainer
        trainer = Trainer(model, config, log_dir, device=device)

        # Warm-up forward to materialize any lazily created layers
        # (e.g., renderer layers) BEFORE optimizer is built
        # Grab one batch from the train loader safely
        warmup_batch = None
        for wb in train_loader:
            warmup_batch = wb
            break
        if warmup_batch is not None:
            trainer.warmup_with_batch(warmup_batch)
        else:
            # If for some reason loader is empty, still set up optimizer
            trainer.setup_optimizer()

        # Load checkpoint if specified
        if args.checkpoint is not None:
            if osp.exists(args.checkpoint):
                trainer.load_checkpoint(args.checkpoint, load_optimizer=args.restore_optim)
            else:
                print(colorize('Checkpoint file not found. Starting from scratch.', 'red', bold=True))
    
                # ============= DEBUG CODE - ADD THIS BLOCK =============
        print(colorize('🔍 Running pre-training debug analysis...', 'yellow', bold=True))
        
        def debug_first_batch():
            model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(train_loader):
                    if batch_idx > 0:
                        break
                    
                    # Move to device
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(device)
                    
                    print(f"📊 Batch shape: {batch['image'].shape}")
                    print(f"📊 Input image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
                    print(f"📊 Target image range: [{batch['future_image'].min():.3f}, {batch['future_image'].max():.3f}]")
                    
                    # Forward pass
                    outputs = model(batch)
                    pred_range = f"[{outputs['future_im_pred'].min():.3f}, {outputs['future_im_pred'].max():.3f}]"
                    print(f"📊 Prediction range: {pred_range}")
                    
                    # Loss computation
                    loss = model.compute_loss(outputs, batch)
                    print(f"📊 Loss value: {loss.item():.6f}")
                    
                    # Diagnostic checks
                    if batch['image'].max() < 2.0:
                        print(colorize('❌ ISSUE FOUND: Images in [0,1] range, should be [0,255]!', 'red', bold=True))
                        return 'range_issue'
                    elif loss.item() < 0.001:
                        print(colorize('❌ ISSUE FOUND: Loss too small, likely range problem!', 'red', bold=True))
                        return 'loss_too_small'
                    elif torch.isnan(outputs['future_im_pred']).any():
                        print(colorize('❌ ISSUE FOUND: NaN in predictions!', 'red', bold=True))
                        return 'nan_predictions'
                    else:
                        print(colorize('✅ Initial checks passed', 'green', bold=True))
                        return 'ok'
            
            model.train()
            return 'ok'
        
        # Run diagnostic
        debug_result = debug_first_batch()
        print(f"🔍 Debug result: {debug_result}")
        # ============= END DEBUG CODE =============

        # Train
        # trainer.train(train_loader, test_loader, num_epochs=args.num_epochs)
        
        # Train
        trainer.train(train_loader, test_loader, num_epochs=args.num_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train IMM Model in PyTorch')
    
    parser.add_argument('--configs', nargs='+', default=[], required=True,
                       help='Paths to the config files.')
    parser.add_argument('--ngpus', type=int, default=1, required=False,
                       help='Number of GPUs to use for training.')
    parser.add_argument('--num-epochs', type=int, default=100, required=False,
                       help='Number of epochs to train.')
    parser.add_argument('--lr-multiple', type=float, default=1.0,
                       help='Multiplier on the learning rate.')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint file to restore from.')
    parser.add_argument('--restore-optim', action='store_true',
                       help='Restore the optimizer variables.')
    parser.add_argument('--reset-global-step', type=int, default=-1,
                       help='Force the value of global step.')
    parser.add_argument('--ignore-missing-vars', action='store_true',
                       help='Skip restoring vars not in the checkpoint file.')
    
    args = parser.parse_args()
    main(args)
