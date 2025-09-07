#!/usr/bin/env python
"""
Training diagnostic script to help debug IMM training issues.

This script monitors key training metrics and can help identify problems.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from imm.models.imm_model import IMMModel
from imm.utils.box import Box
from imm.datasets.celeba_dataset import CelebADataset
from torch.utils.data import DataLoader
import metayaml


def diagnose_model_gradients(model, loss):
    """Diagnose gradient flow through the model."""
    print("\n🔍 GRADIENT ANALYSIS:")
    
    total_norm = 0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            # Check for problematic gradients
            if torch.isnan(param.grad).any():
                print(f"❌ NaN gradients in {name}")
            elif torch.isinf(param.grad).any():
                print(f"❌ Inf gradients in {name}")
            elif param_norm > 10.0:
                print(f"⚠️  Large gradients in {name}: {param_norm:.4f}")
                
    total_norm = total_norm ** (1. / 2)
    print(f"   Total gradient norm: {total_norm:.4f}")
    print(f"   Parameters with gradients: {param_count}")
    
    return total_norm


def diagnose_loss_components(model, outputs, inputs):
    """Break down loss components for analysis."""
    print("\n🔍 LOSS BREAKDOWN:")
    
    future_im = inputs['future_image']
    future_im_pred = outputs['future_im_pred']
    
    # Reconstruction loss
    if model.config.reconstruction_loss == 'l2':
        recon_loss = torch.mean((future_im_pred - future_im) ** 2)
        scaled_recon_loss = 1000 * recon_loss
        print(f"   Raw L2 loss: {recon_loss.item():.6f}")
        print(f"   Scaled L2 loss (×1000): {scaled_recon_loss.item():.2f}")
    else:
        print(f"   Perceptual loss: [complex - not broken down]")
    
    # Weight decay
    weight_decay_loss = 0.0
    for param in model.parameters():
        if param.dim() > 1:
            weight_decay_loss += torch.sum(param ** 2)
    weight_decay_loss *= model._conv_opts['weight_decay']
    
    print(f"   Weight decay loss: {weight_decay_loss.item():.6f}")
    
    # Image statistics
    print(f"\n   Future image range: [{future_im.min():.2f}, {future_im.max():.2f}]")
    print(f"   Predicted image range: [{future_im_pred.min():.2f}, {future_im_pred.max():.2f}]")
    print(f"   Prediction mean: {future_im_pred.mean():.2f}")
    print(f"   Prediction std: {future_im_pred.std():.2f}")
    
    # Landmark analysis
    gauss_yx = outputs['gauss_yx']
    print(f"   Landmark range: [{gauss_yx.min():.3f}, {gauss_yx.max():.3f}]")
    print(f"   Landmark std: {gauss_yx.std():.3f}")


def diagnose_data_sample(inputs):
    """Analyze a data sample for anomalies."""
    print("\n🔍 DATA SAMPLE ANALYSIS:")
    
    image = inputs['image']
    future_image = inputs['future_image']
    mask = inputs.get('mask', None)
    
    print(f"   Image shape: {image.shape}")
    print(f"   Image range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"   Future image range: [{future_image.min():.2f}, {future_image.max():.2f}]")
    
    if mask is not None:
        print(f"   Mask shape: {mask.shape}")
        print(f"   Mask range: [{mask.min():.2f}, {mask.max():.2f}]")
        print(f"   Mask coverage: {mask.mean():.3f}")
    
    # Check for NaN/Inf
    if torch.isnan(image).any():
        print("❌ NaN values in image!")
    if torch.isnan(future_image).any():
        print("❌ NaN values in future_image!")


def run_diagnostic_training_steps(config, num_steps=10):
    """Run a few training steps with detailed diagnostics."""
    print("🚀 RUNNING DIAGNOSTIC TRAINING STEPS...")
    
    # Create model
    model = IMMModel(config.model)
    model.train()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr.start_val)
    
    # Create dataset
    from imm.utils.dataset_import import import_dataset
    dataset_class = import_dataset(config.training.dset)
    
    train_dataset = dataset_class(
        config.training.datadir, 
        subset='train',
        max_samples=1000  # Small sample for debugging
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch,
        shuffle=True, 
        num_workers=0  # Avoid multiprocessing issues
    )
    
    print(f"✅ Model and data loaded. Starting {num_steps} diagnostic steps...\n")
    
    losses = []
    grad_norms = []
    
    for step, batch in enumerate(train_loader):
        if step >= num_steps:
            break
            
        print(f"📊 STEP {step + 1}/{num_steps}")
        print("=" * 50)
        
        # Diagnose input data
        diagnose_data_sample(batch)
        
        # Forward pass
        optimizer.zero_grad()
        
        start_time = time.time()
        outputs = model(batch, training=True)
        forward_time = time.time() - start_time
        
        # Compute loss
        start_time = time.time()
        loss = model.compute_loss(outputs, batch, training=True)
        loss_time = time.time() - start_time
        
        # Diagnose loss
        diagnose_loss_components(model, outputs, batch)
        
        print(f"\n   Total loss: {loss.item():.2f}")
        print(f"   Forward time: {forward_time:.3f}s")
        print(f"   Loss time: {loss_time:.3f}s")
        
        # Backward pass
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        
        # Diagnose gradients
        grad_norm = diagnose_model_gradients(model, loss)
        
        # Apply gradient clipping
        if hasattr(config.training, 'gradclip'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradclip)
        
        # Optimizer step
        optimizer.step()
        
        print(f"   Backward time: {backward_time:.3f}s")
        print(f"   Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        losses.append(loss.item())
        grad_norms.append(grad_norm)
        
        print("")
    
    # Summary
    print("📈 DIAGNOSTIC SUMMARY:")
    print(f"   Loss trend: {losses[0]:.2f} → {losses[-1]:.2f}")
    print(f"   Loss change: {((losses[-1] - losses[0]) / losses[0] * 100):+.1f}%")
    print(f"   Average gradient norm: {np.mean(grad_norms):.4f}")
    print(f"   Gradient norm std: {np.std(grad_norms):.4f}")
    
    if losses[-1] < losses[0]:
        print("✅ Loss is decreasing - good sign!")
    elif abs(losses[-1] - losses[0]) / losses[0] < 0.01:
        print("⚠️  Loss is stable but not decreasing")
    else:
        print("❌ Loss is increasing - problem detected!")
    
    return losses, grad_norms


def main():
    """Run training diagnostics."""
    print("🔧 IMM TRAINING DIAGNOSTICS")
    print("=" * 50)
    
    # Load config
    try:
        config = Box(metayaml.read([
            'configs/paths/default.yaml',
            'configs/experiments/celeba-10pts-mini.yaml'
        ]))
        print("✅ Config loaded successfully")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return
    
    # Print key config settings
    print(f"\n⚙️  KEY SETTINGS:")
    print(f"   Reconstruction loss: {config.model.reconstruction_loss}")
    print(f"   Learning rate: {config.training.lr.start_val}")
    print(f"   LR decay step: {config.training.lr.step}")
    print(f"   Batch size: {config.training.batch}")
    print(f"   Gradient clip: {getattr(config.training, 'gradclip', 'None')}")
    print(f"   Channels bug fix: {getattr(config.model, 'channels_bug_fix', 'None')}")
    
    # Run diagnostic steps
    try:
        losses, grad_norms = run_diagnostic_training_steps(config, num_steps=5)
        
        print(f"\n🎯 RECOMMENDATIONS:")
        
        if np.mean(losses) > 10000:
            print("⚠️  Very high loss values - check loss scaling or learning rate")
        
        if np.std(grad_norms) > np.mean(grad_norms):
            print("⚠️  High gradient norm variance - consider gradient clipping")
            
        if np.mean(grad_norms) < 0.001:
            print("⚠️  Very small gradients - learning rate might be too low")
            
        if np.mean(grad_norms) > 10:
            print("⚠️  Very large gradients - learning rate might be too high")
            
    except Exception as e:
        print(f"❌ Diagnostic training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
