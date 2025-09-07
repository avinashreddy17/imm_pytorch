#!/usr/bin/env python
"""
Visualize landmarks from current checkpoint.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from imm.models.imm_model import IMMModel
from imm.utils.box import Box
from imm.utils.dataset_import import import_dataset
from imm.utils.utils import colorize_landmark_maps
import metayaml


def load_model_from_checkpoint(checkpoint_path, config):
    """Load model from checkpoint."""
    model = IMMModel(config.model)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load with strict=False to handle dynamic renderer layers
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if missing_keys:
        print(f"⚠️  Missing keys: {len(missing_keys)} (this is normal for renderer layers)")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys: {len(unexpected_keys)} (will be ignored)")
    
    print(f"✅ Loaded model from {checkpoint_path}")
    print(f"📊 Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"📊 Step: {checkpoint.get('global_step', 'unknown')}")
    
    return model


def warmup_model(model, dataloader, device, checkpoint_path):
    """Warmup model to create dynamic layers, then reload checkpoint."""
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 1:  # Only need first batch
                break
                
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Run forward pass to create renderer layers
            print("🔧 Running warmup forward pass to create renderer layers...")
            try:
                outputs = model(batch, training=False)
                print("✅ Renderer layers created successfully")
                break
            except Exception as e:
                print(f"⚠️  Warmup failed: {e}")
                return False
    
    # Now reload the checkpoint properly for renderer layers
    print("🔧 Reloading checkpoint for renderer layers...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load only the renderer layers from checkpoint
    model_state = checkpoint['model_state_dict']
    current_state = model.state_dict()
    
    # Update only the keys that exist in both
    updated_keys = 0
    for key in model_state:
        if key in current_state:
            current_state[key] = model_state[key]
            updated_keys += 1
    
    model.load_state_dict(current_state)
    print(f"✅ Updated {updated_keys} parameters from checkpoint")
    return True


def visualize_landmarks(model, dataloader, device, num_samples=8):
    """Visualize landmarks on sample images."""
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 1:  # Only process first batch
                break
                
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            outputs = model(batch, training=False)
            
            # Get samples to visualize
            batch_size = min(num_samples, batch['image'].shape[0])
            
            # Create visualization
            fig, axes = plt.subplots(3, batch_size, figsize=(batch_size * 3, 9))
            if batch_size == 1:
                axes = axes.reshape(3, 1)
            
            for i in range(batch_size):
                # Original images
                input_img = batch['image'][i].cpu().permute(1, 2, 0).numpy()
                target_img = batch['future_image'][i].cpu().permute(1, 2, 0).numpy()
                
                # Normalize for display
                def normalize_img(img):
                    if img.max() > 2.0:
                        img = img / 255.0
                    return np.clip(img, 0, 1)
                
                input_img = normalize_img(input_img)
                target_img = normalize_img(target_img)
                
                # Show input image
                axes[0, i].imshow(input_img)
                axes[0, i].set_title(f'Input {i+1}')
                axes[0, i].axis('off')
                
                # Show target image
                axes[1, i].imshow(target_img)
                axes[1, i].set_title(f'Target {i+1}')
                axes[1, i].axis('off')
                
                # Show landmarks on target
                if 'pose_embeddings' in outputs and len(outputs['pose_embeddings']) > 0:
                    # Get landmark maps (largest resolution)
                    pose_maps = outputs['pose_embeddings'][0][i:i+1]  # [1, H, W, N]
                    
                    # Colorize landmarks
                    colored_landmarks = colorize_landmark_maps(pose_maps)  # [1, H, W, 3]
                    colored_landmarks = colored_landmarks.squeeze(0).cpu().numpy()
                    colored_landmarks = np.clip(colored_landmarks, 0, 1)
                    
                    # Overlay landmarks on target image
                    overlay = target_img * 0.7 + colored_landmarks * 0.3
                    axes[2, i].imshow(overlay)
                    axes[2, i].set_title(f'Landmarks {i+1}')
                else:
                    axes[2, i].text(0.5, 0.5, 'No landmarks', ha='center', va='center')
                    axes[2, i].set_title(f'No Landmarks {i+1}')
                
                axes[2, i].axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            save_path = 'current_landmarks_visualization.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"🎯 Visualization saved: {save_path}")
            
            # Print landmark coordinates
            if 'gauss_yx' in outputs:
                coords = outputs['gauss_yx'][0].cpu().numpy()  # [N_MAPS, 2]
                print(f"\n📍 Landmark coordinates for first image:")
                for j, (y, x) in enumerate(coords):
                    print(f"  Landmark {j+1}: ({x:.3f}, {y:.3f})")
            
            break


def main():
    parser = argparse.ArgumentParser(description='Visualize landmarks from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, default='configs/experiments/celeba-10pts-mini.yaml')
    parser.add_argument('--paths', type=str, default='configs/paths/default.yaml')
    parser.add_argument('--num-samples', type=int, default=8, help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Box(metayaml.read([args.paths, args.config]))
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, config)
    model = model.to(device)
    
    # Create dataset
    dataset_class = import_dataset(config.training.dset)
    
    # Test dataset params
    test_params = {}
    if hasattr(config.training, 'test_dset_params'):
        test_params.update(config.training.test_dset_params)
        test_subset = test_params.pop('subset', 'test')
    else:
        test_subset = 'test'
    
    # Remove max_samples if it exists in test_params to avoid conflict
    test_params.pop('max_samples', None)
    
    # Create test dataset
    test_dataset = dataset_class(
        config.training.datadir, 
        subset=test_subset, 
        max_samples=50,  # Only load 50 images for quick viz
        **test_params
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.num_samples, shuffle=True)
    
    # Warmup model to create renderer layers and reload checkpoint
    print(f"🔧 Warming up model and loading checkpoint properly...")
    success = warmup_model(model, test_loader, device, args.checkpoint)
    
    if not success:
        print("❌ Failed to warmup model. Exiting.")
        return
    
    # Visualize landmarks
    print(f"🔍 Visualizing landmarks from {args.checkpoint}")
    visualize_landmarks(model, test_loader, device, args.num_samples)


if __name__ == '__main__':
    main()
