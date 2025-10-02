#!/usr/bin/env python
"""
Visualize landmarks from trained IMM model.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from imm.models.imm_model import IMMModel
from imm.utils.box import Box
from imm.utils.dataset_import import import_dataset
from torch.utils.data import DataLoader


def load_model_from_checkpoint(checkpoint_path, config):
    """Load model from checkpoint."""
    model = IMMModel(config.model)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load with strict=False to handle dynamic renderer layers
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if missing_keys:
        print(f"⚠️  Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys: {len(unexpected_keys)} (will be ignored)")
    
    print(f"✅ Loaded model from {checkpoint_path}")
    print(f"📊 Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"📊 Step: {checkpoint.get('global_step', 'unknown')}")
    
    return model


def create_dummy_batch(device, batch_size=6, image_size=128):
    """Create dummy batch for visualization when dataset is not available."""
    # Create random images
    images = torch.randint(0, 255, (batch_size, 3, image_size, image_size), dtype=torch.float32)
    
    # Create slightly different target images (add some noise)
    target_images = images + torch.randint(-20, 20, images.shape, dtype=torch.float32)
    target_images = torch.clamp(target_images, 0, 255)
    
    batch = {
        'image': images.to(device),              # Input/warped image
        'future_image': target_images.to(device),  # Target image to reconstruct
    }
    
    return batch


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
            
            # Get images and landmarks
            input_images = batch['image'][:num_samples].cpu()
            target_images = batch['future_image'][:num_samples].cpu()
            pred_images = outputs['future_im_pred'][:num_samples].cpu()
            landmarks = outputs['gauss_yx'][:num_samples].cpu()  # [B, N, 2]
        
        # Convert images from [0,255] to [0,1] for matplotlib
        input_images = input_images / 255.0
        target_images = target_images / 255.0
        pred_images = torch.clamp(pred_images / 255.0, 0, 1)
        
        # Convert landmarks from model coordinates to image coordinates
        # Landmarks are in [-1,1] range, convert to [0, image_size]
        H, W = input_images.shape[2], input_images.shape[3]
        landmarks_img = (landmarks + 1) * 0.5  # [-1,1] -> [0,1]
        landmarks_img[:, :, 0] *= W  # x coordinates
        landmarks_img[:, :, 1] *= H  # y coordinates
        
        # Create visualization
        fig, axes = plt.subplots(4, num_samples, figsize=(2*num_samples, 8))
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_samples):
            # Row 1: Input images
            ax = axes[0, i]
            ax.imshow(input_images[i].permute(1, 2, 0))
            ax.set_title(f'Input {i+1}')
            ax.axis('off')
            
            # Row 2: Target images  
            ax = axes[1, i]
            ax.imshow(target_images[i].permute(1, 2, 0))
            ax.set_title(f'Target {i+1}')
            ax.axis('off')
            
            # Row 3: Predicted images
            ax = axes[2, i]
            ax.imshow(pred_images[i].permute(1, 2, 0))
            ax.set_title(f'Predicted {i+1}')
            ax.axis('off')
            
            # Row 4: Landmarks on target images
            ax = axes[3, i]
            ax.imshow(target_images[i].permute(1, 2, 0))
            
            # Plot landmarks as colored dots
            n_landmarks = landmarks_img.shape[1]
            colors = plt.cm.rainbow(np.linspace(0, 1, n_landmarks))
            
            for j in range(n_landmarks):
                x, y = landmarks_img[i, j, 0], landmarks_img[i, j, 1]
                ax.scatter(x, y, c=[colors[j]], s=50, marker='o', edgecolors='white', linewidth=1)
                ax.text(x+2, y+2, str(j), fontsize=8, color='white', weight='bold')
            
            ax.set_title(f'Landmarks {i+1} ({n_landmarks} pts)')
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = 'current_landmarks_visualization.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to {output_path}")
        
        # Print landmark coordinates
        print(f"\n📍 Landmark Coordinates (image coordinates):")
        for i in range(min(2, num_samples)):  # Show first 2 samples
            print(f"Sample {i+1}:")
            for j in range(landmarks_img.shape[1]):
                x, y = landmarks_img[i, j, 0].item(), landmarks_img[i, j, 1].item()
                print(f"  Landmark {j}: ({x:.1f}, {y:.1f})")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize landmarks from trained model')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--num-samples', type=int, default=6, help='Number of samples to visualize')
    parser.add_argument('--config', default='configs/paths/default.yaml configs/experiments/celeba-10pts-mini.yaml', 
                       help='Config files (space separated)')
    
    args = parser.parse_args()
    
    # Load configuration
    import yaml
    import os
    config_files = args.config.split()
    
    # Load and merge config files manually
    config_dict = {}
    for config_file in config_files:
        with open(config_file, 'r') as f:
            file_config = yaml.safe_load(f)
            if file_config:
                config_dict.update(file_config)
    
    # Resolve path variables
    if 'celeba_data_dir' in config_dict:
        celeba_path = config_dict['celeba_data_dir']
        if celeba_path.startswith('../'):
            # Convert relative path to absolute
            celeba_path = os.path.abspath(os.path.join(os.path.dirname(__file__), celeba_path))
            config_dict['celeba_data_dir'] = celeba_path
            print(f"🔧 Resolved celeba_data_dir to: {celeba_path}")
    
    config = Box(config_dict)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, config)
    model = model.to(device)
    
    # Load dataset
    print(f"🔧 Loading real dataset...")
    dataset_class = import_dataset(config.training.train_dset_params.dataset)
    
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
        config.celeba_data_dir,  # Use resolved celeba_data_dir
        subset=test_subset, 
        max_samples=50,  # Only load 50 images for quick viz
        **test_params
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.num_samples, shuffle=True)
    
    # Warmup model to create renderer layers and reload checkpoint
    print(f"🔧 Warming up model with real data...")
    success = False
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 1:  # Only need first batch
                break
                
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            try:
                # Forward pass to create dynamic layers
                print("🔧 Running warmup forward pass...")
                outputs = model(batch, training=False)
                print(f"✅ Warmup successful! Model ready.")
                
                # Reload checkpoint to ensure all layers are properly loaded
                checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print(f"✅ Checkpoint reloaded after warmup")
                
                success = True
                break
                
            except Exception as e:
                print(f"❌ Warmup failed: {e}")
                break
    
    if not success:
        print("❌ Failed to warmup model. Exiting.")
        return
    
    # Visualize landmarks
    print(f"🔍 Visualizing landmarks from {args.checkpoint}")
    print(f"📸 Using real CelebA images for landmark detection")
    visualize_landmarks(model, test_loader, device, args.num_samples)


if __name__ == '__main__':
    main()