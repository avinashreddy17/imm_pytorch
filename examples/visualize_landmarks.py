#!/usr/bin/env python
"""
Simple visualization script for IMM landmarks.

This script can be used to visualize predicted landmarks on images
without requiring the full Jupyter notebook setup.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imm.models.imm_model import IMMModel
from imm.utils.box import Box


def visualize_landmarks(image_tensor, landmarks_tensor, title="Landmarks"):
    """
    Visualize landmarks on an image.
    
    Args:
        image_tensor: Image tensor [C, H, W] in range [0, 255]
        landmarks_tensor: Landmarks tensor [N, 2] in range [-1, 1]
        title: Plot title
    """
    # Convert image to numpy
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy() / 255.0
    image_np = np.clip(image_np, 0, 1)
    
    # Convert landmarks to image coordinates
    h, w = image_tensor.shape[1], image_tensor.shape[2]
    landmarks_np = landmarks_tensor.cpu().numpy()
    landmarks_np = ((landmarks_np + 1) / 2.0) * np.array([h, w])
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)
    plt.scatter(landmarks_np[:, 1], landmarks_np[:, 0], c='red', s=50, alpha=0.7)
    
    # Add landmark numbers
    for i, (y, x) in enumerate(landmarks_np):
        plt.text(x + 2, y + 2, str(i), color='yellow', fontsize=8, fontweight='bold')
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()


def main():
    """Demo landmark visualization."""
    print("=== IMM Landmark Visualization Demo ===\n")
    
    # Create model
    config = Box({
        'n_maps': 10,
        'gauss_std': 0.1,
        'gauss_mode': 'rot',
        'n_filters': 32,
        'n_filters_render': 32,
        'renderer_stride': 2,
        'min_res': 16,
        'reconstruction_loss': 'l2'
    })
    
    model = IMMModel(config)
    model.eval()
    
    print("✓ Model loaded successfully")
    
    # Create dummy input
    batch_size = 1
    image_size = 128
    
    # Generate a simple test image (gradient + noise)
    y_coords = torch.linspace(0, 1, image_size).unsqueeze(1).repeat(1, image_size)
    x_coords = torch.linspace(0, 1, image_size).unsqueeze(0).repeat(image_size, 1)
    
    test_image = torch.zeros(3, image_size, image_size)
    test_image[0] = y_coords * 255  # Red gradient
    test_image[1] = x_coords * 255  # Green gradient
    test_image[2] = (y_coords * x_coords) * 255  # Blue gradient
    
    # Add some noise
    test_image += torch.randn_like(test_image) * 10
    test_image = torch.clamp(test_image, 0, 255)
    
    inputs = {
        'image': test_image.unsqueeze(0),
        'future_image': test_image.unsqueeze(0)
    }
    
    print("✓ Test image created")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(inputs, training=False)
    
    landmarks = outputs['gauss_yx'][0]  # Remove batch dimension
    
    print(f"✓ Predicted {landmarks.shape[0]} landmarks")
    print(f"  - Landmark coordinates range: [{landmarks.min():.3f}, {landmarks.max():.3f}]")
    
    # Visualize
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(test_image.permute(1, 2, 0).numpy() / 255.0)
    plt.title("Input Image")
    plt.axis('off')
    
    # Image with landmarks
    plt.subplot(1, 3, 2)
    visualize_landmarks(test_image, landmarks, "Predicted Landmarks")
    
    # Landmark positions plot
    plt.subplot(1, 3, 3)
    landmarks_np = landmarks.cpu().numpy()
    plt.scatter(landmarks_np[:, 1], landmarks_np[:, 0], c='red', s=100, alpha=0.7)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.grid(True, alpha=0.3)
    plt.title("Landmark Positions")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    
    # Add landmark numbers
    for i, (y, x) in enumerate(landmarks_np):
        plt.text(x + 0.05, y + 0.05, str(i), fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('landmarks_demo.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'landmarks_demo.png'")
    
    # Show plot if in interactive mode
    try:
        plt.show()
    except:
        print("  (Note: Display not available, only saved to file)")


if __name__ == '__main__':
    main()
