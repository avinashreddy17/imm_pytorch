"""
Utility functions for PyTorch implementation.

Converted from TensorFlow implementation.
Original Authors: Ankush Gupta, Tomas Jakab
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union


def get_n_colors(n: int, pastel_factor: float = 0.5) -> List[List[float]]:
    """
    Generate n distinct colors for visualization.
    
    Args:
        n: Number of colors to generate
        pastel_factor: Factor for making colors more pastel-like
        
    Returns:
        List of RGB color triplets
    """
    colors = []
    for i in range(n):
        hue = i / n
        lightness = (50 + np.random.rand() * 10) / 100.0
        saturation = (90 + np.random.rand() * 10) / 100.0
        
        # Convert HSL to RGB
        c = (1 - abs(2 * lightness - 1)) * saturation
        x = c * (1 - abs((hue * 6) % 2 - 1))
        m = lightness - c / 2
        
        if hue < 1/6:
            r, g, b = c, x, 0
        elif hue < 2/6:
            r, g, b = x, c, 0
        elif hue < 3/6:
            r, g, b = 0, c, x
        elif hue < 4/6:
            r, g, b = 0, x, c
        elif hue < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
            
        r, g, b = (r + m), (g + m), (b + m)
        
        # Apply pastel factor
        r = r * (1 - pastel_factor) + pastel_factor
        g = g * (1 - pastel_factor) + pastel_factor
        b = b * (1 - pastel_factor) + pastel_factor
        
        colors.append([r, g, b])
    
    return colors


def split_tensors(tensors: dict, num_splits: int, dim: int = 0) -> List[dict]:
    """
    Split a dictionary of tensors along a given dimension.
    
    Args:
        tensors: Dictionary of tensors to split
        num_splits: Number of splits
        dim: Dimension to split along
        
    Returns:
        List of tensor dictionaries
    """
    splits = []
    
    for i in range(num_splits):
        split_dict = {}
        for key, tensor in tensors.items():
            if tensor is not None:
                split_tensor = torch.chunk(tensor, num_splits, dim=dim)[i]
                split_dict[key] = split_tensor
            else:
                split_dict[key] = None
        splits.append(split_dict)
    
    return splits


def get_gaussian_maps(mu: torch.Tensor, shape_hw: Tuple[int, int], 
                     inv_std: float, mode: str = 'ankush') -> torch.Tensor:
    """
    Generate 2D Gaussian maps given centers.
    
    Args:
        mu: Gaussian centers [B, N_MAPS, 2] (y, x coordinates)
        shape_hw: Output shape (height, width)
        inv_std: Inverse standard deviation
        mode: Generation mode ('ankush', 'rot', 'flat')
        
    Returns:
        Gaussian maps [B, H, W, N_MAPS]
    """
    device = mu.device
    batch_size, n_maps = mu.shape[:2]
    
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]  # [B, N_MAPS, 1]
    
    # Create coordinate grids
    y = torch.linspace(-1.0, 1.0, shape_hw[0], device=device)
    x = torch.linspace(-1.0, 1.0, shape_hw[1], device=device)
    
    if mode in ['rot', 'flat']:
        # Expand dimensions for broadcasting
        mu_y = mu_y.unsqueeze(-1)  # [B, N_MAPS, 1, 1]
        mu_x = mu_x.unsqueeze(-1)  # [B, N_MAPS, 1, 1]
        
        y = y.view(1, 1, shape_hw[0], 1)  # [1, 1, H, 1]
        x = x.view(1, 1, 1, shape_hw[1])  # [1, 1, 1, W]
        
        g_y = (y - mu_y) ** 2
        g_x = (x - mu_x) ** 2
        dist = (g_y + g_x) * (inv_std ** 2)
        
        if mode == 'rot':
            g_yx = torch.exp(-dist)
        else:  # mode == 'flat'
            g_yx = torch.exp(-torch.pow(dist + 1e-5, 0.25))
            
    elif mode == 'ankush':
        y = y.view(1, 1, shape_hw[0])  # [1, 1, H]
        x = x.view(1, 1, shape_hw[1])  # [1, 1, W]
        
        g_y = torch.exp(-torch.sqrt(1e-4 + torch.abs((mu_y - y) * inv_std)))
        g_x = torch.exp(-torch.sqrt(1e-4 + torch.abs((mu_x - x) * inv_std)))
        
        g_y = g_y.unsqueeze(3)  # [B, N_MAPS, H, 1]
        g_x = g_x.unsqueeze(2)  # [B, N_MAPS, 1, W]
        g_yx = torch.matmul(g_y, g_x)  # [B, N_MAPS, H, W]
    else:
        raise ValueError(f'Unknown mode: {mode}')
    
    # Transpose to [B, H, W, N_MAPS]
    g_yx = g_yx.permute(0, 2, 3, 1)
    return g_yx


def colorize_landmark_maps(maps: torch.Tensor) -> torch.Tensor:
    """
    Colorize landmark maps with different colors for each landmark.
    
    Args:
        maps: Landmark maps [B, H, W, N_MAPS]
        
    Returns:
        Colored landmark maps [B, H, W, 3]
    """
    batch_size, height, width, n_maps = maps.shape
    device = maps.device
    
    # Get colors for each map
    colors = get_n_colors(n_maps, pastel_factor=0.0)
    colors = torch.tensor(colors, device=device)  # [N_MAPS, 3]
    
    # Expand maps for color multiplication
    maps_expanded = maps.unsqueeze(-1)  # [B, H, W, N_MAPS, 1]
    colors_expanded = colors.view(1, 1, 1, n_maps, 3)  # [1, 1, 1, N_MAPS, 3]
    
    # Multiply and take maximum across maps
    colored_maps = maps_expanded * colors_expanded  # [B, H, W, N_MAPS, 3]
    colored_output = torch.max(colored_maps, dim=3)[0]  # [B, H, W, 3]
    
    return colored_output


def resize_points(points: torch.Tensor, original_size: torch.Tensor, 
                 new_size: List[int]) -> torch.Tensor:
    """
    Resize landmark points from original image size to new size.
    
    Args:
        points: Points tensor [B, N_POINTS, 2] or [N_POINTS, 2]
        original_size: Original image size [height, width] or [B, 2]
        new_size: New image size [height, width]
        
    Returns:
        Resized points
    """
    if points.dim() == 2:
        points = points.unsqueeze(0)  # Add batch dimension
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    if original_size.dim() == 1:
        original_size = original_size.unsqueeze(0)  # Add batch dimension
    
    # Convert to tensor if needed
    new_size = torch.tensor(new_size, device=points.device, dtype=points.dtype)
    
    # Calculate scale factors
    scale_y = new_size[0] / original_size[:, 0:1]  # [B, 1]
    scale_x = new_size[1] / original_size[:, 1:2]  # [B, 1]
    
    scale = torch.cat([scale_y, scale_x], dim=1)  # [B, 2]
    scale = scale.unsqueeze(1)  # [B, 1, 2]
    
    # Apply scaling
    resized_points = points * scale
    
    if squeeze_batch:
        resized_points = resized_points.squeeze(0)
    
    return resized_points


def create_meshgrid(height: int, width: int, device: torch.device) -> torch.Tensor:
    """
    Create a meshgrid for coordinate-based operations.
    
    Args:
        height: Grid height
        width: Grid width
        device: Device to create tensor on
        
    Returns:
        Meshgrid tensor [H, W, 2] with (y, x) coordinates
    """
    y_coords = torch.arange(height, device=device, dtype=torch.float32)
    x_coords = torch.arange(width, device=device, dtype=torch.float32)
    
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    meshgrid = torch.stack([y_grid, x_grid], dim=-1)  # [H, W, 2]
    
    return meshgrid
