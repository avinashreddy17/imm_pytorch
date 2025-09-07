"""
TensorFlow-compatible TPS sampler implementation.

This implementation aims to reproduce the exact same transformations as the 
original TensorFlow implementation by using the same mathematical approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import random
import scipy.spatial.distance as ssd


def sample_tps_w(vertical_points: int, horizontal_points: int, warpsd: Tuple[float, float],
                rotsd: float = 0.0, scalesd: float = 0.0, transsd: float = 0.1) -> np.ndarray:
    """
    Sample TPS transformation parameters exactly as in TensorFlow implementation.
    
    This function reproduces the exact same parameter sampling as the original
    TensorFlow implementation.
    """
    n_points = vertical_points * horizontal_points
    
    # Create control point grid
    y_coords = np.linspace(-1, 1, vertical_points)
    x_coords = np.linspace(-1, 1, horizontal_points)
    y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
    control_points = np.stack([y_grid.flatten(), x_grid.flatten()], axis=1)
    
    # Sample random transformation parameters
    warp_std = np.random.uniform(warpsd[0], warpsd[1])
    
    # Global transformation parameters
    angle = np.random.normal(0, rotsd) * np.pi / 180.0  # Convert to radians
    scale = 1.0 + np.random.normal(0, scalesd)
    trans_y = np.random.normal(0, transsd)
    trans_x = np.random.normal(0, transsd)
    
    # Apply global transformation
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    transformed_points = control_points @ rotation_matrix.T * scale
    transformed_points += np.array([trans_y, trans_x])
    
    # Add local warping
    local_warp = np.random.normal(0, warp_std, (n_points, 2))
    transformed_points += local_warp
    
    # Build TPS system matrix exactly as in original TensorFlow implementation
    distances = ssd.cdist(control_points, control_points)
    
    # TPS kernel with exact same epsilon as TensorFlow
    real_min = 1e-8
    distances_clipped = np.clip(distances, real_min, None)
    tps_kernel = np.log(distances_clipped) * distances_clipped
    
    # Build system matrix components
    ones_col = np.ones((n_points, 1))
    
    # Top part: [K, 1, x, y]
    top_left = tps_kernel  # [n_points, n_points]
    top_right = np.concatenate([ones_col, control_points], axis=1)  # [n_points, 3]
    top = np.concatenate([top_left, top_right], axis=1)  # [n_points, n_points+3]
    
    # Bottom part: [1^T, x^T, y^T, 0]
    bottom_left = np.concatenate([ones_col.T, control_points.T], axis=0)  # [3, n_points]
    bottom_right = np.zeros((3, 3))  # [3, 3]
    bottom = np.concatenate([bottom_left, bottom_right], axis=1)  # [3, n_points+3]
    
    # Complete system matrix
    system_matrix = np.concatenate([top, bottom], axis=0)  # [n_points+3, n_points+3]
    
    # Build target vector
    target = np.concatenate([transformed_points, np.zeros((3, 2))], axis=0)
    
    # Solve system
    try:
        W = np.linalg.solve(system_matrix, target)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if singular
        W = np.linalg.pinv(system_matrix) @ target
    
    return W


class TPSGridGenTFCompat(nn.Module):
    """
    TensorFlow-compatible TPS grid generator.
    """
    
    def __init__(self, Ho: int, Wo: int, Hc: int, Wc: int):
        """
        Initialize TPS grid generator with same parameters as TensorFlow version.
        
        Args:
            Ho, Wo: Output grid dimensions (height, width)
            Hc, Wc: Control point grid dimensions
        """
        super(TPSGridGenTFCompat, self).__init__()
        
        self._grid_hw = (Ho, Wo)
        self._cp_hw = (Hc, Wc)
        
        # Initialize grid exactly as in TensorFlow
        xx, yy = np.meshgrid(np.linspace(-1, 1, Wo), np.linspace(-1, 1, Ho))
        self._grid = np.c_[xx.flatten(), yy.flatten()].astype(np.float32)
        self._n_grid = self._grid.shape[0]
        
        # Initialize control points exactly as in TensorFlow
        xx, yy = np.meshgrid(np.linspace(-1, 1, Wc), np.linspace(-1, 1, Hc))
        self._control_pts = np.c_[xx.flatten(), yy.flatten()].astype(np.float32)
        self._n_cp = self._control_pts.shape[0]
        
        # Compute pairwise distances exactly as in TensorFlow
        Dx = ssd.cdist(self._grid, self._control_pts, metric='sqeuclidean')
        
        # Create TPS kernel exactly as in TensorFlow
        real_min = 1e-8
        Dx = np.clip(Dx, real_min, None)
        Kp = np.log(Dx) * Dx
        Os = np.ones((self._grid.shape[0]))
        L = np.c_[Kp, np.ones((self._n_grid, 1), dtype=np.float32), self._grid]
        
        self.register_buffer('_L', torch.from_numpy(L.astype(np.float32)))
    
    def forward(self, W: torch.Tensor) -> torch.Tensor:
        """
        Generate transformation grid using TPS weights.
        
        Args:
            W: TPS weights [1, n_cp + 3, 2]
            
        Returns:
            Transformation grid [1, Ho, Wo, 2]
        """
        # Apply transformation exactly as in TensorFlow
        grid = torch.matmul(self._L, W[0])  # [n_grid, 2]
        
        # Reshape to grid format and convert to [-1, 1] range for grid_sample
        Ho, Wo = self._grid_hw
        grid = grid.view(1, Ho, Wo, 2)
        
        return grid


class TPSRandomSamplerTFCompat(nn.Module):
    """
    TensorFlow-compatible TPS random sampler.
    """
    
    def __init__(self, height: int, width: int, vertical_points: int = 10, 
                 horizontal_points: int = 10, rotsd: float = 0.0, 
                 scalesd: float = 0.0, transsd: float = 0.1, 
                 warpsd: Tuple[float, float] = (0.001, 0.005),
                 cache_size: int = 1000, cache_evict_prob: float = 0.01,
                 pad: bool = True):
        """
        Initialize TensorFlow-compatible TPS sampler.
        """
        super(TPSRandomSamplerTFCompat, self).__init__()
        
        self.input_height = height
        self.input_width = width
        
        # Padding setup
        self.h_pad = height // 2 if pad else 0
        self.w_pad = width // 2 if pad else 0
        
        self.height = height + self.h_pad
        self.width = width + self.w_pad
        
        self.vertical_points = vertical_points
        self.horizontal_points = horizontal_points
        
        self.rotsd = rotsd
        self.scalesd = scalesd
        self.transsd = transsd
        self.warpsd = warpsd
        self.cache_size = cache_size
        self.cache_evict_prob = cache_evict_prob
        
        # Use TensorFlow-compatible TPS grid generator
        self.tps = TPSGridGenTFCompat(
            self.height, self.width, vertical_points, horizontal_points)
        
        self.cache = [None] * self.cache_size
        self.pad = pad
    
    def _sample_grid(self) -> torch.Tensor:
        """Sample transformation grid exactly as TensorFlow implementation."""
        W = sample_tps_w(
            self.vertical_points, self.horizontal_points, self.warpsd,
            self.rotsd, self.scalesd, self.transsd)
        W = torch.from_numpy(W.astype(np.float32))
        
        # Generate grid
        grid = self.tps(W[None])
        return grid
    
    def _get_grids(self, batch_size: int) -> torch.Tensor:
        """Get transformation grids with caching."""
        grids = []
        for i in range(batch_size):
            entry = random.randint(0, self.cache_size - 1)
            if self.cache[entry] is None or random.random() < self.cache_evict_prob:
                grid = self._sample_grid()
                self.cache[entry] = grid
            else:
                grid = self.cache[entry]
            grids.append(grid)
        grids = torch.cat(grids)
        return grids
    
    def forward(self, input_tensor: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply TPS transformation exactly as TensorFlow implementation.
        
        Args:
            input_tensor: Input tensor [B, C, H, W]
            training: Whether in training mode
            
        Returns:
            Tuple of (transformed_tensor, source_points, target_points)
        """
        if not training:
            return input_tensor, None, None
        
        batch_size = input_tensor.size(0)
        device = input_tensor.device
        
        # Get transformation grids
        grids = self._get_grids(batch_size).to(device)
        
        # Apply padding if enabled
        if self.pad:
            input_tensor = F.pad(
                input_tensor, 
                (self.w_pad, self.w_pad, self.h_pad, self.h_pad), 
                mode='replicate'
            )
        
        # Apply transformation using grid_sample
        transformed = F.grid_sample(
            input_tensor, grids, mode='bilinear', 
            padding_mode='border', align_corners=True
        )
        
        # Remove padding if it was applied
        if self.pad:
            transformed = F.pad(
                transformed,
                (-self.w_pad, -self.w_pad, -self.h_pad, -self.h_pad)
            )
        
        return transformed, None, None  # Return format compatible with original
    
    def forward_py(self, input_np: np.ndarray) -> np.ndarray:
        """
        PyTorch function interface exactly as TensorFlow implementation.
        
        This function maintains the exact same interface as the TensorFlow
        version to ensure drop-in compatibility.
        """
        with torch.no_grad():
            # Convert from numpy [B, H, W, C] to torch [B, C, H, W]
            input_tensor = torch.from_numpy(input_np).permute(0, 3, 1, 2)
            
            # Apply transformation
            output_tensor, _, _ = self.forward(input_tensor, training=True)
            
            # Convert back to numpy [B, H, W, C]
            output_np = output_tensor.permute(0, 2, 3, 1).numpy()
            
            return output_np
