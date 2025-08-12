"""
Thin Plate Spline (TPS) sampling for data augmentation.

Converted from TensorFlow implementation.
Original Authors: Ankush Gupta, Tomas Jakab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.spatial.distance as ssd
from typing import Tuple, Optional, List
import random


class TPSGridGen(nn.Module):
    """
    Thin Plate Spline Grid Generator for spatial transformations.
    """
    
    def __init__(self, height: int, width: int, vertical_points: int = 10, 
                 horizontal_points: int = 10):
        """
        Initialize TPS grid generator.
        
        Args:
            height: Grid height
            width: Grid width
            vertical_points: Number of control points vertically
            horizontal_points: Number of control points horizontally
        """
        super(TPSGridGen, self).__init__()
        
        self.height = height
        self.width = width
        self.vertical_points = vertical_points
        self.horizontal_points = horizontal_points
        
        # Generate control point grid
        self.control_points = self._generate_control_points()
        
        # Precompute target grid coordinates
        self.target_grid = self._generate_target_grid()
    
    def _generate_control_points(self) -> torch.Tensor:
        """Generate regular grid of control points."""
        y_coords = torch.linspace(0, self.height - 1, self.vertical_points)
        x_coords = torch.linspace(0, self.width - 1, self.horizontal_points)
        
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        control_points = torch.stack([
            y_grid.flatten(), x_grid.flatten()
        ], dim=1)  # [N_points, 2]
        
        return control_points
    
    def _generate_target_grid(self) -> torch.Tensor:
        """Generate target grid coordinates for transformation."""
        y_coords = torch.arange(self.height, dtype=torch.float32)
        x_coords = torch.arange(self.width, dtype=torch.float32)
        
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        target_grid = torch.stack([
            y_grid.flatten(), x_grid.flatten()
        ], dim=1)  # [H*W, 2]
        
        return target_grid
    
    def _compute_tps_weights(self, source_points: torch.Tensor, 
                           target_points: torch.Tensor) -> torch.Tensor:
        """
        Compute TPS transformation weights.
        
        Args:
            source_points: Source control points [N, 2]
            target_points: Target control points [N, 2]
            
        Returns:
            TPS weights [N+3, 2]
        """
        n_points = source_points.shape[0]
        device = source_points.device
        
        # Compute pairwise distances
        distances = torch.cdist(source_points, source_points)  # [N, N]
        
        # TPS kernel: r^2 * log(r)
        tps_kernel = distances ** 2 * torch.log(distances + 1e-8)
        tps_kernel[distances == 0] = 0  # Handle r=0 case
        
        # Build system matrix
        ones = torch.ones(n_points, 1, device=device)
        system_matrix = torch.zeros(n_points + 3, n_points + 3, device=device)
        
        # Top-left: TPS kernel
        system_matrix[:n_points, :n_points] = tps_kernel
        
        # Top-right: [1, x, y]
        system_matrix[:n_points, n_points] = 1
        system_matrix[:n_points, n_points + 1] = source_points[:, 1]  # x
        system_matrix[:n_points, n_points + 2] = source_points[:, 0]  # y
        
        # Bottom-left: transpose of top-right
        system_matrix[n_points:, :n_points] = system_matrix[:n_points, n_points:].T
        
        # Build target vector
        target_vector = torch.zeros(n_points + 3, 2, device=device)
        target_vector[:n_points] = target_points
        
        # Solve linear system
        weights = torch.linalg.solve(system_matrix, target_vector)
        
        return weights
    
    def forward(self, source_points: torch.Tensor) -> torch.Tensor:
        """
        Generate transformation grid using TPS.
        
        Args:
            source_points: Displaced control points [N, 2]
            
        Returns:
            Transformation grid [H, W, 2]
        """
        device = source_points.device
        control_points = self.control_points.to(device)
        target_grid = self.target_grid.to(device)
        
        # Compute TPS weights
        weights = self._compute_tps_weights(control_points, source_points)
        
        # Apply transformation to target grid
        n_points = control_points.shape[0]
        
        # Compute distances from grid points to control points
        grid_to_control_dist = torch.cdist(target_grid, control_points)  # [H*W, N]
        
        # TPS kernel for grid points
        tps_kernel_grid = grid_to_control_dist ** 2 * torch.log(grid_to_control_dist + 1e-8)
        tps_kernel_grid[grid_to_control_dist == 0] = 0
        
        # Build transformation matrix for grid
        ones_grid = torch.ones(target_grid.shape[0], 1, device=device)
        transform_matrix = torch.cat([
            tps_kernel_grid,  # [H*W, N]
            ones_grid,  # [H*W, 1]
            target_grid[:, 1:2],  # x coordinates [H*W, 1]
            target_grid[:, 0:1]   # y coordinates [H*W, 1]
        ], dim=1)  # [H*W, N+3]
        
        # Apply transformation
        transformed_points = torch.matmul(transform_matrix, weights)  # [H*W, 2]
        
        # Reshape to grid
        transformed_grid = transformed_points.view(self.height, self.width, 2)
        
        return transformed_grid


class TPSRandomSampler(nn.Module):
    """
    Random TPS sampler for data augmentation.
    """
    
    def __init__(self, height: int, width: int, vertical_points: int = 10, 
                 horizontal_points: int = 10, rotsd: float = 0.0, 
                 scalesd: float = 0.0, transsd: float = 0.1, 
                 warpsd: Tuple[float, float] = (0.001, 0.005),
                 cache_size: int = 1000, cache_evict_prob: float = 0.01,
                 pad: bool = True):
        """
        Initialize TPS random sampler.
        
        Args:
            height: Input height
            width: Input width
            vertical_points: Number of vertical control points
            horizontal_points: Number of horizontal control points
            rotsd: Rotation standard deviation (degrees)
            scalesd: Scale standard deviation
            transsd: Translation standard deviation
            warpsd: Warp standard deviation (min, max)
            cache_size: Size of transformation cache
            cache_evict_prob: Probability of cache eviction
            pad: Whether to pad images
        """
        super(TPSRandomSampler, self).__init__()
        
        self.input_height = height
        self.input_width = width
        
        # Padding
        self.h_pad = height // 2 if pad else 0
        self.w_pad = width // 2 if pad else 0
        
        self.height = height + self.h_pad
        self.width = width + self.w_pad
        
        self.vertical_points = vertical_points
        self.horizontal_points = horizontal_points
        
        # Augmentation parameters
        self.rotsd = rotsd
        self.scalesd = scalesd
        self.transsd = transsd
        self.warpsd = warpsd
        
        # Cache for transformations
        self.cache_size = cache_size
        self.cache_evict_prob = cache_evict_prob
        self.cache = [None] * cache_size
        
        # TPS grid generator
        self.tps = TPSGridGen(self.height, self.width, vertical_points, horizontal_points)
        
        self.pad = pad
    
    def _generate_random_transformation(self, device: torch.device) -> torch.Tensor:
        """Generate a random TPS transformation."""
        # Get control points
        control_points = self.tps.control_points.clone().to(device)
        n_points = control_points.shape[0]
        
        # Add random displacement to control points
        if isinstance(self.warpsd, (list, tuple)):
            warp_std = random.uniform(self.warpsd[0], self.warpsd[1]) * max(self.height, self.width)
        else:
            warp_std = self.warpsd * max(self.height, self.width)
        
        # Random warping
        warp_displacement = torch.randn(n_points, 2, device=device) * warp_std
        
        # Global transformations
        if self.rotsd > 0:
            angle = torch.randn(1, device=device) * self.rotsd * np.pi / 180
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            rotation_matrix = torch.tensor([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ], device=device)
        else:
            rotation_matrix = torch.eye(2, device=device)
        
        if self.scalesd > 0:
            scale = 1.0 + torch.randn(1, device=device) * self.scalesd
            scale_matrix = torch.diag(torch.tensor([scale, scale], device=device))
        else:
            scale_matrix = torch.eye(2, device=device)
        
        if self.transsd > 0:
            translation = torch.randn(2, device=device) * self.transsd * max(self.height, self.width)
        else:
            translation = torch.zeros(2, device=device)
        
        # Apply global transformations
        center = torch.tensor([self.height / 2, self.width / 2], device=device)
        centered_points = control_points - center
        
        # Apply scale and rotation
        transformed_points = torch.matmul(centered_points, scale_matrix.T)
        transformed_points = torch.matmul(transformed_points, rotation_matrix.T)
        
        # Add translation and recenter
        transformed_points = transformed_points + center + translation
        
        # Add local warping
        displaced_points = transformed_points + warp_displacement
        
        return displaced_points
    
    def _get_cached_transformation(self, device: torch.device) -> Optional[torch.Tensor]:
        """Get a cached transformation or generate a new one."""
        # Try to get from cache
        valid_indices = [i for i, t in enumerate(self.cache) if t is not None]
        
        if valid_indices and random.random() < (1.0 - self.cache_evict_prob):
            idx = random.choice(valid_indices)
            return self.cache[idx].to(device)
        
        # Generate new transformation
        transformation = self._generate_random_transformation(device)
        
        # Add to cache
        cache_idx = random.randint(0, self.cache_size - 1)
        self.cache[cache_idx] = transformation.cpu()
        
        return transformation
    
    def forward(self, image: torch.Tensor, landmarks: Optional[torch.Tensor] = None,
               training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Apply random TPS transformation to image and landmarks.
        
        Args:
            image: Input image [B, C, H, W]
            landmarks: Optional landmarks [B, N, 2]
            training: Whether in training mode
            
        Returns:
            Tuple of (transformed_image, transformed_landmarks, transformation_grid)
        """
        if not training:
            # No augmentation during evaluation
            if landmarks is not None:
                return image, landmarks, None
            else:
                return image, None, None
        
        batch_size = image.shape[0]
        device = image.device
        
        # Pad image if needed
        if self.pad:
            image = F.pad(image, (self.w_pad//2, self.w_pad//2, self.h_pad//2, self.h_pad//2))
        
        transformed_images = []
        transformed_landmarks_list = []
        
        for b in range(batch_size):
            # Get transformation
            displaced_points = self._get_cached_transformation(device)
            
            # Generate transformation grid
            transformation_grid = self.tps(displaced_points)  # [H, W, 2]
            
            # Normalize grid to [-1, 1] for grid_sample
            norm_grid = transformation_grid.clone()
            norm_grid[:, :, 0] = 2.0 * norm_grid[:, :, 0] / (self.height - 1) - 1.0  # y
            norm_grid[:, :, 1] = 2.0 * norm_grid[:, :, 1] / (self.width - 1) - 1.0   # x
            
            # Swap x and y for grid_sample (expects [x, y])
            norm_grid = norm_grid[:, :, [1, 0]]  # [H, W, 2] with [x, y]
            
            # Apply transformation to image
            img_single = image[b:b+1]  # [1, C, H, W]
            norm_grid_batch = norm_grid.unsqueeze(0)  # [1, H, W, 2]
            
            transformed_img = F.grid_sample(
                img_single, norm_grid_batch, 
                mode='bilinear', padding_mode='border', align_corners=True
            )
            
            # Crop back to original size if padded
            if self.pad:
                h_start = self.h_pad // 2
                w_start = self.w_pad // 2
                transformed_img = transformed_img[
                    :, :, h_start:h_start+self.input_height, w_start:w_start+self.input_width
                ]
            
            transformed_images.append(transformed_img)
            
            # Transform landmarks if provided
            if landmarks is not None:
                landmarks_single = landmarks[b]  # [N, 2]
                
                # Convert landmarks to grid coordinates
                if self.pad:
                    landmarks_grid = landmarks_single + torch.tensor(
                        [self.h_pad//2, self.w_pad//2], device=device
                    )
                else:
                    landmarks_grid = landmarks_single.clone()
                
                # Apply inverse transformation to landmarks
                # This is a simplified approach - could be improved with proper inverse TPS
                transformed_landmarks = landmarks_grid  # Placeholder
                
                # Convert back to original coordinate system
                if self.pad:
                    transformed_landmarks = transformed_landmarks - torch.tensor(
                        [self.h_pad//2, self.w_pad//2], device=device
                    )
                
                transformed_landmarks_list.append(transformed_landmarks)
        
        # Concatenate results
        transformed_image = torch.cat(transformed_images, dim=0)
        
        if landmarks is not None:
            transformed_landmarks = torch.stack(transformed_landmarks_list, dim=0)
        else:
            transformed_landmarks = None
        
        return transformed_image, transformed_landmarks, transformation_grid
