"""
IMM Model implementation in PyTorch.

Converted from TensorFlow implementation.
Original Authors: Ankush Gupta, Tomas Jakab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union

from .base_model import BaseModel
from .vgg16 import VGG16Features
from ..utils.utils import get_gaussian_maps, colorize_landmark_maps


class IMMModel(BaseModel):
    """
    IMM (Image Manipulation Model) for unsupervised landmark learning.
    
    This model learns to predict object landmarks in an unsupervised manner
    by training on image pairs and learning to generate future images from
    current images using predicted landmarks.
    """
    
    def __init__(self, config, dtype: torch.dtype = torch.float32, name: str = 'IMMModel'):
        """
        Initialize the IMM model.
        
        Args:
            config: Model configuration object
            dtype: Data type for model parameters
            name: Model name
        """
        super(IMMModel, self).__init__(dtype, name)
        self.config = config
        
        # Initialize VGG16 for perceptual loss if needed
        if hasattr(config, 'perceptual') and config.reconstruction_loss == 'perceptual':
            try:
                self.vgg16 = VGG16Features(
                    config.perceptual.net_file,
                    config.perceptual.comp
                )
            except Exception as e:
                print(f"Warning: Could not load VGG16 for perceptual loss: {e}")
                print("Falling back to L2 loss")
                config.reconstruction_loss = 'l2'
                self.vgg16 = None
        else:
            self.vgg16 = None
        
        # Build model components
        self._build_model()
    
    def _build_model(self):
        """Build the model components."""
        
        # Image encoder
        self.image_encoder = ImageEncoder(
            n_filters=self.config.n_filters,
            dtype=self.dtype
        )
        
        # Pose encoder
        self.pose_encoder = PoseEncoder(
            n_filters=self.config.n_filters,
            n_maps=self.config.n_maps,
            gauss_std=self.config.gauss_std,
            gauss_mode=self.config.gauss_mode,
            dtype=self.dtype
        )
        
        # Renderer
        self.renderer = SimpleRenderer(
            n_filters_render=self.config.n_filters_render,
            renderer_stride=self.config.renderer_stride,
            min_res=self.config.min_res,
            dtype=self.dtype
        )
    
    def forward(self, inputs: Dict[str, torch.Tensor], training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the IMM model.
        
        Args:
            inputs: Dictionary containing 'image' and 'future_image' tensors
            training: Whether in training mode
            
        Returns:
            Dictionary containing model outputs and intermediate results
        """
        im = inputs['image']  # [B, C, H, W]
        future_im = inputs['future_image']  # [B, C, H, W]
        
        batch_size, channels, height, width = future_im.shape
        assert height == width, "Only square images are supported"
        max_size = height
        
        # Determine renderer sizes
        render_sizes = []
        size = max_size
        stride = self.config.renderer_stride
        while True:
            render_sizes.append(size)
            if size <= self.config.min_res:
                break
            size = size // stride
        
        # Extract image features
        image_embeddings = self.image_encoder(im)
        
        # Extract pose features and landmarks
        gauss_mu, pose_embeddings = self.pose_encoder(future_im, render_sizes)
        
        # Group embeddings by size
        grouped_embeddings = self._group_embeddings_by_size(image_embeddings)
        grouped_pose_embeddings = self._group_embeddings_by_size(pose_embeddings)
        
        # Resize embeddings to match render sizes
        for render_size in render_sizes:
            if render_size not in grouped_embeddings:
                self._add_resized_embeddings(grouped_embeddings, render_size)
        
        # Create joint embeddings
        joint_embeddings = {}
        for rs in render_sizes:
            joint_embeddings[rs] = torch.cat(
                grouped_embeddings[rs] + grouped_pose_embeddings[rs], dim=1
            )
        
        # Generate future image
        future_im_pred = self.renderer(joint_embeddings, max_size)
        
        # Handle channel bug fix if needed
        workaround_channels = 0
        if hasattr(self.config, 'channels_bug_fix') and self.config.channels_bug_fix:
            workaround_channels = len(self.config.perceptual.comp)
        
        color_channels = future_im_pred.shape[1] - workaround_channels
        if workaround_channels > 0:
            future_im_pred_mu = future_im_pred[:, :color_channels]
        else:
            future_im_pred_mu = future_im_pred
        
        # Prepare outputs
        outputs = {
            'future_im_pred': future_im_pred_mu,
            'gauss_yx': gauss_mu,
            'pose_embeddings': pose_embeddings,
            'image_embeddings': image_embeddings
        }
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    inputs: Dict[str, torch.Tensor], training: bool = True) -> torch.Tensor:
        """
        Compute the total loss for the model.
        
        Args:
            outputs: Model outputs from forward pass
            inputs: Input data
            training: Whether in training mode
            
        Returns:
            Total loss tensor
        """
        future_im = inputs['future_image']
        future_im_pred = outputs['future_im_pred']
        
        # Mask for loss computation
        loss_mask = inputs.get('mask', None)
        
        # Reconstruction loss
        if self.config.reconstruction_loss == 'perceptual' and self.vgg16 is not None:
            reconstruction_loss = self._perceptual_loss(
                future_im, future_im_pred, training, loss_mask
            )
            w_reconstruct = 1.0
        elif self.config.reconstruction_loss == 'l2':
            l = F.mse_loss(future_im_pred, future_im, reduction='none')
            if loss_mask is not None:
                l = self._apply_loss_mask(l, loss_mask)
            reconstruction_loss = 1000 * torch.mean(l)
            w_reconstruct = 1.0 / 255.0
        else:
            raise ValueError(f'Unknown reconstruction loss: {self.config.reconstruction_loss}')
        
        # Weight decay loss
        weight_decay_loss = 0.0
        for param in self.parameters():
            if param.dim() > 1:  # Only apply to weight matrices, not biases
                weight_decay_loss += torch.sum(param ** 2)
        weight_decay_loss *= self._conv_opts['weight_decay']
        
        # Total loss
        total_loss = w_reconstruct * reconstruction_loss + weight_decay_loss
        
        return total_loss
    
    def _perceptual_loss(self, gt_image: torch.Tensor, pred_image: torch.Tensor, 
                        training: bool, loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute perceptual loss using VGG16 features.
        
        Args:
            gt_image: Ground truth image [B, C, H, W]
            pred_image: Predicted image [B, C, H, W]
            training: Whether in training mode
            loss_mask: Optional mask for loss computation
            
        Returns:
            Perceptual loss tensor
        """
        # Concatenate images for efficient processing
        ims = torch.cat([gt_image, pred_image], dim=0)
        
        # Extract VGG features
        feats = self.vgg16(ims)
        
        # Split features back
        feat_names = list(feats.keys())
        feat_gt = {}
        feat_pred = {}
        
        batch_size = gt_image.shape[0]
        for name in feat_names:
            feat_gt[name] = feats[name][:batch_size]
            feat_pred[name] = feats[name][batch_size:]
        
        # Compute losses with adaptive weighting
        losses = []
        weights = [100.0, 1.6, 2.3, 1.8, 2.8, 100.0]
        
        # Use L2 or L1 loss
        loss_fn = F.mse_loss if self.config.perceptual.l2 else F.l1_loss
        
        for i, name in enumerate(feat_names):
            if i >= len(weights):
                break
                
            # Compute feature loss
            feat_loss = loss_fn(feat_pred[name], feat_gt[name], reduction='none')
            
            # Apply mask if provided
            if loss_mask is not None:
                feat_loss = self._apply_loss_mask(feat_loss, loss_mask)
            
            # Adaptive weighting using exponential moving average
            weight = self._exp_running_avg(
                torch.mean(feat_loss), training, 
                init_val=weights[i], name=f'perceptual_weight_{name}'
            )
            
            # Normalize by weight and compute mean
            normalized_loss = torch.mean(feat_loss / weight)
            losses.append(normalized_loss)
        
        # Combine losses
        total_loss = 1000.0 * sum(losses)
        return total_loss
    
    def _apply_loss_mask(self, loss_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply loss mask to loss map."""
        # Resize mask to match loss map size
        if mask.shape[-2:] != loss_map.shape[-2:]:
            mask = F.interpolate(mask, size=loss_map.shape[-2:], mode='bilinear', align_corners=False)
        return loss_map * mask
    
    def _group_embeddings_by_size(self, embeddings: List[torch.Tensor]) -> Dict[int, List[torch.Tensor]]:
        """Group embeddings by their spatial size."""
        grouped = defaultdict(list)
        for embedding in embeddings:
            size = embedding.shape[-1]  # Assume square tensors
            grouped[size].append(embedding)
        return dict(grouped)
    
    def _add_resized_embeddings(self, grouped_embeddings: Dict[int, List[torch.Tensor]], 
                               target_size: int):
        """Add resized embeddings for target size."""
        # Find the closest larger size
        available_sizes = sorted(grouped_embeddings.keys())
        source_size = None
        for size in available_sizes:
            if size >= target_size:
                source_size = size
                break
        
        if source_size is None:
            return
        
        # Resize embeddings
        resized_embeddings = []
        for embedding in grouped_embeddings[source_size]:
            resized = F.interpolate(
                embedding, size=(target_size, target_size), 
                mode='bilinear', align_corners=True
            )
            resized_embeddings.append(resized)
        
        grouped_embeddings[target_size] = grouped_embeddings.get(target_size, []) + resized_embeddings


class ImageEncoder(nn.Module):
    """CNN encoder for extracting image features at multiple scales."""
    
    def __init__(self, n_filters: int = 32, dtype: torch.dtype = torch.float32):
        super(ImageEncoder, self).__init__()
        self.n_filters = n_filters
        self.dtype = dtype
        
        # Build encoder blocks
        filters = n_filters
        
        # Block 1 (full resolution)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, filters, 7, padding=3, bias=True),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 3, padding=1, bias=True),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        # Block 2 (1/2 resolution)
        filters *= 2
        self.block2 = nn.Sequential(
            nn.Conv2d(self.n_filters, filters, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 3, padding=1, bias=True),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        # Block 3 (1/4 resolution)
        prev_filters = filters
        filters *= 2
        self.block3 = nn.Sequential(
            nn.Conv2d(prev_filters, filters, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 3, padding=1, bias=True),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        # Block 4 (1/8 resolution)
        prev_filters = filters
        filters *= 2
        self.block4 = nn.Sequential(
            nn.Conv2d(prev_filters, filters, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 3, padding=1, bias=True),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights similar to TensorFlow implementation."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the image encoder.
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            List of feature maps at different resolutions
        """
        features = [x]  # Include input image
        
        # Block 1
        x1 = self.block1(x)
        features.append(x1)
        
        # Block 2
        x2 = self.block2(x1)
        features.append(x2)
        
        # Block 3
        x3 = self.block3(x2)
        features.append(x3)
        
        # Block 4
        x4 = self.block4(x3)
        features.append(x4)
        
        return features


class PoseEncoder(nn.Module):
    """CNN encoder for predicting landmark positions as Gaussian heatmaps."""
    
    def __init__(self, n_filters: int = 32, n_maps: int = 10, 
                 gauss_std: float = 0.1, gauss_mode: str = 'rot',
                 dtype: torch.dtype = torch.float32):
        super(PoseEncoder, self).__init__()
        self.n_filters = n_filters
        self.n_maps = n_maps
        self.gauss_std = gauss_std
        self.gauss_mode = gauss_mode
        self.dtype = dtype
        
        # Build encoder (same as image encoder)
        filters = n_filters
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, filters, 7, padding=3, bias=True),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 3, padding=1, bias=True),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        # Block 2
        filters *= 2
        self.block2 = nn.Sequential(
            nn.Conv2d(self.n_filters, filters, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 3, padding=1, bias=True),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        # Block 3
        prev_filters = filters
        filters *= 2
        self.block3 = nn.Sequential(
            nn.Conv2d(prev_filters, filters, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 3, padding=1, bias=True),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        # Block 4
        prev_filters = filters
        filters *= 2
        self.block4 = nn.Sequential(
            nn.Conv2d(prev_filters, filters, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 3, padding=1, bias=True),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        # Final layer to predict heatmaps
        self.final_conv = nn.Conv2d(filters, n_maps, 1, bias=True)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights similar to TensorFlow implementation."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, map_sizes: List[int]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the pose encoder.
        
        Args:
            x: Input image [B, 3, H, W]
            map_sizes: List of sizes for which to generate Gaussian maps
            
        Returns:
            Tuple of (landmark coordinates, Gaussian maps at different sizes)
        """
        # Encode features
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        
        # Predict heatmaps
        heatmaps = self.final_conv(x4)  # [B, n_maps, H, W]
        
        # Extract coordinates from heatmaps
        gauss_mu = self._extract_coordinates(heatmaps)
        
        # Generate Gaussian maps at different sizes
        gaussian_maps = []
        inv_std = 1.0 / self.gauss_std
        
        for map_size in map_sizes:
            gauss_maps = get_gaussian_maps(
                gauss_mu, (map_size, map_size), inv_std, self.gauss_mode
            )
            # Convert from [B, H, W, N] to [B, N, H, W]
            gauss_maps = gauss_maps.permute(0, 3, 1, 2)
            gaussian_maps.append(gauss_maps)
        
        return gauss_mu, gaussian_maps
    
    def _extract_coordinates(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Extract landmark coordinates from heatmaps using soft-argmax.
        
        Args:
            heatmaps: Heatmaps [B, n_maps, H, W]
            
        Returns:
            Landmark coordinates [B, n_maps, 2] in range [-1, 1]
        """
        batch_size, n_maps, height, width = heatmaps.shape
        device = heatmaps.device
        
        # Get y coordinates (average over x dimension)
        y_probs = torch.mean(heatmaps, dim=3)  # [B, n_maps, H]
        y_probs = F.softmax(y_probs, dim=2)
        
        y_coords = torch.linspace(-1.0, 1.0, height, device=device)  # [H]
        y_coords = y_coords.view(1, 1, height)  # [1, 1, H]
        gauss_y = torch.sum(y_probs * y_coords, dim=2)  # [B, n_maps]
        
        # Get x coordinates (average over y dimension)
        x_probs = torch.mean(heatmaps, dim=2)  # [B, n_maps, W]
        x_probs = F.softmax(x_probs, dim=2)
        
        x_coords = torch.linspace(-1.0, 1.0, width, device=device)  # [W]
        x_coords = x_coords.view(1, 1, width)  # [1, 1, W]
        gauss_x = torch.sum(x_probs * x_coords, dim=2)  # [B, n_maps]
        
        # Stack coordinates
        gauss_mu = torch.stack([gauss_y, gauss_x], dim=2)  # [B, n_maps, 2]
        
        return gauss_mu


class SimpleRenderer(nn.Module):
    """Simple renderer that generates images from joint embeddings."""
    
    def __init__(self, n_filters_render: int = 32, renderer_stride: int = 2,
                 min_res: int = 16, dtype: torch.dtype = torch.float32):
        super(SimpleRenderer, self).__init__()
        self.n_filters_render = n_filters_render
        self.renderer_stride = renderer_stride
        self.min_res = min_res
        self.dtype = dtype
        
        # Renderer layers will be built dynamically based on input sizes
        self.conv_layers = nn.ModuleDict()
    
    def forward(self, joint_embeddings: Dict[int, torch.Tensor], 
               final_res: int, n_final_out: int = 3) -> torch.Tensor:
        """
        Forward pass through the renderer.
        
        Args:
            joint_embeddings: Dictionary mapping sizes to joint feature tensors
            final_res: Final output resolution
            n_final_out: Number of output channels
            
        Returns:
            Generated image [B, n_final_out, final_res, final_res]
        """
        # Start with the smallest resolution
        sizes = sorted(joint_embeddings.keys())
        x = joint_embeddings[sizes[0]]  # Start with smallest size (e.g., 16x16)
        
        current_size = sizes[0]
        filters = self.n_filters_render * 8
        conv_id = 1
        
        while current_size <= final_res:
            # Create or get convolution layer
            layer_name = f'conv_{conv_id}'
            if layer_name not in self.conv_layers:
                in_channels = x.shape[1]
                
                if current_size == final_res:
                    # Final layer
                    self.conv_layers[layer_name] = nn.Conv2d(
                        in_channels, n_final_out, 3, padding=1, bias=True
                    )
                else:
                    # Intermediate layer
                    self.conv_layers[layer_name] = nn.Sequential(
                        nn.Conv2d(in_channels, filters, 3, padding=1, bias=True),
                        nn.BatchNorm2d(filters),
                        nn.ReLU(inplace=True)
                    )
            
            # Apply convolution
            x = self.conv_layers[layer_name](x)
            
            if current_size == final_res:
                break
            else:
                # Add second convolution for this resolution
                layer_name2 = f'conv_{conv_id + 1}'
                if layer_name2 not in self.conv_layers:
                    self.conv_layers[layer_name2] = nn.Sequential(
                        nn.Conv2d(filters, filters, 3, padding=1, bias=True),
                        nn.BatchNorm2d(filters),
                        nn.ReLU(inplace=True)
                    )
                
                x = self.conv_layers[layer_name2](x)
                
                # Upsample to next resolution
                next_size = current_size * 2
                x = F.interpolate(x, size=(next_size, next_size), mode='bilinear', align_corners=False)
                current_size = next_size
                
                conv_id += 2
                if filters >= 8:
                    filters //= 2
        
        return x
