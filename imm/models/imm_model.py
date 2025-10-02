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
        
        # Initialize VGG-16 backend for perceptual loss if needed
        self.vgg16 = None
        if hasattr(config, 'perceptual') and config.reconstruction_loss == 'perceptual':
            backend = getattr(config.perceptual, 'backend', 'selfsup')
            if backend == 'selfsup':
                # Self-supervised backend: grayscale + TF centering
                try:
                    self.vgg16 = VGG16Features(
                        getattr(config.perceptual, 'net_file', ''),
                        config.perceptual.comp
                    )
                except Exception as e:
                    print(f"Warning: Selfsup VGG16 init failed: {e}")
                    print("Falling back to L2 loss")
                    config.reconstruction_loss = 'l2'
                    self.vgg16 = None
            else:
                # TorchVision fallback only if explicitly requested
                try:
                    from .vgg16 import TorchVisionVGG16Features
                    self.vgg16 = TorchVisionVGG16Features(config.perceptual.comp)
                except Exception as e:
                    print(f"Warning: TorchVision VGG16 init failed: {e}")
                    print("Falling back to L2 loss")
                    config.reconstruction_loss = 'l2'
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
        print(f"\n🔍 === IMM MODEL FORWARD PASS (training={training}) ===")
        
        im = inputs['image']  # [B, C, H, W]
        future_im = inputs['future_image']  # [B, C, H, W]
        
        print(f"🔍 Input shapes: im={im.shape}, future_im={future_im.shape}")
        print(f"🔍 Input ranges: im=[{im.min():.3f}, {im.max():.3f}], future_im=[{future_im.min():.3f}, {future_im.max():.3f}]")
        print(f"🔍 Input means: im={im.mean():.3f}, future_im={future_im.mean():.3f}")
        
        batch_size, channels, height, width = future_im.shape
        assert height == width, "Only square images are supported"
        max_size = height
        print(f"🔍 Max size: {max_size}")
        
        # Determine renderer sizes
        render_sizes = []
        size = max_size
        stride = self.config.renderer_stride
        while True:
            render_sizes.append(size)
            if size <= self.config.min_res:
                break
            size = size // stride
        
        print(f"🔍 Render sizes: {render_sizes}")
        
        # Extract image features
        print(f"\n🔍 === IMAGE ENCODER ===")
        image_embeddings = self.image_encoder(im)
        print(f"🔍 Image embeddings count: {len(image_embeddings)}")
        for i, emb in enumerate(image_embeddings):
            print(f"🔍 Image embedding {i}: shape={emb.shape}, range=[{emb.min():.3f}, {emb.max():.3f}], mean={emb.mean():.3f}")
        
        # Extract pose features and landmarks
        print(f"\n🔍 === POSE ENCODER ===")
        gauss_mu, pose_embeddings = self.pose_encoder(future_im, render_sizes)
        print(f"🔍 Pose landmarks (gauss_mu): shape={gauss_mu.shape}, range=[{gauss_mu.min():.3f}, {gauss_mu.max():.3f}], mean={gauss_mu.mean():.3f}")
        print(f"🔍 Pose embeddings count: {len(pose_embeddings)}")
        for i, pe in enumerate(pose_embeddings):
            print(f"🔍 Pose embedding {i}: shape={pe.shape}, range=[{pe.min():.3f}, {pe.max():.3f}], mean={pe.mean():.3f}")
        
        # Group embeddings by size
        print(f"\n🔍 === GROUPING EMBEDDINGS ===")
        grouped_embeddings = self._group_embeddings_by_size(image_embeddings)
        print(f"🔍 Grouped image embeddings by size: {list(grouped_embeddings.keys())}")
        
        # Convert pose embeddings from [B, H, W, N] to [B, N, H, W] for grouping
        pose_embeddings_transposed = [pe.permute(0, 3, 1, 2) for pe in pose_embeddings]
        grouped_pose_embeddings = self._group_embeddings_by_size(pose_embeddings_transposed)
        print(f"🔍 Grouped pose embeddings by size: {list(grouped_pose_embeddings.keys())}")
        
        # Resize embeddings to match render sizes
        for render_size in render_sizes:
            if render_size not in grouped_embeddings:
                print(f"🔍 Resizing embeddings for size {render_size}")
                self._add_resized_embeddings(grouped_embeddings, render_size)
        
        # Create joint embeddings
        print(f"\n🔍 === JOINT EMBEDDINGS ===")
        joint_embeddings = {}
        for rs in render_sizes:
            joint_embeddings[rs] = torch.cat(
                grouped_embeddings[rs] + grouped_pose_embeddings[rs], dim=1
            )
            print(f"🔍 Joint embedding {rs}: shape={joint_embeddings[rs].shape}, range=[{joint_embeddings[rs].min():.3f}, {joint_embeddings[rs].max():.3f}], mean={joint_embeddings[rs].mean():.3f}")
        
        # Determine extra channels for perceptual workaround (match original TF behavior)
        workaround_channels = 0
        if (hasattr(self.config, 'channels_bug_fix') and self.config.channels_bug_fix and 
            hasattr(self.config, 'perceptual') and hasattr(self.config.perceptual, 'comp')):
            workaround_channels = len(self.config.perceptual.comp)

        print(f"\n🔍 === RENDERER ===")
        print(f"🔍 Workaround channels: {workaround_channels}")

        # Generate future image with correct number of output channels
        n_final_out = 3 + workaround_channels
        print(f"🔍 Final output channels: {n_final_out}")
        future_im_pred = self.renderer(joint_embeddings, max_size, n_final_out=n_final_out)

        print(f"🔍 Raw renderer output: shape={future_im_pred.shape}, range=[{future_im_pred.min():.3f}, {future_im_pred.max():.3f}], mean={future_im_pred.mean():.3f}")
        # Keep only RGB for the reconstruction loss; drop workaround channels if present
        # Keep only RGB for reconstruction loss (no scaling/clamping here to match TF)
        future_im_pred_mu = future_im_pred[:, :3]
        
        # Prepare outputs
        outputs = {
            'future_im_pred': future_im_pred_mu,
            'gauss_yx': gauss_mu,
            'pose_embeddings': pose_embeddings,  # Keep original [B, H, W, N] format
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
        print(f"\n🔍 === LOSS COMPUTATION (training={training}) ===")
        
        future_im = inputs['future_image']
        future_im_pred = outputs['future_im_pred']
        
        print(f"🔍 Loss inputs: future_im shape={future_im.shape}, future_im_pred shape={future_im_pred.shape}")
        print(f"🔍 Loss ranges: future_im=[{future_im.min():.3f}, {future_im.max():.3f}], future_im_pred=[{future_im_pred.min():.3f}, {future_im_pred.max():.3f}]")
        print(f"🔍 Loss means: future_im={future_im.mean():.3f}, future_im_pred={future_im_pred.mean():.3f}")
        
        # SAFETY CHECK: Ensure correct image range for loss computation
        if future_im.max() <= 2.0:
            print("⚠️  WARNING: Images in [0,1] range detected, scaling to [0,255] for loss")
            future_im = future_im * 255.0
            # future_im_pred is already scaled in forward method
        
        # Additional validation
        if torch.isnan(future_im_pred).any() or torch.isinf(future_im_pred).any():
            print("⚠️  WARNING: NaN/Inf detected in predictions")
        # Mask for loss computation
        loss_mask = inputs.get('mask', None)
        
        # Reconstruction loss
        print(f"🔍 Reconstruction loss type: {self.config.reconstruction_loss}")
        if self.config.reconstruction_loss == 'perceptual' and self.vgg16 is not None:
            print(f"🔍 Computing perceptual loss...")
            reconstruction_loss = self._perceptual_loss(
                future_im, future_im_pred, training, loss_mask
            )
            w_reconstruct = 1.0
            print(f"🔍 Perceptual loss: {reconstruction_loss.item():.6f}, w_reconstruct: {w_reconstruct:.6f}")
        elif self.config.reconstruction_loss == 'l2':
            print(f"🔍 Computing L2 loss...")
            l = F.mse_loss(future_im_pred, future_im, reduction='none')
            print(f"🔍 Raw L2 loss: mean={torch.mean(l).item():.6f}, shape={l.shape}")
            if loss_mask is not None:
                print(f"🔍 Applying loss mask...")
                l = self._apply_loss_mask(l, loss_mask)
                print(f"🔍 Masked L2 loss: mean={torch.mean(l).item():.6f}")
            reconstruction_loss = 1000.0 * torch.mean(l)  # Match TF scaling exactly
            print(f"🔍 L2 loss (scaled by 1000): {reconstruction_loss.item():.6f}")
            w_reconstruct = 1.0/255.0  # No additional scaling needed
        else:
            raise ValueError(f'Unknown reconstruction loss: {self.config.reconstruction_loss}')

        # Weight decay loss (match TF implementation)
        print(f"🔍 Computing weight decay loss...")
        weight_decay_loss = self._decay()
        print(f"🔍 Weight decay loss: {weight_decay_loss.item():.6f}")
        
        # Total loss
        total_loss = w_reconstruct * reconstruction_loss + weight_decay_loss
        print(f"🔍 Total loss components: recon={w_reconstruct * reconstruction_loss:.6f}, decay={weight_decay_loss:.6f}, total={total_loss.item():.6f}")

        
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
        
        # Use only the specified feature layers (match TF behavior)
        feat_names = self.config.perceptual.comp
        feat_gt = {}
        feat_pred = {}
        
        batch_size = gt_image.shape[0]
        for name in feat_names:
            if name in feats:
                feat_gt[name] = feats[name][:batch_size]
                feat_pred[name] = feats[name][batch_size:]
        
        # Compute losses with adaptive weighting
        losses = []
        # Match TF weights exactly: [100.0, 1.6, 2.3, 1.8, 2.8, 100.0]
        # These correspond to ['input','conv1_2','conv2_2','conv3_2','conv4_2','conv5_2']
        weights = [100.0, 1.6, 2.3, 1.8, 2.8, 100.0]
        
        # Use L2 or L1 loss
        loss_fn = F.mse_loss if self.config.perceptual.l2 else F.l1_loss
        
        for i, name in enumerate(feat_names):
            if i >= len(weights) or name not in feat_gt:
                continue
                
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
            normalized_loss = torch.mean(_apply_loss_mask(feat_loss / weight, loss_mask))
            losses.append(normalized_loss)
        
        # Combine losses
        total_loss = 1000.0 * sum(losses)
        return total_loss
    
    def _decay(self) -> torch.Tensor:
        """Compute weight decay loss (L2 regularization) - matches TF wd=1e-5."""
        weight_decay = 0.0
        for param in self.parameters():
            if param.requires_grad:
                weight_decay += torch.sum(param ** 2)
        return self._conv_opts['weight_decay'] * weight_decay  # Match TF's wd=1e-5
    
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
                nn.init.trunc_normal_(m.weight)
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
        print(f"🔍 ImageEncoder input: shape={x.shape}, range=[{x.min():.3f}, {x.max():.3f}], mean={x.mean():.3f}")
        
        features = [x]  # Include input image
        
        # Block 1
        x1 = self.block1(x)
        print(f"🔍 ImageEncoder block1: shape={x1.shape}, range=[{x1.min():.3f}, {x1.max():.3f}], mean={x1.mean():.3f}")
        features.append(x1)
        
        # Block 2
        x2 = self.block2(x1)
        print(f"🔍 ImageEncoder block2: shape={x2.shape}, range=[{x2.min():.3f}, {x2.max():.3f}], mean={x2.mean():.3f}")
        features.append(x2)
        
        # Block 3
        x3 = self.block3(x2)
        print(f"🔍 ImageEncoder block3: shape={x3.shape}, range=[{x3.min():.3f}, {x3.max():.3f}], mean={x3.mean():.3f}")
        features.append(x3)
        
        # Block 4
        x4 = self.block4(x3)
        print(f"🔍 ImageEncoder block4: shape={x4.shape}, range=[{x4.min():.3f}, {x4.max():.3f}], mean={x4.mean():.3f}")
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
                nn.init.trunc_normal_(m.weight)
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
        print(f"🔍 PoseEncoder input: shape={x.shape}, range=[{x.min():.3f}, {x.max():.3f}], mean={x.mean():.3f}")
        print(f"🔍 PoseEncoder map_sizes: {map_sizes}")
        
        # Encode features
        x1 = self.block1(x)
        print(f"🔍 PoseEncoder block1: shape={x1.shape}, range=[{x1.min():.3f}, {x1.max():.3f}], mean={x1.mean():.3f}")
        x2 = self.block2(x1)
        print(f"🔍 PoseEncoder block2: shape={x2.shape}, range=[{x2.min():.3f}, {x2.max():.3f}], mean={x2.mean():.3f}")
        x3 = self.block3(x2)
        print(f"🔍 PoseEncoder block3: shape={x3.shape}, range=[{x3.min():.3f}, {x3.max():.3f}], mean={x3.mean():.3f}")
        x4 = self.block4(x3)
        print(f"🔍 PoseEncoder block4: shape={x4.shape}, range=[{x4.min():.3f}, {x4.max():.3f}], mean={x4.mean():.3f}")
        
        # Predict heatmaps
        heatmaps = self.final_conv(x4)  # [B, n_maps, H, W]
        print(f"🔍 PoseEncoder heatmaps: shape={heatmaps.shape}, range=[{heatmaps.min():.3f}, {heatmaps.max():.3f}], mean={heatmaps.mean():.3f}")
        
        # Extract coordinates from heatmaps
        gauss_mu = self._extract_coordinates(heatmaps)
        print(f"🔍 PoseEncoder coordinates (gauss_mu): shape={gauss_mu.shape}, range=[{gauss_mu.min():.3f}, {gauss_mu.max():.3f}], mean={gauss_mu.mean():.3f}")
        
        # Generate Gaussian maps at different sizes
        gaussian_maps = []
        inv_std = 1.0 / self.gauss_std
        print(f"🔍 PoseEncoder inv_std: {inv_std}, gauss_mode: {self.gauss_mode}")
        
        for i, map_size in enumerate(map_sizes):
            gauss_maps = get_gaussian_maps(
                gauss_mu, (map_size, map_size), inv_std, self.gauss_mode
            )
            print(f"🔍 PoseEncoder gaussian_map {i} (size {map_size}): shape={gauss_maps.shape}, range=[{gauss_maps.min():.3f}, {gauss_maps.max():.3f}], mean={gauss_maps.mean():.3f}")
            # Keep as [B, H, W, N] to match TensorFlow implementation
            # Do NOT transpose to [B, N, H, W] here
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
    
    def _initialize_layer(self, layer):
        """Initialize a layer with TensorFlow-compatible weights."""
        if isinstance(layer, nn.Conv2d):
            # Use same initialization as TensorFlow
            nn.init.trunc_normal_(layer.weight, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
        elif isinstance(layer, nn.Sequential):
            for sublayer in layer:
                self._initialize_layer(sublayer)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)
    
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
        print(f"🔍 SimpleRenderer input:")
        print(f"🔍   joint_embeddings sizes: {list(joint_embeddings.keys())}")
        print(f"🔍   final_res: {final_res}, n_final_out: {n_final_out}")
        
        # Start with the smallest resolution
        sizes = sorted(joint_embeddings.keys())
        x = joint_embeddings[sizes[0]]  # Start with smallest size (e.g., 16x16)
        print(f"🔍 SimpleRenderer starting with size {sizes[0]}: shape={x.shape}, range=[{x.min():.3f}, {x.max():.3f}], mean={x.mean():.3f}")
        
        current_size = sizes[0]
        filters = self.n_filters_render * 8
        conv_id = 1
        print(f"🔍 SimpleRenderer initial filters: {filters}")
        
        while current_size <= final_res:
            print(f"🔍 SimpleRenderer processing size {current_size} (target: {final_res})")
            
            # Create or get convolution layer
            layer_name = f'conv_{conv_id}'
            if layer_name not in self.conv_layers:
                in_channels = x.shape[1]
                print(f"🔍 SimpleRenderer creating layer {layer_name}: in_channels={in_channels}")
                
                if current_size == final_res:
                    # Final layer
                    print(f"🔍 SimpleRenderer creating FINAL layer with {n_final_out} output channels (NO SIGMOID)")
                    self.conv_layers[layer_name] = nn.Conv2d(in_channels, n_final_out, 3, padding=1, bias=True)
                    self.conv_layers[layer_name] = nn.Sequential(
                        nn.Conv2d(in_channels, n_final_out, 3, padding=1, bias=True),
                        # nn.Sigmoid()
                    )
                    self.conv_layers[layer_name] = self.conv_layers[layer_name].to(x.device)
                else:
                    # Intermediate layer
                    print(f"🔍 SimpleRenderer creating intermediate layer with {filters} filters")
                    self.conv_layers[layer_name] = nn.Sequential(
                        nn.Conv2d(in_channels, filters, 3, padding=1, bias=True),
                        nn.BatchNorm2d(filters),
                        nn.ReLU(inplace=True)
                    )
                    self.conv_layers[layer_name] = self.conv_layers[layer_name].to(x.device)
            
            # Apply convolution
            x_before = x
            x = self.conv_layers[layer_name](x)
            print(f"🔍 SimpleRenderer {layer_name}: {x_before.shape} -> {x.shape}, range=[{x.min():.3f}, {x.max():.3f}], mean={x.mean():.3f}")
            
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
                    self.conv_layers[layer_name2] = self.conv_layers[layer_name2].to(x.device)
                
                x = self.conv_layers[layer_name2](x)
                
            # Upsample to next resolution
            next_size = current_size * 2
            x = F.interpolate(x, size=(next_size, next_size), mode='bilinear', align_corners=False)
            current_size = next_size
            
            conv_id += 2
            if filters >= 8:
                filters //= 2
        
        return x