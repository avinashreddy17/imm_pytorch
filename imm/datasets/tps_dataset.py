"""
TPS Dataset implementation in PyTorch.

Converted from TensorFlow implementation.
Original Authors: Tomas Jakab, Ankush Gupta
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os.path as osp
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

from ..utils.tps_sampler_tf_compat import TPSRandomSamplerTFCompat
# from ..utils.tps_sampler_normalized import TPSRandomSamplerNormalized


class ImagePairDataset(Dataset, ABC):
    """Abstract base class for datasets returning image pairs."""
    
    def __init__(self, data_dir: str, subset: str, image_size: List[int] = [128, 128],
                 bbox_padding: List[int] = [10, 10], crop_to_bbox: bool = False,
                 jittering: Optional[bool] = None, augmentations: List[str] = ['flip', 'swap'],
                 name: str = 'PairDataset'):
        """
        Initialize image pair dataset.
        
        Args:
            data_dir: Directory containing the dataset
            subset: Dataset subset ('train', 'val', 'test')
            image_size: Target image size [height, width]
            bbox_padding: Padding around bounding boxes
            crop_to_bbox: Whether to crop to bounding box
            jittering: Whether to apply jittering (None = True for train)
            augmentations: List of augmentations to apply
            name: Dataset name
        """
        super(ImagePairDataset, self).__init__()
        
        self._data_dir = data_dir
        self._subset = subset
        self._image_size = image_size
        self.image_size = image_size
        self._bbox_padding = bbox_padding
        self._crop_to_bbox = crop_to_bbox
        
        if jittering is None:
            self._jittering = (subset == 'train')
        else:
            self._jittering = jittering
            
        self._augmentations = augmentations
        self._name = name
    
    def _read_image(self, image_path: str, channels: int = 3) -> torch.Tensor:
        """
        Read image from file path and convert to tensor.
        
        Args:
            image_path: Path to image file
            channels: Number of channels
            
        Returns:
            Image tensor [C, H, W] in range [0, 255]
        """
        if isinstance(image_path, torch.Tensor):
            return image_path
        
        # Read image using PIL
        image = Image.open(image_path).convert('RGB' if channels == 3 else 'L')
        
        # Convert to numpy array
        image_np = np.array(image,dtype=np.float32)
        
        # Convert to tensor [H, W, C] -> [C, H, W]
        if image_np.ndim == 2:
            image_np = image_np[:, :, np.newaxis]
        
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()

        assert image_tensor.min() >= 0.0 and image_tensor.max() <= 255.0, \
            f"Image range [{image_tensor.min():.1f}, {image_tensor.max():.1f}] invalid, should be [0,255]"
        
        return image_tensor
    
    def _resize_image(self, image: torch.Tensor, size: List[int], 
                     keep_aspect: bool = True) -> torch.Tensor:
        """
        Resize image to target size.
        
        Args:
            image: Input image [C, H, W]
            size: Target size [height, width]
            keep_aspect: Whether to keep aspect ratio
            
        Returns:
            Resized image
        """
        if keep_aspect:
            # Resize keeping aspect ratio
            c, h, w = image.shape
            target_h, target_w = size
            
            # Calculate scale factor
            scale = min(target_h / h, target_w / w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize image
            image = F.interpolate(
                image.unsqueeze(0), size=(new_h, new_w), 
                mode='bilinear', align_corners=False
            ).squeeze(0)
            
            # Pad to target size
            pad_h = target_h - new_h
            pad_w = target_w - new_w
            
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), value=0)
        else:
            # Direct resize
            image = F.interpolate(
                image.unsqueeze(0), size=size, 
                mode='bilinear', align_corners=False
            ).squeeze(0)
        
        return image
    
    def _resize_points(self, points: torch.Tensor, original_size: List[int], 
                      new_size: List[int]) -> torch.Tensor:
        """
        Resize landmark points from original image size to new size.
        
        Args:
            points: Points tensor [N, 2] in (y, x) format
            original_size: Original image size [height, width]
            new_size: New image size [height, width]
            
        Returns:
            Resized points
        """
        scale_y = new_size[0] / original_size[0]
        scale_x = new_size[1] / original_size[1]
        
        scale = torch.tensor([scale_y, scale_x], device=points.device, dtype=points.dtype)
        return points * scale
    
    def _apply_augmentations(self, image: torch.Tensor, landmarks: Optional[torch.Tensor] = None,
                           training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply data augmentations.
        
        Args:
            image: Input image [C, H, W]
            landmarks: Optional landmarks [N, 2]
            training: Whether in training mode
            
        Returns:
            Augmented image and landmarks
        """
        if not training:
            return image, landmarks
        
        # Horizontal flip
        if 'flip' in self._augmentations and torch.rand(1) < 0.5:
            image = torch.flip(image, dims=[2])  # Flip width dimension
            if landmarks is not None:
                landmarks = landmarks.clone()
                landmarks[:, 1] = image.shape[2] - 1 - landmarks[:, 1]  # Flip x coordinate
        
        return image, landmarks
    
    @abstractmethod
    def _get_sample(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return dataset length."""
        pass
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item for PyTorch DataLoader."""
        return self._get_sample(idx)


class TPSDataset(ImagePairDataset):
    """Dataset with Thin Plate Spline (TPS) augmentation."""
    
    def __init__(self, data_dir: str, subset: str, max_samples: Optional[int] = None,
                 image_size: List[int] = [128, 128], order_stream: bool = False,
                 landmarks: bool = False, tps: bool = True, vertical_points: int = 10,
                 horizontal_points: int = 10, rotsd: List[float] = [0.0, 5.0],
                 scalesd: List[float] = [0.0, 0.1], transsd: List[float] = [0.1, 0.1],
                 warpsd: List[float] = [0.001, 0.005, 0.001, 0.01],
                 name: str = 'TPSDataset'):
        """
        Initialize TPS dataset.
        
        Args:
            data_dir: Directory containing the dataset
            subset: Dataset subset
            max_samples: Maximum number of samples
            image_size: Target image size [height, width]
            order_stream: Whether to iterate in order
            landmarks: Whether to output landmarks
            tps: Whether to apply TPS transformation
            vertical_points: Number of vertical control points
            horizontal_points: Number of horizontal control points
            rotsd: Rotation standard deviations for [target, source]
            scalesd: Scale standard deviations for [target, source]
            transsd: Translation standard deviations for [target, source]
            warpsd: Warp standard deviations [target_min, target_max, source_min, source_max]
            name: Dataset name
        """
        super(TPSDataset, self).__init__(
            data_dir, subset, image_size=image_size, jittering=False, name=name
        )
        
        if landmarks and tps:
            raise ValueError('Outputting landmarks is not supported with TPS transform.')
        
        self._max_samples = max_samples
        self._order_stream = order_stream
        self._landmarks_output = landmarks
        
        self._tps = tps
        if tps:
            self._target_sampler = TPSRandomSamplerTFCompat(
                image_size[0], image_size[1], 
                vertical_points=vertical_points, horizontal_points=horizontal_points,
                rotsd=rotsd[0], scalesd=scalesd[0], transsd=transsd[0],
                warpsd=(warpsd[0], warpsd[1]), pad=False
            )
            self._source_sampler = TPSRandomSamplerTFCompat(
                image_size[0], image_size[1],
                vertical_points=vertical_points, horizontal_points=horizontal_points,
                rotsd=rotsd[1], scalesd=scalesd[1], transsd=transsd[1],
                warpsd=(warpsd[2], warpsd[3]), pad=False
            )
        
        # Initialize data - to be implemented by subclasses
        self._initialize_data()
    
    @abstractmethod
    def _initialize_data(self):
        """Initialize dataset-specific data (images, landmarks, etc.)."""
        pass
    
    def _get_smooth_step(self, n: int, b: float) -> torch.Tensor:
        """Generate smooth step function."""
        x = torch.linspace(-1.0, 1.0, n)
        y = 0.5 + 0.5 * torch.tanh(x / b)
        return y
    
    def _get_smooth_mask(self, h: int, w: int, margin: int, step: int) -> torch.Tensor:
        """Generate smooth mask for TPS transformation."""
        b = 0.4
        step_up = self._get_smooth_step(step, b)
        step_down = self._get_smooth_step(step, -b)
        
        def create_strip(size):
            return torch.cat([
                torch.zeros(margin),
                step_up,
                torch.ones(size - 2 * margin - 2 * step),
                step_down,
                torch.zeros(margin)
            ])
        
        mask_x = create_strip(w)
        mask_y = create_strip(h)
        mask2d = mask_y[:, None] * mask_x[None, :]
        
        return mask2d
    
    def _apply_tps(self, inputs: Dict[str, torch.Tensor], training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Apply TPS transformation to create image pairs.
        
        Args:
            inputs: Dictionary containing image and mask
            training: Whether in training mode
            
        Returns:
            Dictionary with transformed image pair
        """
        if not self._tps or not training:
            # No TPS transformation
            inputs['future_image'] = inputs['image'].clone()
            return inputs
        
        # image = inputs['image']  # [C, H, W]
        # mask = inputs.get('mask', torch.ones(1, *image.shape[1:]))  # [1, H, W]
        
        # # Combine image and mask for joint transformation
        # image_with_mask = torch.cat([mask, image], dim=0)  # [C+1, H, W]
        # image_with_mask = image_with_mask.unsqueeze(0)  # [1, C+1, H, W]
        
        # # Apply target transformation using TensorFlow-compatible method
        # future_image_with_mask, _, _ = self._target_sampler(
        #     image_with_mask, training=training
        # )
        
        # # Apply source transformation using TensorFlow-compatible method
        # image_with_mask, _, _ = self._source_sampler(
        #     future_image_with_mask, training=training
        # )
        
        # # Separate mask and image
        # future_mask = future_image_with_mask[0, 0:1]  # [1, H, W]
        # future_image = future_image_with_mask[0, 1:]  # [C, H, W]
        
        # mask = image_with_mask[0, 0:1]  # [1, H, W]
        # image = image_with_mask[0, 1:]  # [C, H, W]
        
        # inputs['image'] = image
        # inputs['future_image'] = future_image
        # inputs['mask'] = future_mask
        original_image = inputs['image']  # [C, H, W] - this will be our target
        mask = inputs.get('mask', torch.ones(1, *original_image.shape[1:]))  # [1, H, W]
        
        # The target (future_image) is the original, unwarped image
        inputs['future_image'] = original_image.clone()
        
        # Combine image and mask for joint transformation
        image_with_mask = torch.cat([mask, original_image], dim=0)  # [C+1, H, W]
        image_with_mask = image_with_mask.unsqueeze(0)  # [1, C+1, H, W]
        
        # Apply single warp transformation to create the input image
        warped_image_with_mask, _, _ = self._target_sampler(
            image_with_mask, training=training
        )
        
        # Separate mask and image from warped result
        warped_mask = warped_image_with_mask[0, 0:1]  # [1, H, W]
        warped_image = warped_image_with_mask[0, 1:]  # [C, H, W]
        
        # Set the warped image as input and keep original mask for loss computation
        inputs['image'] = warped_image
        inputs['mask'] = mask  # Use original mask for loss computation
        return inputs
    
    def _proc_im_pair(self, inputs: Dict[str, Any], training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Process image pair with augmentations and transformations.
        
        Args:
            inputs: Dictionary containing raw inputs
            training: Whether in training mode
            
        Returns:
            Processed inputs as tensors
        """
        # Read image
        if isinstance(inputs['image'], str):
            image = self._read_image(inputs['image'])
        else:
            image = inputs['image']
        
        # Handle landmarks
        landmarks = inputs.get('landmarks', None)
        if landmarks is not None and not isinstance(landmarks, torch.Tensor):
            landmarks = torch.tensor(landmarks, dtype=torch.float32)
        
        # Get original size
        original_size = [image.shape[1], image.shape[2]]  # [H, W]
        
        # Resize image
        image = self._resize_image(image, self._image_size, keep_aspect=True)
        
        # Resize landmarks if present
        if landmarks is not None:
            landmarks = self._resize_points(landmarks, original_size, self._image_size)
        
        # Apply augmentations
        image, landmarks = self._apply_augmentations(image, landmarks, training)
        
        # Create smooth mask for TPS
        h, w = self._image_size
        margin = min(h, w) // 10
        step = min(h, w) // 20
        mask = self._get_smooth_mask(h, w, margin, step)
        mask = mask.unsqueeze(0)  # Add channel dimension
        
        # Prepare inputs dictionary
        processed_inputs = {
            'image': image,
            'mask': mask
        }
        
        if landmarks is not None and self._landmarks_output:
            processed_inputs['landmarks'] = landmarks
        
        # Copy other inputs
        for key, value in inputs.items():
            if key not in processed_inputs and key not in ['image', 'landmarks']:
                if isinstance(value, (int, float)):
                    processed_inputs[key] = torch.tensor(value)
                elif isinstance(value, (list, np.ndarray)):
                    processed_inputs[key] = torch.tensor(value)
                else:
                    processed_inputs[key] = value
        
        # Apply TPS transformation
        processed_inputs = self._apply_tps(processed_inputs, training)
        
        return processed_inputs
    
    def _get_image(self, idx: int) -> Dict[str, Any]:
        """Get image data for given index - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_image")
    
    def _get_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get processed sample for given index."""
        # Get raw data
        raw_inputs = self._get_image(idx)
        
        # Process the data
        training = (self._subset == 'train')
        processed_inputs = self._proc_im_pair(raw_inputs, training=training)
        
        return processed_inputs
    
    def __len__(self) -> int:
        """Return dataset length."""
        if hasattr(self, '_images'):
            length = len(self._images)
            if self._max_samples is not None:
                length = min(length, self._max_samples)
            return length
        else:
            return 0
