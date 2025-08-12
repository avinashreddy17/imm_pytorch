"""
AFLW Dataset implementation in PyTorch.

Converted from TensorFlow implementation.
Original Author: Tomas Jakab
"""

import os
import os.path as osp
import numpy as np
import torch
from scipy.io import loadmat
from typing import Dict, List, Tuple, Optional, Any

from .tps_dataset import TPSDataset


def load_dataset(data_dir: str, subset: str) -> Tuple[str, List[str], np.ndarray, np.ndarray]:
    """
    Load AFLW dataset.
    
    Args:
        data_dir: Path to AFLW dataset directory
        subset: Dataset subset ('train', 'val', 'test')
        
    Returns:
        Tuple of (image_dir, image_files, keypoints, sizes)
    """
    load_subset = 'train' if subset in ['train', 'val'] else 'test'
    
    # Load image files
    images_file = os.path.join(data_dir, f'aflw_{load_subset}_images.txt')
    with open(images_file, 'r') as f:
        images = f.read().splitlines()
    
    # Load keypoints and sizes
    keypoints_file = os.path.join(data_dir, f'aflw_{load_subset}_keypoints.mat')
    mat = loadmat(keypoints_file)
    keypoints = mat['gt'][:, :, [1, 0]]  # Convert from (x,y) to (y,x)
    sizes = mat['hw']
    
    if subset in ['train', 'val']:
        # Put last 10% of training aside for validation
        n_validation = int(round(0.1 * len(images)))
        if subset == 'train':
            images = images[:-n_validation]
            keypoints = keypoints[:-n_validation]
            sizes = sizes[:-n_validation]
        elif subset == 'val':
            images = images[-n_validation:]
            keypoints = keypoints[-n_validation:]
            sizes = sizes[-n_validation:]
        else:
            raise ValueError()
    
    image_dir = os.path.join(data_dir, 'output')
    return image_dir, images, keypoints, sizes


class AFLWDataset(TPSDataset):
    """AFLW dataset for facial landmark learning."""
    
    LANDMARK_LABELS = {'left_eye': 0, 'right_eye': 1}
    N_LANDMARKS = 5
    
    def __init__(self, data_dir: str, subset: str, max_samples: Optional[int] = None,
                 image_size: List[int] = [128, 128], order_stream: bool = False,
                 landmarks: bool = False, tps: bool = True, vertical_points: int = 10,
                 horizontal_points: int = 10, rotsd: List[float] = [0.0, 5.0],
                 scalesd: List[float] = [0.0, 0.1], transsd: List[float] = [0.1, 0.1],
                 warpsd: List[float] = [0.001, 0.005, 0.001, 0.01],
                 name: str = 'AFLWDataset'):
        """
        Initialize AFLW dataset.
        
        Args:
            data_dir: Path to dataset directory
            subset: Dataset subset ('train', 'val', 'test')
            max_samples: Maximum number of samples
            image_size: Target image size [height, width]
            order_stream: Whether to iterate in order
            landmarks: Whether to output landmarks
            tps: Whether to apply TPS transformation
            vertical_points: Number of vertical control points for TPS
            horizontal_points: Number of horizontal control points for TPS
            rotsd: Rotation standard deviations for TPS
            scalesd: Scale standard deviations for TPS
            transsd: Translation standard deviations for TPS
            warpsd: Warp standard deviations for TPS
            name: Dataset name
        """
        
        super(AFLWDataset, self).__init__(
            data_dir, subset, max_samples=max_samples, image_size=image_size,
            order_stream=order_stream, landmarks=landmarks, tps=tps,
            vertical_points=vertical_points, horizontal_points=horizontal_points,
            rotsd=rotsd, scalesd=scalesd, transsd=transsd, warpsd=warpsd, name=name
        )
    
    def _initialize_data(self):
        """Load AFLW dataset files and landmarks."""
        self._image_dir, self._images, self._keypoints, self._sizes = load_dataset(
            self._data_dir, self._subset
        )
    
    def _get_sample_dtype(self) -> Dict[str, type]:
        """Get sample data types for dataset creation."""
        d = {
            'image': str,
            'landmarks': np.float32,
            'size': np.int32
        }
        d.update({k: int for k in self.LANDMARK_LABELS.keys()})
        return d
    
    def _get_sample_shape(self) -> Dict[str, Optional[List[int]]]:
        """Get sample shapes for dataset creation."""
        d = {
            'image': None,  # Variable size string
            'landmarks': [self.N_LANDMARKS, 2],
            'size': [2]
        }
        d.update({k: [] for k in self.LANDMARK_LABELS.keys()})  # Scalar landmarks
        return d
    
    def _get_image(self, idx: int) -> Dict[str, Any]:
        """
        Get image data for the given index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing image path, landmarks, and size
        """
        image_path = os.path.join(self._image_dir, self._images[idx])
        landmarks = self._keypoints[idx]  # Already in (y,x) format
        size = self._sizes[idx]
        
        inputs = {
            'image': image_path,
            'landmarks': landmarks,
            'size': size
        }
        
        # Add individual landmark labels  
        inputs.update({k: v for k, v in self.LANDMARK_LABELS.items()})
        
        return inputs
    
    def _proc_im_pair(self, inputs: Dict[str, Any], training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Process image pair with AFLW-specific handling.
        
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
        
        # Handle landmarks and size
        landmarks = inputs.get('landmarks', None)
        if landmarks is not None and not isinstance(landmarks, torch.Tensor):
            landmarks = torch.tensor(landmarks, dtype=torch.float32)
        
        original_size = inputs.get('size', None)
        if original_size is not None and not isinstance(original_size, torch.Tensor):
            original_size = torch.tensor(original_size, dtype=torch.float32)
        
        # Resize landmarks based on original size
        if landmarks is not None and original_size is not None:
            final_size = self._image_size[0]  # Assume square images
            landmarks = self._resize_points(
                landmarks, original_size.tolist(), [final_size, final_size]
            )
        
        # Get image size for fallback
        if original_size is None:
            original_size = [image.shape[1], image.shape[2]]  # [H, W]
        else:
            original_size = original_size.tolist()
        
        # Resize image
        image = self._resize_image(image, self._image_size, keep_aspect=True)
        
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
            if key not in processed_inputs and key not in ['image', 'landmarks', 'size']:
                if isinstance(value, (int, float)):
                    processed_inputs[key] = torch.tensor(value)
                elif isinstance(value, (list, np.ndarray)):
                    processed_inputs[key] = torch.tensor(value)
                else:
                    processed_inputs[key] = value
        
        # Apply TPS transformation
        processed_inputs = self._apply_tps(processed_inputs, training)
        
        return processed_inputs
    
    def __len__(self) -> int:
        """Return dataset length."""
        length = len(self._images)
        if self._max_samples is not None:
            length = min(length, self._max_samples)
        return length
