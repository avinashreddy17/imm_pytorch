"""
CelebA Dataset implementation in PyTorch.

Converted from TensorFlow implementation.
Original Author: Tomas Jakab
"""

import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any

from .tps_dataset import TPSDataset


def load_dataset(data_root: str, dataset: str, subset: str) -> Tuple[str, List[str], np.ndarray]:
    """
    Load CelebA or MAFL dataset.
    
    Args:
        data_root: Root directory of the dataset
        dataset: Dataset name ('celeba' or 'mafl')
        subset: Subset name ('train', 'val', 'test')
        
    Returns:
        Tuple of (image_dir, image_files, keypoints)
    """
    image_dir = os.path.join(data_root, 'Img', 'img_align_celeba_hq')
    
    # Load landmarks
    landmarks_file = os.path.join(data_root, 'Anno', 'list_landmarks_align_celeba.txt')
    with open(landmarks_file, 'r') as f:
        lines = f.read().splitlines()
    
    # Skip header
    lines = lines[2:]
    image_files = []
    keypoints = []
    
    for line in lines:
        parts = line.split()
        image_files.append(parts[0])
        keypoints.append([int(x) for x in parts[1:]])
    
    keypoints = np.array(keypoints, dtype=np.float32)
    assert image_files[0] == '000001.jpg'
    
    # Load MAFL training set
    mafl_train_file = os.path.join(data_root, 'MAFL', 'training.txt')
    with open(mafl_train_file, 'r') as f:
        mafl_train = set(f.read().splitlines())
    
    mafl_train_overlap = []
    for i, image_file in enumerate(image_files):
        if image_file in mafl_train:
            mafl_train_overlap.append(i)
    
    # Initialize image set labels
    images_set = np.zeros(len(image_files), dtype=np.int32)
    
    if dataset == 'celeba':
        # Load CelebA partition
        partition_file = os.path.join(data_root, 'Eval', 'list_eval_partition.txt')
        with open(partition_file, 'r') as f:
            celeba_set = [int(line.split()[1]) for line in f.readlines()]
        images_set[:] = celeba_set
        images_set += 1  # Convert to 1-based indexing
    elif dataset == 'mafl':
        images_set[mafl_train_overlap] = 1
    else:
        raise ValueError(f'Dataset = {dataset} not recognized.')
    
    # Set test set
    mafl_test_file = os.path.join(data_root, 'MAFL', 'testing.txt')
    with open(mafl_test_file, 'r') as f:
        mafl_test = set(f.read().splitlines())
    
    mafl_test_overlap = []
    for i, image_file in enumerate(image_files):
        if image_file in mafl_test:
            mafl_test_overlap.append(i)
    
    images_set[mafl_test_overlap] = 4
    
    # Put last 10% of MAFL training aside for validation
    n_validation = int(round(0.1 * len(mafl_train_overlap)))
    mafl_validation = mafl_train_overlap[-n_validation:]
    images_set[mafl_validation] = 5
    
    # Determine subset label
    if dataset == 'celeba':
        if subset == 'train':
            label = 1
        elif subset == 'val':
            label = 2
        else:
            raise ValueError(f'subset = {subset} for celeba dataset not recognized.')
    elif dataset == 'mafl':
        if subset == 'train':
            label = 1
        elif subset == 'test':
            label = 4
        elif subset == 'train10':
            label = 5
        else:
            raise ValueError(f'subset = {subset} for mafl dataset not recognized.')
    
    # Filter data for subset
    image_files = np.array(image_files)
    images = image_files[images_set == label]
    keypoints = keypoints[images_set == label]
    
    # Convert keypoints to correct format
    # [[lefteye_x, lefteye_y], [righteye_x, righteye_y], [nose_x, nose_y],
    #  [leftmouth_x, leftmouth_y], [rightmouth_x, rightmouth_y]]
    keypoints = np.reshape(keypoints, [-1, 5, 2])
    
    return image_dir, images.tolist(), keypoints


class CelebADataset(TPSDataset):
    """CelebA dataset for facial landmark learning."""
    
    LANDMARK_LABELS = {'left_eye': 0, 'right_eye': 1}
    N_LANDMARKS = 5
    
    def __init__(self, data_dir: str, subset: str, dataset: Optional[str] = None,
                 max_samples: Optional[int] = None, image_size: List[int] = [128, 128],
                 order_stream: bool = False, landmarks: bool = False, tps: bool = True,
                 vertical_points: int = 10, horizontal_points: int = 10,
                 rotsd: List[float] = [0.0, 5.0], scalesd: List[float] = [0.0, 0.1],
                 transsd: List[float] = [0.1, 0.1], warpsd: List[float] = [0.001, 0.005, 0.001, 0.01],
                 name: str = 'CelebADataset'):
        """
        Initialize CelebA dataset.
        
        Args:
            data_dir: Path to dataset directory
            subset: Dataset subset ('train', 'val', 'test')
            dataset: Specific dataset ('celeba' or 'mafl')
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
        
        # Set default dataset
        if dataset is None:
            dataset = 'celeba'
        
        self._dataset = dataset
        
        super(CelebADataset, self).__init__(
            data_dir, subset, max_samples=max_samples, image_size=image_size,
            order_stream=order_stream, landmarks=landmarks, tps=tps,
            vertical_points=vertical_points, horizontal_points=horizontal_points,
            rotsd=rotsd, scalesd=scalesd, transsd=transsd, warpsd=warpsd, name=name
        )
    
    def _initialize_data(self):
        """Load CelebA dataset files and landmarks."""
        self._image_dir, self._images, self._keypoints = load_dataset(
            self._data_dir, self._dataset, self._subset
        )
    
    def _get_sample_dtype(self) -> Dict[str, type]:
        """Get sample data types for dataset creation."""
        d = {
            'image': str,
            'landmarks': np.float32,
        }
        d.update({k: int for k in self.LANDMARK_LABELS.keys()})
        return d
    
    def _get_sample_shape(self) -> Dict[str, Optional[List[int]]]:
        """Get sample shapes for dataset creation."""
        d = {
            'image': None,  # Variable size string
            'landmarks': [self.N_LANDMARKS, 2],
        }
        d.update({k: [] for k in self.LANDMARK_LABELS.keys()})  # Scalar landmarks
        return d
    
    def _get_image(self, idx: int) -> Dict[str, Any]:
        """
        Get image data for the given index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing image path and landmarks
        """
        image_path = os.path.join(self._image_dir, self._images[idx])
        landmarks = self._keypoints[idx][:, [1, 0]]  # Convert from (x,y) to (y,x)
        
        inputs = {
            'image': image_path,
            'landmarks': landmarks
        }
        
        # Add individual landmark labels
        inputs.update({k: v for k, v in self.LANDMARK_LABELS.items()})
        
        return inputs
    
    def __len__(self) -> int:
        """Return dataset length."""
        length = len(self._images)
        if self._max_samples is not None:
            length = min(length, self._max_samples)
        return length
