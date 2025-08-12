"""
Dynamic dataset import utility.

Converted from TensorFlow implementation.
"""

from typing import Any


def import_dataset(dataset_name: str) -> Any:
    """
    Dynamically import dataset class by name.
    
    Args:
        dataset_name: Name of the dataset ('celeba', 'aflw', etc.)
        
    Returns:
        Dataset class
    """
    if dataset_name.lower() == 'celeba':
        from ..datasets.celeba_dataset import CelebADataset
        return CelebADataset
    elif dataset_name.lower() == 'aflw':
        from ..datasets.aflw_dataset import AFLWDataset
        return AFLWDataset
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
