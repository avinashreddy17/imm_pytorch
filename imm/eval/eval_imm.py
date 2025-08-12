"""
Evaluation utilities for IMM model.

Converted from TensorFlow implementation.
Original Author: Tomas Jakab
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Any

from ..utils.box import Box
import metayaml


def load_configs(file_names: List[str]) -> Box:
    """Load configuration from YAML files."""
    config = Box(metayaml.read(file_names))
    return config


def evaluate_model(dataset, net_class, model_config: Box, net_file: str, 
                  training_config: Box, batch_size: int = 100,
                  random_seed: int = 0, eval_tensors: List[str] = None) -> Dict[str, List]:
    """
    Evaluate model on a dataset.
    
    Args:
        dataset: Dataset to evaluate on
        net_class: Model class
        model_config: Model configuration
        net_file: Path to model checkpoint
        training_config: Training configuration
        batch_size: Batch size for evaluation
        random_seed: Random seed
        eval_tensors: List of tensor names to extract
        
    Returns:
        Dictionary of extracted tensors
    """
    if eval_tensors is None:
        eval_tensors = ['gauss_yx', 'future_landmarks']
    
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = net_class(model_config)
    model = model.to(device)
    model.eval()
    
    # Load checkpoint
    checkpoint = torch.load(net_file, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Collect results
    results = {tensor_name: [] for tensor_name in eval_tensors}
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch, training=False)
            
            # Extract requested tensors
            for tensor_name in eval_tensors:
                if tensor_name in outputs:
                    results[tensor_name].append(outputs[tensor_name].cpu().numpy())
                elif tensor_name in batch:
                    results[tensor_name].append(batch[tensor_name].cpu().numpy())
                else:
                    # Handle special cases
                    if tensor_name == 'future_landmarks' and 'landmarks' in batch:
                        results[tensor_name].append(batch['landmarks'].cpu().numpy())
    
    return results
