#!/usr/bin/env python
"""
Testing script for PyTorch IMM implementation.

Converted from TensorFlow implementation.
Original Author: Tomas Jakab
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import os.path as osp
import argparse
import sklearn.linear_model
from typing import Dict, Any

# Add project root to path
import sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from imm.models.imm_model import IMMModel
from imm.eval.eval_imm import evaluate_model
from imm.utils.box import Box
from imm.utils.dataset_import import import_dataset
import metayaml


def load_configs(file_names: list) -> Box:
    """Load configuration from YAML files."""
    config = Box(metayaml.read(file_names))
    return config


def evaluate(net_class, net_file: str, model_config: Box, training_config: Box,
            train_dset, test_dset, batch_size: int = 100, bias: bool = False) -> float:
    """
    Evaluate model on landmark regression task.
    
    Args:
        net_class: Model class
        net_file: Model checkpoint file
        model_config: Model configuration
        training_config: Training configuration
        train_dset: Training dataset
        test_dset: Test dataset
        batch_size: Batch size for evaluation
        bias: Whether to use bias in regressor
        
    Returns:
        Mean normalized error
    """
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
    
    def evaluate_dataset(dataset):
        """Evaluate model on a dataset."""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        gauss_yx_list = []
        future_landmarks_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(batch, training=False)
                
                # Collect outputs
                gauss_yx_list.append(outputs['gauss_yx'].cpu())
                if 'landmarks' in batch:
                    future_landmarks_list.append(batch['landmarks'].cpu())
        
        # Concatenate results
        gauss_yx = torch.cat(gauss_yx_list, dim=0).numpy()
        future_landmarks = torch.cat(future_landmarks_list, dim=0).numpy() if future_landmarks_list else None
        
        return {'gauss_yx': gauss_yx, 'future_landmarks': future_landmarks}
    
    # Evaluate on both datasets
    train_tensors = evaluate_dataset(train_dset)
    test_tensors = evaluate_dataset(test_dset)
    
    # Convert landmarks
    def convert_landmarks(tensors, im_size):
        landmarks = tensors['gauss_yx']
        landmarks_gt = tensors['future_landmarks'].astype(np.float32)
        im_size = np.array(im_size)
        landmarks = ((landmarks + 1) / 2.0) * im_size
        n_samples = landmarks.shape[0]
        landmarks = landmarks.reshape((n_samples, -1))
        landmarks_gt = landmarks_gt.reshape((n_samples, -1))
        return landmarks, landmarks_gt
    
    X_train, y_train = convert_landmarks(train_tensors, train_dset.image_size)
    X_test, y_test = convert_landmarks(test_tensors, train_dset.image_size)
    
    # Train regressor
    regr = sklearn.linear_model.Ridge(alpha=0.0, fit_intercept=bias)
    regr.fit(X_train, y_train)
    y_predict = regr.predict(X_test)
    
    landmarks_gt = test_tensors['future_landmarks'].astype(np.float32)
    landmarks_regressed = y_predict.reshape(landmarks_gt.shape)
    
    # Compute normalized error with respect to inter-ocular distance
    eyes = landmarks_gt[:, :2, :]
    ocular_distances = np.sqrt(np.sum((eyes[:, 0, :] - eyes[:, 1, :]) ** 2, axis=-1))
    distances = np.sqrt(np.sum((landmarks_gt - landmarks_regressed) ** 2, axis=-1))
    mean_error = np.mean(distances / ocular_distances[:, None])
    
    return mean_error


def main(args):
    """Main testing function."""
    experiment_name = args.experiment_name
    iteration = args.iteration
    im_size = args.im_size
    bias = args.bias
    batch_size = args.batch_size
    
    postfix = ''
    if bias:
        postfix += '-bias'
    else:
        postfix += '-no_bias'
    postfix += '-' + args.test_dataset
    postfix += '-' + args.test_split
    
    # Load config
    config = load_configs([
        args.paths_config,
        osp.join('configs', 'experiments', experiment_name + '.yaml')
    ])
    
    # Setup training dataset
    if args.train_dataset == 'mafl':
        train_dataset_class = import_dataset('celeba')
        train_dset = train_dataset_class(
            config.training.datadir, dataset='mafl', subset='train',
            order_stream=True, tps=False, image_size=[im_size, im_size]
        )
    elif args.train_dataset == 'aflw':
        train_dataset_class = import_dataset('aflw')
        train_dset = train_dataset_class(
            config.training.datadir, subset='train',
            order_stream=True, tps=False, image_size=[im_size, im_size]
        )
    else:
        raise ValueError(f'Dataset {args.train_dataset} not supported.')
    
    # Setup test dataset
    if args.test_dataset == 'mafl':
        test_dataset_class = import_dataset('celeba')
        test_dset = test_dataset_class(
            config.training.datadir, dataset='mafl', subset=args.test_split,
            order_stream=True, tps=False, image_size=[im_size, im_size]
        )
    elif args.test_dataset == 'aflw':
        test_dataset_class = import_dataset('aflw')
        test_dset = test_dataset_class(
            config.training.datadir, subset=args.test_split,
            order_stream=True, tps=False, image_size=[im_size, im_size]
        )
    else:
        raise ValueError(f'Dataset {args.test_dataset} not supported.')
    
    # Get model file
    if iteration is not None:
        net_file = f'model_epoch_{iteration}.pth'
    else:
        net_file = 'model_epoch_40.pth'
    
    checkpoint_file = osp.join(config.training.logdir, net_file)
    if not osp.isfile(checkpoint_file):
        raise ValueError(f'Checkpoint file {checkpoint_file} not found.')
    
    # Evaluate
    mean_error = evaluate(
        IMMModel, checkpoint_file, config.model, config.training,
        train_dset, test_dset, batch_size=batch_size, bias=bias
    )
    
    # Determine model dataset
    if hasattr(config.training, 'train_dset_params') and hasattr(config.training.train_dset_params, 'dataset'):
        model_dataset = config.training.train_dset_params.dataset
    else:
        model_dataset = config.training.dset
    
    # Print results
    print('')
    print('========================= RESULTS =========================')
    print(f'model trained in unsupervised way on {model_dataset} dataset')
    print(f'regressor trained on {args.train_dataset} training set')
    print(f'error on {args.test_dataset} dataset {args.test_split} set: {mean_error:.5f} ({mean_error * 100.0:.3f} percent)')
    print('===========================================================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model on face datasets.')
    
    parser.add_argument('--experiment-name', type=str, required=True,
                       help='Name of the experiment to evaluate.')
    parser.add_argument('--train-dataset', type=str, required=True,
                       help='Training dataset for regressor (mafl|aflw).')
    parser.add_argument('--test-dataset', type=str, required=True,
                       help='Testing dataset for regressed landmarks (mafl|aflw).')
    
    parser.add_argument('--paths-config', type=str, default='configs/paths/default.yaml',
                       required=False, help='Path to the paths config.')
    parser.add_argument('--iteration', type=int, default=None, required=False,
                       help='Checkpoint iteration to evaluate.')
    parser.add_argument('--test-split', type=str, default='test', required=False,
                       help='Test split (val|test).')
    parser.add_argument('--im-size', type=int, default=128, required=False,
                       help='Image size.')
    parser.add_argument('--bias', action='store_true', required=False,
                       help='Use bias in the regressor.')
    parser.add_argument('--batch-size', type=int, default=100, required=False,
                       help='Batch size for evaluation.')
    
    args = parser.parse_args()
    main(args)
