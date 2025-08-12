"""
Abstract base model class for PyTorch implementation.

Converted from TensorFlow implementation.
Original Author: Ankush Gupta
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseModel(nn.Module, ABC):
    """Base class for all models in the IMM framework."""
    
    def __init__(self, dtype: torch.dtype = torch.float32, name: str = 'BaseModel'):
        super(BaseModel, self).__init__()
        self.dtype = dtype
        self._name = name
        
        # For tracking moving averages (equivalent to TF's moving averages)
        self._moving_averages = {}
        
        # Options for convolution layers
        self._conv_opts = {
            'weight_decay': 1e-5,
            'std': 0.01,
        }
    
    def _get_conv_opts(self) -> Dict[str, Any]:
        """Get convolution layer options."""
        return self._conv_opts
    
    def _exp_running_avg(self, x: torch.Tensor, training: bool, 
                        init_val: float = 0.0, rho: float = 0.99, 
                        name: str = 'x') -> torch.Tensor:
        """
        Exponential running average similar to TensorFlow's implementation.
        
        Args:
            x: Input tensor
            training: Whether in training mode
            init_val: Initial value for the average
            rho: Decay rate
            name: Name for the average variable
            
        Returns:
            Updated average tensor
        """
        if name not in self._moving_averages:
            self._moving_averages[name] = torch.full_like(x, init_val)
        
        avg = self._moving_averages[name]
        
        if training:
            w_update = 1.0 - rho
            new_avg = avg + w_update * (x - avg)
            self._moving_averages[name] = new_avg.detach()
            return new_avg
        else:
            return avg
    
    def conv_block(self, in_channels: int, out_channels: int, 
                   kernel_size: int = 3, stride: int = 1, 
                   padding: Optional[int] = None, bias: bool = True,
                   batch_norm: bool = True, activation: Optional[nn.Module] = None,
                   name: str = 'conv_block') -> nn.Sequential:
        """
        Create a convolution block with optional batch normalization and activation.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Padding size (auto-calculated if None)
            bias: Whether to use bias
            batch_norm: Whether to apply batch normalization
            activation: Activation function (ReLU if None and batch_norm=True)
            name: Block name
            
        Returns:
            Sequential model containing the convolution block
        """
        if padding is None:
            padding = kernel_size // 2
            
        if activation is None and batch_norm:
            activation = nn.ReLU(inplace=True)
        
        layers = []
        
        # Convolution layer
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                        stride=stride, padding=padding, bias=bias)
        
        # Initialize weights similar to TensorFlow
        nn.init.trunc_normal_(conv.weight, std=self._conv_opts['std'])
        if bias:
            nn.init.constant_(conv.bias, 0.0)
            
        layers.append(conv)
        
        # Batch normalization
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # Activation
        if activation is not None:
            layers.append(activation)
        
        return nn.Sequential(*layers)
    
    def uncertainty_weighted_mtl(self, losses: list, name: str = 'uw_mtloss') -> torch.Tensor:
        """
        Implements uncertainty-weighted multi-task loss [Kendall et al., 2017].
        Loss-total = Sum_i 1/s_i^2 * loss_i + log(s_i)
        
        Args:
            losses: List of loss tensors
            name: Name for the loss parameters
            
        Returns:
            Combined uncertainty-weighted loss
        """
        uw_losses = []
        
        # Create learnable log-variance parameters
        if not hasattr(self, f'_{name}_log_vars'):
            log_vars = nn.Parameter(torch.zeros(len(losses)))
            setattr(self, f'_{name}_log_vars', log_vars)
        
        log_vars = getattr(self, f'_{name}_log_vars')
        
        for i, loss in enumerate(losses):
            log_var = log_vars[i]
            precision = torch.exp(-log_var)
            weighted_loss = precision * loss + log_var
            uw_losses.append(weighted_loss)
        
        return torch.stack(uw_losses).sum()
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass - must be implemented by subclasses."""
        pass
