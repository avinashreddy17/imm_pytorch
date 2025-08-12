"""
VGG16 implementation for perceptual loss.

Converted from TensorFlow implementation.
Original code adapted from: Colorization as a Proxy Task for Visual Understanding, 
Larsson, Maire, Shakhnarovich, CVPR 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, List, Optional


class VGG16Features(nn.Module):
    """
    VGG16 feature extractor for perceptual loss computation.
    
    This implementation loads weights from the original colorization network
    and extracts features at multiple layers for perceptual loss computation.
    """
    
    def __init__(self, pretrained_file: str, feature_layers: List[str] = None):
        """
        Initialize VGG16 feature extractor.
        
        Args:
            pretrained_file: Path to the pretrained VGG16 model (.h5 file)
            feature_layers: List of layer names to extract features from
        """
        super(VGG16Features, self).__init__()
        
        if feature_layers is None:
            feature_layers = ['input', 'conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2']
        
        self.feature_layers = feature_layers
        
        # Load pretrained weights if available. Requires h5py; otherwise skip.
        self.data = {}
        if pretrained_file and os.path.exists(pretrained_file):
            try:
                import h5py  # local import to avoid hard dependency
                self.data = self._load_pretrained_data_h5(pretrained_file)
            except Exception as e:
                print(f"Warning: Failed to load VGG16 weights from {pretrained_file}: {e}")
                self.data = {}
        
        # Build the network
        self._build_network()
        
        # Set to evaluation mode and freeze weights
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def _build_network(self):
        """Build the VGG16 network architecture."""
        
        # Conv1 layers
        self.conv1_1 = self._make_conv_layer('conv1_1', 1, 64)
        self.conv1_2 = self._make_conv_layer('conv1_2', 64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2 layers  
        self.conv2_1 = self._make_conv_layer('conv2_1', 64, 128)
        self.conv2_2 = self._make_conv_layer('conv2_2', 128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv3 layers
        self.conv3_1 = self._make_conv_layer('conv3_1', 128, 256)
        self.conv3_2 = self._make_conv_layer('conv3_2', 256, 256)
        self.conv3_3 = self._make_conv_layer('conv3_3', 256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv4 layers
        self.conv4_1 = self._make_conv_layer('conv4_1', 256, 512)
        self.conv4_2 = self._make_conv_layer('conv4_2', 512, 512)
        self.conv4_3 = self._make_conv_layer('conv4_3', 512, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv5 layers
        self.conv5_1 = self._make_conv_layer('conv5_1', 512, 512)
        self.conv5_2 = self._make_conv_layer('conv5_2', 512, 512)
        self.conv5_3 = self._make_conv_layer('conv5_3', 512, 512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def _make_conv_layer(self, name: str, in_channels: int, out_channels: int) -> nn.Module:
        """
        Create a convolution layer with pretrained weights.
        
        Args:
            name: Layer name in the pretrained model
            in_channels: Number of input channels
            out_channels: Number of output channels
            
        Returns:
            Convolution layer with loaded weights
        """
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)

        # Load weights from pretrained model (shape-safe)
        if name in self.data:
            w_np = self.data[name].get('0', None)
            b_np = self.data[name].get('1', None)
            if w_np is not None:
                w = torch.from_numpy(w_np)
                # Many HDF5s store as [H, W, in, out]; convert to [out, in, H, W]
                if w.dim() == 4 and w.shape[0] == 3 and w.shape[1] == 3:
                    # Likely already [H, W, in, out]; fallback to transpose below
                    pass
                # Try common HWIO -> OIHW
                if w.shape != conv.weight.data.shape:
                    if w.dim() == 4 and w.shape[-1] == out_channels and w.shape[2] == in_channels:
                        w = w.permute(3, 2, 0, 1).contiguous()  # HWIO -> OIHW
                    elif w.dim() == 4 and w.shape[-1] == out_channels and w.shape[2] == 3 and in_channels == 1:
                        # HWIO with 3 input channels; average to grayscale then transpose
                        w = w.mean(dim=2, keepdim=True).permute(3, 2, 0, 1).contiguous()
                    elif w.dim() == 4 and w.shape[0] == out_channels and w.shape[1] == in_channels:
                        # Already [out, in, H, W]
                        pass
                # Special handling for conv1_1 BGR->RGB if 3 input channels
                if name == 'conv1_1' and w.dim() == 4 and w.shape[1] == 3 and in_channels == 3:
                    w = w[:, [2, 1, 0], :, :]
                # Final shape check
                if w.shape == conv.weight.data.shape:
                    conv.weight.data = w
            if b_np is not None:
                b = torch.from_numpy(b_np)
                if b.shape == conv.bias.data.shape:
                    conv.bias.data = b
        
        return conv

    def _load_pretrained_data_h5(self, file_path: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load pretrained parameters from an HDF5 file produced for the colorization model.

        Expected structure (typical):
        - Root or '/data' group contains groups per layer name, each with datasets
          named '0' (weights) and '1' (biases). Some variants may use 'weights'/'biases'.
        """
        data: Dict[str, Dict[str, np.ndarray]] = {}
        import h5py
        with h5py.File(file_path, 'r') as f:
            # If dataset stored under '/data', use that; else use root
            root = f['/data'] if '/data' in f else f['/']

            def load_layer_group(layer_group, layer_name: str):
                data[layer_name] = {}
                # Try common keys
                for k in ['0', '1', 'weights', 'biases', 'W', 'b']:
                    if k in layer_group:
                        arr = np.array(layer_group[k])
                        data[layer_name][k] = arr

            # Iterate immediate children as layers
            for layer_name in root.keys():
                obj = root[layer_name]
                if isinstance(obj, h5py.Group):
                    load_layer_group(obj, layer_name)
        return data
    
    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input images for VGG16.
        
        Args:
            x: Input images [B, C, H, W] in range [0, 255]
            
        Returns:
            Preprocessed images
        """
        # Convert RGB to grayscale
        if x.shape[1] == 3:
            x = torch.mean(x, dim=1, keepdim=True)
        
        # Normalize to [0, 1] and center
        x = x / 255.0
        x = x - 114.451 / 255.0
        
        return x
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VGG16.
        
        Args:
            x: Input images [B, 3, H, W] in range [0, 255]
            
        Returns:
            Dictionary of feature maps at specified layers
        """
        # Store original input
        input_raw = x
        
        # Preprocess input
        x = self._preprocess_input(x)
        
        features = {}
        
        # Store input if requested
        if 'input' in self.feature_layers:
            features['input'] = input_raw
        
        # Conv1 block
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        if 'conv1_2' in self.feature_layers:
            features['conv1_2'] = x
        x = self.pool1(x)
        
        # Conv2 block
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        if 'conv2_2' in self.feature_layers:
            features['conv2_2'] = x
        x = self.pool2(x)
        
        # Conv3 block
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        if 'conv3_2' in self.feature_layers:
            features['conv3_2'] = x
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)
        
        # Conv4 block
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        if 'conv4_2' in self.feature_layers:
            features['conv4_2'] = x
        x = F.relu(self.conv4_3(x))
        x = self.pool4(x)
        
        # Conv5 block
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        if 'conv5_2' in self.feature_layers:
            features['conv5_2'] = x
        x = F.relu(self.conv5_3(x))
        x = self.pool5(x)
        
        return features


def build_vgg16(input_tensor: torch.Tensor, pretrained_file: str, 
               feature_layers: List[str] = None) -> Dict[str, torch.Tensor]:
    """
    Build VGG16 and extract features (functional interface).
    
    Args:
        input_tensor: Input images [B, 3, H, W] in range [0, 255]
        pretrained_file: Path to pretrained VGG16 model
        feature_layers: List of layer names to extract features from
        
    Returns:
        Dictionary of feature maps
    """
    vgg = VGG16Features(pretrained_file, feature_layers)
    return vgg(input_tensor)
