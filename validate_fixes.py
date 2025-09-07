#!/usr/bin/env python
"""
Validation script to test the fixes for PyTorch IMM implementation.

This script runs a quick training validation to ensure the fixes are working.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from imm.models.imm_model import IMMModel
from imm.utils.box import Box
from imm.utils.tps_sampler_tf_compat import TPSRandomSamplerTFCompat
import metayaml


def test_model_creation():
    """Test that the model can be created with fixed config."""
    print("🔧 Testing model creation...")
    
    # Load the fixed config
    config = Box(metayaml.read([
        'configs/paths/default.yaml',
        'configs/experiments/celeba-10pts-mini.yaml'
    ]))
    
    try:
        model = IMMModel(config.model)
        print("✅ Model created successfully!")
        return model, config
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None, None


def test_forward_pass(model, config):
    """Test forward pass with dummy data."""
    print("🔧 Testing forward pass...")
    
    try:
        # Create dummy input
        batch_size = 2
        image_size = 128
        
        dummy_input = {
            'image': torch.randn(batch_size, 3, image_size, image_size),
            'future_image': torch.randn(batch_size, 3, image_size, image_size),
            'mask': torch.ones(batch_size, 1, image_size, image_size)
        }
        
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input, training=False)
        
        # Check output shapes
        expected_keys = ['future_im_pred', 'gauss_yx', 'pose_embeddings', 'image_embeddings']
        for key in expected_keys:
            if key not in outputs:
                print(f"❌ Missing output key: {key}")
                return False
        
        # Check tensor shapes
        future_im_pred = outputs['future_im_pred']
        gauss_yx = outputs['gauss_yx']
        
        if future_im_pred.shape != (batch_size, 3, image_size, image_size):
            print(f"❌ Wrong future_im_pred shape: {future_im_pred.shape}")
            return False
            
        if gauss_yx.shape != (batch_size, config.model.n_maps, 2):
            print(f"❌ Wrong gauss_yx shape: {gauss_yx.shape}")
            return False
            
        print("✅ Forward pass successful!")
        print(f"   - future_im_pred shape: {future_im_pred.shape}")
        print(f"   - gauss_yx shape: {gauss_yx.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False


def test_loss_computation(model, config):
    """Test loss computation."""
    print("🔧 Testing loss computation...")
    
    try:
        # Create dummy input
        batch_size = 2
        image_size = 128
        
        inputs = {
            'image': torch.randn(batch_size, 3, image_size, image_size),
            'future_image': torch.randn(batch_size, 3, image_size, image_size),
            'mask': torch.ones(batch_size, 1, image_size, image_size)
        }
        
        model.train()
        outputs = model(inputs, training=True)
        loss = model.compute_loss(outputs, inputs, training=True)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ Loss is NaN or Inf: {loss}")
            return False
            
        print(f"✅ Loss computation successful! Loss: {loss.item():.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        return False


def test_tps_sampler():
    """Test TensorFlow-compatible TPS sampler."""
    print("🔧 Testing TF-compatible TPS sampler...")
    
    try:
        sampler = TPSRandomSamplerTFCompat(
            height=128, width=128, 
            vertical_points=10, horizontal_points=10,
            rotsd=5.0, scalesd=0.1, transsd=0.1,
            warpsd=(0.001, 0.005), pad=False
        )
        
        # Test with dummy image
        dummy_image = torch.randn(1, 4, 128, 128)  # 3 channels + mask
        
        transformed, _, _ = sampler(dummy_image, training=True)
        
        if transformed.shape != dummy_image.shape:
            print(f"❌ Wrong output shape: {transformed.shape}")
            return False
            
        print("✅ TPS sampler working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ TPS sampler failed: {e}")
        return False


def test_channels_bug_fix(config):
    """Test that channels_bug_fix is correctly set."""
    print("🔧 Testing channels_bug_fix configuration...")
    
    if not hasattr(config.model, 'channels_bug_fix'):
        print("❌ channels_bug_fix not found in config")
        return False
        
    if not config.model.channels_bug_fix:
        print("❌ channels_bug_fix should be True")
        return False
        
    print("✅ channels_bug_fix correctly set to True!")
    return True


def main():
    """Run all validation tests."""
    print("🚀 Starting validation of PyTorch IMM fixes...\n")
    
    # Test 1: Model creation
    model, config = test_model_creation()
    if model is None:
        print("💥 CRITICAL: Model creation failed. Cannot continue.")
        return False
    
    # Test 2: Config validation
    config_ok = test_channels_bug_fix(config)
    
    # Test 3: Forward pass
    forward_ok = test_forward_pass(model, config)
    
    # Test 4: Loss computation
    loss_ok = test_loss_computation(model, config)
    
    # Test 5: TPS sampler
    tps_ok = test_tps_sampler()
    
    # Summary
    all_tests = [config_ok, forward_ok, loss_ok, tps_ok]
    passed = sum(all_tests)
    total = len(all_tests)
    
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"   Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The fixes should resolve your training issues.")
        print("\n🏃‍♂️ Ready to retrain with:")
        print("   python scripts/train.py --configs configs/paths/default.yaml configs/experiments/celeba-10pts-mini.yaml --ngpus 1 --num-epochs 50")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
