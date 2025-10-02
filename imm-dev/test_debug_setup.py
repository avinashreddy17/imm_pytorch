#!/usr/bin/env python
"""
Test script to verify the debug training setup works correctly.
This script will test the basic functionality without running full training.
"""

import tensorflow as tf
import numpy as np
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    try:
        from imm.models.imm_model import IMMModel
        from imm.utils.box import Box
        import imm.train.cnn_train_multi as tru
        from imm.utils.colorize import colorize
        import metayaml
        from imm.utils.dataset_import import import_dataset
        print("All imports successful")
        return True
    except ImportError as e:
        print('Import error: {}'.format(e))
        return False

def test_tensorflow_setup():
    """Test TensorFlow setup and GPU availability."""
    print("\nTesting TensorFlow setup...")
    print("TensorFlow version: {}".format(tf.__version__))
    print("GPU available: {}".format(tf.test.is_gpu_available()))
    
    # Test basic TensorFlow operations
    with tf.Session() as sess:
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        result = sess.run(c)
        print("Basic TF operation test: {}".format(result))
        print(" TensorFlow setup working")
        return True

def test_model_creation():
    """Test if the IMM model can be created."""
    print("\nTesting model creation...")
    try:
        from imm.utils.box import Box
        from imm.models.imm_model import IMMModel
        # Create a simple config
        config = Box({
            'gauss_std': 0.10,
            'gauss_mode': 'rot',
            'n_maps': 10,
            'n_filters': 16,
            'block_sizes': [1, 1, 1],
            'n_filters_render': 16,
            'renderer_stride': 2,
            'min_res': 16,
            'same_n_filt': False,
            'reconstruction_loss': 'l2',
            'loss_mask': False,
            'confidence': False,
            'channels_bug_fix': False
        })
        
        # Create model
        model = IMMModel(config)
        print(" IMM model creation successful")
        return True
    except Exception as e:
        print(" Model creation error: {}".format(e))
        return False

def test_debug_functions():
    """Test the debug functions."""
    print("\nTesting debug functions...")
    try:
        # Import debug functions
        print("  Importing debug functions...")
        from debug_train import debug_tensor_info, debug_weights, debug_gradients
        print("  Debug functions imported successfully")
        
        # Test with a simple tensor
        print("  Testing debug_tensor_info...")
        with tf.Session() as sess:
            test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            debug_tensor_info(test_tensor, "test_tensor", sess)
            print("  debug_tensor_info completed successfully")
        
        print("  Debug functions working")
        print("  Returning True from debug functions test")
        return True
    except Exception as e:
        print("  Debug functions error: {}".format(e))
        import traceback
        print("  Traceback: {}".format(traceback.format_exc()))
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("IMM Debug Training Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_tensorflow_setup,
        test_model_creation,
        test_debug_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for i, test in enumerate(tests):
        print("Running test {}: {}".format(i+1, test.__name__))
        result = test()
        print("Test {} result: {}".format(i+1, result))
        if result:
            passed += 1
        print()
    
    print("=" * 50)
    print("Test Results: {}/{} tests passed".format(passed, total))
    print("=" * 50)
    
    if passed == total:
        print(" All tests passed! Debug training setup is ready.")
        return True
    else:
        print(" Some tests failed. Please check the errors above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
