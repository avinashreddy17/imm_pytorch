#!/usr/bin/env python
# ==========================================================
# Enhanced Debug Training Script for IMM TensorFlow Model with Perceptual Loss
# ==========================================================
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import os
import os.path as osp
import time
from datetime import datetime

# network definition:
from imm.models.imm_model import IMMModel
from imm.utils.box import Box
import imm.train.cnn_train_multi as tru
from imm.utils.colorize import colorize
import metayaml
from imm.utils.dataset_import import import_dataset

class DebugModelFactory():
    """
    Factory which can be used to instantiate models with debug capabilities.
    """
    def __init__(self, network, **kwargs):
        self.network = network
        self.net_args = kwargs

    def create(self):
        return self.network(**self.net_args)

def load_configs(file_names):
    """
    Loads the yaml config files.
    """
    config = Box(metayaml.read(file_names))
    return config

def debug_tensor_info(tensor, name, session=None, step=None):
    """
    Print detailed information about a tensor.
    """
    print("\n=== DEBUG: {} ===".format(name))
    print("Shape: {}".format(tensor.shape))
    print("Dtype: {}".format(tensor.dtype))
    print("Name: {}".format(tensor.name))
    
    if session is not None:
        try:
            value = session.run(tensor)
            print("Min: {:.6f}".format(np.min(value)))
            print("Max: {:.6f}".format(np.max(value)))
            print("Mean: {:.6f}".format(np.mean(value)))
            print("Std: {:.6f}".format(np.std(value)))
            print("Has NaN: {}".format(np.any(np.isnan(value))))
            print("Has Inf: {}".format(np.any(np.isinf(value))))
            if step is not None:
                print("Step: {}".format(step))
        except Exception as e:
            print("Error evaluating tensor: {}".format(e))

def debug_weights(session, step=None):
    """
    Debug all trainable weights in the model.
    """
    print("\n{}".format('='*60))
    print("DEBUG: WEIGHTS ANALYSIS - Step {}".format(step))
    print("{}".format('='*60))
    
    trainable_vars = tf.trainable_variables()
    total_params = 0
    
    for var in trainable_vars:
        var_value = session.run(var)
        print("\nVariable: {}".format(var.name))
        print("  Shape: {}".format(var.shape))
        print("  Min: {:.6f}".format(np.min(var_value)))
        print("  Max: {:.6f}".format(np.max(var_value)))
        print("  Mean: {:.6f}".format(np.mean(var_value)))
        print("  Std: {:.6f}".format(np.std(var_value)))
        print("  Has NaN: {}".format(np.any(np.isnan(var_value))))
        print("  Has Inf: {}".format(np.any(np.isinf(var_value))))
        print("  L2 Norm: {:.6f}".format(np.linalg.norm(var_value)))
        
        total_params += np.prod(var.shape)
    
    print("\nTotal trainable parameters: {}".format(total_params))

def debug_gradients(session, grads_and_vars, step=None):
    """
    Debug gradients for all variables.
    """
    print("\n{}".format('='*60))
    print("DEBUG: GRADIENTS ANALYSIS - Step {}".format(step))
    print("{}".format('='*60))
    
    for grad, var in grads_and_vars:
        if grad is not None:
            # grad is already a numpy array from tf.gradients()
            grad_value = grad
            print("\nGradient: {}".format(var.name))
            print("  Shape: {}".format(grad_value.shape))
            print("  Min: {:.6f}".format(np.min(grad_value)))
            print("  Max: {:.6f}".format(np.max(grad_value)))
            print("  Mean: {:.6f}".format(np.mean(grad_value)))
            print("  Std: {:.6f}".format(np.std(grad_value)))
            print("  Has NaN: {}".format(np.any(np.isnan(grad_value))))
            print("  Has Inf: {}".format(np.any(np.isinf(grad_value))))
            print("  L2 Norm: {:.6f}".format(np.linalg.norm(grad_value)))
        else:
            print("\nGradient: {} - None".format(var.name))

def debug_loss_components(session, loss_tensors, step=None):
    """
    Debug individual loss components.
    """
    print("\n{}".format('='*60))
    print("DEBUG: LOSS COMPONENTS - Step {}".format(step))
    print("{}".format('='*60))
    
    for name, tensor in loss_tensors.items():
        try:
            value = session.run(tensor)
            print("{}: {:.6f}".format(name, value))
        except Exception as e:
            print("{}: Error - {}".format(name, e))

def debug_perceptual_loss_components(session, model, step=None):
    """
    Debug perceptual loss components in detail.
    """
    print("\n{}".format('='*60))
    print("DEBUG: PERCEPTUAL LOSS COMPONENTS - Step {}".format(step))
    print("{}".format('='*60))
    
    # Get all variables related to perceptual loss
    perceptual_vars = [v for v in tf.global_variables() if 'perceptual' in v.name.lower() or 'vgg' in v.name.lower()]
    
    if perceptual_vars:
        print("Perceptual loss related variables:")
        for var in perceptual_vars:
            try:
                value = session.run(var)
                print("  {}: shape={}, mean={:.6f}, std={:.6f}".format(
                    var.name, var.shape, np.mean(value), np.std(value)))
            except Exception as e:
                print("  {}: Error - {}".format(var.name, e))
    else:
        print("No perceptual loss variables found")

def debug_batch_data(session, batch_data, step=None):
    """
    Debug input batch data in detail.
    """
    print("\n{}".format('='*60))
    print("DEBUG: BATCH DATA ANALYSIS - Step {}".format(step))
    print("{}".format('='*60))
    
    for key, value in batch_data.items():
        if isinstance(value, np.ndarray):
            print("\n{}:".format(key))
            print("  Shape: {}".format(value.shape))
            print("  Dtype: {}".format(value.dtype))
            print("  Min: {:.6f}".format(np.min(value)))
            print("  Max: {:.6f}".format(np.max(value)))
            print("  Mean: {:.6f}".format(np.mean(value)))
            print("  Std: {:.6f}".format(np.std(value)))
            print("  Has NaN: {}".format(np.any(np.isnan(value))))
            print("  Has Inf: {}".format(np.any(np.isinf(value))))
            
            # For image data, show additional statistics
            if len(value.shape) == 4 and value.shape[-1] in [1, 3]:  # Image data
                print("  Image Statistics:")
                print("    Per-channel mean: {}".format(np.mean(value, axis=(0,1,2))))
                print("    Per-channel std: {}".format(np.std(value, axis=(0,1,2))))

def debug_model_forward_pass(session, model, inputs, step=None):
    """
    Debug the model forward pass step by step.
    """
    print("\n{}".format('='*60))
    print("DEBUG: MODEL FORWARD PASS - Step {}".format(step))
    print("{}".format('='*60))
    
    try:
        # Get model tensors if available
        if hasattr(model, 'tensors'):
            tensors = model.tensors
            print("Available model tensors:")
            for name, tensor in tensors.items():
                try:
                    value = session.run(tensor)
                    print("  {}: shape={}, mean={:.6f}, std={:.6f}".format(
                        name, tensor.shape, np.mean(value), np.std(value)))
                except Exception as e:
                    print("  {}: Error - {}".format(name, e))
        else:
            print("Model tensors not available for debugging")
    except Exception as e:
        print("Error in model forward pass debugging: {}".format(e))

def create_debug_config():
    """
    Create a debug configuration with small batch size and perceptual loss.
    """
    # Load the default paths configuration
    paths_config = load_configs(['configs/paths/default.yaml'])
    
    config = {
        'name': 'debug-celeba-10pts-perceptual',
        'training': {
            'ncheckpoint': 50,  # Save more frequently
            'n_test': 25,       # Test more frequently
            'gradclip': 1.0,
            'dset': 'celeba',
            'train_dset_params': {
                'dataset': 'celeba',
                'subset': 'train'
            },
            'test_dset_params': {
                'dataset': 'mafl',
                'subset': 'test',
                'order_stream': True,
                'max_samples': 50  # Small test set
            },
            'logdir': 'data/logs/debug_perceptual',
            'datadir': paths_config.celeba_data_dir,  # Use path from config
            'batch': 2,  # Very small batch size for debugging
            'allow_growth': True,
            'optim': 'Adam',
            'lr': {
                'start_val': 0.0001,  # Lower learning rate for perceptual loss
                'step': 1000,  # Shorter decay steps
                'decay': 0.95
            }
        },
        'model': {
            'gauss_std': 0.10,
            'gauss_mode': 'rot',
            'n_maps': 10,
            'n_filters': 16,  # Smaller filters for debugging
            'block_sizes': [1, 1, 1],
            'n_filters_render': 16,  # Smaller renderer filters
            'renderer_stride': 2,
            'min_res': 16,
            'same_n_filt': False,
            'reconstruction_loss': 'perceptual',  # Use perceptual loss
            'perceptual': {
                'l2': True,
                'comp': ['input', 'conv1_2','conv2_2','conv3_2','conv4_2','conv5_2'],
                'net_file': paths_config.vgg16_path  # Use path from config
            },
            'loss_mask': False,  # Disable for simpler debugging
            'confidence': False,
            'channels_bug_fix': True  # Enable for perceptual loss
        }
    }
    return Box(config)

def debug_training_loop(session, loss, train_op, train_summary_op, test_summary_op,
                       train_dset, test_dset, training_pl, handle_pl, 
                       global_step, model, num_steps=100):
    """
    Enhanced debug training loop with step-by-step analysis.
    """
    # Initialize iterators
    train_iterator = train_dset.make_initializable_iterator()
    test_iterator = test_dset.make_initializable_iterator()
    
    train_handle = session.run(train_iterator.string_handle())
    session.run(train_iterator.initializer)
    
    test_handle = session.run(test_iterator.string_handle())
    session.run(test_iterator.initializer)
    
    # Get initial step
    start_step = session.run(global_step)
    
    print("\n{}".format('='*80))
    print("STARTING ENHANCED DEBUG TRAINING WITH PERCEPTUAL LOSS")
    print("Initial step: {}".format(start_step))
    print("Total steps: {}".format(num_steps))
    print("Batch size: 2")
    print("Loss type: Perceptual")
    print("{}".format('='*80))
    
    for step in range(start_step, start_step + num_steps):
        print("\n{}".format('='*60))
        print("STEP {} - DETAILED ANALYSIS".format(step))
        print("{}".format('='*60))
        
        # Training step
        feed_dict = {handle_pl: train_handle, training_pl: True}
        
        # Get current batch for debugging
        if step % 5 == 0:  # Debug every 5 steps
            print("\n--- DEBUGGING BATCH DATA ---")
            batch_data = session.run(train_iterator.get_next())
            debug_batch_data(session, batch_data, step)
        
        # Run training step with detailed debugging
        if step % 5 == 0:
            # Debug step - run with summaries and detailed analysis
            print("\n--- RUNNING TRAINING STEP WITH DEBUGGING ---")
            
            # Debug weights before training step
            debug_weights(session, step)
            
            # Run training step
            loss_value, _, summary_str = session.run(
                [loss, train_op, train_summary_op], feed_dict=feed_dict)
            
            # Debug gradients after training step
            grads_and_vars = session.run(tf.gradients(loss, tf.trainable_variables()), 
                                        feed_dict={training_pl: True, handle_pl: train_handle})
            debug_gradients(session, list(zip(grads_and_vars, tf.trainable_variables())), step)
            
            # Debug perceptual loss components
            debug_perceptual_loss_components(session, model, step)
            
            # Debug model forward pass
            debug_model_forward_pass(session, model, batch_data, step)
            
        else:
            # Regular step
            loss_value, _ = session.run([loss, train_op], feed_dict=feed_dict)
        
        # Print loss
        print("\n--- LOSS SUMMARY ---")
        print("Loss: {:.6f}".format(loss_value))
        
        # Check for NaN/Inf
        if np.isnan(loss_value) or np.isinf(loss_value):
            print("WARNING: Loss is {}!".format('NaN' if np.isnan(loss_value) else 'Inf'))
            print("Stopping training due to numerical instability")
            break
        
        # Test step every 10 steps
        if step % 10 == 0 and test_dset is not None:
            print("\n--- TESTING ---")
            test_feed_dict = {handle_pl: test_handle, training_pl: False}
            test_loss = session.run(loss, feed_dict=test_feed_dict)
            print("Test Loss: {:.6f}".format(test_loss))
            
            # Reset test iterator
            session.run(test_iterator.initializer)
        
        # Save checkpoint every 25 steps
        if step % 25 == 0:
            checkpoint_path = "data/logs/debug_perceptual/model_debug_step_{}.ckpt".format(step)
            saver = tf.train.Saver()
            saver.save(session, checkpoint_path)
            print("Saved checkpoint: {}".format(checkpoint_path))

def main():
    """
    Main debug training function with perceptual loss.
    """
    print("Starting IMM Enhanced Debug Training with Perceptual Loss...")
    
    # Create debug configuration
    config = create_debug_config()
    train_config = config.training
    
    # Create log directory
    if not os.path.exists(train_config.logdir):
        os.makedirs(train_config.logdir)
    
    # Create graph
    graph = tf.Graph()
    with graph.as_default():
        # Global step
        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        # Learning rate
        lr = tf.train.exponential_decay(
            train_config.lr.start_val,
            global_step,
            train_config.lr.step,
            train_config.lr.decay,
            staircase=True
        )
        
        # Optimizer
        if train_config.optim.lower() == 'adam':
            optim = tf.train.AdamOptimizer(lr, name='Adam')
        else:
            raise ValueError('Optimizer {} not supported'.format(train_config.optim))
        
        # Model factory
        factory = DebugModelFactory(IMMModel, config=config.model, global_step=global_step)
        
        # Dataset
        dset_class = import_dataset(train_config.dset)
        train_dset_params = train_config.train_dset_params.copy()
        if 'subset' in train_dset_params:
            del train_dset_params['subset']
        train_dset = dset_class(
            train_config.datadir, 
            subset='train',
            **train_dset_params
        )
        train_dset = train_dset.get_dataset(
            train_config.batch, 
            repeat=True, 
            shuffle=False,
            num_preprocess_threads=1  # Single thread for debugging
        )
        
        test_dset_params = train_config.test_dset_params.copy()
        if 'subset' in test_dset_params:
            del test_dset_params['subset']
        test_dset = dset_class(
            train_config.datadir,
            subset='test',
            **test_dset_params
        )
        test_dset = test_dset.get_dataset(
            train_config.batch,
            repeat=False,
            shuffle=False,
            num_preprocess_threads=1
        )
        
        # Input placeholders
        training_pl = tf.placeholder(tf.bool, name='training_pl')
        handle_pl = tf.placeholder(tf.string, shape=[], name='handle_pl')
        
        # Iterator
        base_iterator = tf.data.Iterator.from_string_handle(
            handle_pl, train_dset.output_types, train_dset.output_shapes)
        inputs = base_iterator.get_next()
        
        # Setup training
        loss, train_op, train_summary_op, test_summary_op, model = tru.setup_training(
            {'gpu_ids': [0], 'batch_size': train_config.batch},
            graph, optim, inputs, training_pl, factory, global_step,
            clip_value=train_config.gradclip, split_gpus=False
        )
        
        # Session configuration
        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
        session_config.gpu_options.allow_growth = train_config.allow_growth
        
        # Run training
        with tf.Session(config=session_config) as session:
            # Initialize variables
            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())
            
            # Debug initial weights
            debug_weights(session, 0)
            
            # Run enhanced debug training loop
            debug_training_loop(
                session, loss, train_op, train_summary_op, test_summary_op,
                train_dset, test_dset, training_pl, handle_pl, global_step, model,
                num_steps=1000
            )

if __name__ == '__main__':
    main()
