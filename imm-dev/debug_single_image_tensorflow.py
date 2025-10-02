#!/usr/bin/env python
"""
Single image debugging script for TensorFlow IMM implementation.

This script trains the model on a single image for multiple epochs to 
compare with PyTorch implementation for debugging purposes.
"""

from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import os
import os.path as osp
import sys
import time
import json
import yaml
import traceback
from datetime import datetime

# network definition:
from imm.models.imm_model import IMMModel
import imm.train.cnn_train_multi as tru
from imm.utils.colorize import colorize
from imm.utils.dataset_import import import_dataset


class DebugModelFactory():
    """Factory for creating models with debug capabilities."""
    def __init__(self, network, **kwargs):
        self.network = network
        self.net_args = kwargs

    def create(self):
        return self.network(**self.net_args)


class SimpleBox(dict):
    """Simple box-like dict for config."""
    def __getattr__(self, name):
        try:
            value = self[name]
            if isinstance(value, dict):
                return SimpleBox(value)
            return value
        except KeyError:
            raise AttributeError(name)
    
    def __setattr__(self, name, value):
        self[name] = value


class Tee(object):
    """Write to multiple streams (console + file)."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

def create_debug_config():
    """Create debug configuration for single image training."""
    
    def resolve_celeba_dir(default_hint='../Mtp/data/celeba'):
        """Resolve CelebA data directory robustly.
        Order:
          1) CELEBA_DIR env var
          2) imm-dev/configs/paths/default.yaml
          3) repo-root/configs/paths/default.yaml
          4) heuristics relative to this file
        """
        def is_valid(dir_path):
            if not dir_path or not osp.isdir(dir_path):
                return False
            anno = osp.join(dir_path, 'Anno', 'list_landmarks_align_celeba.txt')
            img_hq = osp.join(dir_path, 'Img', 'img_align_celeba_hq')
            img_std = osp.join(dir_path, 'Img', 'img_align_celeba')
            return osp.isfile(anno) and (osp.isdir(img_hq) or osp.isdir(img_std))

        # 1) Env var
        env_dir = os.environ.get('CELEBA_DIR')
        if is_valid(env_dir):
            print("Using CELEBA_DIR from environment: {}".format(env_dir))
            return osp.abspath(env_dir)

        candidates = []

        # 2) imm-dev paths config
        imm_dev_dir = osp.dirname(osp.abspath(__file__))
        imm_dev_paths = osp.join(imm_dev_dir, 'configs', 'paths', 'default.yaml')
        if osp.isfile(imm_dev_paths):
            try:
                with open(imm_dev_paths, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
                cdir = cfg.get('celeba_data_dir')
                if cdir:
                    cdir = cdir if osp.isabs(cdir) else osp.abspath(osp.join(osp.dirname(imm_dev_paths), cdir))
                    candidates.append(cdir)
            except Exception:
                pass

        # 3) repo-root paths config
        repo_root = osp.dirname(imm_dev_dir)
        root_paths = osp.join(repo_root, 'configs', 'paths', 'default.yaml')
        if osp.isfile(root_paths):
            try:
                with open(root_paths, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
                cdir = cfg.get('celeba_data_dir')
                if cdir:
                    cdir = cdir if osp.isabs(cdir) else osp.abspath(osp.join(osp.dirname(root_paths), cdir))
                    candidates.append(cdir)
            except Exception:
                pass

        # 4) heuristics
        candidates.extend([
            osp.abspath(osp.join(repo_root, '..', 'data', 'celeba')),
            osp.abspath(osp.join(repo_root, 'data', 'celeba')),
            osp.abspath(osp.join(repo_root, '..', 'Mtp', 'data', 'celeba')),
            osp.abspath(osp.join(repo_root, '..', 'MTP', 'data', 'celeba')),
            osp.abspath(osp.join(imm_dev_dir, default_hint))
        ])

        for c in candidates:
            print("Probing CelebA dir: {}".format(c))
            if is_valid(c):
                print("Resolved CelebA dir: {}".format(c))
                return c

        print("WARNING: Failed to resolve CelebA dir from known locations. Last tried: {}".format(candidates[-1] if candidates else 'None'))
        return osp.abspath(osp.join(imm_dev_dir, default_hint))

    celeba_dir = resolve_celeba_dir()

    def resolve_vgg16_path(default_hint='../../vgg16.caffemodel.h5'):
        def is_valid(path):
            return path and osp.isfile(path) and path.lower().endswith('.h5')

        # Env var
        env_path = os.environ.get('VGG16_WEIGHTS')
        if is_valid(env_path):
            print("Using VGG16_WEIGHTS from environment: {}".format(env_path))
            return osp.abspath(env_path)

        candidates = []

        # imm-dev paths config
        imm_dev_dir = osp.dirname(osp.abspath(__file__))
        imm_dev_paths = osp.join(imm_dev_dir, 'configs', 'paths', 'default.yaml')
        if osp.isfile(imm_dev_paths):
            try:
                with open(imm_dev_paths, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
                vgg = cfg.get('vgg16_path')
                if vgg:
                    vgg = vgg if osp.isabs(vgg) else osp.abspath(osp.join(osp.dirname(imm_dev_paths), vgg))
                    candidates.append(vgg)
            except Exception:
                pass

        # repo-root paths config
        repo_root = osp.dirname(imm_dev_dir)
        root_paths = osp.join(repo_root, 'configs', 'paths', 'default.yaml')
        if osp.isfile(root_paths):
            try:
                with open(root_paths, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
                vgg = cfg.get('vgg16_path')
                if vgg:
                    vgg = vgg if osp.isabs(vgg) else osp.abspath(osp.join(osp.dirname(root_paths), vgg))
                    candidates.append(vgg)
            except Exception:
                pass

        # heuristics
        candidates.extend([
            osp.abspath(osp.join(repo_root, '..', 'vgg16.caffemodel.h5')),
            osp.abspath(osp.join(repo_root, 'vgg16.caffemodel.h5')),
            osp.abspath(osp.join(imm_dev_dir, default_hint)),
        ])

        for c in candidates:
            print("Probing VGG16 path: {}".format(c))
            if is_valid(c):
                print("Resolved VGG16 path: {}".format(c))
                return c

        print("WARNING: Failed to resolve VGG16 path from known locations.")
        return osp.abspath(osp.join(imm_dev_dir, default_hint))

    vgg16_path = resolve_vgg16_path()

    config = {
        'name': 'debug-tf-single-image',
        'training': {
            'ncheckpoint': 1000,  # Save very infrequently for debugging
            'n_test': 1000,       # Test very infrequently
            'n_summary': 10,      # Summary every 10 steps
            'gradclip': 1.0,
            'dset': 'celeba',
            'train_dset_params': {
                'dataset': 'celeba',
                'subset': 'train',
                'max_samples': 1,
                'order_stream': True  # deterministic single image
            },
            'test_dset_params': {
                'dataset': 'mafl',
                'subset': 'test',
                'order_stream': True,
                'max_samples': 1  # Only one test image
            },
            'logdir': 'data/logs/debug_tf_single',
            'datadir': celeba_dir,  # resolved absolute dir
            'batch': 1,  # Single image
            'allow_growth': True,
            'optim': 'Adam',
            'lr': {
                'start_val': 0.0001,
                'step': 50,
                'decay': 0.95
            }
        },
        'model': {
            'gauss_std': 0.10,
            'gauss_mode': 'rot',
            'n_maps': 10,
            'n_filters': 32,
            'block_sizes': [1, 1, 1],
            'n_filters_render': 32,
            'renderer_stride': 2,
            'min_res': 16,
            'same_n_filt': False,
            'reconstruction_loss': 'perceptual',
            'perceptual': {
                'l2': True,
                'comp': ['input','conv1_2','conv2_2','conv3_2','conv4_2','conv5_2'],
                'net_file': vgg16_path
            },
            'loss_mask': False,
            'confidence': False,
            'channels_bug_fix': True
        }
    }
    return SimpleBox(config)


def debug_tensor_values(session, tensors_or_values, step, feed_dict=None):
    """Debug tensor values during training.
    tensors_or_values: dict name -> tf.Tensor or np.ndarray
    If tf.Tensor, runs session.run with provided feed_dict; if ndarray, prints directly.
    """
    print("\n=== TF TENSOR VALUES AT STEP {} ===".format(step))
    
    for name, obj in tensors_or_values.items():
        try:
            if isinstance(obj, np.ndarray):
                value = obj
            else:
                value = session.run(obj, feed_dict=feed_dict) if feed_dict is not None else session.run(obj)
            if isinstance(value, np.ndarray):
                print("{}: shape={}, min={:.6f}, max={:.6f}, mean={:.6f}, std={:.6f}".format(
                    name, value.shape, np.min(value), np.max(value), np.mean(value), np.std(value)))
            else:
                print("{}: {}".format(name, value))
        except Exception as e:
            print("{}: Error - {}".format(name, e))


def debug_weights(session, step):
    """Debug all trainable weights."""
    print("\n=== TF MODEL WEIGHTS AT STEP {} ===".format(step))
    
    trainable_vars = tf.trainable_variables()
    total_params = 0
    
    for var in trainable_vars:
        try:
            var_value = session.run(var)
            total_params += np.prod(var.shape)
            print("{}: shape={}, mean={:.6f}, std={:.6f}, norm={:.6f}".format(
                var.name, var.shape, np.mean(var_value), np.std(var_value), np.linalg.norm(var_value)))
        except Exception as e:
            print("{}: Error - {}".format(var.name, e))
    
    print("Total trainable parameters: {}".format(total_params))


def create_dummy_dataset(batch_size=1):
    """Create a dummy dataset for debugging when real data is unavailable."""
    import tensorflow as tf
    import numpy as np
    
    # Create random images
    def generator():
        while True:
            # Random image and future image
            image = np.random.uniform(0, 255, (128, 128, 3)).astype(np.float32)
            future_image = np.random.uniform(0, 255, (128, 128, 3)).astype(np.float32)
            yield {'image': image, 'future_image': future_image}
    
    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types={'image': tf.float32, 'future_image': tf.float32},
        output_shapes={'image': (128, 128, 3), 'future_image': (128, 128, 3)}
    )
    
    # Batch dataset
    dataset = dataset.batch(batch_size)
    
    return dataset


def save_tf_debug_log(step, loss_value, tensor_values, log_data):
    """Save debug information to log."""
    step_data = {
        'step': step,
        'timestamp': datetime.now().isoformat(),
        'loss': float(loss_value),
        'tensor_values': {}
    }
    
    # Convert tensor values to JSON-serializable format
    for name, value in tensor_values.items():
        if isinstance(value, np.ndarray):
            step_data['tensor_values'][name] = {
                'shape': list(value.shape),
                'min': float(np.min(value)),
                'max': float(np.max(value)),
                'mean': float(np.mean(value)),
                'std': float(np.std(value))
            }
        else:
            step_data['tensor_values'][name] = str(value)
    
    log_data['steps'].append(step_data)


def debug_training_loop(session, loss, train_op, train_summary_op, 
                       train_dset, training_pl, handle_pl, global_step, 
                       num_steps=100):
    """Debug training loop with extensive logging."""
    
    # Initialize iterator
    train_iterator = train_dset.make_initializable_iterator()
    train_handle = session.run(train_iterator.string_handle())
    session.run(train_iterator.initializer)
    
    # Get initial step
    start_step = session.run(global_step)
    
    print("\nTF STARTING DEBUG TRAINING")
    print("Initial step: {}".format(start_step))
    print("Total steps: {}".format(num_steps))
    print("=" * 60)
    
    losses = []
    log_data = {
        'framework': 'tensorflow',
        'num_steps': num_steps,
        'start_step': start_step,
        'steps': []
    }
    
    # Print initial weights
    debug_weights(session, 0)
    
    # Get tensor references for debugging (expanded list)
    debug_tensors = {}
    tensor_collection = tf.get_collection('tensors')
    for name, tensor in tensor_collection:
        debug_tensors[name] = tensor
    
    for step in range(start_step, start_step + num_steps):
        print("\nTF STEP {}/{}".format(step + 1, start_step + num_steps))
        print("-" * 40)
        
        # Training step
        feed_dict = {handle_pl: train_handle, training_pl: True}
        
        start_time = time.time()
        
        # Run training step with summary every 10 steps
        if step % 10 == 0:
            loss_value, _, summary_str = session.run(
                [loss, train_op, train_summary_op], feed_dict=feed_dict)
        else:
            loss_value, _ = session.run([loss, train_op], feed_dict=feed_dict)
        
        step_time = time.time() - start_time
        losses.append(loss_value)
        
        print("Step {} completed in {:.3f}s".format(step + 1, step_time))
        print("Loss: {:.6f}".format(loss_value))
        
        # Debug tensor values every step with detailed ranges
        if step % 1 == 0:  # Every step for single image debugging
            tensor_values = {}
            for name, tensor in debug_tensors.items():
                try:
                    value = session.run(tensor, feed_dict={training_pl: False, handle_pl: train_handle})
                    tensor_values[name] = value
                except Exception:
                    pass
            # Print using precomputed values to avoid missing feeds
            debug_tensor_values(session, tensor_values, step)
            save_tf_debug_log(step, loss_value, tensor_values, log_data)
        
        # Debug weights every 10 steps
        if step % 10 == 0:
            debug_weights(session, step)
        
        # Check for divergence
        if np.isnan(loss_value) or np.isinf(loss_value):
            divergence_type = 'NaN' if np.isnan(loss_value) else 'Inf'
            print("TF TRAINING DIVERGED - {} loss detected!".format(divergence_type))
            break
        
        # Reset iterator to repeat the same image
        session.run(train_iterator.initializer)
    
    # Final summary
    print("\nTF TRAINING SUMMARY")
    print("=" * 60)
    print("Completed {} steps".format(len(losses)))
    print("Initial loss: {:.6f}".format(losses[0]))
    print("Final loss: {:.6f}".format(losses[-1]))
    print("Loss change: {:+.2f}%".format((losses[-1] - losses[0]) / losses[0] * 100))
    print("Average loss: {:.6f}".format(np.mean(losses)))
    print("Loss std: {:.6f}".format(np.std(losses)))
    
    # Final weights
    debug_weights(session, len(losses))
    
    return log_data, losses


def main():
    """Main debugging function."""
    print("TENSORFLOW SINGLE IMAGE DEBUG TRAINING")
    print("=" * 60)
    
    # Create debug configuration
    config = create_debug_config()
    train_config = config.training
    
    print("Config created")
    print("Batch size: {}".format(config.training.batch))
    print("Learning rate: {}".format(config.training.lr.start_val))
    print("Reconstruction loss: {}".format(config.model.reconstruction_loss))
    
    # Create log directory
    if not os.path.exists(train_config.logdir):
        os.makedirs(train_config.logdir)
    
    # Ensure log directory exists and set up tee logger for console + file
    resolved_logdir = train_config.logdir
    try:
        resolved_logdir = osp.abspath(resolved_logdir)
        if not os.path.exists(resolved_logdir):
            os.makedirs(resolved_logdir)
        log_file_path = osp.join(resolved_logdir, 'train.log')
        log_fp = open(log_file_path, 'a')
        sys.stdout = Tee(sys.stdout, log_fp)
        sys.stderr = Tee(sys.stderr, log_fp)
        print("Log dir: {}".format(resolved_logdir))
        print("Saving terminal logs to: {}".format(log_file_path))
    except Exception as e:
        print("Warning: failed to set up log file writer: {}".format(e))

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
        
        # Try to use the real dataset first
        try:
            print("Trying to load real dataset from {}".format(train_config.datadir))
            if not osp.isdir(train_config.datadir):
                raise IOError("Data dir does not exist: {}".format(train_config.datadir))

            dset_class = import_dataset(train_config.dset)
            train_dset_params = train_config.train_dset_params.copy()
            if 'subset' in train_dset_params:
                del train_dset_params['subset']
            # Build dataset instance to inspect files
            train_dset_instance = dset_class(
                train_config.datadir, 
                subset='train',
                **train_dset_params
            )
            # Sanity-check at least one file
            try:
                first_path = osp.join(train_dset_instance._image_dir, train_dset_instance._images[0])
                print("First image path: {} (exists={} )".format(first_path, osp.isfile(first_path)))
                print("Num images in subset: {}".format(len(train_dset_instance._images)))
            except Exception as ie:
                print("Unable to inspect dataset instance: {}".format(ie))

            train_dset = train_dset_instance.get_dataset(
                train_config.batch, 
                repeat=True, 
                shuffle=False,  # Don't shuffle for consistent debugging
                num_preprocess_threads=1
            )
            print("Successfully loaded real dataset")
        except Exception as e:
            print("Failed to load real dataset: {}".format(repr(e)))
            traceback.print_exc()
            print("Using dummy dataset as fallback")
            train_dset = create_dummy_dataset(batch_size=train_config.batch)
        
        # Input placeholders
        training_pl = tf.placeholder(tf.bool, name='training_pl')
        handle_pl = tf.placeholder(tf.string, shape=[], name='handle_pl')
        
        # Iterator
        base_iterator = tf.data.Iterator.from_string_handle(
            handle_pl, train_dset.output_types, train_dset.output_shapes)
        inputs = base_iterator.get_next()
        
        # Debug: Print input info
        print("TF Input types: {}".format(train_dset.output_types))
        print("TF Input shapes: {}".format(train_dset.output_shapes))
        
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
        
        # Use original training loop
        opts = {
            'print_'
            'gpu_ids': [0],
            'log_dir': train_config.logdir,
            'n_summary': train_config.n_summary,
            'n_test': train_config.n_test,
            'n_checkpoint': train_config.ncheckpoint,
            'batch_size': train_config.batch,
            'print_weights_every_step': True,
            'print_loss_each_step': True,
            'print_tensor_ranges': True,
            'tensor_ranges_every': 1,
        }

        # Also create a tiny test dataset (single image)
        try:
            test_dset_params = train_config.test_dset_params.copy()
            if 'subset' in test_dset_params:
                del test_dset_params['subset']
            test_dset_instance = dset_class(
                train_config.datadir,
                subset='test',
                **test_dset_params
            )
            test_dset = test_dset_instance.get_dataset(
                train_config.batch,
                repeat=False,
                shuffle=False,
                num_preprocess_threads=1
            )
        except Exception:
            test_dset = None

        checkpoint_fname = osp.join(train_config.logdir, 'INVALID')

        # Run the canonical training loop for a short debug run
        tru.train_loop(
            opts, graph, loss, train_dset, training_pl, handle_pl,
            train_op, train_summary_op, test_summary_op,
            num_steps=100, global_step=global_step,
            checkpoint_fname=checkpoint_fname,
            test_dataset=test_dset,
            allow_growth=train_config.allow_growth)
    
    print("\nTENSORFLOW SINGLE-IMAGE TRAINING (PERCEPTUAL) COMPLETED!")


if __name__ == '__main__':
    main()
