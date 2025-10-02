#!/bin/bash

# Debug training script for IMM TensorFlow model
echo "Starting IMM Debug Training..."

# Create necessary directories
mkdir -p data/logs/debug
mkdir -p data/datasets/celeba
mkdir -p data/models

# Check if VGG16 model exists
if [ ! -f "data/models/vgg16.caffemodel.h5" ]; then
    echo "Warning: VGG16 model not found at data/models/vgg16.caffemodel.h5"
    echo "Please ensure the VGG16 model is available for perceptual loss"
fi

# Run debug training
python debug_train.py

echo "Debug training completed!"
