# IMM Debug Training Setup

This directory contains debug training scripts and configurations for the IMM (Implicit Motion Model) TensorFlow implementation.

## Files Created

### Debug Training Scripts
- `debug_train.py` - Basic debug training script with L2 loss
- `debug_train_perceptual.py` - **Enhanced debug training with perceptual loss**
- `debug_train.bat` - Windows batch file to run basic debug training
- `debug_train_perceptual.bat` - **Windows batch file for enhanced debug training**
- `debug_train.sh` - Linux/Mac shell script to run debug training
- `test_debug_setup.py` - Test script to verify setup

### Configuration Files
- `configs/experiments/debug-celeba-10pts.yaml` - Debug configuration with small batch size

## Features

### Debug Capabilities

#### Basic Debug Training (`debug_train.py`)
1. **Comprehensive Weight Monitoring**
   - Prints all trainable variables with statistics (min, max, mean, std)
   - Detects NaN and Inf values in weights
   - Calculates L2 norms for weight analysis

2. **Gradient Analysis**
   - Monitors gradients for all trainable variables
   - Detects gradient explosion/vanishing
   - Tracks gradient statistics and norms

3. **Loss Component Debugging**
   - Breaks down loss into individual components
   - Monitors loss behavior over time
   - Detects NaN/Inf in loss values

4. **Tensor Information**
   - Detailed tensor shape, dtype, and value analysis
   - Batch data inspection
   - Image tensor statistics

#### Enhanced Debug Training (`debug_train_perceptual.py`)
**All basic features PLUS:**

5. **Perceptual Loss Analysis**
   - Detailed VGG16 feature extraction debugging
   - Perceptual loss component breakdown
   - Feature map statistics at each layer

6. **Step-by-Step Model Analysis**
   - Forward pass debugging through each model component
   - Intermediate tensor analysis
   - Model state monitoring

7. **Enhanced Batch Data Analysis**
   - Detailed image statistics per channel
   - Data distribution analysis
   - Input validation

8. **Frequent Debugging**
   - Debug every 5 steps (vs 10 in basic)
   - More detailed logging
   - Perceptual loss specific monitoring

### Training Configuration
- **Small Batch Size**: 2 samples per batch for easier debugging
- **Reduced Model Size**: Smaller filters (16 instead of 32) for faster training
- **Frequent Checkpointing**: Saves model every 50 steps
- **Simplified Loss**: Uses L2 loss instead of perceptual loss for easier debugging
- **Disabled Loss Mask**: Simplified training without mask complexity

## Usage

### 1. Test Setup
First, verify that everything is working:
```bash
python test_debug_setup.py
```

### 2. Run Enhanced Debug Training (Recommended)
On Windows:
```bash
debug_train_perceptual.bat
```

Or directly:
```bash
python debug_train_perceptual.py
```

### 3. Run Basic Debug Training (L2 Loss)
On Windows:
```bash
debug_train.bat
```

On Linux/Mac:
```bash
./debug_train.sh
```

Or directly:
```bash
python debug_train.py
```

### 3. Monitor Training
The debug training will output:
- Weight statistics every 10 steps
- Gradient analysis every 10 steps
- Loss values every step
- Test loss every 20 steps
- Checkpoint saves every 50 steps

## Debug Output Example

```
=== DEBUG: WEIGHTS ANALYSIS - Step 10 ===
Variable: encoder/conv_1/weights:0
  Shape: (7, 7, 3, 16)
  Min: -0.123456
  Max: 0.234567
  Mean: 0.001234
  Std: 0.056789
  Has NaN: False
  Has Inf: False
  L2 Norm: 0.123456

=== DEBUG: GRADIENTS ANALYSIS - Step 10 ===
Gradient: encoder/conv_1/weights:0
  Shape: (7, 7, 3, 16)
  Min: -0.001234
  Max: 0.002345
  Mean: 0.000123
  Std: 0.000456
  Has NaN: False
  Has Inf: False
  L2 Norm: 0.012345
```

## Configuration

The debug configuration is optimized for debugging:
- **Batch Size**: 2 (very small for easy debugging)
- **Learning Rate**: 0.001 with exponential decay
- **Model Size**: Reduced filters for faster training
- **Loss Type**: L2 (simpler than perceptual loss)
- **Checkpointing**: Every 50 steps
- **Testing**: Every 20 steps

## Troubleshooting

### Common Issues
1. **Import Errors**: Make sure you're in the correct directory and all dependencies are installed
2. **CUDA Errors**: The script will fall back to CPU if GPU is not available
3. **Memory Issues**: The small batch size should prevent memory problems
4. **Dataset Issues**: Ensure the dataset path is correct in the configuration

### Debug Tips
1. Watch for NaN/Inf values in weights or gradients
2. Monitor gradient norms - they should be reasonable (not too large or too small)
3. Check that loss is decreasing over time
4. Verify that weights are updating (not stuck at initialization values)

## Next Steps

After running the debug training:
1. Analyze the loss behavior and weight updates
2. Check for any numerical instabilities
3. Adjust hyperparameters based on observations
4. Scale up to larger batch sizes once debugging is complete
5. Switch to perceptual loss once L2 loss is working well

## Files Structure

```
imm-dev/
├── debug_train.py              # Main debug training script
├── debug_train.bat             # Windows batch file
├── debug_train.sh              # Linux/Mac shell script
├── test_debug_setup.py         # Setup verification script
├── DEBUG_TRAINING_README.md    # This file
└── configs/experiments/
    └── debug-celeba-10pts.yaml # Debug configuration
```
