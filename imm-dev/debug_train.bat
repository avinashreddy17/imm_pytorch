@echo off
REM Debug training script for IMM TensorFlow model
echo Starting IMM Debug Training...

REM Create necessary directories
if not exist "data\logs\debug" mkdir "data\logs\debug"
if not exist "data\datasets\celeba" mkdir "data\datasets\celeba"
if not exist "data\models" mkdir "data\models"

REM Check if VGG16 model exists
if not exist "data\models\vgg16.caffemodel.h5" (
    echo Warning: VGG16 model not found at data\models\vgg16.caffemodel.h5
    echo Please ensure the VGG16 model is available for perceptual loss
)

REM Run debug training
python debug_train.py

echo Debug training completed!
pause
