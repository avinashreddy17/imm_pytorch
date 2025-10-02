@echo off
REM Enhanced Debug training script for IMM TensorFlow model with Perceptual Loss
echo Starting IMM Enhanced Debug Training with Perceptual Loss...

REM Create necessary directories
if not exist "data\logs\debug_perceptual" mkdir "data\logs\debug_perceptual"
if not exist "data\datasets\celeba" mkdir "data\datasets\celeba"
if not exist "data\models" mkdir "data\models"

REM Check if VGG16 model exists
if not exist "data\models\vgg16.caffemodel.h5" (
    echo Warning: VGG16 model not found at data\models\vgg16.caffemodel.h5
    echo Please ensure the VGG16 model is available for perceptual loss
    echo Download from: http://www.robots.ox.ac.uk/~vgg/research/unsupervised_landmarks/resources/vgg16.caffemodel.h5
)

REM Check if CelebA dataset exists
if not exist "data\datasets\celeba" (
    echo Warning: CelebA dataset not found at data\datasets\celeba
    echo Please download and extract CelebA dataset
    echo Download from: http://www.robots.ox.ac.uk/~vgg/research/unsupervised_landmarks/resources/celeba.zip
)

REM Run enhanced debug training
python debug_train_perceptual.py

echo Enhanced debug training completed!
pause
