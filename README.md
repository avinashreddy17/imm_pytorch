# PyTorch Implementation of IMM: Unsupervised Learning of Object Landmarks

This is a PyTorch implementation of "Unsupervised Learning of Object Landmarks through Conditional Image Generation" by Tomas Jakab*, Ankush Gupta*, Hakan Bilen, Andrea Vedaldi (NeurIPS 2018).

**Original TensorFlow implementation**: https://github.com/tomasjakab/imm

## Features

- Complete PyTorch reimplementation of the original TensorFlow code
- Multi-GPU training support with DistributedDataParallel
- SLURM cluster support for distributed training
- Exact replication of the original model architecture and training procedure
- Support for CelebA and AFLW datasets
- VGG16-based perceptual loss
- Thin Plate Spline (TPS) data augmentation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for complete dependencies

## Installation

```bash
# Clone the repository
git clone <this-repo>
cd imm_to_pytorch

# Install dependencies
pip install -r requirements.txt

# Add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Dataset Setup

### CelebA Dataset
Download the CelebA dataset and set the path in `configs/paths/default.yaml`:
```yaml
celeba_data_dir: /path/to/celeba
```

### AFLW Dataset  
Download the AFLW dataset and set the path in `configs/paths/default.yaml`:
```yaml
aflw_data_dir: /path/to/aflw
```

### VGG16 Model
Download the pre-trained VGG16 model for perceptual loss and set the path:
```yaml
vgg16_path: /path/to/vgg16.caffemodel.h5
```

## Training

### Single GPU Training
```bash
python scripts/train.py --configs configs/paths/default.yaml configs/experiments/celeba-10pts.yaml --ngpus 1
```

### Multi-GPU Training
```bash
python scripts/train.py --configs configs/paths/default.yaml configs/experiments/celeba-10pts.yaml --ngpus 4
```

### SLURM Training
```bash
sbatch scripts/slurm_train.sh configs/paths/default.yaml configs/experiments/celeba-10pts.yaml
```

## Testing

```bash
python scripts/test.py --experiment-name celeba-10pts --train-dataset mafl --test-dataset mafl
```

## Model Architecture

The model consists of:
1. **Image Encoder**: CNN encoder for extracting image features
2. **Pose Encoder**: CNN that predicts landmark coordinates as Gaussian heatmaps
3. **Renderer**: Decoder that generates images from joint image+pose features
4. **VGG16 Perceptual Loss**: Pre-trained VGG16 for computing perceptual reconstruction loss

## Configuration

Model and training configurations are specified in YAML files:
- `configs/paths/default.yaml`: Dataset and model paths
- `configs/experiments/`: Experiment-specific configurations

## Differences from Original

This PyTorch implementation maintains exact functional equivalence with the original TensorFlow code while providing:
- Modern PyTorch training practices
- Better multi-GPU scaling
- SLURM integration
- Cleaner code organization

## Citation

```bibtex
@inproceedings{jakab2018unsupervised,
  title={Unsupervised learning of object landmarks through conditional image generation},
  author={Jakab, Tomas and Gupta, Ankush and Bilen, Hakan and Vedaldi, Andrea},
  booktitle={Advances in Neural Information Processing Systems},
  pages={4016--4027},
  year={2018}
}
```
