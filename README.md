# DenseNet Implementation

This repository contains two DenseNet implementations:

## Main Implementation (Current)
Uses custom DenseNet implementation for inference.

### Files:
- `inference.py` - Main inference script using custom DenseNet
- `model.py` - Custom DenseNet implementation
- `config.yaml` - Configuration for custom model
- `download_weights.py` - Script to download pre-trained weights
- `requirements.txt` - Dependencies

### Usage:
```bash
# Basic inference with default config
python inference.py --image path/to/image.jpg

# With custom config
python inference.py --config custom_config.yaml --image path/to/image.jpg

# Save results
python inference.py --image path/to/image.jpg --output results.yaml
```

### Downloading Weights:
```bash
# Download torchvision DenseNet-121 weights
python download_weights.py --model densenet121 --type torchvision

# Download different model weights
python download_weights.py --model densenet169 --type torchvision

# Force re-download
python download_weights.py --model densenet121 --type torchvision --force

# Update config automatically
python download_weights.py --model densenet121 --type torchvision --update-config config.yaml
```

### Supported Models:
- `densenet121` (default)
- `densenet169`
- `densenet201`
- `densenet161`

## Alternative Implementation
Custom DenseNet implementation located in `alternative_implementation/` folder.

### Files:
- `alternative_implementation/model.py` - Custom DenseNet implementation
- `alternative_implementation/config.yaml` - Configuration for custom model
- `alternative_implementation/inference_alt.py` - Inference script for custom model

## Installation
```bash
pip install -r requirements.txt
```

## Configuration
Edit `config.yaml` to modify:
- Model architecture parameters (growth rate, block config, etc.)
- Number of classes
- Pre-trained weights usage
- Inference settings (device, image size, normalization)

## Weight Management
The `download_weights.py` script provides:
- Automatic download of torchvision pre-trained weights
- MD5 verification for downloaded files
- Automatic config file updates
- Support for multiple DenseNet variants
- Fallback to creating models locally if download fails
