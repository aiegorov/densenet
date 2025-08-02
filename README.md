# DenseNet Implementation

This repository contains two DenseNet implementations:

## Main Implementation (Current)
Uses torchvision's pre-trained DenseNet models for inference.

### Files:
- `inference.py` - Main inference script using torchvision DenseNet
- `config.yaml` - Configuration for torchvision models
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

## Installation
```bash
pip install -r requirements.txt
```

## Configuration
Edit `config.yaml` to modify:
- Model type (densenet121, densenet169, etc.)
- Number of classes
- Pre-trained weights usage
- Inference settings (device, image size, normalization)
