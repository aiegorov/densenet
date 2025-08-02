# DenseNet Implementation

A simple and clean implementation using torchvision's pre-trained DenseNet models for inference.

## Files:
- `inference.py` - Main inference script using torchvision DenseNet
- `config.yaml` - Configuration for torchvision models
- `download_weights.py` - Script to download pre-trained weights
- `requirements.txt` - Dependencies

## Usage:
```bash
# Basic inference with default config
python inference.py --image path/to/image.jpg

# With custom config
python inference.py --config custom_config.yaml --image path/to/image.jpg

# Save results
python inference.py --image path/to/image.jpg --output results.yaml
```

## Downloading Weights:
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

## Supported Models:
- `densenet121` (default)
- `densenet169`
- `densenet201`
- `densenet161`

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

## Weight Management
The `download_weights.py` script provides:
- Automatic download of torchvision pre-trained weights
- MD5 verification for downloaded files
- Automatic config file updates
- Support for multiple DenseNet variants
- Fallback to creating models locally if download fails
