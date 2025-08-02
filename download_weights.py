#!/usr/bin/env python3
"""
Script to download pre-trained model weights for DenseNet implementations.
Supports both custom DenseNet and torchvision models.
"""

import os
import torch
import torchvision.models as models
import argparse
import yaml
from urllib.request import urlretrieve
import hashlib
import zipfile
import tarfile

# Pre-defined weight URLs and checksums
WEIGHT_URLS = {
    'densenet121': {
        'url': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
        'filename': 'densenet121_pretrained.pth',
        'md5': 'a639ec97c8e4b0b0c0c0c0c0c0c0c0c0'  # Placeholder
    },
    'densenet169': {
        'url': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
        'filename': 'densenet169_pretrained.pth',
        'md5': 'b2777c0a8e4b0b0c0c0c0c0c0c0c0c0c0'  # Placeholder
    },
    'densenet201': {
        'url': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
        'filename': 'densenet201_pretrained.pth',
        'md5': 'c11035718e4b0b0c0c0c0c0c0c0c0c0c0'  # Placeholder
    },
    'densenet161': {
        'url': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
        'filename': 'densenet161_pretrained.pth',
        'md5': '8d451a508e4b0b0c0c0c0c0c0c0c0c0c0'  # Placeholder
    }
}

def calculate_md5(filepath):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url, filename, expected_md5=None):
    """Download a file with optional MD5 verification."""
    print(f"Downloading {filename} from {url}...")
    
    # Create weights directory if it doesn't exist
    os.makedirs('weights', exist_ok=True)
    filepath = os.path.join('weights', filename)
    
    # Download the file
    try:
        urlretrieve(url, filepath)
        print(f"Download completed: {filepath}")
        
        # Verify MD5 if provided
        if expected_md5:
            actual_md5 = calculate_md5(filepath)
            if actual_md5 == expected_md5:
                print("MD5 verification passed!")
            else:
                print(f"Warning: MD5 mismatch. Expected: {expected_md5}, Got: {actual_md5}")
        
        return filepath
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return None

def download_torchvision_weights(model_name, output_dir='weights'):
    """Download torchvision pre-trained weights."""
    if model_name not in WEIGHT_URLS:
        print(f"Error: {model_name} not supported")
        return None
    
    weight_info = WEIGHT_URLS[model_name]
    url = weight_info['url']
    filename = weight_info['filename']
    expected_md5 = weight_info['md5']
    
    return download_file(url, filename, expected_md5)

def download_custom_weights(model_name, output_dir='weights'):
    """Download custom DenseNet weights (placeholder for custom trained models)."""
    print(f"Custom weights for {model_name} not available for download.")
    print("You can train your own model or use torchvision pre-trained weights.")
    return None

def create_model_and_save_weights(model_name, output_dir='weights'):
    """Create a torchvision model and save its weights."""
    print(f"Creating {model_name} model and saving weights...")
    
    # Create the model
    if model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif model_name == 'densenet169':
        model = models.densenet169(pretrained=True)
    elif model_name == 'densenet201':
        model = models.densenet201(pretrained=True)
    elif model_name == 'densenet161':
        model = models.densenet161(pretrained=True)
    else:
        print(f"Error: {model_name} not supported")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'{model_name}_pretrained.pth')
    
    # Save the model weights
    torch.save(model.state_dict(), filepath)
    print(f"Model weights saved to: {filepath}")
    
    return filepath

def update_config_with_weights(config_path, weights_path, model_name):
    """Update config file with the downloaded weights path."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['weights']['path'] = weights_path
        config['weights']['load_pretrained'] = True
        config['model']['name'] = model_name
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Updated {config_path} with weights path: {weights_path}")
    except Exception as e:
        print(f"Error updating config: {e}")

def main():
    parser = argparse.ArgumentParser(description='Download DenseNet model weights')
    parser.add_argument('--model', type=str, default='densenet121',
                       choices=['densenet121', 'densenet169', 'densenet201', 'densenet161'],
                       help='Model to download weights for')
    parser.add_argument('--type', type=str, default='torchvision',
                       choices=['torchvision', 'custom'],
                       help='Type of weights to download')
    parser.add_argument('--output-dir', type=str, default='weights',
                       help='Output directory for weights')
    parser.add_argument('--update-config', type=str, default='config.yaml',
                       help='Config file to update with weights path')
    parser.add_argument('--force', action='store_true',
                       help='Force download even if file exists')
    
    args = parser.parse_args()
    
    print(f"Downloading weights for {args.model} ({args.type})")
    
    # Check if weights already exist
    weights_path = os.path.join(args.output_dir, f'{args.model}_pretrained.pth')
    if os.path.exists(weights_path) and not args.force:
        print(f"Weights already exist at {weights_path}")
        print("Use --force to re-download")
        return
    
    # Download weights based on type
    if args.type == 'torchvision':
        # Try direct download first, fallback to creating model
        downloaded_path = download_torchvision_weights(args.model, args.output_dir)
        if downloaded_path is None:
            print("Direct download failed, creating model and saving weights...")
            downloaded_path = create_model_and_save_weights(args.model, args.output_dir)
    else:  # custom
        downloaded_path = download_custom_weights(args.model, args.output_dir)
    
    if downloaded_path:
        print(f"Successfully downloaded weights to: {downloaded_path}")
        
        # Update config if requested
        if args.update_config and os.path.exists(args.update_config):
            update_config_with_weights(args.update_config, downloaded_path, args.model)
    else:
        print("Failed to download weights")

if __name__ == "__main__":
    main() 