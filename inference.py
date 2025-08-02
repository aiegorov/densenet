import yaml
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import os

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def initialize_model(config):
    """Initialize DenseNet model with configuration."""
    model_config = config['model']
    
    # Use torchvision's pre-trained DenseNet
    model_name = model_config.get('name', 'densenet121').lower()
    
    if model_name == 'densenet121':
        model = models.densenet121(pretrained=model_config.get('pretrained', False))
    elif model_name == 'densenet169':
        model = models.densenet169(pretrained=model_config.get('pretrained', False))
    elif model_name == 'densenet201':
        model = models.densenet201(pretrained=model_config.get('pretrained', False))
    elif model_name == 'densenet161':
        model = models.densenet161(pretrained=model_config.get('pretrained', False))
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Modify the classifier if num_classes is different from default (1000)
    num_classes = model_config.get('num_classes', 1000)
    if num_classes != 1000:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    
    # Load custom weights if specified
    if config['weights']['load_pretrained'] and os.path.exists(config['weights']['path']):
        print(f"Loading custom weights from {config['weights']['path']}")
        model.load_state_dict(torch.load(config['weights']['path'], map_location='cpu'))
    
    # Set device
    device = torch.device(config['inference']['device'] if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    return model, device

def create_transforms(config):
    """Create image transforms for inference."""
    inference_config = config['inference']
    
    transform = transforms.Compose([
        transforms.Resize(inference_config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=inference_config['mean'],
            std=inference_config['std']
        )
    ])
    
    return transform

def predict(model, device, image_path, transform):
    """Perform inference on a single image."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence

def main():
    parser = argparse.ArgumentParser(description='DenseNet Inference')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output (optional)')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Initialize model
    print("Initializing DenseNet model...")
    model, device = initialize_model(config)
    print(f"Model loaded on device: {device}")
    
    # Create transforms
    transform = create_transforms(config)
    
    # Perform inference
    print(f"Performing inference on {args.image}")
    predicted_class, confidence = predict(model, device, args.image, transform)
    
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    
    # Save results if output path is specified
    if args.output:
        results = {
            'image_path': args.image,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'model_config': config['model']
        }
        
        with open(args.output, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
