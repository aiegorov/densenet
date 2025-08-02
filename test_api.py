#!/usr/bin/env python3
"""
Test script for the DenseNet FastAPI application.
Demonstrates how to use the API endpoints.
"""

import requests
import base64
import json
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_config():
    """Test the config endpoint."""
    print("Testing config endpoint...")
    response = requests.get(f"{BASE_URL}/config")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_predict_image(image_path):
    """Test image prediction endpoint."""
    print(f"Testing image prediction with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"Error: Image file {image_path} not found")
        return
    
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{BASE_URL}/predict", files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Model: {result['model_name']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_predict_base64(image_path):
    """Test base64 image prediction endpoint."""
    print(f"Testing base64 image prediction with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"Error: Image file {image_path} not found")
        return
    
    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
    
    # Send request
    payload = {"image": base64_image}
    response = requests.post(f"{BASE_URL}/predict_base64", json=payload)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Model: {result['model_name']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_batch_predict(image_paths):
    """Test batch prediction endpoint."""
    print(f"Testing batch prediction with {len(image_paths)} images...")
    
    files = []
    for path in image_paths:
        if Path(path).exists():
            files.append(("files", open(path, "rb")))
    
    if not files:
        print("Error: No valid image files found")
        return
    
    response = requests.post(f"{BASE_URL}/predict_batch", files=files)
    
    # Close files
    for _, f in files:
        f.close()
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Total images processed: {result['total_images']}")
        print(f"Model: {result['model_name']}")
        for pred in result['predictions']:
            print(f"  {pred['filename']}: class {pred['predicted_class']} (confidence: {pred['confidence']:.4f})")
    else:
        print(f"Error: {response.text}")
    print()

def main():
    """Run all tests."""
    print("DenseNet FastAPI Test Script")
    print("=" * 40)
    
    # Test basic endpoints
    test_health()
    test_config()
    
    # Test image prediction (you'll need to provide an image path)
    # test_predict_image("path/to/your/image.jpg")
    # test_predict_base64("path/to/your/image.jpg")
    
    # Test batch prediction
    # test_batch_predict(["path/to/image1.jpg", "path/to/image2.jpg"])
    
    print("Tests completed!")
    print("\nTo test with actual images, uncomment the test calls in main()")
    print("and provide valid image paths.")

if __name__ == "__main__":
    main() 