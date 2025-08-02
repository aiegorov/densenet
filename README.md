# DenseNet Implementation

A simple and clean implementation using torchvision's pre-trained DenseNet models for inference, with both command-line and FastAPI interfaces.

## Files:
- `inference.py` - Command-line inference script using torchvision DenseNet
- `app.py` - FastAPI application for REST API inference
- `config.yaml` - Configuration for torchvision models
- `download_weights.py` - Script to download pre-trained weights
- `test_api.py` - Test script for the FastAPI endpoints
- `requirements.txt` - Dependencies

## Command-Line Usage:
```bash
# Basic inference with default config
python inference.py --image path/to/image.jpg

# With custom config
python inference.py --config custom_config.yaml --image path/to/image.jpg

# Save results
python inference.py --image path/to/image.jpg --output results.yaml
```

## FastAPI Usage:

### Start the API Server:
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py

# Or using uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000
```

### API Endpoints:

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Get Configuration
```bash
curl http://localhost:8000/config
```

#### Single Image Prediction
```bash
# Upload image file
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/image.jpg"

# Send base64 encoded image
curl -X POST "http://localhost:8000/predict_base64" \
     -H "Content-Type: application/json" \
     -d '{"image": "base64_encoded_image_data"}'
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict_batch" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg"
```

#### Update Configuration
```bash
curl -X POST "http://localhost:8000/config" \
     -H "Content-Type: application/json" \
     -d '{"model_name": "densenet169", "pretrained": true}'
```

### Interactive API Documentation:
Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

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

## Testing the API:
```bash
# Run the test script
python test_api.py

# Or test individual endpoints
curl http://localhost:8000/health
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
