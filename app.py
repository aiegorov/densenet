from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import yaml
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import os
import uvicorn
from pathlib import Path

# Import inference functions from the existing script
from inference import load_config, initialize_model, create_transforms, predict

app = FastAPI(
    title="DenseNet Inference API",
    description="A FastAPI wrapper for DenseNet image classification",
    version="1.0.0"
)

# Global variables to store model and transforms
model = None
device = None
transform = None
config = None

class PredictionResponse(BaseModel):
    predicted_class: int
    confidence: float
    model_name: str
    image_size: list

class ConfigRequest(BaseModel):
    model_name: str = "densenet121"
    num_classes: int = 1000
    pretrained: bool = True
    device: str = "cuda"

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global model, device, transform, config
    
    print("Initializing DenseNet model...")
    
    # Load configuration
    config = load_config("config.yaml")
    
    # Initialize model
    model, device = initialize_model(config)
    print(f"Model loaded on device: {device}")
    
    # Create transforms
    transform = create_transforms(config)
    print("Model initialization complete!")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "DenseNet Inference API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Upload image for classification",
            "POST /predict_base64": "Send base64 encoded image for classification",
            "GET /health": "Health check",
            "GET /config": "Get current configuration",
            "POST /config": "Update configuration"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    }

@app.get("/config")
async def get_config():
    """Get current configuration."""
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")
    
    return {
        "model": config.get("model", {}),
        "inference": config.get("inference", {}),
        "weights": config.get("weights", {})
    }

@app.post("/config")
async def update_config(config_request: ConfigRequest):
    """Update configuration and reload model."""
    global model, device, transform, config
    
    try:
        # Update config
        config["model"]["name"] = config_request.model_name
        config["model"]["num_classes"] = config_request.num_classes
        config["model"]["pretrained"] = config_request.pretrained
        config["inference"]["device"] = config_request.device
        
        # Save updated config
        with open("config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Reinitialize model
        model, device = initialize_model(config)
        transform = create_transforms(config)
        
        return {
            "message": "Configuration updated successfully",
            "model_name": config_request.model_name,
            "device": str(device)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Predict class for uploaded image."""
    global model, device, transform, config
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Perform prediction
        predicted_class, confidence = predict(model, device, image, transform)
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            model_name=config["model"]["name"],
            image_size=config["inference"]["image_size"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_base64", response_model=PredictionResponse)
async def predict_base64_image(image_data: dict):
    """Predict class for base64 encoded image."""
    global model, device, transform, config
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Decode base64 image
        if "image" not in image_data:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request")
        
        image_bytes = base64.b64decode(image_data["image"])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Perform prediction
        predicted_class, confidence = predict(model, device, image, transform)
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            model_name=config["model"]["name"],
            image_size=config["inference"]["image_size"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Predict classes for multiple images."""
    global model, device, transform, config
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    
    try:
        for file in files:
            if not file.content_type.startswith("image/"):
                continue  # Skip non-image files
            
            # Read and process image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Perform prediction
            predicted_class, confidence = predict(model, device, image, transform)
            
            results.append({
                "filename": file.filename,
                "predicted_class": predicted_class,
                "confidence": confidence
            })
        
        return {
            "predictions": results,
            "total_images": len(results),
            "model_name": config["model"]["name"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 