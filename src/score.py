import json
import os
import glob
import joblib
import logging
import numpy as np
import pandas as pd

# Global model variable
model = None

def init():
    """
    Called once when the container starts. Loads the trained model into memory.
    """
    global model
    
    logging.info("Initializing fraud detection model")
    
    # Get model directory from environment
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    
    if model_dir is None:
        raise RuntimeError("AZUREML_MODEL_DIR environment variable not set")
    
    logging.info(f"Model directory: {model_dir}")
    
    # Find model file
    model_files = glob.glob(os.path.join(model_dir, "**/*.joblib"), recursive=True)
    
    if not model_files:
        raise FileNotFoundError(f"No .joblib model file found in {model_dir}")
    
    model_path = model_files[0]
    logging.info(f"Loading model from: {model_path}")
    
    # Load model
    model = joblib.load(model_path)
    logging.info("Model loaded successfully")

def run(raw_data):
    """
    Called for each inference request.
    """
    try:
        # Parse input
        if isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            data = raw_data
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        logging.info(f"Received request with {len(df)} records")
        
        # Make predictions
        predictions = model.predict(df)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)[:, 1]
        else:
            probabilities = predictions.astype(float)
        
        # Return response
        response = {
            "prediction": predictions.tolist(),
            "fraud_probability": probabilities.tolist()
        }
        
        logging.info(f"Predictions: {response}")
        return response
        
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON input: {str(e)}"
        logging.error(error_msg)
        return {"error": error_msg}
    
    except ValueError as e:
        error_msg = f"Prediction error: {str(e)}"
        logging.error(error_msg)
        return {"error": error_msg}
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logging.error(error_msg)
        return {"error": error_msg}