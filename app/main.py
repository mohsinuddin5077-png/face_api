from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import numpy as np
import uvicorn
import json
import os
from PIL import Image 
from io import BytesIO
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
ENCODINGS_FILE = "encodings.json"
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

def initialize_encodings():
    """Initialize encodings file if it doesn't exist"""
    if not os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "w") as f:
            json.dump({}, f)
        logger.info("Created new encodings file")

def load_encodings() -> Dict[str, List[float]]:
    """Load face encodings from JSON file"""
    try:
        with open(ENCODINGS_FILE, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} face encodings")
        return {k: np.array(v) for k, v in data.items()}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Error loading encodings: {str(e)}")
        return {}

def save_encoding(name: str, encoding: np.ndarray) -> None:
    """Save a new face encoding"""
    data = load_encodings()
    data[name] = encoding.tolist()
    with open(ENCODINGS_FILE, "w") as f:
        json.dump(data, f)
    logger.info(f"Saved encoding for {name}")

def validate_image(file: UploadFile) -> None:
    """Validate uploaded image file"""
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    
    extension = file.filename.split(".")[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Invalid file type. Only JPG/JPEG/PNG allowed.")

@app.post("/register_face")
async def register_face(name: str = Form(...), image: UploadFile = File(...)):
    try:
        # Validate inputs
        if not name.strip():
            raise HTTPException(400, "Name cannot be empty")
        
        validate_image(image)
        
        # Read image
        contents = await image.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(400, "File too large (max 5MB)")

        # Process image
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_array = np.array(img)

        # Detect faces
        face_locations = face_recognition.face_locations(img_array)
        if not face_locations:
            raise HTTPException(400, "No face detected")

        # Get encoding
        encoding = face_recognition.face_encodings(img_array, face_locations)[0]
        save_encoding(name, encoding)

        return {"status": "success", "name": name}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(500, f"Server error: {str(e)}")

@app.post("/verify_face")
async def verify_face(image: UploadFile = File(...)):
    try:
        validate_image(image)
        
        # Read image
        contents = await image.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_array = np.array(img)

        # Detect faces
        face_locations = face_recognition.face_locations(img_array)
        if not face_locations:
            raise HTTPException(400, "No face detected")

        # Get encoding
        unknown_encoding = face_recognition.face_encodings(img_array, face_locations)[0]
        known_encodings = load_encodings()

        # Compare faces
        for name, known_encoding in known_encodings.items():
            matches = face_recognition.compare_faces(
                [np.array(known_encoding)], 
                unknown_encoding,
                tolerance=0.6
            )
            if matches[0]:
                distance = face_recognition.face_distance(
                    [np.array(known_encoding)], 
                    unknown_encoding
                )[0]
                return {
                    "status": "success",
                    "name": name,
                    "confidence": float(1 - distance)
                }

        return {"status": "failed", "message": "Face not recognized"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification error: {str(e)}")
        raise HTTPException(500, f"Server error: {str(e)}")

# Initialize on startup
initialize_encodings()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)