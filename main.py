from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

app = FastAPI()

# Global variable to track if the model is loaded
model_loaded = False

# Load the model at startup
model = None

class SentenceInput(BaseModel):
    sentences: List[str]

class EncodingOutput(BaseModel):
    encodings: List[List[float]]

@app.on_event("startup")
async def startup_event():
    global model, model_loaded
    model = SentenceTransformer("BAAI/bge-m3")
    model_loaded = True

@app.post("/encode", response_model=EncodingOutput)
async def encode_sentences(input: SentenceInput):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        encodings = model.encode(input.sentences)
        # Convert numpy arrays to lists for JSON serialization
        encodings_list = encodings.tolist()
        return EncodingOutput(encodings=encodings_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/readiness")
async def readiness_probe():
    if model_loaded:
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Model not loaded yet")

@app.get("/liveness")
async def liveness_probe():
    return {"status": "alive"}
