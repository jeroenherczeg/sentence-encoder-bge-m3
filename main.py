from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List

app = FastAPI()

model_loaded = False
model = None

class SentenceInput(BaseModel):
    sentences: List[str]

class EncodingOutput(BaseModel):
    encodings: List[List[float]]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_loaded
    model = SentenceTransformer("BAAI/bge-m3")
    model_loaded = True
    yield
    model_loaded = False

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
