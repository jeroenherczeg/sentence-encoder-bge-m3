import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List

model_loaded = False
model = None

class SentenceInput(BaseModel):
    sentences: List[str]

class EncodingOutput(BaseModel):
    encodings: List[List[float]]

async def load_model():
    global model, model_loaded
    loop = asyncio.get_event_loop()
    model = await loop.run_in_executor(None, SentenceTransformer, "BAAI/bge-m3")
    model_loaded = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(load_model())
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/encode", response_model=EncodingOutput)
async def encode_sentences(input: SentenceInput):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        encodings = model.encode(input.sentences)
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
