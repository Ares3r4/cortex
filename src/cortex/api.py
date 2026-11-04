from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

app = FastAPI(title="Cortex V3 Toxic Comment API")

print("Loading model... (this takes a few seconds)")
model_path = "../artifacts/model_v2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval() 

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    is_toxic: bool
    confidence: float
    label: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):

    inputs = tokenizer(
        request.text, 
        truncation=True, 
        padding=True, 
        max_length=128, 
        return_tensors="pt"
    )
    

    with torch.no_grad():
        outputs = model(**inputs)
        

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1).numpy()[0]
    

    prediction = np.argmax(probabilities)
    confidence = float(probabilities[prediction])
    is_toxic = bool(prediction == 1)
    label = "Toxic" if is_toxic else "Safe"

    return PredictionResponse(
        text=request.text,
        is_toxic=is_toxic,
        confidence=round(confidence, 4),
        label=label
    )

@app.get("/health")
def health_check():
    return {"status": "healthy"}