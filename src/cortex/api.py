from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from prometheus_fastapi_instrumentator import Instrumentator
from evidently import Report
from evidently.presets import DataDriftPreset

app = FastAPI(title="Cortex V4 Toxic Comment API")
Instrumentator().instrument(app).expose(app)

# --- IN-MEMORY STORAGE (No DB needed) ---
live_logs = []

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
    live_logs.append(request.text)

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

@app.get("/monitor", response_class=HTMLResponse)
async def monitor_drift():
    csv_path = Path("../data/raw/train.csv")
    if not csv_path.exists():
        raise HTTPException(status_code=500, detail="train.csv not found.")
    
    df_train = pd.read_csv(csv_path, nrows=500) 
    reference_data = pd.DataFrame({"text": df_train['comment_text'].astype(str)})

    if len(live_logs) < 5:
        raise HTTPException(status_code=400, detail="Need at least 5 live predictions to calculate drift. Go send some texts to /predict first.")

    current_data = pd.DataFrame({"text": live_logs})

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    
    return HTMLResponse(content=report.get_html(), status_code=200)