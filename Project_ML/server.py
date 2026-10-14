"""
BERT INFERENCE API  (loads the fine-tuned model saved by train_bert.py)
Run:  uvicorn server:app --port 8000
POST /predict  {"text": "..."}  ->  {"label": 0|1, "confidence": float}   (0 = Negative, 1 = Positive)
"""
import os, torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "./bert_sentiment_model"
if not os.path.isdir(MODEL_DIR):
    raise SystemExit("bert_sentiment_model/ not found. Run train_bert.py first (or unzip the saved model here).")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)                       # WordPiece tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()  # fine-tuned BERT

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Review(BaseModel):
    text: str

@app.post("/predict")
def predict(r: Review):
    x = tokenizer(r.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        probs = torch.softmax(model(**x).logits, dim=1)                   # [P(neg), P(pos)]
    label = int(torch.argmax(probs, dim=1))
    return {"label": label, "confidence": float(probs[0][label])}
