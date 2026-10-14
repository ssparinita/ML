"""
BERT FINE-TUNING  (this is where the BERT model is implemented/trained)
Run:  python train_bert.py --csv amazon_review.csv
Outputs: ./bert_sentiment_model/ (weights + tokenizer), metrics.json, loss_curve.png, confusion_matrix.png
Same data steps as the EDA notebook: drop missing/blank reviewText, label = 1 if overall >= 4 else 0,
stratified 80/10/10 split with random_state=42.
"""
import argparse, json
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="amazon_review.csv"); ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--out", default="bert_sentiment_model"); a = ap.parse_args()
MODEL, MAXLEN = "bert-base-uncased", 128

# 1) data + preprocessing 
df = pd.read_csv(a.csv)
df = df.dropna(subset=["reviewText"]); df = df[df["reviewText"].astype(str).str.strip() != ""].copy()
df["sentiment"] = np.where(df["overall"] >= 4, 1, 0)           # 0 = Negative, 1 = Positive
X, y = df["reviewText"].astype(str), df["sentiment"]
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

# 2) BERT tokenizer: WordPiece tokens + [CLS]/[SEP], truncated to 128
tok = AutoTokenizer.from_pretrained(MODEL)
class DS(Dataset):
    def __init__(s, X, y): s.e = tok(list(X), truncation=True, padding="max_length", max_length=MAXLEN); s.y = list(y)
    def __len__(s): return len(s.y)
    def __getitem__(s, i):
        d = {k: torch.tensor(v[i]) for k, v in s.e.items()}; d["labels"] = torch.tensor(int(s.y[i])); return d

# 3) BERT model: pretrained encoder + 2-class classification head on the [CLS] representation
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

# 4) fine-tuning
args = TrainingArguments(output_dir="./runs", num_train_epochs=a.epochs, learning_rate=2e-5,
    per_device_train_batch_size=16, per_device_eval_batch_size=32, eval_strategy="epoch",
    save_strategy="no", logging_strategy="epoch", fp16=torch.cuda.is_available(), report_to="none")
trainer = Trainer(model=model, args=args, train_dataset=DS(X_tr, y_tr), eval_dataset=DS(X_val, y_val))
trainer.train()

# 5) evaluation on the held-out test split
pred = np.argmax(trainer.predict(DS(X_te, y_te)).predictions, axis=1)
m = dict(accuracy=accuracy_score(y_te, pred), precision=precision_score(y_te, pred), recall=recall_score(y_te, pred), f1=f1_score(y_te, pred))
m = {k: round(v * 100, 2) for k, v in m.items()}; print(m); json.dump(m, open("metrics.json", "w"), indent=2)

# 6) charts used on the website
log = trainer.state.log_history
tr = [l["loss"] for l in log if "loss" in l]; ev = [l["eval_loss"] for l in log if "eval_loss" in l]
plt.figure(figsize=(6, 4)); plt.plot(range(1, len(tr) + 1), tr, marker="o", label="Train"); plt.plot(range(1, len(ev) + 1), ev, marker="o", label="Validation")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("BERT training / validation loss"); plt.legend(); plt.tight_layout(); plt.savefig("loss_curve.png", dpi=150); plt.close()
ConfusionMatrixDisplay(confusion_matrix(y_te, pred), display_labels=["Negative", "Positive"]).plot(cmap="viridis"); plt.title("BERT confusion matrix"); plt.savefig("confusion_matrix.png", dpi=150)

# 7) save the fine-tuned model -> loaded by server.py
model.save_pretrained(a.out); tok.save_pretrained(a.out); print("saved to", a.out)
