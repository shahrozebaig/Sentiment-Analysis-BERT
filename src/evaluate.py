import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("data/processed/test.csv")
df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})

tokenizer = BertTokenizer.from_pretrained("models/bert_sentiment_model")
model = BertForSequenceClassification.from_pretrained("models/bert_sentiment_model")
model.to(device)
model.eval()

preds = []

with torch.no_grad():
    for text in df["review"]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = model(**inputs)
        preds.append(torch.argmax(outputs.logits, dim=1).item())

print("BERT Accuracy:", accuracy_score(df["label"], preds))
