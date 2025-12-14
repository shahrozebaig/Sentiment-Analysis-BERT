import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

MODEL = "bert-base-uncased"
EPOCHS = 2
BATCH_SIZE = 8
MAX_LEN = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DatasetIMDB(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx])
        }

df = pd.read_csv("data/processed/train.csv")
df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})

tokenizer = BertTokenizer.from_pretrained(MODEL)
dataset = DatasetIMDB(df["review"].tolist(), df["label"].tolist(), tokenizer)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = BertForSequenceClassification.from_pretrained(MODEL, num_labels=2)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

model.train()
for epoch in range(EPOCHS):
    for batch in tqdm(loader):
        optimizer.zero_grad()
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device)
        )
        out.loss.backward()
        optimizer.step()

model.save_pretrained("models/bert_sentiment_model")
tokenizer.save_pretrained("models/bert_sentiment_model")

print("✅ BERT model trained & saved")
