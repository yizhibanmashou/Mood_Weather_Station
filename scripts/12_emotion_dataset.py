"""
Script 13b: Emotion Dataset
PyTorch Dataset: text → RoBERTa tokenization + 6-dim softmax labels.
"""
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class EmotionDataset(Dataset):
    EMOTION_KEYS = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]

    def __init__(self, csv_path, tokenizer_name="hfl/chinese-roberta-wwm-ext",
                 max_length=128, temperature=1.0):
        import pandas as pd
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.temperature = temperature

        # Verify required columns
        missing_text = "content_clean" not in self.df.columns
        missing_labels = any(k not in self.df.columns for k in self.EMOTION_KEYS)
        if missing_text:
            raise ValueError(f"CSV missing 'content_clean' column. Got: {list(self.df.columns)}")
        if missing_labels:
            raise ValueError(f"CSV missing emotion columns. Need: {self.EMOTION_KEYS}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import pandas as pd
        row = self.df.iloc[idx]
        text = str(row["content_clean"]) if pd.notna(row["content_clean"]) else ""

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        raw = torch.tensor(
            [float(row[k]) for k in self.EMOTION_KEYS], dtype=torch.float32
        )
        # Temperature scaling: T<1 sharpens teacher distribution, T>1 softens
        if self.temperature != 1.0:
            raw = raw.clamp(min=1e-8)
            raw = torch.softmax(torch.log(raw) / self.temperature, dim=-1)
        labels = raw

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }
