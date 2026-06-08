"""
Script 13c: Emotion Model
RoBERTa + classifier head -> 6-dim logits.
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class EmotionClassifier(nn.Module):
    """hfl/chinese-roberta-wwm-ext -> [CLS] -> MLP -> 6-dim logits."""

    def __init__(self, model_name="hfl/chinese-roberta-wwm-ext", dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 6),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(cls_embedding)
