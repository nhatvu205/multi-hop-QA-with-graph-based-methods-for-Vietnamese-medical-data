import torch
import torch.nn as nn
from transformers import AutoModel
from config import MODEL_CONFIG, RELATION_TYPES

class ViHealthBERTRelationExtractor(nn.Module):
    def __init__(self, num_relations=len(RELATION_TYPES)):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_CONFIG['model_name'])
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_relations)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

def load_model(checkpoint_path=None):
    """Load model, optionally from checkpoint"""
    model = ViHealthBERTRelationExtractor()
    
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model from {checkpoint_path}")
    
    return model

