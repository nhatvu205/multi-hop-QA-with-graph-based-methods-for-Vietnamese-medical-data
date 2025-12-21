import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config import MODEL_CONFIG, RELATION2ID
from text_utils import segment_text

class RelationExtractionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        head = sample['head_entity']
        tail = sample['tail_entity']
        context = sample['context']
        relation = sample['relation']
        
        context_seg = segment_text(context)
        head_seg = segment_text(head)
        tail_seg = segment_text(tail)
        
        text = f"{head_seg} [SEP] {tail_seg} [SEP] {context_seg}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        label = RELATION2ID[relation]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_tokenizer():
    """Load ViHealthBERT tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
    return tokenizer

