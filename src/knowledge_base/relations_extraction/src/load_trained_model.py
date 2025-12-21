import json
import os
import torch
from transformers import AutoTokenizer
from model import ViHealthBERTRelationExtractor
from config import OUTPUT_PATHS

def load_trained_model(model_dir=None, device=None):
    """
    Load trained relation extraction model
    
    Args:
        model_dir: Path to model directory (default: from config)
        device: Device to load model on (default: auto-detect)
    
    Returns:
        model, tokenizer, config
    """
    if model_dir is None:
        model_dir = OUTPUT_PATHS['model_dir']
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from: {model_dir}")
    print(f"Device: {device}")
    
    config_path = os.path.join(model_dir, 'training_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"Model: {config['model_name']}")
    print(f"Relations: {config['num_relations']}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    model = ViHealthBERTRelationExtractor(num_relations=config['num_relations'])
    
    model_path = os.path.join(model_dir, 'pytorch_model.bin')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    return model, tokenizer, config

if __name__ == '__main__':
    model, tokenizer, config = load_trained_model()
    
    print("\nModel info:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Max length: {config['max_length']}")
    print(f"  Relation types: {', '.join(config['relation_types'])}")

