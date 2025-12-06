import torch
from load_trained_model import load_trained_model
from text_utils import segment_text
from config import ID2RELATION

def predict_relation(head_entity, tail_entity, context, model, tokenizer, config, device):
    """
    Predict relation between two entities in context
    
    Args:
        head_entity: First entity text
        tail_entity: Second entity text
        context: Context text containing both entities
        model: Trained model
        tokenizer: Tokenizer
        config: Model config
        device: Device
    
    Returns:
        relation_type, confidence
    """
    context_seg = segment_text(context)
    head_seg = segment_text(head_entity)
    tail_seg = segment_text(tail_entity)
    
    text = f"{head_seg} [SEP] {tail_seg} [SEP] {context_seg}"
    
    encoding = tokenizer(
        text,
        max_length=config['max_length'],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs, dim=1)[0].item()
    
    relation_type = ID2RELATION[pred_class]
    
    return relation_type, confidence

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, config = load_trained_model(device=device)
    
    print("\n" + "="*60)
    print("RELATION EXTRACTION - DEMO")
    print("="*60)
    
    test_cases = [
        {
            'head': 'tiểu đường',
            'tail': 'Insulin',
            'context': 'Bệnh nhân tiểu đường được điều trị bằng Insulin'
        },
        {
            'head': 'Metformin',
            'tail': 'đau bụng',
            'context': 'Metformin có thể gây ra tác dụng phụ như đau bụng'
        },
        {
            'head': 'viêm gan B',
            'tail': 'xét nghiệm máu',
            'context': 'Viêm gan B được chẩn đoán bằng xét nghiệm máu'
        },
        {
            'head': 'COVID-19',
            'tail': 'sốt',
            'context': 'Bệnh nhân COVID-19 thường có triệu chứng sốt cao'
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        print(f"  Head: {test['head']}")
        print(f"  Tail: {test['tail']}")
        print(f"  Context: {test['context']}")
        
        relation, confidence = predict_relation(
            test['head'],
            test['tail'],
            test['context'],
            model,
            tokenizer,
            config,
            device
        )
        
        print(f"  Prediction: {relation} (confidence: {confidence:.4f})")
    
    print("\n" + "="*60)

