import json
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import OUTPUT_PATHS, ID2RELATION, MODEL_CONFIG, RELATION_TYPES
from dataset import RelationExtractionDataset
from load_trained_model import load_trained_model
from data_preparation import (
    load_knowledge_base, 
    load_preextracted_entities,
    get_kb_contexts,
    format_entities_for_relations
)
from itertools import combinations

def extract_relations_from_kb(model, tokenizer, kb, preextracted_entities, device, confidence_threshold=0.7):
    """Extract relations from knowledge base using trained model"""
    print("\nExtracting relations from knowledge base...")
    
    model.eval()
    all_relations = []
    relation_id = 0
    
    for kb_record in tqdm(kb, desc="Processing KB records"):
        qa_id = kb_record.get('qa_id', '')
        entities_data = preextracted_entities.get(qa_id, {})
        raw_entities = entities_data.get('context', [])
        
        if not raw_entities or len(raw_entities) < 2:
            continue
        
        contexts = get_kb_contexts(kb_record)
        
        if not contexts:
            continue
        
        for context_obj in contexts:
            context_text = context_obj['text']
            if not context_text:
                continue
            
            entities = format_entities_for_relations(raw_entities, context_text)
            
            if len(entities) < 2:
                continue
            
            samples = []
            entity_pairs = []
            
            for e1, e2 in combinations(entities, 2):
                distance = abs(e1['position'] - e2['position'])
                
                if distance > 100:
                    continue
                
                # Generate entity IDs based on text and type
                e1_id = f"{e1['type']}_{e1['text'].replace(' ', '_')}"
                e2_id = f"{e2['type']}_{e2['text'].replace(' ', '_')}"
                
                samples.append({
                    'head_entity': e1['text'],
                    'head_entity_id': e1_id,
                    'head_entity_type': e1['type'],
                    'tail_entity': e2['text'],
                    'tail_entity_id': e2_id,
                    'tail_entity_type': e2['type'],
                    'context': context_text,
                    'relation': 'NO_RELATION',
                    'confidence': 0.0,
                    'source_qa_id': context_obj['qa_id'],
                })
                
                entity_pairs.append((e1, e2))
            
            if not samples:
                continue
            
            dataset = RelationExtractionDataset(samples, tokenizer, MODEL_CONFIG['max_length'])
            dataloader = DataLoader(dataset, batch_size=MODEL_CONFIG['batch_size'], shuffle=False)
            
            predictions = []
            confidences = []
            
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    logits = model(input_ids, attention_mask)
                    probs = torch.softmax(logits, dim=1)
                    
                    pred_classes = torch.argmax(probs, dim=1)
                    pred_confidences = torch.max(probs, dim=1)[0]
                    
                    predictions.extend(pred_classes.cpu().numpy())
                    confidences.extend(pred_confidences.cpu().numpy())
            
            for idx, (pred, conf) in enumerate(zip(predictions, confidences)):
                relation_type = ID2RELATION[pred]
                
                if relation_type == 'NO_RELATION' or conf < confidence_threshold:
                    continue
                
                sample = samples[idx]
                
                all_relations.append({
                    'relation_id': f'rel_{relation_id:06d}',
                    'head_entity_id': sample['head_entity_id'],
                    'head_entity_text': sample['head_entity'],
                    'head_entity_type': sample['head_entity_type'],
                    'tail_entity_id': sample['tail_entity_id'],
                    'tail_entity_text': sample['tail_entity'],
                    'tail_entity_type': sample['tail_entity_type'],
                    'relation_type': relation_type,
                    'confidence': float(conf),
                    'evidence': {
                        'text': sample['context'],
                        'source_qa_id': sample['source_qa_id']
                    }
                })
                
                relation_id += 1
    
    print(f"\nExtracted {len(all_relations)} relations")
    
    return all_relations

def compute_statistics(relations):
    """Compute statistics about extracted relations"""
    stats = {
        'total_relations': len(relations),
        'relation_type_counts': {},
        'avg_confidence': 0,
        'unique_head_entities': len(set(r['head_entity_id'] for r in relations)),
        'unique_tail_entities': len(set(r['tail_entity_id'] for r in relations)),
    }
    
    for rel in relations:
        rel_type = rel['relation_type']
        stats['relation_type_counts'][rel_type] = stats['relation_type_counts'].get(rel_type, 0) + 1
        stats['avg_confidence'] += rel['confidence']
    
    if relations:
        stats['avg_confidence'] /= len(relations)
    
    return stats

def run_extraction(confidence_threshold=0.7):
    """Main extraction function"""
    print("\n" + "="*60)
    print("EXTRACTING RELATIONS FROM KNOWLEDGE BASE")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    print("\nLoading trained model...")
    model, tokenizer, config = load_trained_model(device=device)
    print(f"Model loaded successfully!")
    print(f"Number of relation types: {len(RELATION_TYPES)}")
    
    print("\nLoading knowledge base...")
    kb = load_knowledge_base()
    print(f"KB records: {len(kb)}")
    
    print("\nLoading pre-extracted entities...")
    preextracted_entities = load_preextracted_entities()
    print(f"Entities loaded for {len(preextracted_entities)} QAs")
    
    print(f"\nExtracting relations (confidence threshold: {confidence_threshold})...")
    relations = extract_relations_from_kb(
        model,
        tokenizer,
        kb,
        preextracted_entities,
        device,
        confidence_threshold=confidence_threshold
    )
    
    os.makedirs(os.path.dirname(OUTPUT_PATHS['extracted_relations']), exist_ok=True)
    
    with open(OUTPUT_PATHS['extracted_relations'], 'w', encoding='utf-8') as f:
        json.dump(relations, f, ensure_ascii=False, indent=2)
    print(f"\nSaved relations to: {OUTPUT_PATHS['extracted_relations']}")
    
    stats = compute_statistics(relations)
    
    with open(OUTPUT_PATHS['relation_stats'], 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Saved statistics to: {OUTPUT_PATHS['relation_stats']}")
    
    print("\n" + "="*60)
    print("EXTRACTION STATISTICS")
    print("="*60)
    print(f"Total relations: {stats['total_relations']}")
    print(f"Unique head entities: {stats['unique_head_entities']}")
    print(f"Unique tail entities: {stats['unique_tail_entities']}")
    print(f"Average confidence: {stats['avg_confidence']:.4f}")
    print("\nRelation type distribution:")
    for rel_type, count in sorted(stats['relation_type_counts'].items(), key=lambda x: -x[1]):
        print(f"  {rel_type}: {count}")
    print("="*60)
    
    return relations, stats

if __name__ == '__main__':
    import sys
    
    confidence_threshold = 0.7
    if len(sys.argv) > 1:
        confidence_threshold = float(sys.argv[1])
        print(f"Using confidence threshold: {confidence_threshold}")
    
    run_extraction(confidence_threshold=confidence_threshold)

