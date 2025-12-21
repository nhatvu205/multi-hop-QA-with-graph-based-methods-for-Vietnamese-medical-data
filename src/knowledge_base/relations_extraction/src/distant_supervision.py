import json
import os
from itertools import combinations
from tqdm import tqdm
from config import RELATION_SCHEMA, DISTANT_SUPERVISION_CONFIG, OUTPUT_PATHS
from data_preparation import (
    load_knowledge_base, 
    load_preextracted_entities,
    get_kb_contexts,
    format_entities_for_relations
)
from text_utils import extract_sentence_with_entities
import random

def get_relation_by_types(type1, type2):
    """Get relation type based on entity type pair"""
    for relation, schema in RELATION_SCHEMA.items():
        type_pairs = schema['type_pairs']
        if (type1, type2) in type_pairs:
            return relation, 0.6
        if (type2, type1) in type_pairs:
            return relation, 0.5
    return 'NO_RELATION', 0.3

def generate_training_data(kb, preextracted_entities, max_samples=None):
    """Generate training data using pre-extracted entities"""
    print("\nGenerating training data using pre-extracted entities...")
    
    training_data = []
    max_distance = DISTANT_SUPERVISION_CONFIG['max_distance']
    min_confidence = DISTANT_SUPERVISION_CONFIG['min_confidence']
    
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
            
            for e1, e2 in combinations(entities, 2):
                distance = abs(e1['position'] - e2['position'])
                
                if distance > max_distance:
                    continue
                
                relation, confidence = get_relation_by_types(e1['type'], e2['type'])
                
                if confidence < min_confidence:
                    continue
                
                sentence = extract_sentence_with_entities(
                    context_text,
                    e1['position'],
                    e2['position'],
                    window=50
                )
                
                training_data.append({
                    'head_entity': e1['text'],
                    'head_entity_type': e1['type'],
                    'tail_entity': e2['text'],
                    'tail_entity_type': e2['type'],
                    'context': sentence,
                    'relation': relation,
                    'confidence': confidence,
                    'source_qa_id': context_obj['qa_id'],
                    'source_passage_id': context_obj['passage_id'],
                })
                
                if max_samples and len(training_data) >= max_samples:
                    break
            
            if max_samples and len(training_data) >= max_samples:
                break
        
        if max_samples and len(training_data) >= max_samples:
            break
    
    print(f"\nGenerated {len(training_data)} training samples")
    
    relation_counts = {}
    for sample in training_data:
        rel = sample['relation']
        relation_counts[rel] = relation_counts.get(rel, 0) + 1
    
    print("\nRelation distribution:")
    for rel, count in sorted(relation_counts.items(), key=lambda x: -x[1]):
        print(f"  {rel}: {count}")
    
    return training_data

def split_data(training_data):
    """Split data into train/val/test"""
    random.shuffle(training_data)
    
    total = len(training_data)
    train_size = int(total * DISTANT_SUPERVISION_CONFIG['train_ratio'])
    val_size = int(total * DISTANT_SUPERVISION_CONFIG['val_ratio'])
    
    train_data = training_data[:train_size]
    val_data = training_data[train_size:train_size + val_size]
    test_data = training_data[train_size + val_size:]
    
    print(f"\nData split:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")
    print(f"  Test: {len(test_data)}")
    
    return train_data, val_data, test_data

def save_training_data(train_data, val_data, test_data):
    """Save training data splits"""
    os.makedirs(os.path.dirname(OUTPUT_PATHS['train_split']), exist_ok=True)
    
    with open(OUTPUT_PATHS['train_split'], 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {OUTPUT_PATHS['train_split']}")
    
    with open(OUTPUT_PATHS['val_split'], 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {OUTPUT_PATHS['val_split']}")
    
    with open(OUTPUT_PATHS['test_split'], 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {OUTPUT_PATHS['test_split']}")

def run_distant_supervision():
    """Main function to run distant supervision"""
    kb = load_knowledge_base()
    preextracted_entities = load_preextracted_entities()
    
    training_data = generate_training_data(kb, preextracted_entities)
    
    train_data, val_data, test_data = split_data(training_data)
    
    save_training_data(train_data, val_data, test_data)
    
    return train_data, val_data, test_data

if __name__ == '__main__':
    run_distant_supervision()