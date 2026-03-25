import json
import os
from itertools import combinations
from tqdm import tqdm
from config import RELATION_SCHEMA, DISTANT_SUPERVISION_CONFIG, OUTPUT_PATHS, RELATION_TYPES
from data_preparation import (
    load_knowledge_base, 
    load_preextracted_entities,
    get_kb_contexts,
    format_entities_for_relations,
    load_entity_id_mapping
)
from text_utils import extract_sentence_with_entities
import random
from sklearn.model_selection import train_test_split
from collections import defaultdict

def get_relation_by_types(type1, type2):
    """Get relation type based on entity types"""
    for relation, schema in RELATION_SCHEMA.items():
        type_pairs = schema['type_pairs']
        if (type1, type2) in type_pairs:
            return relation, 0.6
        if (type2, type1) in type_pairs:
            return relation, 0.5
    return 'NO_RELATION', 0.3

def extract_relations_from_text(raw_entities, text, qa_id, kb_record, word_to_id_map, source_type):
    """Extract relations from a single text (question/answer/context)"""
    if not raw_entities or len(raw_entities) < 2 or not text:
        return []
    
    entities = format_entities_for_relations(raw_entities, text, word_to_id_map)
    
    if len(entities) < 2:
        return []
    
    relations = []
    max_distance = DISTANT_SUPERVISION_CONFIG['max_distance']
    min_confidence = DISTANT_SUPERVISION_CONFIG['min_confidence']
    
    for e1, e2 in combinations(entities, 2):
        distance = abs(e1['position'] - e2['position'])
        
        if distance > max_distance:
            continue
        
        relation, confidence = get_relation_by_types(e1['type'], e2['type'])
        
        if relation == 'NO_RELATION':
            continue
        
        if confidence < min_confidence:
            continue
        
        sentence = extract_sentence_with_entities(
            text,
            e1['position'],
            e2['position'],
            window=50
        )
        
        relations.append({
            'head_entity': e1['text'],
            'head_entity_type': e1['type'],
            'tail_entity': e2['text'],
            'tail_entity_type': e2['type'],
            'context': sentence,
            'relation': relation,
            'confidence': confidence,
            'source_qa_id': qa_id,
            'source_type': source_type,
            'context_title': kb_record.get('context_title', ''),
            'context_url': kb_record.get('context_url', ''),
        })
    
    return relations

def generate_training_data(kb, preextracted_entities, word_to_id_map=None, max_samples=None):
    """Generate training data using pre-extracted entities from Q+A+C"""
    print("\nGenerating training data from Question + Answer + Context...")
    
    training_data = []
    stats = {'question': 0, 'answer': 0, 'context': 0}
    
    for kb_record in tqdm(kb, desc="Processing KB records"):
        qa_id = kb_record.get('qa_id', '')
        entities_data = preextracted_entities.get(qa_id, {})
        
        question_text = kb_record.get('question', '')
        answer_text = kb_record.get('answer', '')
        context_text = entities_data.get('context_text', '') or kb_record.get('context_text', '')
        
        question_entities = entities_data.get('question', [])
        answer_entities = entities_data.get('answer', [])
        context_entities = entities_data.get('context', [])
        
        q_relations = extract_relations_from_text(
            question_entities, question_text, qa_id, kb_record, word_to_id_map, 'question'
        )
        stats['question'] += len(q_relations)
        training_data.extend(q_relations)
        
        a_relations = extract_relations_from_text(
            answer_entities, answer_text, qa_id, kb_record, word_to_id_map, 'answer'
        )
        stats['answer'] += len(a_relations)
        training_data.extend(a_relations)
        
        c_relations = extract_relations_from_text(
            context_entities, context_text, qa_id, kb_record, word_to_id_map, 'context'
        )
        stats['context'] += len(c_relations)
        training_data.extend(c_relations)
        
        if max_samples and len(training_data) >= max_samples:
            break
    
    print(f"\nGenerated {len(training_data)} training samples")
    if training_data:
        print(f"  From Question: {stats['question']} ({stats['question']/len(training_data)*100:.1f}%)")
        print(f"  From Answer: {stats['answer']} ({stats['answer']/len(training_data)*100:.1f}%)")
        print(f"  From Context: {stats['context']} ({stats['context']/len(training_data)*100:.1f}%)")
    
    relation_counts = {}
    for sample in training_data:
        rel = sample['relation']
        relation_counts[rel] = relation_counts.get(rel, 0) + 1
    
    print("\nRelation distribution:")
    for rel, count in sorted(relation_counts.items(), key=lambda x: -x[1]):
        print(f"  {rel}: {count}")
    
    return training_data

def split_data(training_data):
    """Split data with stratification by relation type"""
    print("\nSplitting data into train/val/test (stratified by relation type)...")
    
    data_by_relation = defaultdict(list)
    for sample in training_data:
        data_by_relation[sample['relation']].append(sample)
    
    train_data = []
    val_data = []
    test_data = []
    
    for relation_type, samples_for_type in data_by_relation.items():
        if len(samples_for_type) < 3:
            train_data.extend(samples_for_type)
            continue
        
        train_val, test = train_test_split(
            samples_for_type, 
            test_size=DISTANT_SUPERVISION_CONFIG['test_ratio'], 
            random_state=42
        )
        train, val = train_test_split(
            train_val, 
            test_size=DISTANT_SUPERVISION_CONFIG['val_ratio'] / (DISTANT_SUPERVISION_CONFIG['train_ratio'] + DISTANT_SUPERVISION_CONFIG['val_ratio']), 
            random_state=42
        )
        
        train_data.extend(train)
        val_data.extend(val)
        test_data.extend(test)
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    print(f"\nFinal split:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    train_rels = sorted(list(set(s['relation'] for s in train_data)))
    val_rels = sorted(list(set(s['relation'] for s in val_data)))
    test_rels = sorted(list(set(s['relation'] for s in test_data)))
    
    print("\nRelation types per split:")
    print(f"  Train: {len(train_rels)} types - {train_rels}")
    print(f"  Val: {len(val_rels)} types - {val_rels}")
    print(f"  Test: {len(test_rels)} types - {test_rels}")
    
    return train_data, val_data, test_data

def save_training_data(train_data, val_data, test_data):
    """Save training splits"""
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
    """Main distant supervision function"""
    kb = load_knowledge_base()
    preextracted_entities = load_preextracted_entities()
    word_to_id_map = load_entity_id_mapping()
    
    training_data = generate_training_data(kb, preextracted_entities, word_to_id_map)
    
    train_data, val_data, test_data = split_data(training_data)
    
    save_training_data(train_data, val_data, test_data)
    
    return train_data, val_data, test_data

if __name__ == '__main__':
    run_distant_supervision()
