import json
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import OUTPUT_PATHS, ID2RELATION, MODEL_CONFIG
from dataset import RelationExtractionDataset, load_tokenizer
from model import ViHealthBERTRelationExtractor
from data_preparation import (
    load_knowledge_base, 
    load_preextracted_entities,
    load_entity_id_mapping,
    get_kb_contexts,
    format_entities_for_relations
)
from itertools import combinations

def extract_relations_from_text_inference(model, tokenizer, raw_entities, text, qa_id, word_to_id_map, device, confidence_threshold, source_type):
    """Extract relations from a single text using model inference"""
    if not raw_entities or len(raw_entities) < 2 or not text:
        return []
    
    entities = format_entities_for_relations(raw_entities, text, word_to_id_map)
    
    if len(entities) < 2:
        return []
    
    samples = []
    
    for e1, e2 in combinations(entities, 2):
        distance = abs(e1['position'] - e2['position'])
        
        if distance > 100:
            continue
        
        samples.append({
            'head_entity': e1['text'],
            'head_entity_id': e1['entity_id'],
            'head_entity_type': e1['type'],
            'tail_entity': e2['text'],
            'tail_entity_id': e2['entity_id'],
            'tail_entity_type': e2['type'],
            'context': text,
            'relation': 'TREATED_BY',
            'confidence': 0.0,
            'source_qa_id': qa_id,
            'source_type': source_type,
        })
    
    if not samples:
        return []
    
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
    
    relations = []
    for idx, (pred, conf) in enumerate(zip(predictions, confidences)):
        if conf < confidence_threshold:
            continue
        
        relation_type = ID2RELATION[pred]
        sample = samples[idx]
        
        relations.append({
            'head_entity_id': sample['head_entity_id'],
            'head_entity_text': sample['head_entity'],
            'head_entity_type': sample['head_entity_type'],
            'tail_entity_id': sample['tail_entity_id'],
            'tail_entity_text': sample['tail_entity'],
            'tail_entity_type': sample['tail_entity_type'],
            'relation_type': relation_type,
            'confidence': float(conf),
            'source_type': source_type,
            'evidence': {
                'text': sample['context'],
                'source_qa_id': sample['source_qa_id']
            }
        })
    
    return relations

def extract_relations_from_kb(model, tokenizer, kb, preextracted_entities, word_to_id_map, device, confidence_threshold=0.7):
    """Extract relations from knowledge base using trained model (Q+A+C)"""
    print("\nExtracting relations from Question + Answer + Context...")
    
    model.eval()
    all_relations = []
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
        
        q_relations = extract_relations_from_text_inference(
            model, tokenizer, question_entities, question_text, qa_id,
            word_to_id_map, device, confidence_threshold, 'question'
        )
        stats['question'] += len(q_relations)
        all_relations.extend(q_relations)
        
        a_relations = extract_relations_from_text_inference(
            model, tokenizer, answer_entities, answer_text, qa_id,
            word_to_id_map, device, confidence_threshold, 'answer'
        )
        stats['answer'] += len(a_relations)
        all_relations.extend(a_relations)
        
        c_relations = extract_relations_from_text_inference(
            model, tokenizer, context_entities, context_text, qa_id,
            word_to_id_map, device, confidence_threshold, 'context'
        )
        stats['context'] += len(c_relations)
        all_relations.extend(c_relations)
    
    for idx, rel in enumerate(all_relations):
        rel['relation_id'] = f"rel_{idx:06d}"
    
    print(f"\nExtracted {len(all_relations)} relations")
    if all_relations:
        print(f"  From Question: {stats['question']} ({stats['question']/len(all_relations)*100:.1f}%)")
        print(f"  From Answer: {stats['answer']} ({stats['answer']/len(all_relations)*100:.1f}%)")
        print(f"  From Context: {stats['context']} ({stats['context']/len(all_relations)*100:.1f}%)")
    
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

def run_extraction():
    """Main extraction function"""
    print("\n" + "="*60)
    print("EXTRACTING RELATIONS")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    print("\nLoading model...")
    tokenizer = load_tokenizer()
    model = ViHealthBERTRelationExtractor()
    
    model_path = os.path.join(OUTPUT_PATHS['model_dir'], 'pytorch_model.bin')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"Loaded model from {model_path}")
    
    print("\nLoading data...")
    kb = load_knowledge_base()
    preextracted_entities = load_preextracted_entities()
    word_to_id_map = load_entity_id_mapping()
    
    relations = extract_relations_from_kb(
        model,
        tokenizer,
        kb,
        preextracted_entities,
        word_to_id_map,
        device,
        confidence_threshold=0.7
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
    run_extraction()
