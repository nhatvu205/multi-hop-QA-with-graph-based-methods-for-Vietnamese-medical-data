import json
import os
from config import DATA_PATHS

def load_knowledge_base():
    """Load knowledge base từ Kaggle input"""
    print(f"Loading KB from: {DATA_PATHS['kb']}")
    with open(DATA_PATHS['kb'], 'r', encoding='utf-8') as f:
        kb = json.load(f)
    print(f"Loaded {len(kb)} KB records")
    return kb

def load_preextracted_entities():
    """Load pre-extracted entities từ Kaggle input"""
    entities_full_path = DATA_PATHS['entities_full']
    
    print(f"Loading pre-extracted entities from: {entities_full_path}")
    
    with open(entities_full_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    entities_by_qa = {}
    for record in data:
        qa_id = record['qa_id']
        entities_by_qa[qa_id] = {
            'context': record['entities'].get('context', []),
            'question': record['entities'].get('question', []),
            'answer': record['entities'].get('answer', []),
            'context_text': record.get('context_text', '')
        }
    
    print(f"Loaded entities for {len(entities_by_qa)} QAs")
    return entities_by_qa

def load_entity_id_mapping():
    """Load entity ID mapping từ medical_entities_dict.json"""
    dict_path = DATA_PATHS['original_entities']
    print(f"Loading entity ID mapping from: {dict_path}")
    
    with open(dict_path, 'r', encoding='utf-8') as f:
        entity_dict = json.load(f)
    
    word_to_id = {}
    for entity_id, entity_info in entity_dict.items():
        canonical = entity_info['canonical_form'].lower()
        entity_type = entity_info['entity_type']
        word_to_id[(canonical, entity_type)] = entity_id
        
        for alias in entity_info.get('aliases', []):
            word_to_id[(alias.lower(), entity_type)] = entity_id
    
    print(f"Loaded {len(word_to_id)} entity mappings")
    return word_to_id

def get_kb_contexts(kb_record):
    """Get contexts from KB record"""
    contexts = []
    if 'context_text' in kb_record and kb_record.get('context_text'):
        contexts.append({
            'text': kb_record['context_text'],
            'qa_id': kb_record.get('qa_id', ''),
            'passage_id': kb_record.get('passage_id', ''),
            'context_title': kb_record.get('context_title', '')
        })
    return contexts

def format_entities_for_relations(raw_entities, context_text, word_to_id_map=None):
    """Format entities with entity_id for relation extraction"""
    formatted_entities = []
    seen_texts = set()
    
    for ent in raw_entities:
        word = ent.get('word', '')
        entity_type = ent.get('type', 'UNKNOWN')
        
        if not word or len(word) < 2:
            continue
        
        if word.lower() in seen_texts:
            continue
        
        pos = context_text.lower().find(word.lower())
        if pos == -1:
            continue
        
        entity_id = None
        if word_to_id_map:
            entity_id = word_to_id_map.get((word.lower(), entity_type))
        
        if not entity_id:
            entity_id = f"temp_{hash(word.lower() + entity_type) % 1000000:06d}"
        
        formatted_entities.append({
            'text': word,
            'type': entity_type,
            'position': pos,
            'score': ent.get('score', 1.0),
            'entity_id': entity_id
        })
        
        seen_texts.add(word.lower())
    
    return formatted_entities
