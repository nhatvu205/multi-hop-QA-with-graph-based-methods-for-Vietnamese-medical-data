import json
import os
from config import DATA_PATHS

def load_knowledge_base():
    """Load knowledge base"""
    print(f"Loading KB from: {DATA_PATHS['kb']}")
    with open(DATA_PATHS['kb'], 'r', encoding='utf-8') as f:
        kb = json.load(f)
    print(f"Loaded {len(kb)} KB records")
    return kb

def load_normalized_entities():
    """Load normalized entity dictionary"""
    print(f"Loading normalized entities from: {DATA_PATHS['normalized_entities']}")
    with open(DATA_PATHS['normalized_entities'], 'r', encoding='utf-8') as f:
        entities = json.load(f)
    print(f"Loaded {len(entities)} normalized entities")
    return entities

def load_preextracted_entities():
    """Load pre-extracted entities from medical_entities_full.json"""
    entities_full_path = os.path.join(
        os.path.dirname(DATA_PATHS['kb']),
        '..',
        'entity_data',
        'original',
        'medical_entities_full.json'
    )
    entities_full_path = os.path.normpath(entities_full_path)
    
    print(f"Loading pre-extracted entities from: {entities_full_path}")
    
    with open(entities_full_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    entities_by_qa = {}
    for record in data:
        qa_id = record['qa_id']
        entities_by_qa[qa_id] = {
            'context': record['entities'].get('context', []),
            'question': record['entities'].get('question', []),
            'answer': record['entities'].get('answer', [])
        }
    
    print(f"Loaded entities for {len(entities_by_qa)} QAs")
    return entities_by_qa

def get_kb_contexts(kb_record):
    """Get all rank=1 contexts from a KB record"""
    contexts = []
    
    if 'contexts' in kb_record:
        for ctx in kb_record['contexts']:
            if ctx.get('rank', 1) == 1:
                contexts.append({
                    'text': ctx.get('text', ''),
                    'qa_id': kb_record.get('qa_id', ''),
                    'passage_id': ctx.get('passage_id', ''),
                })
    
    return contexts

def format_entities_for_relations(raw_entities, context_text):
    """Convert pre-extracted entities to relation format with positions"""
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
        
        formatted_entities.append({
            'text': word,
            'type': entity_type,
            'position': pos,
            'score': ent.get('score', 1.0)
        })
        
        seen_texts.add(word.lower())
    
    return formatted_entities