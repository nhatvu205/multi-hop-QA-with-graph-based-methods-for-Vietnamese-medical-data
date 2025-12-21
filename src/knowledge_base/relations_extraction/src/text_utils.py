from pyvi import ViTokenizer

def segment_text(text):
    """Segment Vietnamese text using pyvi"""
    if not text or not isinstance(text, str):
        return ""
    return ViTokenizer.tokenize(text)

def segment_batch(texts):
    """Segment batch of texts"""
    return [segment_text(text) for text in texts]

def extract_sentence_with_entities(text, entity1_pos, entity2_pos, window=50):
    """Extract sentence containing both entities with context window"""
    start_pos = max(0, min(entity1_pos, entity2_pos) - window)
    end_pos = min(len(text), max(entity1_pos, entity2_pos) + window)
    return text[start_pos:end_pos]

