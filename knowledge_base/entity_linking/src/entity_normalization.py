"""
Entity Normalization & Clustering
- Merge similar entities (aliases)
- Create canonical forms
- Assign normalized entity IDs
"""

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
import unicodedata


def normalize_text(text):
    """
    Normalize Vietnamese text for comparison
    """
    # Lowercase
    text = text.lower().strip()
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize Unicode
    text = unicodedata.normalize('NFC', text)
    
    return text


def compute_similarity(text1, text2):
    """
    Compute string similarity (0-1)
    """
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    return SequenceMatcher(None, norm1, norm2).ratio()


def are_entities_similar(entity1, entity2, similarity_threshold=0.85):
    """
    Check if two entities are similar enough to merge
    
    Rules:
    1. Must be same entity type
    2. String similarity > threshold
    3. OR one is substring of another (after normalization)
    """
    # Rule 1: Same type
    if entity1['entity_type'] != entity2['entity_type']:
        return False
    
    text1 = entity1['canonical_form']
    text2 = entity2['canonical_form']
    
    # Rule 2: String similarity
    similarity = compute_similarity(text1, text2)
    if similarity >= similarity_threshold:
        return True
    
    # Rule 3: Substring matching (for variants like "tiểu đường" vs "bệnh tiểu đường")
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    if norm1 in norm2 or norm2 in norm1:
        # Only merge if similarity is still reasonably high
        if similarity >= 0.7:
            return True
    
    return False


def cluster_entities(entity_dict, similarity_threshold=0.85):
    """
    Cluster similar entities together
    
    Returns:
        List of clusters, where each cluster is a list of entity IDs
    """
    entity_ids = list(entity_dict.keys())
    clusters = []
    processed = set()
    
    print(f"Clustering {len(entity_ids)} entities...")
    
    for i, entity_id in enumerate(entity_ids):
        if entity_id in processed:
            continue
        
        # Start new cluster
        cluster = [entity_id]
        processed.add(entity_id)
        
        entity1 = entity_dict[entity_id]
        
        # Find similar entities
        for other_id in entity_ids[i+1:]:
            if other_id in processed:
                continue
            
            entity2 = entity_dict[other_id]
            
            if are_entities_similar(entity1, entity2, similarity_threshold):
                cluster.append(other_id)
                processed.add(other_id)
        
        clusters.append(cluster)
        
        # Progress
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(entity_ids)} entities...")
    
    # Filter out single-entity clusters for reporting
    multi_entity_clusters = [c for c in clusters if len(c) > 1]
    
    print(f"\nClustering complete!")
    print(f"   • Total clusters: {len(clusters)}")
    print(f"   • Merged clusters: {len(multi_entity_clusters)}")
    print(f"   • Reduction: {len(entity_ids)} → {len(clusters)} entities")
    
    return clusters


def merge_cluster(cluster_ids, entity_dict):
    """
    Merge entities in a cluster into one canonical entity
    
    Strategy:
    - Canonical form: Most frequent variant
    - Aliases: All other variants
    - Frequency: Sum of all frequencies
    - Confidence: Weighted average
    """
    if len(cluster_ids) == 1:
        # No merging needed
        return entity_dict[cluster_ids[0]]
    
    # Collect all data
    all_forms = []
    all_aliases = []
    total_frequency = 0
    all_scores = []
    all_sources = []
    entity_type = entity_dict[cluster_ids[0]]['entity_type']
    
    for entity_id in cluster_ids:
        entity = entity_dict[entity_id]
        
        # Collect forms
        all_forms.append((entity['canonical_form'], entity['frequency']))
        all_aliases.extend(entity.get('aliases', []))
        
        # Aggregate stats
        total_frequency += entity['frequency']
        all_scores.append(entity['avg_confidence'] * entity['frequency'])  # Weighted
        all_sources.extend(entity.get('sources', []))
    
    # Choose canonical form (most frequent)
    canonical_form = max(all_forms, key=lambda x: x[1])[0]
    
    # Collect unique aliases (excluding canonical)
    unique_aliases = list(set([form for form, _ in all_forms] + all_aliases) - {canonical_form})
    
    # Compute weighted average confidence
    avg_confidence = sum(all_scores) / total_frequency if total_frequency > 0 else 0
    
    # Limit sources to top 10
    all_sources = sorted(all_sources, key=lambda x: x.get('score', 0), reverse=True)[:10]
    
    return {
        'canonical_form': canonical_form,
        'entity_type': entity_type,
        'aliases': unique_aliases,
        'frequency': total_frequency,
        'avg_confidence': avg_confidence,
        'sources': all_sources,
        'merged_from': cluster_ids if len(cluster_ids) > 1 else []
    }


def create_normalized_dictionary(entity_dict, similarity_threshold=0.85):
    """
    Create normalized entity dictionary with clustering
    """
    print("\n" + "="*80)
    print("ENTITY NORMALIZATION & CLUSTERING")
    print("="*80)
    
    print(f"\nInput:")
    print(f"   • Total entities: {len(entity_dict)}")
    
    # Cluster entities
    clusters = cluster_entities(entity_dict, similarity_threshold)
    
    # Merge clusters
    print(f"\nMerging clusters...")
    normalized_dict = {}
    entity_id = 1
    
    for cluster in clusters:
        merged_entity = merge_cluster(cluster, entity_dict)
        normalized_dict[f"ent_{entity_id:05d}"] = merged_entity
        entity_id += 1
    
    print(f"Merging complete!")
    
    # Statistics
    print(f"\nOutput:")
    print(f"   • Normalized entities: {len(normalized_dict)}")
    print(f"   • Reduction rate: {(1 - len(normalized_dict)/len(entity_dict))*100:.1f}%")
    
    # Show merged examples
    merged_entities = [
        (eid, ent) for eid, ent in normalized_dict.items() 
        if len(ent.get('merged_from', [])) > 1
    ]
    
    if merged_entities:
        print(f"\nExample merged entities:")
        for eid, ent in merged_entities[:5]:
            print(f"\n   {eid}: {ent['canonical_form']}")
            print(f"      Aliases: {', '.join(ent['aliases'][:3])}")
            print(f"      Merged from {len(ent['merged_from'])} entities")
    
    return normalized_dict


def save_normalized_dictionary(normalized_dict, output_file='normalized_entity_dict.json'):
    """
    Save normalized dictionary
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(normalized_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved: {output_file}")


def create_entity_mapping(original_dict, normalized_dict):
    """
    Create mapping from original entity IDs to normalized IDs
    
    For backward compatibility
    """
    mapping = {}
    
    for norm_id, norm_entity in normalized_dict.items():
        # Map canonical form
        canonical = norm_entity['canonical_form'].lower()
        mapping[canonical] = norm_id
        
        # Map aliases
        for alias in norm_entity.get('aliases', []):
            mapping[alias.lower()] = norm_id
    
    return mapping


if __name__ == "__main__":
    print("Starting entity normalization...\n")
    
    # Load original dictionary
    print("Loading original entity dictionary...")
    with open('../medical_entities_dict.json', 'r', encoding='utf-8') as f:
        original_dict = json.load(f)
    
    print(f"Loaded {len(original_dict)} entities\n")
    
    # Normalize
    normalized_dict = create_normalized_dictionary(
        original_dict,
        similarity_threshold=0.85
    )
    
    # Save
    save_normalized_dictionary(normalized_dict, 'normalized_entity_dict.json')
    
    # Create mapping
    mapping = create_entity_mapping(original_dict, normalized_dict)
    with open('entity_id_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    print(f"Saved: entity_id_mapping.json")
    print(f"   • Mapping entries: {len(mapping)}")
    print("ENTITY NORMALIZATION COMPLETE!")

