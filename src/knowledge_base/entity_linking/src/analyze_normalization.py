import json
import os
from collections import Counter, defaultdict

def load_data():
    """Load original and normalized dictionaries"""
    print("Loading data...\n")
    
    # Load files
    with open('medical_entities_dict.json', 'r', encoding='utf-8') as f:
        original = json.load(f)
    
    with open('entity_linking/normalized_entity_dict.json', 'r', encoding='utf-8') as f:
        normalized = json.load(f)
    
    with open('entity_linking/entity_id_mapping.json', 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    return original, normalized, mapping


def analyze_normalization(original, normalized, mapping):
    """Compute statistics"""
    
    print("="*80)
    print("ENTITY NORMALIZATION ANALYSIS")
    print("="*80)
    
    # Basic stats
    print(f"\nBasic Statistics:")
    print(f"   • Original entities: {len(original)}")
    print(f"   • Normalized entities: {len(normalized)}")
    print(f"   • Reduction: {len(original) - len(normalized)} entities ({(1-len(normalized)/len(original))*100:.1f}%)")
    print(f"   • Mapping entries: {len(mapping)}")
    
    # Merged entities
    merged = [ent for ent in normalized.values() if len(ent.get('merged_from', [])) > 1]
    print(f"\nMerging Statistics:")
    print(f"   • Entities created by merging: {len(merged)} ({len(merged)/len(normalized)*100:.1f}%)")
    print(f"   • Original entities merged: {sum(len(e['merged_from']) for e in merged)}")
    
    if merged:
        merge_counts = [len(e['merged_from']) for e in merged]
        print(f"   • Avg entities per merge: {sum(merge_counts)/len(merge_counts):.1f}")
        print(f"   • Max entities merged: {max(merge_counts)}")
    
    # Entity type distribution
    print(f"\nEntity Type Distribution:")
    
    orig_types = Counter(e['entity_type'] for e in original.values())
    norm_types = Counter(e['entity_type'] for e in normalized.values())
    
    print(f"\n   {'Type':<25s} {'Original':>10s} {'Normalized':>10s} {'Reduction':>10s}")
    print(f"   {'-'*60}")
    
    for etype in sorted(orig_types.keys()):
        orig_count = orig_types[etype]
        norm_count = norm_types[etype]
        reduction = (1 - norm_count/orig_count)*100 if orig_count > 0 else 0
        print(f"   {etype:<25s} {orig_count:>10d} {norm_count:>10d} {reduction:>9.1f}%")
    
    # Top merged entities
    if merged:
        print(f"\nTop 10 Most Merged Entities:")
        top_merged = sorted(merged, key=lambda x: len(x['merged_from']), reverse=True)[:10]
        
        for i, ent in enumerate(top_merged, 1):
            print(f"\n   {i}. {ent['canonical_form']} ({ent['entity_type']})")
            print(f"      • Merged from: {len(ent['merged_from'])} entities")
            print(f"      • Aliases: {', '.join(ent['aliases'][:5])}")
            if len(ent['aliases']) > 5:
                print(f"        ... and {len(ent['aliases']) - 5} more")
            print(f"      • Total frequency: {ent['frequency']}")
    
    # Frequency analysis
    print(f"\nFrequency Analysis:")
    
    orig_freq = [e['frequency'] for e in original.values()]
    norm_freq = [e['frequency'] for e in normalized.values()]
    
    print(f"   Original:")
    print(f"      • Total mentions: {sum(orig_freq)}")
    print(f"      • Avg per entity: {sum(orig_freq)/len(orig_freq):.1f}")
    
    print(f"   Normalized:")
    print(f"      • Total mentions: {sum(norm_freq)}")
    print(f"      • Avg per entity: {sum(norm_freq)/len(norm_freq):.1f}")
    
    # Confidence analysis
    print(f"\nConfidence Scores:")
    
    orig_conf = [e['avg_confidence'] for e in original.values()]
    norm_conf = [e['avg_confidence'] for e in normalized.values()]
    
    print(f"   Original: mean={sum(orig_conf)/len(orig_conf):.3f}")
    print(f"   Normalized: mean={sum(norm_conf)/len(norm_conf):.3f}")
    
    # Examples of mappings
    print(f"\nExample Mappings (Aliases → Canonical):")
    
    # Group mapping by normalized ID
    id_to_texts = defaultdict(list)
    for text, norm_id in mapping.items():
        id_to_texts[norm_id].append(text)
    
    # Find entities with multiple mappings
    multi_mappings = [(norm_id, texts) for norm_id, texts in id_to_texts.items() if len(texts) > 2]
    multi_mappings = sorted(multi_mappings, key=lambda x: len(x[1]), reverse=True)[:5]
    
    for norm_id, texts in multi_mappings:
        canonical = normalized[norm_id]['canonical_form']
        print(f"\n   {norm_id} → {canonical}")
        for text in texts[:5]:
            if text != canonical.lower():
                print(f"      • {text}")
        if len(texts) > 5:
            print(f"      ... and {len(texts)-5} more variants")
    
    print("\n" + "="*80)


def save_statistics(original, normalized, mapping):
    """Save statistics to JSON"""
    
    merged = [ent for ent in normalized.values() if len(ent.get('merged_from', [])) > 1]
    
    stats = {
        'original_count': len(original),
        'normalized_count': len(normalized),
        'reduction_count': len(original) - len(normalized),
        'reduction_percentage': (1 - len(normalized)/len(original)) * 100,
        'merged_entities_count': len(merged),
        'total_merges': sum(len(e['merged_from']) for e in merged),
        'avg_merge_size': sum(len(e['merged_from']) for e in merged) / len(merged) if merged else 0,
        'entity_type_distribution': {
            etype: sum(1 for e in normalized.values() if e['entity_type'] == etype)
            for etype in set(e['entity_type'] for e in normalized.values())
        }
    }
    
if __name__ == "__main__":
    original, normalized, mapping = load_data()
    analyze_normalization(original, normalized, mapping)
    save_statistics(original, normalized, mapping)

