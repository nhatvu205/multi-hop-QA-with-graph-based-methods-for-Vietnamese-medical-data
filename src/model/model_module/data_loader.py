"""Data loading utilities for Knowledge Graph and QA dataset"""

import sys
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_knowledge_graph_data(kg_triples_path, neo4j_nodes_path, neo4j_rels_path):
    """
    Load Knowledge Graph data from CSV files
    
    Args:
        kg_triples_path: Path to knowledge_graph_triples.csv
        neo4j_nodes_path: Path to neo4j_nodes.csv
        neo4j_rels_path: Path to neo4j_relationships.csv
    
    Returns:
        tuple: (kg_triples_df, neo4j_nodes_df, neo4j_rels_df)
    """
    print('Loading Knowledge Graph data...')
    
    kg_triples = pd.read_csv(kg_triples_path)
    print(f'KG Triples: {len(kg_triples):,} triples')
    
    neo4j_nodes = pd.read_csv(neo4j_nodes_path)
    print(f'Nodes: {len(neo4j_nodes):,} nodes')
    
    neo4j_rels = pd.read_csv(neo4j_rels_path)
    print(f'Relationships: {len(neo4j_rels):,} relationships')
    
    unique_entities = set(kg_triples['subject'].unique()) | set(kg_triples['object'].unique())
    unique_relations = kg_triples['predicate'].unique()
    
    print(f'\nStatistics:')
    print(f'  Unique entities: {len(unique_entities):,}')
    print(f'  Unique relations: {len(unique_relations):,}')
    
    return kg_triples, neo4j_nodes, neo4j_rels


def load_qa_dataset(qa_dataset_path):
    """
    Load and process QA dataset from CSV file
    
    Args:
        qa_dataset_path: Path to 6000_samples.csv
    
    Returns:
        list: List of QA samples with format:
            {
                'question': str,
                'context': str,
                'answers': list,
                'entities': list,
                'hops': int,
                'qa_id': str
            }
    """
    print(f'\nLoading real QA dataset from {qa_dataset_path}...')
    qa_df = pd.read_csv(qa_dataset_path)
    
    print(f'Dataset loaded: {len(qa_df):,} samples')
    print(f'Columns: {list(qa_df.columns)}')
    
    if 'is_multihop' in qa_df.columns:
        print(f'\nMulti-hop distribution:')
        print(qa_df['is_multihop'].value_counts())
    
    print('\nProcessing dataset...')
    qa_dataset = []
    
    for idx, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc='Processing QA', leave=False, ncols=100):
        try:
            # Parse entities
            entities_str = row['entities']
            if pd.isna(entities_str) or entities_str == '[]':
                entities = []
            else:
                try:
                    entities = ast.literal_eval(entities_str)
                    if not isinstance(entities, list):
                        entities = []
                except:
                    entities = str(entities_str).replace('[', '').replace(']', '').replace("'", "").split(',')
                    entities = [e.strip() for e in entities if e.strip()]
            
            if not entities or len(entities) == 0:
                continue
            
            # Get question, answer, context
            question = str(row['question']) if not pd.isna(row['question']) else ''
            answer = str(row['answer']) if not pd.isna(row['answer']) else ''
            
            # Load context_text - CRITICAL for model performance!
            if 'context_text' in row:
                context = str(row['context_text']) if not pd.isna(row['context_text']) else ''
            else:
                continue
            
            if not question or not answer or len(question) < 10:
                continue
            
            # IMPORTANT: Ensure context is not empty
            # If context is empty, use answer as context (better than nothing)
            if not context or len(context.strip()) == 0:
                context = answer[:300]  # Use answer as context if context is missing
                # This ensures model always has context to work with
            
            # Truncate long context (but keep it meaningful)
            if len(context) > 500:
                context = context[:500]
            
            # Determine if multi-hop
            is_multihop = row['is_multihop'] if 'is_multihop' in row and not pd.isna(row['is_multihop']) else False
            
            qa_sample = {
                'question': question,
                'context': context,  # Always use context (never empty now)
                'answers': [answer[:100]],
                'entities': entities[:15],
                'hops': 2 if is_multihop else 1,
                'qa_id': row['qa_id'] if 'qa_id' in row else f'qa_{idx}'
            }
            
            qa_dataset.append(qa_sample)
        
        except Exception as e:
            continue
    
    print(f'\nProcessed {len(qa_dataset):,} valid QA pairs')
    
    multi_hop_count = sum(1 for qa in qa_dataset if qa['hops'] > 1)
    single_hop_count = len(qa_dataset) - multi_hop_count
    print(f'  - Single-hop: {single_hop_count:,}')
    print(f'  - Multi-hop: {multi_hop_count:,}')
    
    return qa_dataset


def print_qa_examples(qa_dataset, num_examples=3):
    """Print example QA pairs"""
    print('\nExample QA pairs:')
    for i, sample in enumerate(qa_dataset[:num_examples]):
        print(f'\n--- Example {i+1} ({sample["hops"]}-hop) ---')
        print(f'ID: {sample["qa_id"]}')
        print(f'Question: {sample["question"][:100]}...')
        print(f'Context: {sample["context"][:100]}...' if sample["context"] else 'Context: [EMPTY]')
        print(f'Answer: {sample["answers"][0][:80]}...')
        print(f'Entities ({len(sample["entities"])}): {sample["entities"][:5]}')
        
        # Check if context is meaningful
        if not sample["context"] or len(sample["context"].strip()) < 10:
            print(f'  ⚠️  WARNING: Context is empty or too short!')

