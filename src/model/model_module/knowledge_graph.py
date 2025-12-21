"""Medical Knowledge Graph class"""

import torch
from collections import defaultdict


class MedicalKnowledgeGraph:
    """
    Knowledge Graph for medical domain
    
    Supports:
    - Loading from DataFrame
    - Finding neighbors of entities
    - Extracting subgraphs for multi-hop reasoning
    - Converting to PyTorch Geometric format
    """
    
    def __init__(self, triples_df, min_confidence=0.9):
        """
        Initialize Knowledge Graph
        
        Args:
            triples_df: DataFrame with columns [subject/START_ID, predicate/TYPE, object/END_ID]
            min_confidence: Minimum confidence threshold for filtering triples
        """
        self.nodes = []
        self.node_to_idx = {}
        self.edges = []
        self.relation_to_idx = {}
        self.idx_to_relation = {}
        self.entity_to_neighbors = defaultdict(list)
        
        self._build_from_dataframe(triples_df, min_confidence)
    
    def _build_from_dataframe(self, df, min_confidence):
        """Build KG from DataFrame"""
        # Filter by confidence if available
        if 'confidence' in df.columns:
            df = df[df['confidence'] >= min_confidence]
        
        # Build graph
        for _, row in df.iterrows():
            subject = str(row['subject'] if 'subject' in row else row[':START_ID']).strip()
            predicate = str(row['predicate'] if 'predicate' in row else row[':TYPE']).strip()
            obj = str(row['object'] if 'object' in row else row[':END_ID']).strip()
            
            self.add_node(subject)
            self.add_node(obj)
            self.add_edge(subject, predicate, obj)
    
    def add_node(self, node_name):
        """Add a node to the graph"""
        if node_name not in self.node_to_idx:
            idx = len(self.nodes)
            self.nodes.append(node_name)
            self.node_to_idx[node_name] = idx
    
    def add_edge(self, source, relation, target):
        """Add an edge (triple) to the graph"""
        if relation not in self.relation_to_idx:
            idx = len(self.relation_to_idx)
            self.relation_to_idx[relation] = idx
            self.idx_to_relation[idx] = relation
        
        self.edges.append((source, relation, target))
        self.entity_to_neighbors[source].append((relation, target))
    
    def get_neighbors(self, entity, relation_type=None):
        """
        Get neighbors of an entity
        
        Args:
            entity: Entity name
            relation_type: Optional filter by relation type
        
        Returns:
            list: List of (relation, target) tuples
        """
        neighbors = self.entity_to_neighbors.get(entity, [])
        if relation_type:
            neighbors = [(r, t) for r, t in neighbors if r == relation_type]
        return neighbors
    
    def get_subgraph(self, entities, max_hops=2):
        """
        Extract subgraph around given entities
        
        Args:
            entities: List of entity names
            max_hops: Maximum number of hops
        
        Returns:
            tuple: (list of entities in subgraph, list of edges in subgraph)
        """
        entities_set = set(entities)
        subgraph_entities = set(entities)
        subgraph_edges = []
        
        for hop in range(max_hops):
            new_entities = set()
            for entity in list(subgraph_entities):
                neighbors = self.get_neighbors(entity)
                for rel, target in neighbors:
                    new_entities.add(target)
                    subgraph_edges.append((entity, rel, target))
            subgraph_entities.update(new_entities)
        
        return list(subgraph_entities), subgraph_edges
    
    def get_pyg_data(self, entities=None, use_pretrained_emb=False, emb_dim=300):
        """
        Convert to PyTorch Geometric format
        
        Args:
            entities: List of entities to include (None = all)
            use_pretrained_emb: Whether to use pretrained embeddings
            emb_dim: Embedding dimension
        
        Returns:
            tuple: (node_features, edge_index, entity_list)
        """
        if entities is None:
            entities = self.nodes
            edges = self.edges
        else:
            entities, edges = self.get_subgraph(entities, max_hops=2)
        
        local_node_to_idx = {e: i for i, e in enumerate(entities)}
        num_nodes = len(entities)
        
        # Create node features
        if use_pretrained_emb:
            # TODO: Implement pretrained embeddings
            node_features = torch.randn(num_nodes, emb_dim)
        else:
            node_features = torch.randn(num_nodes, emb_dim)
        
        # Create edge index
        edge_list = []
        for source, relation, target in edges:
            if source in local_node_to_idx and target in local_node_to_idx:
                src_idx = local_node_to_idx[source]
                tgt_idx = local_node_to_idx[target]
                edge_list.append([src_idx, tgt_idx])
        
        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return node_features, edge_index, entities
    
    def __repr__(self):
        return f'MedicalKnowledgeGraph(nodes={len(self.nodes)}, edges={len(self.edges)}, relations={len(self.relation_to_idx)})'

