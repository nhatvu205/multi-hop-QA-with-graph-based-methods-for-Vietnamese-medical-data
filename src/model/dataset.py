"""PyTorch Dataset for KG-enhanced QA"""

import torch
from torch.utils.data import Dataset


class KGQADataset(Dataset):
    """
    Dataset for QA with optional Knowledge Graph enhancement
    
    Returns:
        dict with keys:
            - question_ids: Tokenized question
            - context_ids: Tokenized context
            - answer_ids: Tokenized answer
            - kg_node_features: Node features from KG (None if use_kg=False)
            - kg_edge_index: Edge connectivity from KG (None if use_kg=False)
            - answer_text: Original answer text
    """
    
    def __init__(self, qa_data, kg, tokenizer, kg_emb_dim=300, use_kg=True):
        """
        Initialize dataset
        
        Args:
            qa_data: List of QA dictionaries
            kg: MedicalKnowledgeGraph instance (can be None if use_kg=False)
            tokenizer: Tokenizer instance
            kg_emb_dim: KG embedding dimension
            use_kg: Whether to use KG enhancement
        """
        self.qa_data = qa_data
        self.kg = kg
        self.tokenizer = tokenizer
        self.kg_emb_dim = kg_emb_dim
        self.use_kg = use_kg
    
    def __len__(self):
        return len(self.qa_data)
    
    def __getitem__(self, idx):
        sample = self.qa_data[idx]
        
        # Tokenize question and context
        question_ids = torch.tensor(self.tokenizer.encode(sample['question']), dtype=torch.long)
        context_ids = torch.tensor(self.tokenizer.encode(sample['context']), dtype=torch.long)
        
        # Tokenize answer
        answer_text = sample['answers'][0] if sample['answers'] else '<UNK>'
        answer_ids = torch.tensor(self.tokenizer.encode(answer_text), dtype=torch.long)
        
        # Get KG subgraph (only if use_kg=True)
        if self.use_kg and self.kg is not None:
            entities = sample['entities']
            node_features, edge_index, subgraph_entities = self.kg.get_pyg_data(
                entities=entities, 
                emb_dim=self.kg_emb_dim
            )
        else:
            # Dummy KG data for transformer-only mode
            node_features = torch.zeros(1, self.kg_emb_dim)
            edge_index = torch.zeros(2, 0, dtype=torch.long)
        
        return {
            'question_ids': question_ids,
            'context_ids': context_ids,
            'answer_ids': answer_ids,
            'kg_node_features': node_features,
            'kg_edge_index': edge_index,
            'answer_text': answer_text
        }


def collate_fn(batch):
    """
    Collate function for DataLoader
    
    Handles variable-sized KG subgraphs by padding.
    Works for both KG-enhanced and transformer-only modes.
    
    Args:
        batch: List of samples from Dataset
    
    Returns:
        dict: Batched tensors
    """
    question_ids = torch.stack([item['question_ids'] for item in batch])
    context_ids = torch.stack([item['context_ids'] for item in batch])
    answer_ids = torch.stack([item['answer_ids'] for item in batch])
    
    # Pad KG node features to max_nodes in batch (works even for dummy KG data)
    max_nodes = max([item['kg_node_features'].size(0) for item in batch])
    kg_emb_dim = batch[0]['kg_node_features'].size(1)
    
    kg_node_features = torch.zeros(len(batch), max_nodes, kg_emb_dim)
    kg_edge_indices = []
    
    for i, item in enumerate(batch):
        num_nodes = item['kg_node_features'].size(0)
        kg_node_features[i, :num_nodes, :] = item['kg_node_features']
        kg_edge_indices.append(item['kg_edge_index'])
    
    answer_texts = [item['answer_text'] for item in batch]
    
    return {
        'question_ids': question_ids,
        'context_ids': context_ids,
        'answer_ids': answer_ids,
        'kg_node_features': kg_node_features,
        'kg_edge_index': kg_edge_indices,
        'answer_texts': answer_texts
    }

