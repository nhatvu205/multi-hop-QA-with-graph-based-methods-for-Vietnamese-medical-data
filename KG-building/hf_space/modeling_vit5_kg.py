"""
Custom ViT5 model with Knowledge Graph encoder for Hugging Face
This file should be uploaded to Hugging Face model repository
MATCHES model_module/models.py architecture
"""
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config, T5PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Optional, List, Tuple, Union
import torch.nn.functional as F
import math

try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.utils import to_dense_batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not available. KG encoder will be disabled.")


class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for Knowledge Graph
    MATCHES model_module/models.py GNNEncoder
    
    Supports: GCN, GAT, GraphSAGE
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels,
                 gnn_type='gcn', num_layers=2, dropout=0.1):
        super().__init__()
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        if gnn_type == 'gcn':
            self.convs.append(GCNConv(in_channels, hidden_channels))
        elif gnn_type == 'gat':
            self.convs.append(GATConv(in_channels, hidden_channels, heads=4, concat=True))
            hidden_channels = hidden_channels * 4
        elif gnn_type == 'sage':
            self.convs.append(SAGEConv(in_channels, hidden_channels))

        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(hidden_channels, hidden_channels // 4, heads=4, concat=True))
            elif gnn_type == 'sage':
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Output layer
        if gnn_type == 'gcn':
            self.convs.append(GCNConv(hidden_channels, out_channels))
        elif gnn_type == 'gat':
            self.convs.append(GATConv(hidden_channels, out_channels, heads=1, concat=False))
        elif gnn_type == 'sage':
            self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, batch=None):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x


class CrossAttention(nn.Module):
    """
    Cross-Attention to fuse KG embeddings into text
    MATCHES model_module/models.py CrossAttention
    """
    
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert d_model % num_heads == 0

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key_value, mask=None):
        batch_size = query.size(0)

        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key_value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(key_value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(context)

        output = self.layer_norm(query + self.dropout(output))

        return output


class KGEnhancedViT5(T5ForConditionalGeneration):
    """
    ViT5 model enhanced with Knowledge Graph encoder
    """
    
    def __init__(self, config: T5Config, kg_node_features=300, gnn_hidden=256, 
                 gnn_type='gcn', gnn_layers=2, dropout=0.1, use_kg=True):
        super().__init__(config)
        
        self.use_kg = use_kg
        
        # Initialize KG encoder if torch_geometric is available
        # MATCHES model_module/models.py architecture
        if TORCH_GEOMETRIC_AVAILABLE and use_kg:
            # Use GNNEncoder class (matches training)
            self.kg_encoder = GNNEncoder(
                in_channels=kg_node_features,  # 300
                hidden_channels=gnn_hidden,     # 256
                out_channels=config.d_model,    # 768
                gnn_type=gnn_type,              # 'gcn', 'gat', 'sage'
                num_layers=gnn_layers,          # 2
                dropout=dropout                 # 0.1
            )
            
            # Cross-Attention layer to fuse KG into text (CRITICAL - matches training!)
            self.kg_cross_attn = CrossAttention(config.d_model, config.num_heads, dropout)
            
            # Projection to match T5 encoder hidden size
            self.kg_projection = nn.Linear(config.d_model, config.d_model)
        else:
            self.kg_encoder = None
            self.kg_cross_attn = None
            self.kg_projection = None
    
    def encode_with_kg(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kg_node_features: Optional[torch.Tensor] = None,
        kg_edge_index: Optional[List[torch.Tensor]] = None,
        **kwargs
    ):
        """
        Encode input with Knowledge Graph enhancement
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            kg_node_features: KG node features [batch_size, num_nodes, emb_dim]
            kg_edge_index: List of edge indices for each sample in batch
        
        Returns:
            Encoder outputs
        """
        # Standard T5 encoding
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # KG Enhancement (if enabled and KG data is provided)
        # MATCHES model_module/models.py encode_with_kg logic
        if (self.use_kg and self.kg_encoder is not None and 
            kg_node_features is not None and kg_edge_index is not None and
            TORCH_GEOMETRIC_AVAILABLE):
            
            batch_size = input_ids.size(0)
            encoder_hidden_states = encoder_outputs.last_hidden_state
            
            # Encode KG with GNN (matches training)
            kg_embeddings_list = []
            for i in range(batch_size):
                node_feat = kg_node_features[i]  # [num_nodes, kg_emb_dim]
                
                if isinstance(kg_edge_index, list):
                    edge_idx = kg_edge_index[i]
                else:
                    edge_idx = kg_edge_index[i]
                
                # Ensure edge_idx is on the same device as node_feat
                edge_idx = edge_idx.to(node_feat.device)
                
                # Use GNNEncoder (matches training)
                kg_emb = self.kg_encoder(node_feat, edge_idx)
                kg_embeddings_list.append(kg_emb)
            
            # Pad KG embeddings to max_nodes in batch (matches training)
            max_nodes = max([emb.size(0) for emb in kg_embeddings_list])
            kg_embeddings = torch.zeros(batch_size, max_nodes, self.config.d_model,
                                        device=input_ids.device)
            for i, emb in enumerate(kg_embeddings_list):
                kg_embeddings[i, :emb.size(0), :] = emb
            
            # Project KG embeddings
            kg_embeddings = self.kg_projection(kg_embeddings)
            
            # Cross-Attention: Fuse KG into encoder output (CRITICAL - matches training!)
            enhanced_hidden_states = self.kg_cross_attn(encoder_hidden_states, kg_embeddings)
            
            # Update encoder outputs
            encoder_outputs.last_hidden_state = enhanced_hidden_states
        
        return encoder_outputs
    
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kg_node_features: Optional[torch.Tensor] = None,
        kg_edge_index: Optional[List[torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_len: Optional[int] = None,
        repetition_penalty: float = 1.2,
        temperature: float = 1.0,
        num_beams: int = 1,
        **kwargs
    ):
        """
        Generate with KG enhancement
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            kg_node_features: KG node features
            kg_edge_index: KG edge indices
            max_length/max_len: Maximum generation length
            repetition_penalty: Repetition penalty
            temperature: Sampling temperature
            num_beams: Beam search size
            **kwargs: Other generation parameters
        """
        # Use max_len if provided, otherwise max_length
        max_gen_len = max_len if max_len is not None else max_length
        if max_gen_len is None:
            max_gen_len = 50  # Default
        
        # Encode with KG
        encoder_outputs = self.encode_with_kg(
            input_ids=input_ids,
            attention_mask=attention_mask,
            kg_node_features=kg_node_features,
            kg_edge_index=kg_edge_index
        )
        
        # Remove do_sample from kwargs if present to avoid conflict
        # We'll set it explicitly based on temperature and num_beams
        kwargs_clean = {k: v for k, v in kwargs.items() if k != 'do_sample'}
        
        # Determine do_sample value (only sample if temperature != 1.0 or beams > 1)
        do_sample_value = (temperature != 1.0 or num_beams > 1)
        
        # Generate using T5's generate method
        generated_ids = super().generate(
            inputs=None,  # We provide encoder_outputs instead
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_length=max_gen_len,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            num_beams=num_beams,
            do_sample=do_sample_value,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            decoder_start_token_id=self.config.decoder_start_token_id,
            **kwargs_clean
        )
        
        return generated_ids
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load model from pretrained path
        """
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs
        )