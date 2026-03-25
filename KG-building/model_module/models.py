"""Model architectures: ViT5 Encoder-Decoder with optional KG Enhancement"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from transformers import T5ForConditionalGeneration, T5Config
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for Knowledge Graph
    
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
    """Cross-Attention to fuse KG embeddings into text"""
    
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


class KGEnhancedViT5(nn.Module):
    """
    ViT5 Encoder-Decoder with optional KG Enhancement
    
    Architecture:
        Base: Pre-trained ViT5 (T5ForConditionalGeneration)
        Enhancement: Optional GNN + Cross-Attention for KG fusion
    
    Args:
        use_kg: If True, adds KG enhancement layers on top of ViT5
    """
    
    def __init__(self,
                 vit5_model_name='VietAI/vit5-base',
                 kg_node_features=300,
                 gnn_hidden=256,
                 gnn_type='gcn',
                 gnn_layers=2,
                 dropout=0.1,
                 use_kg=True):
        super().__init__()

        self.use_kg = use_kg
        self.d_model = None  # Will be set from T5 config
        
        # Load pre-trained ViT5 model with memory-efficient settings
        print(f"Loading pre-trained ViT5: {vit5_model_name}")
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            vit5_model_name,
            low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
            torch_dtype=torch.float32  # Use float32 to save memory (can use float16 if needed)
        )
        
        # Get model dimensions from T5 config
        t5_config = self.t5_model.config
        self.d_model = t5_config.d_model
        
        # No need to resize vocab - ViT5 tokenizer matches ViT5 model vocab!
        print(f"✓ ViT5 loaded: d_model={self.d_model}, vocab_size={t5_config.vocab_size}")

        # KG components (only if use_kg=True)
        if use_kg:
            self.kg_encoder = GNNEncoder(
                in_channels=kg_node_features,
                hidden_channels=gnn_hidden,
                out_channels=self.d_model,
                gnn_type=gnn_type,
                num_layers=gnn_layers,
                dropout=dropout
            )
            
            # Cross-attention layers to fuse KG into T5 encoder output
            self.kg_cross_attn = CrossAttention(self.d_model, t5_config.num_heads, dropout)
            
            # Projection to match T5 encoder hidden size
            self.kg_projection = nn.Linear(self.d_model, self.d_model)
            
            print(f"✓ KG enhancement enabled (GNN: {gnn_type}, {gnn_layers} layers)")

    def encode_with_kg(self, input_ids, attention_mask, kg_node_features, kg_edge_index):
        """
        Encode input with optional KG enhancement
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            kg_node_features: (batch, num_nodes, kg_features)
            kg_edge_index: list of edge indices
        
        Returns:
            encoder_outputs: T5 encoder outputs with KG enhancement
        """
        # Get T5 encoder outputs
        encoder_outputs = self.t5_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # encoder_outputs.last_hidden_state: (batch, seq_len, d_model)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # KG Enhancement (if enabled and KG data is provided)
        if self.use_kg and kg_node_features is not None and kg_edge_index is not None:
            batch_size = input_ids.size(0)
            
            # Encode KG with GNN
            kg_embeddings_list = []
            for i in range(batch_size):
                node_feat = kg_node_features[i]
                
                if isinstance(kg_edge_index, list):
                    edge_idx = kg_edge_index[i]
                else:
                    edge_idx = kg_edge_index[i]
                
                # Ensure edge_idx is on the same device as node_feat
                edge_idx = edge_idx.to(node_feat.device)
                
                kg_emb = self.kg_encoder(node_feat, edge_idx)
                kg_embeddings_list.append(kg_emb)
            
            # Pad KG embeddings
            max_nodes = max([emb.size(0) for emb in kg_embeddings_list])
            kg_embeddings = torch.zeros(batch_size, max_nodes, self.d_model,
                                        device=input_ids.device)
            for i, emb in enumerate(kg_embeddings_list):
                kg_embeddings[i, :emb.size(0), :] = emb
            
            # Project KG embeddings
            kg_embeddings = self.kg_projection(kg_embeddings)
            
            # Cross-Attention: Fuse KG into encoder output
            enhanced_hidden_states = self.kg_cross_attn(encoder_hidden_states, kg_embeddings)
            
            # Update encoder outputs
            encoder_outputs.last_hidden_state = enhanced_hidden_states
        
        return encoder_outputs

    def forward(self, question_ids, context_ids, answer_ids,
                kg_node_features=None, kg_edge_index=None):
        """
        Full forward pass
        
        Args:
            question_ids: (batch, q_len)
            context_ids: (batch, c_len)
            answer_ids: (batch, a_len) - target sequence
            kg_node_features: (batch, num_nodes, kg_features) - optional
            kg_edge_index: list of edge indices - optional
            
        Returns:
            outputs: T5 model outputs with logits
        """
        # Combine question and context for T5 input
        # T5 format: "question: {question} context: {context}"
        # We'll concatenate them directly
        batch_size = question_ids.size(0)
        device = question_ids.device
        
        # Concatenate question and context
        # Add separator token (we'll use pad token as separator, or can add special token)
        max_q_len = question_ids.size(1)
        max_c_len = context_ids.size(1)
        
        # Create input_ids by concatenating question and context
        input_ids = torch.cat([question_ids, context_ids], dim=1)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        # Use pad_token_id from T5 config (after vocab resize, it should match)
        pad_token_id = self.t5_model.config.pad_token_id
        attention_mask = (input_ids != pad_token_id).long()
        
        # Create decoder input_ids (shift answer_ids for teacher forcing)
        decoder_input_ids = answer_ids[:, :-1]  # Remove last token
        decoder_input_ids = torch.cat([
            torch.full((batch_size, 1), self.t5_model.config.decoder_start_token_id, device=device),
            decoder_input_ids
        ], dim=1)
        
        # Encode with optional KG enhancement
        encoder_outputs = self.encode_with_kg(
            input_ids=input_ids,
            attention_mask=attention_mask,
            kg_node_features=kg_node_features,
            kg_edge_index=kg_edge_index
        )
        
        # Decode with T5 decoder
        decoder_outputs = self.t5_model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get logits from T5's LM head
        logits = self.t5_model.lm_head(decoder_outputs.last_hidden_state)
        
        return logits

    def generate(self, question_ids, context_ids, kg_node_features=None, kg_edge_index=None,
                 max_len=50, repetition_penalty=1.2, temperature=1.0, num_beams=1):
        """
        Autoregressive generation with optional KG enhancement
        
        Args:
            question_ids: (batch, q_len)
            context_ids: (batch, c_len)
            kg_node_features, kg_edge_index: optional KG data
            max_len: Maximum generation length
            repetition_penalty: Penalty for repeating tokens
            temperature: Sampling temperature
            num_beams: Beam search size (1 = greedy)
            
        Returns:
            generated_ids: (batch, gen_len)
        """
        self.eval()
        device = question_ids.device
        
        # Combine question and context
        input_ids = torch.cat([question_ids, context_ids], dim=1)
        attention_mask = (input_ids != self.t5_model.config.pad_token_id).long()
        
        # Encode with optional KG enhancement
        with torch.no_grad():
            encoder_outputs = self.encode_with_kg(
                input_ids=input_ids,
                attention_mask=attention_mask,
                kg_node_features=kg_node_features,
                kg_edge_index=kg_edge_index
            )
            
            # Use T5's built-in generate method
            generated_ids = self.t5_model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                max_length=max_len,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                num_beams=num_beams,
                do_sample=(temperature != 1.0 or num_beams > 1),
                pad_token_id=self.t5_model.config.pad_token_id,
                eos_token_id=self.t5_model.config.eos_token_id,
                decoder_start_token_id=self.t5_model.config.decoder_start_token_id
            )
        
        return generated_ids


# Alias for backward compatibility
KGEnhancedTransformer = KGEnhancedViT5
