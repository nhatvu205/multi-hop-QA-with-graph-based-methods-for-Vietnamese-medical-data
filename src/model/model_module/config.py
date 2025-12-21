"""Configuration settings for KG-Enhanced Transformer"""

import torch

class Config:
    """Configuration class for model training and inference"""
    
    def __init__(self):
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data paths
        self.kg_triples_path = 'data/knowledge_graph_triples.csv'
        self.neo4j_nodes_path = 'data/neo4j_nodes.csv'
        self.neo4j_rels_path = 'data/neo4j_relationships.csv'
        self.qa_dataset_path = 'data/qa_data.csv'
        
        # Knowledge Graph settings
        self.kg_min_confidence = 0.95
        self.kg_max_hops = 2
        self.kg_emb_dim = 300
        
        # Tokenizer settings
        self.tokenizer_type = 'vit5'  # Using ViT5 tokenizer
        self.vocab_size = None
        
        # ViT5 Tokenizer settings
        self.vit5_tokenizer_model = 'VietAI/vit5-base'  # ViT5 tokenizer model name
        self.use_vncorenlp = False  # Not used for ViT5 (kept for compatibility)
        self.vncorenlp_path = None  # Not used for ViT5 (kept for compatibility)
        
        # Model mode
        self.use_kg = True  # True: use KG enhancement, False: transformer only
        
        # ViT5 Pre-trained Model settings
        self.vit5_model_name = 'VietAI/vit5-base'  # Pre-trained ViT5 model
        # Options: 'VietAI/vit5-base', 'VietAI/vit5-large'
        
        # Model architecture (for KG components only, ViT5 has its own architecture)
        self.d_model = 768  # ViT5-base d_model (will be auto-detected from model)
        self.num_heads = 12  # ViT5-base num_heads (will be auto-detected)
        self.num_encoder_layers = 12  # ViT5-base encoder layers (will be auto-detected)
        self.num_decoder_layers = 12  # ViT5-base decoder layers (will be auto-detected)
        self.d_ff = 3072  # ViT5-base d_ff (will be auto-detected)
        self.gnn_hidden = 256
        self.gnn_type = 'gcn'  # 'gcn', 'gat', or 'sage'
        self.gnn_layers = 2
        self.dropout = 0.1
        
        # Training settings
        self.batch_size = 2  # Further reduced for max_seq_len=512
        self.gradient_accumulation_steps = 2  # Effective batch size = 2 * 2 = 4
        self.num_epochs = 5
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.grad_clip_norm = 1.0
        self.max_seq_len = 384  # Keep at 512 for optimal output
        
        # Memory optimization settings
        self.use_mixed_precision = True  # Use FP16/BF16 to reduce memory by ~50%
        self.use_gradient_checkpointing = True  # Trade compute for memory
        self.clear_cache_every_n_batches = 5  # Clear GPU cache more frequently
        
        # Data split
        self.train_ratio = 0.8
        
        # Scheduler
        self.scheduler_factor = 0.5
        self.scheduler_patience = 2
        
        # Early stopping
        self.early_stopping_patience = 2  # Stop if no improvement for N epochs
        self.early_stopping_min_delta = 0.0001  # Minimum change to qualify as improvement
        
        # Save paths
        self.best_model_path = 'best_model.pt'
        self.final_model_path = 'final_kg_enhanced_transformer.pt'
        self.test_predictions_path = 'test_predictions.csv'  # Path to save test set predictions
        
        # Evaluation settings
        self.bertscore_model = 'xlm-roberta-base'  # XLM-RoBERTa (good for Vietnamese)
        self.bertscore_rescale = True  # Note: Vietnamese baseline not available, will use raw scores
        self.eval_samples_per_epoch = None  # Evaluate on full val set (None = all)
        self.compute_qa_metrics_every_n_epochs = 1  # Compute BLEU/ROUGE/BERTScore every N epochs
        
        # Random seed
        self.seed = 42
    
    def __str__(self):
        config_str = "Model Configuration:\n"
        config_str += "=" * 50 + "\n"
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_str += f"{key:25s}: {value}\n"
        return config_str

