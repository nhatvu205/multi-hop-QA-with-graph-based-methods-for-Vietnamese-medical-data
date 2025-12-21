"""
Model Module for KG-Enhanced Transformer QA System
Supports both KG-enhanced and transformer-only modes
"""

from .config import Config
from .data_loader import load_knowledge_graph_data, load_qa_dataset, print_qa_examples
from .knowledge_graph import MedicalKnowledgeGraph
from .vit5_tokenizer import ViT5Tokenizer
from .dataset import KGQADataset, collate_fn
from .models import (
    KGEnhancedViT5,
    KGEnhancedTransformer,  # Alias for backward compatibility
    PositionalEncoding,
    GNNEncoder,
    CrossAttention
)
from .train import train_epoch, evaluate, train_model, compute_qa_metrics_on_samples
from .inference import generate_answer, predict_batch, interactive_qa, evaluate_samples
from .metrics import (
    evaluate_qa_metrics,
    exact_match_score,
    f1_score,
    bleu_score,
    rouge_l_score,
    bert_score,
    compute_accuracy,
    compute_perplexity
)
from .visualization import (
    plot_training_history,
    plot_metrics_history,
    plot_qa_metrics_history,
    plot_qa_metrics,
    plot_complete_summary,
    print_metrics_table
)

__all__ = [
    # Config
    'Config',
    
    # Data
    'load_knowledge_graph_data',
    'load_qa_dataset',
    'print_qa_examples',
    'MedicalKnowledgeGraph',
    
    # Tokenizer
    'ViT5Tokenizer',
    
    # Dataset
    'KGQADataset',
    'collate_fn',
    
    # Models
    'KGEnhancedViT5',
    'KGEnhancedTransformer',  # Alias for backward compatibility
    'PositionalEncoding',
    'GNNEncoder',
    'CrossAttention',
    
    # Training
    'train_epoch',
    'evaluate',
    'train_model',
    'compute_qa_metrics_on_samples',
    
    # Inference
    'generate_answer',
    'predict_batch',
    'interactive_qa',
    'evaluate_samples',
    
    # Metrics
    'evaluate_qa_metrics',
    'exact_match_score',
    'f1_score',
    'bleu_score',
    'rouge_l_score',
    'bert_score',
    'compute_accuracy',
    'compute_perplexity',
    
    # Visualization
    'plot_training_history',
    'plot_metrics_history',
    'plot_qa_metrics_history',
    'plot_qa_metrics',
    'plot_complete_summary',
    'print_metrics_table'
]
