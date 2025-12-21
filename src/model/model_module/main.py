"""
Main script to run Transformer training and evaluation
Supports both KG-Enhanced and Transformer-only modes

Usage:
    python -m model_module.main --mode train --use_kg
    python -m model_module.main --mode train  # transformer-only
    python -m model_module.main --mode eval --use_kg
    python -m model_module.main --mode interactive
"""

import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import argparse

from .config import Config
from .data_loader import load_knowledge_graph_data, load_qa_dataset, print_qa_examples
from .knowledge_graph import MedicalKnowledgeGraph
from .vit5_tokenizer import ViT5Tokenizer
from .dataset import KGQADataset, collate_fn
from .models import KGEnhancedViT5
from .train import train_model
from .inference import evaluate_samples, interactive_qa, generate_answer
from .metrics import evaluate_qa_metrics
from .visualization import plot_complete_summary, plot_qa_metrics, print_metrics_table


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(mode='train', use_kg=None):
    """
    Main function
    
    Args:
        mode: 'train', 'eval', or 'interactive'
        use_kg: Override config.use_kg (True/False/None)
    """
    # Load config
    config = Config()
    
    # Override use_kg if specified
    if use_kg is not None:
        config.use_kg = use_kg
    
    print(config)
    
    # Set seed
    set_seed(config.seed)
    
    print(f'\n{"="*70}')
    print(f'Mode: {mode.upper()}')
    print(f'Model Type: {"KG-Enhanced Transformer" if config.use_kg else "Transformer-only"}')
    print(f'Device: {config.device}')
    print(f'{"="*70}\n')
    
    # =======================
    # STEP 1: Load Data
    # =======================
    print('STEP 1: Loading data...')
    print('-' * 70)
    
    # Load Knowledge Graph (only if use_kg=True)
    medical_kg = None
    if config.use_kg:
        print('Loading Knowledge Graph...')
        kg_triples, neo4j_nodes, neo4j_rels = load_knowledge_graph_data(
            config.kg_triples_path,
            config.neo4j_nodes_path,
            config.neo4j_rels_path
        )
        medical_kg = MedicalKnowledgeGraph(neo4j_rels, min_confidence=config.kg_min_confidence)
        print(f'{medical_kg}')
    else:
        print('Skipping Knowledge Graph (Transformer-only mode)')
    
    # Load QA dataset
    print('\nLoading QA dataset...')
    qa_dataset = load_qa_dataset(config.qa_dataset_path)
    print_qa_examples(qa_dataset, num_examples=3)
    
    # =======================
    # STEP 2: Tokenizer
    # =======================
    print('\nSTEP 2: Building tokenizer...')
    print('-' * 70)
    
    # Use ViT5 tokenizer (matches ViT5 model vocab)
    tokenizer = ViT5Tokenizer(
        model_name=config.vit5_tokenizer_model,
        max_len=config.max_seq_len,
        use_vncorenlp=False,  # Not used for ViT5
        vncorenlp_path=None   # Not used for ViT5
    )
    print(f"Using ViT5 tokenizer: {config.vit5_tokenizer_model}")
    print(f"Vocabulary size: {len(tokenizer.word_to_idx):,}")
    
    # =======================
    # STEP 3: Data Split
    # =======================
    print('\nSTEP 3: Splitting dataset...')
    print('-' * 70)
    
    train_size = int(config.train_ratio * len(qa_dataset))
    remaining = len(qa_dataset) - train_size
    val_size = int(0.5 * remaining)
    test_size = remaining - val_size
    
    train_qa = qa_dataset[:train_size]
    val_qa = qa_dataset[train_size:train_size + val_size]
    test_qa = qa_dataset[train_size + val_size:]
    
    # Create datasets
    train_dataset = KGQADataset(train_qa, medical_kg, tokenizer, config.kg_emb_dim, use_kg=config.use_kg)
    val_dataset = KGQADataset(val_qa, medical_kg, tokenizer, config.kg_emb_dim, use_kg=config.use_kg)
    test_dataset = KGQADataset(test_qa, medical_kg, tokenizer, config.kg_emb_dim, use_kg=config.use_kg)
    
    # Create dataloaders with memory-efficient settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        pin_memory=False,  # Disable pin_memory to save GPU memory
        num_workers=0,  # Use 0 workers to avoid memory issues
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        pin_memory=False,
        num_workers=0,
        persistent_workers=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        pin_memory=False,
        num_workers=0,
        persistent_workers=False
    )
    
    print(f'Train samples: {len(train_dataset):,}')
    print(f'Val samples:   {len(val_dataset):,}')
    print(f'Test samples:  {len(test_dataset):,}')
    
    # =======================
    # STEP 4: Model
    # =======================
    print('\nSTEP 4: Initializing ViT5 model...')
    print('-' * 70)
    
    model = KGEnhancedViT5(
        vit5_model_name=config.vit5_model_name,
        kg_node_features=config.kg_emb_dim,
        gnn_hidden=config.gnn_hidden,
        gnn_type=config.gnn_type,
        gnn_layers=config.gnn_layers,
        dropout=config.dropout,
        use_kg=config.use_kg
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'Model: {"KG-Enhanced " if config.use_kg else ""}Encoder-Decoder Transformer')
    print(f'Total parameters:      {total_params:,}')
    print(f'Trainable parameters:  {trainable_params:,}')
    
    # =======================
    # MODE: TRAIN
    # =======================
    if mode == 'train':
        print(f'\n{"="*70}')
        print('STEP 5: Training...')
        print(f'{"="*70}')
        print(f'Total epochs: {config.num_epochs}')
        print(f'QA Metrics (BLEU/ROUGE-L/BERTScore) will be computed every {config.compute_qa_metrics_every_n_epochs} epochs')
        print(f'Metrics will be computed on epochs: {", ".join([str(i) for i in range(config.compute_qa_metrics_every_n_epochs, config.num_epochs + 1, config.compute_qa_metrics_every_n_epochs)])}')
        print(f'{"-"*70}')
        
        best_val_loss, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            val_qa_samples=val_qa,
            tokenizer=tokenizer,
            config=config,
            kg=medical_kg
        )
        
        # Visualize training history
        print(f'\n{"="*70}')
        print('STEP 6: Visualizing training history...')
        print(f'{"="*70}')
        try:
            plot_complete_summary(history)
        except Exception as e:
            print(f'Warning: Visualization failed ({e})')
        
        # Note: Best model is already saved during training (in train_model function)
        if config.use_kg:
            print(f'\n✓ Training completed! Best model saved to {config.best_model_path}')
        else:
            print(f'\n✓ Training completed! [Model not saved: use_kg=False]')
        
        # Print final metrics summary
        print(f'\n{"="*70}')
        print('TRAINING SUMMARY')
        print(f'{"="*70}')
        print(f'Best validation loss: {best_val_loss:.4f}')
        
        # Get last non-None metrics
        if 'val_bleu' in history and len(history['val_bleu']) > 0:
            # Find last non-None values
            last_bleu = next((x for x in reversed(history['val_bleu']) if x is not None), None)
            last_rouge = next((x for x in reversed(history['val_rouge_l']) if x is not None), None)
            last_bert = next((x for x in reversed(history['val_bertscore_f1']) if x is not None), None)
            
            if last_bleu is not None:
                print(f'\nFinal QA Metrics (last computed epoch):')
                print(f'  BLEU:         {last_bleu:.2f}')
                print(f'  ROUGE-L:      {last_rouge:.2f}')
                print(f'  BERTScore-F1: {last_bert:.2f}')
    
    # =======================
    # MODE: EVAL
    # =======================
    elif mode == 'eval':
        print(f'\n{"="*70}')
        print('STEP 5: Evaluation on TEST SET...')
        print(f'{"="*70}')
        
        # Load best model (only if use_kg = True, otherwise model should be in memory from training)
        if config.use_kg:
            import os
            if not os.path.exists(config.best_model_path):
                print(f'\nERROR: Model file not found: {config.best_model_path}')
                print('Please run training mode first (mode="train")')
                return
            
            # Load checkpoint on CPU first to save GPU memory
            print(f'Loading checkpoint from {config.best_model_path}...')
            checkpoint = torch.load(config.best_model_path, map_location='cpu', weights_only=False)
            
            # Load only model weights (skip optimizer state dict to save memory)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            # Clear checkpoint from CPU memory
            del checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Move model to GPU after loading
            model = model.to(config.device)
            model.eval()
            print(f'✓ Loaded model from {config.best_model_path}')
        else:
            # If use_kg = False, model should be trained in the same session
            # If model is not trained, this will fail - user should run train mode first
            if not hasattr(model, 't5_model'):
                print('\nERROR: Model not found. Please run training mode first (mode="train")')
                return
            model = model.to(config.device)
            model.eval()
            print('✓ Using model from current session (use_kg=False, model not saved)')
        
        # Generate predictions on test set
        predictions = []
        ground_truths = []
        
        print(f'\nGenerating answers on {len(test_qa)} test samples...')
        from tqdm import tqdm
        for i, sample in enumerate(tqdm(test_qa, desc='Evaluating', leave=False, ncols=100)):
            try:
                pred = generate_answer(
                    model=model,
                    question=sample['question'],
                    context=sample['context'],
                    entities=sample.get('entities', []) if config.use_kg else [],
                    kg=medical_kg if config.use_kg else None,
                    tokenizer=tokenizer,
                    device=config.device,
                    use_kg=config.use_kg,
                    max_len=50
                )
                predictions.append(pred)
                ground_truths.append(sample['answers'][0])
            except Exception as e:
                print(f'\nWarning: Failed on sample {i}: {e}')
                continue
        
        if not predictions:
            print('\nERROR: No predictions generated!')
            return
        
        # Compute QA metrics (BLEU, ROUGE-L, BERTScore)
        print(f'\n{"="*70}')
        print('Computing QA Metrics (BLEU, ROUGE-L, BERTScore)...')
        print(f'{"="*70}')
        
        qa_metrics = evaluate_qa_metrics(
            predictions,
            ground_truths,
            compute_bertscore=True,
            bertscore_model=config.bertscore_model,
            bertscore_rescale=config.bertscore_rescale
        )
        
        # Display only BLEU, ROUGE-L, BERTScore
        test_metrics = {
            'BLEU': qa_metrics['bleu'],
            'ROUGE-L': qa_metrics['rouge_l'],
            'BERTScore-F1': qa_metrics.get('bertscore_f1', 0.0)
        }
        
        print_metrics_table(test_metrics, title="Test Set Performance")
        
        # Visualize
        try:
            plot_qa_metrics(test_metrics, title="Test Set QA Metrics")
        except Exception as e:
            print(f'Warning: Visualization failed ({e})')
        
        # Show sample predictions
        print(f'\n{"="*70}')
        print('SAMPLE PREDICTIONS')
        print(f'{"="*70}')
        evaluate_samples(
            model=model,
            samples=test_qa,
            kg=medical_kg,
            tokenizer=tokenizer,
            device=config.device,
            num_samples=5,
            use_kg=config.use_kg
        )
        
        # Save predictions to CSV file
        print(f'\n{"="*70}')
        print('Saving predictions to file...')
        print(f'{"="*70}')
        try:
            import pandas as pd
            import os
            
            # Prepare data for saving
            results_data = []
            for i, sample in enumerate(test_qa):
                if i < len(predictions):
                    result = {
                        'question': sample['question'],
                        'context': sample.get('context', ''),
                        'ground_truth': ground_truths[i],
                        'prediction': predictions[i],
                        'entities': ', '.join(sample.get('entities', [])) if config.use_kg else '',
                        'qa_id': sample.get('qa_id', f'sample_{i}')
                    }
                    results_data.append(result)
            
            # Create DataFrame
            results_df = pd.DataFrame(results_data)
            
            # Save to CSV
            output_path = getattr(config, 'test_predictions_path', 'test_predictions.csv')
            results_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f'✓ Predictions saved to: {output_path}')
            print(f'  Total samples: {len(results_df)}')
            print(f'  File size: {os.path.getsize(output_path) / 1024:.2f} KB')
            
            # Also save summary metrics
            summary_data = {
                'metric': ['BLEU', 'ROUGE-L', 'BERTScore-F1'],
                'score': [test_metrics['BLEU'], test_metrics['ROUGE-L'], test_metrics['BERTScore-F1']]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_path = output_path.replace('.csv', '_metrics.csv')
            summary_df.to_csv(summary_path, index=False, encoding='utf-8')
            print(f'✓ Metrics summary saved to: {summary_path}')
            
        except Exception as e:
            print(f'⚠ Warning: Failed to save predictions: {e}')
            import traceback
            traceback.print_exc()
    
    # =======================
    # MODE: INTERACTIVE
    # =======================
    elif mode == 'interactive':
        print(f'\n{"="*70}')
        print('STEP 5: Interactive QA...')
        print(f'{"="*70}')
        
        # Load best model (only if use_kg = True, otherwise model should be in memory from training)
        if config.use_kg:
            import os
            if not os.path.exists(config.best_model_path):
                print(f'\nERROR: Model file not found: {config.best_model_path}')
                print('Please run training mode first (mode="train")')
                return
            
            # Load checkpoint on CPU first to save GPU memory
            print(f'Loading checkpoint from {config.best_model_path}...')
            checkpoint = torch.load(config.best_model_path, map_location='cpu', weights_only=False)
            
            # Load only model weights (skip optimizer state dict to save memory)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            # Clear checkpoint from CPU memory
            del checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Move model to GPU after loading
            model = model.to(config.device)
            model.eval()
            print(f'✓ Loaded model from {config.best_model_path}')
        else:
            # If use_kg = False, model should be trained in the same session
            if not hasattr(model, 't5_model'):
                print('\nERROR: Model not found. Please run training mode first (mode="train")')
                return
            model = model.to(config.device)
            print('✓ Using model from current session (use_kg=False, model not saved)')
        
        interactive_qa(
            model=model,
            kg=medical_kg,
            tokenizer=tokenizer,
            device=config.device,
            use_kg=config.use_kg
        )
    
    print(f'\n{"="*70}')
    print('COMPLETED!')
    print(f'{"="*70}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Encoder-Decoder Transformer for Medical QA (with optional KG enhancement)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'eval', 'interactive'],
        help='Mode: train, eval, or interactive'
    )
    parser.add_argument(
        '--use_kg',
        action='store_true',
        help='Enable KG enhancement (default: use config.py setting)'
    )
    parser.add_argument(
        '--no_kg',
        action='store_true',
        help='Disable KG enhancement (transformer-only mode)'
    )
    
    args = parser.parse_args()
    
    # Determine use_kg
    if args.use_kg:
        use_kg = True
    elif args.no_kg:
        use_kg = False
    else:
        use_kg = None  # Use config default
    
    main(mode=args.mode, use_kg=use_kg)
