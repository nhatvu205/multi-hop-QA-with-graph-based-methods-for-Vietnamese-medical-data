"""Training and evaluation functions with QA metrics"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from .metrics import compute_accuracy, compute_perplexity, evaluate_qa_metrics


def train_epoch(model, dataloader, criterion, optimizer, device, use_kg=True, 
                use_mixed_precision=False, gradient_accumulation_steps=1, 
                clear_cache_every_n_batches=5, scaler=None):
    """
    Train for one epoch with memory optimizations
    
    Args:
        model: KGEnhancedTransformer
        dataloader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: torch.device
        use_kg: Whether to use KG enhancement
        use_mixed_precision: Use FP16/BF16 mixed precision
        gradient_accumulation_steps: Number of steps to accumulate gradients
        clear_cache_every_n_batches: Clear GPU cache every N batches
        scaler: GradScaler for mixed precision (required if use_mixed_precision=True)
    
    Returns:
        tuple: (avg_loss, avg_accuracy)
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    progress_bar = tqdm(dataloader, desc='Training', leave=False, ncols=100)
    
    for batch_idx, batch in enumerate(progress_bar):
        question_ids = batch['question_ids'].to(device, non_blocking=True)
        context_ids = batch['context_ids'].to(device, non_blocking=True)
        answer_ids = batch['answer_ids'].to(device, non_blocking=True)
        
        # KG data (may be dummy if use_kg=False)
        kg_node_features = batch['kg_node_features'].to(device, non_blocking=True) if use_kg else None
        kg_edge_index = [edge_idx.to(device, non_blocking=True) for edge_idx in batch['kg_edge_index']] if use_kg else None
        
        # Zero gradients only at the start of accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 1:
            optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        if use_mixed_precision and scaler is not None:
            with autocast():
                outputs = model(
                    question_ids=question_ids,
                    context_ids=context_ids,
                    answer_ids=answer_ids,  # Teacher forcing
                    kg_node_features=kg_node_features,
                    kg_edge_index=kg_edge_index
                )
                # Compute loss (outputs: batch, seq_len, vocab_size)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), answer_ids.reshape(-1))
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
        else:
            outputs = model(
                question_ids=question_ids,
                context_ids=context_ids,
                answer_ids=answer_ids,  # Teacher forcing
                kg_node_features=kg_node_features,
                kg_edge_index=kg_edge_index
            )
            # Compute loss (outputs: batch, seq_len, vocab_size)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), answer_ids.reshape(-1))
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
        
        # Compute accuracy
        predictions = torch.argmax(outputs, dim=-1)
        mask = (answer_ids != 0)  # Ignore PAD tokens
        correct = ((predictions == answer_ids) & mask).sum().item()
        total_correct += correct
        total_tokens += mask.sum().item()
        
        # Backward pass with mixed precision if enabled
        if use_mixed_precision and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Step optimizer only after accumulating gradients
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_mixed_precision and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # Accumulate loss (multiply back to get original scale)
        total_loss += loss.item() * gradient_accumulation_steps
        
        acc = 100 * correct / mask.sum().item() if mask.sum().item() > 0 else 0
        progress_bar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}', 'acc': f'{acc:.2f}%'})
        
        # Explicitly delete tensors to free memory
        del outputs, loss, predictions, mask, question_ids, context_ids, answer_ids
        if kg_node_features is not None:
            del kg_node_features
        if kg_edge_index is not None:
            del kg_edge_index
        
        # Clear GPU cache periodically to prevent OOM
        if torch.cuda.is_available() and (batch_idx + 1) % clear_cache_every_n_batches == 0:
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = 100 * total_correct / total_tokens if total_tokens > 0 else 0
    
    # Final cache clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return avg_loss, avg_acc


def evaluate(model, dataloader, criterion, device, use_kg=True):
    """
    Evaluate model
    
    Args:
        model: KGEnhancedTransformer
        dataloader: Validation DataLoader
        criterion: Loss function
        device: torch.device
        use_kg: Whether to use KG enhancement
    
    Returns:
        tuple: (avg_loss, avg_accuracy)
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False, ncols=100):
            question_ids = batch['question_ids'].to(device)
            context_ids = batch['context_ids'].to(device)
            answer_ids = batch['answer_ids'].to(device)
            
            kg_node_features = batch['kg_node_features'].to(device, non_blocking=True) if use_kg else None
            kg_edge_index = [edge_idx.to(device, non_blocking=True) for edge_idx in batch['kg_edge_index']] if use_kg else None
            
            outputs = model(
                question_ids=question_ids,
                context_ids=context_ids,
                answer_ids=answer_ids,
                kg_node_features=kg_node_features,
                kg_edge_index=kg_edge_index
            )
            
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), answer_ids.reshape(-1))
            total_loss += loss.item()
            
            # Compute accuracy
            predictions = torch.argmax(outputs, dim=-1)
            mask = (answer_ids != 0)
            correct = ((predictions == answer_ids) & mask).sum().item()
            total_correct += correct
            total_tokens += mask.sum().item()
            
            # Explicitly delete tensors to free memory
            del outputs, loss, predictions, mask, question_ids, context_ids, answer_ids
            if kg_node_features is not None:
                del kg_node_features
            if kg_edge_index is not None:
                del kg_edge_index
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = 100 * total_correct / total_tokens if total_tokens > 0 else 0
    
    # Clear cache after evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return avg_loss, avg_acc


def compute_qa_metrics_on_samples(model, qa_samples, tokenizer, device, use_kg=True, 
                                   kg=None, max_samples=None, max_gen_len=50):
    """
    Generate answers and compute QA metrics (BLEU, ROUGE-L, BERTScore)
    
    Args:
        model: Trained model
        qa_samples: List of QA dictionaries
        tokenizer: Tokenizer instance
        device: Device
        use_kg: Whether to use KG
        kg: KnowledgeGraph instance (required if use_kg=True)
        max_samples: Maximum samples to evaluate (None = all)
        max_gen_len: Maximum generation length
    
    Returns:
        dict: QA metrics (bleu, rouge_l, bertscore_f1)
    """
    from .inference import generate_answer  # Import here to avoid circular dependency
    
    model.eval()
    predictions = []
    ground_truths = []
    
    samples = qa_samples[:max_samples] if max_samples else qa_samples
    
    tqdm.write(f'\nGenerating answers for {len(samples)} samples...')
    for i, sample in enumerate(tqdm(samples, desc='Generating', leave=False, ncols=100)):
        try:
            # Generate answer
            pred = generate_answer(
                model=model,
                question=sample['question'],
                context=sample['context'],
                entities=sample.get('entities', []) if use_kg else [],
                kg=kg if use_kg else None,
                tokenizer=tokenizer,
                device=device,
                max_len=max_gen_len,
                use_kg=use_kg
            )
            predictions.append(pred)
            ground_truths.append(sample['answers'][0])
        except Exception as e:
            # Skip problematic samples
            continue
    
    if len(predictions) == 0:
        return {'bleu': 0.0, 'rouge_l': 0.0, 'bertscore_f1': 0.0}
    
    # Compute metrics (using BERTScore with XLM-RoBERTa - has baseline)
    metrics = evaluate_qa_metrics(
        predictions,
        ground_truths,
        compute_bertscore=True,
        bertscore_model='xlm-roberta-base',
        bertscore_rescale=True
    )
    
    return {
        'bleu': metrics['bleu'],
        'rouge_l': metrics['rouge_l'],
        'bertscore_f1': metrics.get('bertscore_f1', 0.0)
    }


def train_model(model, train_loader, val_loader, val_qa_samples, tokenizer, config, kg=None):
    """
    Complete training loop with QA metrics evaluation
    
    Args:
        model: KGEnhancedTransformer
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        val_qa_samples: Validation QA samples (for metrics)
        tokenizer: Tokenizer
        config: Config object
        kg: KnowledgeGraph (required if config.use_kg=True)
    
    Returns:
        tuple: (best_val_loss, training_history)
    """
    device = config.device
    model = model.to(device)
    use_kg = config.use_kg
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Enable gradient checkpointing if available (saves memory)
    use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', False)
    if use_gradient_checkpointing and hasattr(model, 't5_model'):
        if hasattr(model.t5_model, 'gradient_checkpointing_enable'):
            model.t5_model.gradient_checkpointing_enable()
            tqdm.write('✓ Gradient checkpointing enabled (saves memory, slower training)')
    
    # Mixed precision training setup
    use_mixed_precision = getattr(config, 'use_mixed_precision', False)
    scaler = None
    if use_mixed_precision and torch.cuda.is_available():
        scaler = GradScaler()
        tqdm.write('✓ Mixed precision training enabled (FP16/BF16)')
    
    # Gradient accumulation steps
    gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    clear_cache_every_n_batches = getattr(config, 'clear_cache_every_n_batches', 5)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD token
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config.scheduler_factor, 
        patience=config.scheduler_patience
    )
    
    best_val_loss = float('inf')
    best_epoch = 0
    early_stopping_counter = 0
    history = {
        'train_loss': [], 
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_ppl': [],
        'val_ppl': [],
        'val_bleu': [],
        'val_rouge_l': [],
        'val_bertscore_f1': []
    }
    
    tqdm.write(f'\n{"="*70}')
    tqdm.write(f'Starting training for {config.num_epochs} epochs...')
    tqdm.write(f'Mode: {"KG-Enhanced" if use_kg else "Transformer-only"}')
    tqdm.write(f'Batch size: {config.batch_size}, Gradient accumulation: {gradient_accumulation_steps} steps')
    tqdm.write(f'Effective batch size: {config.batch_size * gradient_accumulation_steps}')
    tqdm.write(f'Max sequence length: {config.max_seq_len}')
    if use_mixed_precision:
        tqdm.write(f'Mixed precision: Enabled (FP16/BF16)')
    if use_gradient_checkpointing:
        tqdm.write(f'Gradient checkpointing: Enabled')
    if hasattr(config, 'early_stopping_patience') and config.early_stopping_patience > 0:
        tqdm.write(f'Early stopping: patience={config.early_stopping_patience}, min_delta={getattr(config, "early_stopping_min_delta", 0.0001)}')
    tqdm.write(f'{"="*70}')
    
    for epoch in range(config.num_epochs):
        tqdm.write(f'\n{"="*70}')
        tqdm.write(f'Epoch {epoch+1}/{config.num_epochs}')
        tqdm.write(f'{"="*70}')
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, use_kg,
            use_mixed_precision=use_mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            clear_cache_every_n_batches=clear_cache_every_n_batches,
            scaler=scaler
        )
        train_ppl = compute_perplexity(train_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_ppl'].append(train_ppl)
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_kg)
        val_ppl = compute_perplexity(val_loss)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_ppl'].append(val_ppl)
        
        # Compute QA metrics (BLEU, ROUGE-L, BERTScore) every N epochs
        compute_metrics = (
            (epoch + 1) % config.compute_qa_metrics_every_n_epochs == 0 or
            (epoch + 1) == config.num_epochs  # Always compute on last epoch
        )
        
        if compute_metrics:
            tqdm.write(f'\n{"-"*70}')
            tqdm.write(f'Computing QA Metrics (BLEU, ROUGE-L, BERTScore) - Epoch {epoch+1}...')
            tqdm.write(f'{"-"*70}')
            
            qa_metrics = compute_qa_metrics_on_samples(
                model=model,
                qa_samples=val_qa_samples,
                tokenizer=tokenizer,
                device=device,
                use_kg=use_kg,
                kg=kg,
                max_samples=config.eval_samples_per_epoch,
                max_gen_len=50
            )
            
            history['val_bleu'].append(qa_metrics['bleu'])
            history['val_rouge_l'].append(qa_metrics['rouge_l'])
            history['val_bertscore_f1'].append(qa_metrics['bertscore_f1'])
        else:
            tqdm.write(f'\n{"-"*70}')
            tqdm.write(f'Skipping QA metrics (will compute on epoch {((epoch // config.compute_qa_metrics_every_n_epochs + 1) * config.compute_qa_metrics_every_n_epochs)})')
            tqdm.write(f'{"-"*70}')
            # Append None or previous value to maintain list length alignment
            # We'll use None and filter later for plotting
            history['val_bleu'].append(None)
            history['val_rouge_l'].append(None)
            history['val_bertscore_f1'].append(None)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Print all metrics
        tqdm.write(f'\n{"-"*70}')
        tqdm.write('EPOCH SUMMARY')
        tqdm.write(f'{"-"*70}')
        tqdm.write(f'Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | PPL: {train_ppl:.2f}')
        tqdm.write(f'Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | PPL: {val_ppl:.2f}')
        
        if compute_metrics:
            tqdm.write(f'QA Metrics:')
            tqdm.write(f'  BLEU:         {qa_metrics["bleu"]:.2f}')
            tqdm.write(f'  ROUGE-L:      {qa_metrics["rouge_l"]:.2f}')
            tqdm.write(f'  BERTScore-F1: {qa_metrics["bertscore_f1"]:.2f}')
        else:
            tqdm.write(f'QA Metrics: [Skipped this epoch]')
        
        tqdm.write(f'{"-"*70}')
        
        # Save best model and check for improvement
        min_delta = getattr(config, 'early_stopping_min_delta', 0.0001)
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            best_epoch = epoch + 1
            early_stopping_counter = 0
            # Only save model if use_kg = True
            if use_kg:
                # Prepare config dict for easy loading (especially for HuggingFace Space)
                config_dict = {
                    'use_kg': config.use_kg,
                    'vit5_model_name': config.vit5_model_name,
                    'vit5_tokenizer_model': config.vit5_tokenizer_model,
                    'kg_emb_dim': config.kg_emb_dim,
                    'gnn_hidden': config.gnn_hidden,
                    'gnn_type': config.gnn_type,
                    'gnn_layers': config.gnn_layers,
                    'dropout': config.dropout,
                    'max_seq_len': config.max_seq_len,
                    'tokenizer_type': config.tokenizer_type,
                }
                
                # Save best model with all necessary info for HuggingFace deployment
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config,  # Full config object (for backward compatibility)
                    'config_dict': config_dict,  # Dict version (easier to load on HuggingFace)
                    'use_kg': config.use_kg,  # Explicit flag for easy detection
                    'tokenizer_info': {
                        'model_name': config.vit5_tokenizer_model,
                        'max_len': config.max_seq_len,
                        'type': config.tokenizer_type
                    },
                    'model_architecture': {
                        'vit5_model_name': config.vit5_model_name,
                        'kg_node_features': config.kg_emb_dim,
                        'gnn_hidden': config.gnn_hidden,
                        'gnn_type': config.gnn_type,
                        'gnn_layers': config.gnn_layers,
                        'dropout': config.dropout,
                    },
                    'history': history
                }, config.best_model_path)
                tqdm.write(f'\n✓ Best model saved! (val_loss: {val_loss:.4f}, epoch: {best_epoch})')
                tqdm.write(f'  Saved to: {config.best_model_path}')
                tqdm.write(f'  Includes: model weights, config, tokenizer info, architecture params')
            else:
                tqdm.write(f'\n✓ Best model found! (val_loss: {val_loss:.4f}, epoch: {best_epoch}) [Not saved: use_kg=False]')
        else:
            early_stopping_counter += 1
            if hasattr(config, 'early_stopping_patience') and config.early_stopping_patience > 0:
                tqdm.write(f'  No improvement for {early_stopping_counter}/{config.early_stopping_patience} epochs')
        
        # Early stopping check
        if (hasattr(config, 'early_stopping_patience') and 
            config.early_stopping_patience > 0 and 
            early_stopping_counter >= config.early_stopping_patience):
            tqdm.write(f'\n{"="*70}')
            tqdm.write(f'Early stopping triggered!')
            tqdm.write(f'No improvement for {config.early_stopping_patience} epochs.')
            tqdm.write(f'Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}')
            tqdm.write(f'{"="*70}')
            break
    
    tqdm.write(f'\n{"="*70}')
    tqdm.write('Training completed!')
    tqdm.write(f'Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}')
    tqdm.write(f'Total epochs trained: {epoch + 1}/{config.num_epochs}')
    tqdm.write(f'{"="*70}')
    
    return best_val_loss, history
