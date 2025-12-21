import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from config import MODEL_CONFIG, OUTPUT_PATHS, ID2RELATION, RELATION_TYPES
from dataset import RelationExtractionDataset, load_tokenizer
from model import ViHealthBERTRelationExtractor

def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy, all_preds, all_labels

def plot_training_history(history, save_path):
    """Plot training history"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved training plot to: {save_path}")

def resume_training(start_epoch=1, additional_epochs=None):
    """
    Resume training from checkpoint
    
    Args:
        start_epoch: Epoch number already completed (1-indexed)
        additional_epochs: Number of additional epochs to train (default: remaining epochs)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    with open(OUTPUT_PATHS['train_split'], 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(OUTPUT_PATHS['val_split'], 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    
    # Load tokenizer
    tokenizer = load_tokenizer()
    
    # Create datasets
    train_dataset = RelationExtractionDataset(train_data, tokenizer, MODEL_CONFIG['max_length'])
    val_dataset = RelationExtractionDataset(val_data, tokenizer, MODEL_CONFIG['max_length'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Load model from checkpoint
    print("\nLoading model from checkpoint...")
    model = ViHealthBERTRelationExtractor(num_relations=len(RELATION_TYPES))
    
    checkpoint_path = os.path.join(OUTPUT_PATHS['model_dir'], 'pytorch_model.bin')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    
    print(f"Loaded model from: {checkpoint_path}")
    print(f"Already completed: {start_epoch} epoch(s)")
    
    # Calculate remaining epochs
    total_epochs = MODEL_CONFIG['num_epochs']
    if additional_epochs is not None:
        remaining_epochs = additional_epochs
        total_epochs = start_epoch + additional_epochs
    else:
        remaining_epochs = total_epochs - start_epoch
    
    if remaining_epochs <= 0:
        print(f"\nTraining already complete ({start_epoch}/{total_epochs} epochs)")
        print("Evaluating on test set...")
        
        with open(OUTPUT_PATHS['test_split'], 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        test_dataset = RelationExtractionDataset(test_data, tokenizer, MODEL_CONFIG['max_length'])
        test_loader = DataLoader(
            test_dataset,
            batch_size=MODEL_CONFIG['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, device)
        
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Acc: {test_acc:.4f}")
        
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(
            test_labels, 
            test_preds, 
            target_names=RELATION_TYPES,
            digits=4
        ))
        
        return
    
    print(f"Will train for {remaining_epochs} more epoch(s) ({start_epoch+1} -> {total_epochs})")
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=MODEL_CONFIG['learning_rate'])
    
    total_steps = len(train_loader) * remaining_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # Evaluate current model first
    print("\nEvaluating current checkpoint...")
    val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
    best_val_acc = val_acc
    print(f"Current Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    print("\n" + "="*60)
    print(f"RESUMING TRAINING FROM EPOCH {start_epoch + 1}")
    print("="*60)
    
    for epoch in range(remaining_epochs):
        current_epoch = start_epoch + epoch + 1
        
        print(f"\nEpoch {current_epoch}/{total_epochs}")
        print("-" * 60)
        
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Training Epoch {current_epoch}")
        for batch in train_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {current_epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(OUTPUT_PATHS['model_dir'], exist_ok=True)
            
            model_path = os.path.join(OUTPUT_PATHS['model_dir'], 'pytorch_model.bin')
            torch.save(model.state_dict(), model_path)
            
            tokenizer.save_pretrained(OUTPUT_PATHS['model_dir'])
            
            config_path = os.path.join(OUTPUT_PATHS['model_dir'], 'training_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'model_name': MODEL_CONFIG['model_name'],
                    'max_length': MODEL_CONFIG['max_length'],
                    'num_epochs': current_epoch,
                    'batch_size': MODEL_CONFIG['batch_size'],
                    'learning_rate': MODEL_CONFIG['learning_rate'],
                    'num_relations': len(RELATION_TYPES),
                    'relation_types': RELATION_TYPES,
                }, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Saved best model (Val Acc improved to {best_val_acc:.4f})")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"{'='*60}")
    
    plot_path = os.path.join(OUTPUT_PATHS['model_dir'], 'training_history.png')
    plot_training_history(history, plot_path)
    
    # Test evaluation
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    with open(OUTPUT_PATHS['test_split'], 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Test samples: {len(test_data)}")
    
    test_dataset = RelationExtractionDataset(test_data, tokenizer, MODEL_CONFIG['max_length'])
    test_loader = DataLoader(
        test_dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    best_model_path = os.path.join(OUTPUT_PATHS['model_dir'], 'pytorch_model.bin')
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model from: {best_model_path}")
    
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, device)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        test_labels, 
        test_preds, 
        target_names=RELATION_TYPES,
        digits=4
    ))
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(test_labels, test_preds)
    
    print("\nRelation types:")
    for i, rel_type in enumerate(RELATION_TYPES):
        print(f"  [{i}] {rel_type}")
    
    print("\nConfusion Matrix:")
    print("Rows = True labels, Columns = Predictions")
    print(cm)
    
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class Accuracy:")
    for i, (rel_type, acc) in enumerate(zip(RELATION_TYPES, per_class_acc)):
        print(f"  {rel_type}: {acc:.4f} ({int(cm[i,i])}/{int(cm.sum(axis=1)[i])})")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        start_epoch = int(sys.argv[1])
        additional_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else None
        resume_training(start_epoch, additional_epochs)
    else:
        print("Usage:")
        print("  python resume_training.py <completed_epochs> [additional_epochs]")
        print("\nExamples:")
        print("  python resume_training.py 1        # Resume from epoch 1, train remaining epochs")
        print("  python resume_training.py 1 2      # Resume from epoch 1, train 2 more epochs")
        print("  python resume_training.py 5        # Resume from epoch 5, train remaining epochs")


