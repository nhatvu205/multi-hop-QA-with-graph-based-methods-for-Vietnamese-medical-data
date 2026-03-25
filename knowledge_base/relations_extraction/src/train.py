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

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(dataloader), correct / total, all_preds, all_labels

def plot_training_history(history, save_path):
    """Plot and save training history"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.close()
    
    print(f"\nSaved training plot to: {save_path}")

def train_model():
    """Main training function"""
    print("\n" + "="*60)
    print("TRAINING RELATION EXTRACTION MODEL")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    print("\nLoading data...")
    with open(OUTPUT_PATHS['train_split'], 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(OUTPUT_PATHS['val_split'], 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    
    print("\nLoading tokenizer and model...")
    tokenizer = load_tokenizer()
    model = ViHealthBERTRelationExtractor()
    model.to(device)
    
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
    
    optimizer = AdamW(
        model.parameters(),
        lr=MODEL_CONFIG['learning_rate'],
        weight_decay=MODEL_CONFIG['weight_decay']
    )
    
    total_steps = len(train_loader) * MODEL_CONFIG['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=MODEL_CONFIG['warmup_steps'],
        num_training_steps=total_steps
    )
    
    print("\nStarting training...")
    print(f"Epochs: {MODEL_CONFIG['num_epochs']}")
    print(f"Batch size: {MODEL_CONFIG['batch_size']}")
    print(f"Learning rate: {MODEL_CONFIG['learning_rate']}")
    
    best_val_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(MODEL_CONFIG['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{MODEL_CONFIG['num_epochs']}")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
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
                    'num_epochs': MODEL_CONFIG['num_epochs'],
                    'batch_size': MODEL_CONFIG['batch_size'],
                    'learning_rate': MODEL_CONFIG['learning_rate'],
                    'num_relations': len(RELATION_TYPES),
                    'relation_types': RELATION_TYPES,
                }, f, indent=2, ensure_ascii=False)
            
            print(f"\nSaved best model to {model_path}")
            print(f"Saved tokenizer and config to {OUTPUT_PATHS['model_dir']}")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"{'='*60}")
    
    plot_path = os.path.join(OUTPUT_PATHS['model_dir'], 'training_history.png')
    plot_training_history(history, plot_path)
    
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    with open(OUTPUT_PATHS['test_split'], 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Test samples: {len(test_data)}")
    
    actual_relation_types = sorted(set(sample['relation'] for sample in test_data))
    print(f"Actual relation types in test data: {actual_relation_types}")
    
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
    
    unique_labels = sorted(set(test_labels))
    label_to_relation = {i: ID2RELATION[i] for i in unique_labels}
    target_names_actual = [label_to_relation[i] for i in unique_labels]
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        test_labels, 
        test_preds, 
        labels=unique_labels,
        target_names=target_names_actual,
        digits=4
    ))
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(test_labels, test_preds, labels=unique_labels)
    
    print("\nRelation types:")
    for i in unique_labels:
        print(f"  [{i}] {ID2RELATION[i]}")
    
    print("\nConfusion Matrix:")
    print("Rows = True labels, Columns = Predictions")
    print(cm)
    
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class Accuracy:")
    for idx, i in enumerate(unique_labels):
        rel_type = ID2RELATION[i]
        acc = per_class_acc[idx]
        print(f"  {rel_type}: {acc:.4f} ({int(cm[idx,idx])}/{int(cm.sum(axis=1)[idx])})")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    
    return model

if __name__ == '__main__':
    train_model()
