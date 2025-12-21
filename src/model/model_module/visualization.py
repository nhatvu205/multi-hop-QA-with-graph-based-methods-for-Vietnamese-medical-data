"""Visualization utilities for training history and metrics"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def plot_training_history(history):
    """
    Plot training and validation loss over epochs
    
    Args:
        history: Dict with 'train_loss' and 'val_loss' lists
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_metrics_history(history):
    """
    Plot accuracy and perplexity over epochs
    
    Args:
        history: Dict with 'train_acc', 'val_acc', 'train_ppl', 'val_ppl'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history.get('train_acc', [])) + 1)
    
    # Plot accuracy
    if 'train_acc' in history:
        ax1.plot(epochs, history['train_acc'], 'b-o', label='Training', linewidth=2)
    if 'val_acc' in history:
        ax1.plot(epochs, history['val_acc'], 'r-s', label='Validation', linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Token-Level Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot perplexity
    if 'train_ppl' in history:
        ax2.plot(epochs, history['train_ppl'], 'b-o', label='Training', linewidth=2)
    if 'val_ppl' in history:
        ax2.plot(epochs, history['val_ppl'], 'r-s', label='Validation', linewidth=2)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Perplexity', fontsize=12)
    ax2.set_title('Perplexity', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale for perplexity
    plt.tight_layout()
    plt.show()


def plot_qa_metrics_history(history):
    """
    Plot QA metrics (BLEU, ROUGE-L, BERTScore) over epochs
    Handles None values (epochs where metrics were not computed)
    
    Args:
        history: Dict with 'val_bleu', 'val_rouge_l', 'val_bertscore_f1'
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Filter out None values and get corresponding epochs
    def filter_none(values):
        """Return (epochs, values) with None entries removed"""
        filtered_epochs = []
        filtered_values = []
        for i, v in enumerate(values):
            if v is not None:
                filtered_epochs.append(i + 1)
                filtered_values.append(v)
        return filtered_epochs, filtered_values
    
    if 'val_bleu' in history and history['val_bleu']:
        epochs, values = filter_none(history['val_bleu'])
        if values:
            ax.plot(epochs, values, 'b-o', label='BLEU', linewidth=2, markersize=8)
    
    if 'val_rouge_l' in history and history['val_rouge_l']:
        epochs, values = filter_none(history['val_rouge_l'])
        if values:
            ax.plot(epochs, values, 'r-s', label='ROUGE-L', linewidth=2, markersize=8)
    
    if 'val_bertscore_f1' in history and history['val_bertscore_f1']:
        epochs, values = filter_none(history['val_bertscore_f1'])
        if values:
            ax.plot(epochs, values, 'g-^', label='BERTScore-F1', linewidth=2, markersize=8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('QA Metrics Over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_qa_metrics(metrics, title="QA Evaluation Metrics"):
    """
    Plot QA evaluation metrics (optimized for BLEU, ROUGE-L, BERTScore)
    
    Args:
        metrics: Dict with metric names and scores
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter to show only BLEU, ROUGE-L, BERTScore if present
    display_metrics = {}
    priority_keys = ['bleu', 'rouge_l', 'bertscore_f1']
    
    for key in priority_keys:
        if key in metrics:
            display_metrics[key] = metrics[key]
    
    # If none of the priority keys exist, show all metrics
    if not display_metrics:
        display_metrics = metrics
    
    metric_names = [k.replace('_', '-').upper() for k in display_metrics.keys()]
    scores = list(display_metrics.values())
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    bars = ax.bar(metric_names, scores, color=colors[:len(scores)], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(100, max(scores) * 1.1))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_complete_summary(history):
    """
    Create complete visualization with all training metrics
    
    Args:
        history: Training history dict with loss, acc, ppl, and QA metrics
    """
    # Check if QA metrics history exists
    has_qa_metrics = ('val_bleu' in history and len(history['val_bleu']) > 0)
    
    if has_qa_metrics:
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 5))
        gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training', linewidth=2, markersize=6)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    if 'train_acc' in history:
        ax2.plot(epochs, history['train_acc'], 'b-o', label='Training', linewidth=2, markersize=6)
    if 'val_acc' in history:
        ax2.plot(epochs, history['val_acc'], 'r-s', label='Validation', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Token-Level Accuracy', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Perplexity
    ax3 = fig.add_subplot(gs[0, 2])
    if 'train_ppl' in history:
        ax3.plot(epochs, history['train_ppl'], 'b-o', label='Training', linewidth=2, markersize=6)
    if 'val_ppl' in history:
        ax3.plot(epochs, history['val_ppl'], 'r-s', label='Validation', linewidth=2, markersize=6)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Perplexity', fontsize=11)
    ax3.set_title('Perplexity (log scale)', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Second row: QA metrics over epochs (filter None values)
    if has_qa_metrics:
        # Helper function to filter None values
        def filter_none(values):
            filtered_epochs = []
            filtered_values = []
            for i, v in enumerate(values):
                if v is not None:
                    filtered_epochs.append(i + 1)
                    filtered_values.append(v)
            return filtered_epochs, filtered_values
        
        # Plot 4: BLEU
        ax4 = fig.add_subplot(gs[1, 0])
        if 'val_bleu' in history and history['val_bleu']:
            bleu_epochs, bleu_values = filter_none(history['val_bleu'])
            if bleu_values:
                ax4.plot(bleu_epochs, bleu_values, 'b-o', label='BLEU', linewidth=2, markersize=6)
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('BLEU Score', fontsize=11)
        ax4.set_title('BLEU Score Over Epochs', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: ROUGE-L
        ax5 = fig.add_subplot(gs[1, 1])
        if 'val_rouge_l' in history and history['val_rouge_l']:
            rouge_epochs, rouge_values = filter_none(history['val_rouge_l'])
            if rouge_values:
                ax5.plot(rouge_epochs, rouge_values, 'r-s', label='ROUGE-L', linewidth=2, markersize=6)
        ax5.set_xlabel('Epoch', fontsize=11)
        ax5.set_ylabel('ROUGE-L Score', fontsize=11)
        ax5.set_title('ROUGE-L Score Over Epochs', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: BERTScore
        ax6 = fig.add_subplot(gs[1, 2])
        if 'val_bertscore_f1' in history and history['val_bertscore_f1']:
            bert_epochs, bert_values = filter_none(history['val_bertscore_f1'])
            if bert_values:
                ax6.plot(bert_epochs, bert_values, 'g-^', label='BERTScore-F1', linewidth=2, markersize=6)
        ax6.set_xlabel('Epoch', fontsize=11)
        ax6.set_ylabel('BERTScore F1', fontsize=11)
        ax6.set_title('BERTScore-F1 Over Epochs', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
    
    plt.show()

def print_metrics_table(metrics, title="Evaluation Metrics"):
    """
    Print metrics in a formatted table
    
    Args:
        metrics: Dict of metric names and scores
        title: Table title
    """
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    
    max_name_len = max(len(name) for name in metrics.keys())
    
    for name, score in metrics.items():
        if isinstance(score, float):
            print(f"{name:<{max_name_len}} : {score:>8.4f}")
        else:
            print(f"{name:<{max_name_len}} : {score:>8}")
    
    print("=" * 60)


# Test code
if __name__ == '__main__':
    print("Testing Visualization Functions")
    
    # Mock history data
    history = {
        'train_loss': [2.5, 2.0, 1.5, 1.2, 1.0],
        'val_loss': [2.7, 2.2, 1.8, 1.5, 1.3],
        'train_acc': [30, 45, 60, 70, 75],
        'val_acc': [28, 42, 55, 65, 70],
        'train_ppl': [12.2, 7.4, 4.5, 3.3, 2.7],
        'val_ppl': [14.9, 9.0, 6.0, 4.5, 3.7]
    }
    
    qa_metrics = {
        'exact_match': 45.5,
        'f1': 62.3,
        'bleu': 38.7,
        'rouge_l': 55.2
    }
    
    # Test plots
    plot_training_history(history)
    plot_metrics_history(history)
    plot_qa_metrics(qa_metrics)
    plot_complete_summary(history, qa_metrics)
    print_metrics_table(qa_metrics)

