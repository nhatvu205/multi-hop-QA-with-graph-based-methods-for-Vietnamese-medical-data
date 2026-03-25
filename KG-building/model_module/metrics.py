"""Evaluation metrics for QA tasks"""

import torch
import numpy as np
from collections import Counter
import string

# BERTScore
try:
    from bert_score import score as bertscore_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("Warning: bert-score not installed. BERTScore will not be available.")


def normalize_answer(s):
    """Normalize answer text for comparison"""
    def remove_punctuation(text):
        return ''.join(ch if ch not in string.punctuation else ' ' for ch in text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    return white_space_fix(remove_punctuation(s.lower()))


def exact_match_score(prediction, ground_truth):
    """
    Exact Match (EM) score
    
    Args:
        prediction: Predicted answer string
        ground_truth: True answer string
    
    Returns:
        float: 1.0 if exact match, 0.0 otherwise
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    """
    Token-level F1 score
    
    Args:
        prediction: Predicted answer string
        ground_truth: True answer string
    
    Returns:
        float: F1 score
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(len(pred_tokens) == len(truth_tokens))
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def bleu_score(prediction, ground_truth, n=2):
    """
    Simple BLEU score (up to n-grams)
    
    Args:
        prediction: Predicted answer string
        ground_truth: True answer string
        n: Maximum n-gram size
    
    Returns:
        float: BLEU score
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0
    
    scores = []
    for i in range(1, min(n + 1, len(pred_tokens) + 1)):
        pred_ngrams = [tuple(pred_tokens[j:j+i]) for j in range(len(pred_tokens) - i + 1)]
        truth_ngrams = [tuple(truth_tokens[j:j+i]) for j in range(len(truth_tokens) - i + 1)]
        
        if len(pred_ngrams) == 0 or len(truth_ngrams) == 0:
            scores.append(0.0)
            continue
        
        common = Counter(pred_ngrams) & Counter(truth_ngrams)
        num_same = sum(common.values())
        
        precision = num_same / len(pred_ngrams) if len(pred_ngrams) > 0 else 0
        scores.append(precision)
    
    if not scores or all(s == 0 for s in scores):
        return 0.0
    
    # Geometric mean
    bleu = np.exp(np.mean([np.log(s) if s > 0 else -float('inf') for s in scores]))
    return bleu if not np.isnan(bleu) and not np.isinf(bleu) else 0.0


def rouge_l_score(prediction, ground_truth):
    """
    ROUGE-L score (Longest Common Subsequence)
    
    Args:
        prediction: Predicted answer string
        ground_truth: True answer string
    
    Returns:
        float: ROUGE-L F1 score
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0
    
    # LCS using dynamic programming
    m, n = len(pred_tokens), len(truth_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == truth_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    
    if lcs_length == 0:
        return 0.0
    
    precision = lcs_length / len(pred_tokens)
    recall = lcs_length / len(truth_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_accuracy(logits, targets, pad_idx=0):
    """
    Compute token-level accuracy
    
    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len)
        pad_idx: Padding token index to ignore
    
    Returns:
        float: Accuracy
    """
    predictions = torch.argmax(logits, dim=-1)
    mask = (targets != pad_idx)
    
    correct = ((predictions == targets) & mask).sum().item()
    total = mask.sum().item()
    
    return correct / total if total > 0 else 0.0


def compute_perplexity(loss):
    """
    Compute perplexity from cross-entropy loss
    
    Args:
        loss: Cross-entropy loss value
    
    Returns:
        float: Perplexity
    """
    return np.exp(min(loss, 100))  # Cap to avoid overflow


def bert_score(predictions, ground_truths, model_type='xlm-roberta-base', rescale_with_baseline=True):
    """
    Compute BERTScore
    
    Args:
        predictions: List of predicted answer strings
        ground_truths: List of true answer strings
        model_type: BERTScore model ('xlm-roberta-base', 'bert-base-multilingual-cased', etc.)
        rescale_with_baseline: Whether to rescale with baseline
    
    Returns:
        dict: {'precision': float, 'recall': float, 'f1': float} (all in percentage)
    """
    if not BERTSCORE_AVAILABLE:
        print("Warning: BERTScore not available, returning zeros")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    if len(predictions) == 0 or len(ground_truths) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    try:
        # IMPORTANT: Vietnamese ('vi') KHÔNG có baseline file trong bert-score library
        # Baseline chỉ có cho: en, zh, tr, de, ru, fr, ar, es, it, ja, ko, nl, pl, pt
        # Xem: https://github.com/Tiiiger/bert_score/tree/master/bert_score/rescale_baseline
        
        # Solution: Không dùng lang='vi' và tắt rescaling
        if rescale_with_baseline:
            print("Note: Vietnamese baseline not available in bert-score, rescaling disabled")
            print("      Scores will be raw values (~0.8-1.0), scaled to 0-100 for display")
        
        P, R, F1 = bertscore_score(
            predictions,
            ground_truths,
            model_type=model_type,
            num_layers=9,
            verbose=False,
            rescale_with_baseline=False,  # Must be False - no Vietnamese baseline
            lang=None  # Don't specify language to avoid baseline lookup
        )
        
        # Raw BERTScore typically in range [0.8, 1.0]
        # Scale to 0-100 for consistency: multiply by 100
        # Example: 0.85 → 85, 0.92 → 92
        return {
            'precision': P.mean().item() * 100,
            'recall': R.mean().item() * 100,
            'f1': F1.mean().item() * 100
        }
    except Exception as e:
        print(f"Warning: BERTScore computation failed: {e}")
        import traceback
        traceback.print_exc()
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}


def evaluate_qa_metrics(predictions, ground_truths, compute_bertscore=False, 
                        bertscore_model='xlm-roberta-base', bertscore_rescale=True):
    """
    Compute all QA metrics for a batch
    
    Args:
        predictions: List of predicted answer strings
        ground_truths: List of true answer strings
        compute_bertscore: Whether to compute BERTScore (slower)
        bertscore_model: BERTScore model type
        bertscore_rescale: Whether to rescale BERTScore
    
    Returns:
        dict: Dictionary of metric scores
    """
    assert len(predictions) == len(ground_truths), "Mismatch in predictions and ground truths length"
    
    em_scores = []
    f1_scores = []
    bleu_scores = []
    rouge_scores = []
    
    for pred, truth in zip(predictions, ground_truths):
        em_scores.append(exact_match_score(pred, truth))
        f1_scores.append(f1_score(pred, truth))
        bleu_scores.append(bleu_score(pred, truth))
        rouge_scores.append(rouge_l_score(pred, truth))
    
    metrics = {
        'exact_match': np.mean(em_scores) * 100,  # Percentage
        'f1': np.mean(f1_scores) * 100,
        'bleu': np.mean(bleu_scores) * 100,
        'rouge_l': np.mean(rouge_scores) * 100
    }
    
    # Add BERTScore if requested
    if compute_bertscore:
        bs_metrics = bert_score(predictions, ground_truths, bertscore_model, bertscore_rescale)
        metrics['bertscore_p'] = bs_metrics['precision']
        metrics['bertscore_r'] = bs_metrics['recall']
        metrics['bertscore_f1'] = bs_metrics['f1']
    
    return metrics


# Test code
if __name__ == '__main__':
    print("Testing QA Metrics")
    print("=" * 60)
    
    # Test cases
    pred = "Triệu chứng của bệnh viêm gan B là sốt và mệt mỏi"
    truth = "Triệu chứng bệnh viêm gan B gồm sốt, mệt mỏi"
    
    print(f"Prediction: {pred}")
    print(f"Ground Truth: {truth}")
    print()
    
    print(f"Exact Match: {exact_match_score(pred, truth):.4f}")
    print(f"F1 Score: {f1_score(pred, truth):.4f}")
    print(f"BLEU Score: {bleu_score(pred, truth):.4f}")
    print(f"ROUGE-L: {rouge_l_score(pred, truth):.4f}")
    
    # Batch test
    print("\n" + "=" * 60)
    print("Batch Evaluation:")
    preds = [pred, "Bệnh tiểu đường", "Không biết"]
    truths = [truth, "Bệnh tiểu đường type 2", "Không rõ"]
    
    metrics = evaluate_qa_metrics(preds, truths)
    for metric, score in metrics.items():
        print(f"{metric}: {score:.2f}%")

