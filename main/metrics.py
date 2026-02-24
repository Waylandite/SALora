"""
Evaluation metrics for code-related tasks

Implements:
- BLEU: For code2nl (code summarization)
- METEOR: For code2nl
- ROUGE-L: For code2nl
- Exact Match: For code2code and nl2code
- CodeBLEU: For code generation tasks
"""

import numpy as np
from typing import List, Dict
import re


def compute_bleu(predictions: List[str], references: List[str], max_order: int = 4) -> Dict[str, float]:
    """
    Compute BLEU score

    Args:
        predictions: List of predicted strings
        references: List of reference strings (or list of lists for multiple references)
        max_order: Maximum n-gram order

    Returns:
        Dictionary with BLEU scores
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
        import nltk
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    except ImportError:
        print("Warning: NLTK not installed. Install with: pip install nltk")
        return {"bleu": 0.0}

    smooth = SmoothingFunction().method1

    # Tokenize
    pred_tokens = [pred.split() for pred in predictions]
    ref_tokens = [[ref.split()] if isinstance(ref, str) else [r.split() for r in ref]
                  for ref in references]

    # Compute corpus BLEU
    bleu_score = corpus_bleu(ref_tokens, pred_tokens, smoothing_function=smooth)

    # Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4
    weights_list = [
        (1.0, 0, 0, 0),  # BLEU-1
        (0.5, 0.5, 0, 0),  # BLEU-2
        (0.33, 0.33, 0.33, 0),  # BLEU-3
        (0.25, 0.25, 0.25, 0.25),  # BLEU-4
    ]

    bleu_scores = {}
    for i, weights in enumerate(weights_list, 1):
        score = corpus_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
        bleu_scores[f"bleu-{i}"] = score

    bleu_scores["bleu"] = bleu_score

    return bleu_scores


def compute_meteor(predictions: List[str], references: List[str]) -> float:
    """
    Compute METEOR score

    Args:
        predictions: List of predicted strings
        references: List of reference strings

    Returns:
        METEOR score
    """
    try:
        from nltk.translate.meteor_score import meteor_score
        import nltk
        # Download required data
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
    except ImportError:
        print("Warning: NLTK not installed. Install with: pip install nltk")
        return 0.0

    scores = []
    for pred, ref in zip(predictions, references):
        # Tokenize
        pred_tokens = pred.split()
        ref_tokens = ref.split() if isinstance(ref, str) else [r.split() for r in ref]

        if isinstance(ref_tokens[0], list):
            # Multiple references - use max score
            score = max(meteor_score([r], pred_tokens) for r in ref_tokens)
        else:
            score = meteor_score([ref_tokens], pred_tokens)

        scores.append(score)

    return np.mean(scores)


def compute_rouge_l(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE-L score

    Args:
        predictions: List of predicted strings
        references: List of reference strings

    Returns:
        Dictionary with ROUGE-L scores (precision, recall, f1)
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("Warning: rouge-score not installed. Install with: pip install rouge-score")
        return {"rouge-l-f": 0.0, "rouge-l-p": 0.0, "rouge-l-r": 0.0}

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    f_scores = []
    p_scores = []
    r_scores = []

    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        f_scores.append(score['rougeL'].fmeasure)
        p_scores.append(score['rougeL'].precision)
        r_scores.append(score['rougeL'].recall)

    return {
        "rouge-l-f": np.mean(f_scores),
        "rouge-l-p": np.mean(p_scores),
        "rouge-l-r": np.mean(r_scores),
    }


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Compute Exact Match accuracy

    Args:
        predictions: List of predicted strings
        references: List of reference strings

    Returns:
        Exact match ratio
    """
    matches = sum(pred.strip() == ref.strip() for pred, ref in zip(predictions, references))
    return matches / len(predictions) if len(predictions) > 0 else 0.0


def normalize_assertion(assertion: str) -> str:
    """
    Normalize assertion for SAM metric

    Converts assertTrue(x) to assertEquals(true, x) format
    """
    # Remove whitespace
    assertion = re.sub(r'\s+', '', assertion)

    # Normalize assertTrue to assertEquals
    assertion = re.sub(r'assertTrue\((.*?)\)', r'assertEquals(true,\1)', assertion)
    assertion = re.sub(r'assertFalse\((.*?)\)', r'assertEquals(false,\1)', assertion)

    return assertion


def compute_sam(predictions: List[str], references: List[str]) -> float:
    """
    Compute Semantic Assertion Match (SAM) for assertion generation

    Args:
        predictions: List of predicted assertions
        references: List of reference assertions

    Returns:
        SAM score (percentage of semantically equivalent assertions)
    """
    matches = 0
    for pred, ref in zip(predictions, references):
        pred_norm = normalize_assertion(pred)
        ref_norm = normalize_assertion(ref)
        if pred_norm == ref_norm:
            matches += 1

    return matches / len(predictions) if len(predictions) > 0 else 0.0


def compute_code2nl_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute all metrics for code2nl (code summarization) task

    Args:
        predictions: List of predicted summaries
        references: List of reference summaries

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # BLEU scores
    bleu_scores = compute_bleu(predictions, references)
    metrics.update(bleu_scores)

    # METEOR
    meteor = compute_meteor(predictions, references)
    metrics["meteor"] = meteor

    # ROUGE-L
    rouge_scores = compute_rouge_l(predictions, references)
    metrics.update(rouge_scores)

    return metrics


def compute_code2code_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute all metrics for code2code (assertion generation) task

    Args:
        predictions: List of predicted assertions
        references: List of reference assertions

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # Exact Match
    em = compute_exact_match(predictions, references)
    metrics["exact_match"] = em

    # SAM
    sam = compute_sam(predictions, references)
    metrics["sam"] = sam

    # BLEU
    bleu_scores = compute_bleu(predictions, references)
    metrics.update(bleu_scores)

    return metrics


def compute_nl2code_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute all metrics for nl2code (code generation) task

    Args:
        predictions: List of predicted code
        references: List of reference code

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # Exact Match
    em = compute_exact_match(predictions, references)
    metrics["exact_match"] = em

    # BLEU
    bleu_scores = compute_bleu(predictions, references)
    metrics.update(bleu_scores)

    return metrics


def compute_metrics_by_task(
    task_type: str,
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute appropriate metrics based on task type

    Args:
        task_type: One of 'code2nl', 'code2code', 'nl2code'
        predictions: List of predictions
        references: List of references

    Returns:
        Dictionary with computed metrics
    """
    if task_type == 'code2nl':
        return compute_code2nl_metrics(predictions, references)
    elif task_type == 'code2code':
        return compute_code2code_metrics(predictions, references)
    elif task_type == 'nl2code':
        return compute_nl2code_metrics(predictions, references)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
