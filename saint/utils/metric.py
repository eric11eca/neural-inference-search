from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string
    import re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


def compute_rouge(prediction, truth):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(truth, prediction)
    return rouge_scores['rougeL'][1]


def compute_bleu_score(prediction, truth):
    bleu_score = sentence_bleu([truth], prediction, weights=(0, 0, 0, 1))
    return bleu_score

metrics = {
    "F1": compute_f1,
    "EM": compute_exact_match,
    "BLEU-4": compute_bleu_score,
    "ROUGEL": compute_rouge,
}

def compute_metric_score_batch(outputs, target, metric="F1"):
    metric_score_max = 0
    
    for pred in outputs:
        metric_score = metrics[metric](pred, target[0])
        metric_score_max = max(metric_score_max, metric_score)
    return metric_score_max


def compute_batched_metrics(outputs, targets):
    metrics_scores = {
        "F1": [],
        "EM": [],
        "BLEU-4": [],
        "ROUGEL": [],
    }
    
    for preds, target in zip(outputs[0], targets[0]):
        for metric in metrics_scores:
            metrics_scores[metric].append(
                compute_metric_score_batch(preds, target, metric=metric))

    metrics_scores_avg = {}
    for metric in metrics_scores:
        metrics_scores_avg[metric] = sum(metrics_scores[metric]) / len(outputs)
    
    return metrics_scores_avg
