import numpy as np


def midrank_relevance(truth):
    """Linear midrank on the final axis; constant truth is exactly 0.5."""
    truth = np.asarray(truth, dtype=np.float64)
    n = truth.shape[-1]
    if n < 2 or not np.isfinite(truth).all():
        raise ValueError("Finite truth and at least two candidates are required")
    order = np.argsort(truth, axis=-1, kind="stable")
    sorted_y = np.take_along_axis(truth, order, axis=-1)
    pos = np.arange(n)
    start = np.concatenate([np.ones(sorted_y.shape[:-1] + (1,), bool),
                            sorted_y[..., 1:] != sorted_y[..., :-1]], axis=-1)
    end = np.concatenate([sorted_y[..., 1:] != sorted_y[..., :-1],
                          np.ones(sorted_y.shape[:-1] + (1,), bool)], axis=-1)
    left = np.maximum.accumulate(np.where(start, pos, 0), axis=-1)
    right = np.minimum.accumulate(np.where(end, pos, n)[..., ::-1], axis=-1)[..., ::-1]
    sorted_rel = (left + right) / (2.0 * (n - 1))
    inverse = np.argsort(order, axis=-1, kind="stable")
    return np.take_along_axis(sorted_rel, inverse, axis=-1)


def tied_dcg(scores, relevance, k=5):
    """Expected DCG for uniformly random order inside every entire score tie."""
    scores = np.asarray(scores, dtype=np.float64)
    relevance = np.broadcast_to(np.asarray(relevance, dtype=np.float64), scores.shape)
    n = scores.shape[-1]
    order = np.argsort(-scores, axis=-1, kind="stable")
    s = np.take_along_axis(scores, order, axis=-1)
    g = np.take_along_axis(relevance, order, axis=-1)
    pos = np.arange(n)
    start = np.concatenate([np.ones(s.shape[:-1] + (1,), bool), s[..., 1:] != s[..., :-1]], -1)
    end = np.concatenate([s[..., 1:] != s[..., :-1], np.ones(s.shape[:-1] + (1,), bool)], -1)
    left = np.maximum.accumulate(np.where(start, pos, 0), axis=-1)
    right = np.minimum.accumulate(np.where(end, pos, n)[..., ::-1], axis=-1)[..., ::-1]
    accum = np.concatenate([np.zeros(g.shape[:-1] + (1,)), np.cumsum(g, axis=-1)], axis=-1)
    block_mean = (np.take_along_axis(accum, right + 1, -1)
                  - np.take_along_axis(accum, left, -1)) / (right - left + 1)
    kk = min(k, n)
    discounts = 1.0 / np.log2(np.arange(2, kk + 2))
    return np.sum(block_mean[..., :kk] * discounts, axis=-1)


def ndcg(scores, relevance, k=5):
    kk = min(k, relevance.shape[-1])
    ideal = np.sum(np.sort(relevance, axis=-1)[..., ::-1][..., :kk]
                   / np.log2(np.arange(2, kk + 2)), axis=-1)
    if np.any(ideal <= 0):
        raise ValueError("Nonpositive ideal DCG")
    return tied_dcg(scores, relevance, k) / ideal


def both_poles(scores, truth, k=5):
    """Scores (..., method, run, candidate), truth (..., candidate)."""
    rel = midrank_relevance(truth)[..., None, None, :]
    result = 0.5 * (ndcg(scores, rel, k) + ndcg(-scores, 1.0 - rel, k))
    return result.mean(axis=-1)


def auc(y, score):
    pos = np.asarray(score)[np.asarray(y) == 1]
    neg = np.asarray(score)[np.asarray(y) == 0]
    if not len(pos) or not len(neg):
        raise ValueError("AUROC requires both classes")
    delta = pos[:, None] - neg[None, :]
    return float(np.mean((delta > 0) + 0.5 * (delta == 0)))


def category_retrieval(scores, truth, categories, k=5):
    """Average eligible categories and runs; scores are method by run by item."""
    scores, truth = np.asarray(scores), np.asarray(truth)
    categories = np.asarray(categories)
    return np.mean([both_poles(scores[..., categories == c], truth[categories == c], k)
                    for c in np.unique(categories)], axis=0)


def spearman(scores, truth):
    """Return rank correlations over the final axis."""
    x, y = midrank_relevance(scores), midrank_relevance(truth)
    x = x - x.mean(axis=-1, keepdims=True)
    y = y - y.mean(axis=-1, keepdims=True)
    denominator = np.sqrt(np.sum(x*x, axis=-1) * np.sum(y*y, axis=-1))
    numerator = np.sum(x*y, axis=-1)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                     where=denominator > 0)
