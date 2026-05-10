"""
evaluation/metrics.py
---------------------
Retrieval evaluation metrics:
  - Recall@K
  - NDCG@K    (Normalised Discounted Cumulative Gain)
  - mAP@K     (Mean Average Precision)

All three are standard in fashion / image retrieval literature.

Design
------
* Each metric is a pure function: (retrieved_ids, relevant_ids, K) → float.
* `evaluate_all()` computes all metrics for all K values in one pass over
  query results, returning a nested dict.
* Functions are deliberately separate so they can be unit-tested in isolation.

Ground truth
------------
An item is "relevant" if it shares the same item_id as the query.
Two images are a correct match iff they share the same item_id (per spec).
The query image itself is excluded from the relevant set during evaluation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Set


# ─────────────────────────────────────────────────────────────────────────────
# Atomic metric functions
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(
    retrieved: List[str],
    relevant:  Set[str],
    k:         int,
) -> float:
    """
    Recall@K: is at least one relevant item in the top-K results?

    Binary: 1.0 if yes, 0.0 if no.
    This is the most intuitive metric — it checks whether the system
    "found" the product within K results.

    Parameters
    ----------
    retrieved : Ordered list of retrieved item_ids (top-K).
    relevant  : Set of item_ids that are ground-truth matches.
    k         : Cutoff.
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return 1.0 if any(r in relevant for r in top_k) else 0.0


def ndcg_at_k(
    retrieved: List[str],
    relevant:  Set[str],
    k:         int,
) -> float:
    """
    NDCG@K: Normalised Discounted Cumulative Gain.

    Rewards placing relevant items early in the ranking.
    A relevant item at rank 1 contributes log2(2)=1.0; at rank 5, log2(6)≈0.39.

    DCG@K  = Σ_{i=1}^{K} rel_i / log2(i+1)
    IDCG@K = DCG of ideal ranking (all relevant items at top)
    NDCG@K = DCG@K / IDCG@K

    Binary relevance (rel_i ∈ {0,1}).
    """
    if not relevant:
        return 0.0

    dcg = 0.0
    for i, item_id in enumerate(retrieved[:k], start=1):
        if item_id in relevant:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG: place all relevant items first (up to K)
    n_relevant_in_k = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_relevant_in_k + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def average_precision_at_k(
    retrieved: List[str],
    relevant:  Set[str],
    k:         int,
) -> float:
    """
    Average Precision@K.

    AP@K = (1 / min(|R|, K)) * Σ_{i=1}^{K} precision_at_i * rel_i

    where precision_at_i = (number of relevant in top-i) / i
    and   rel_i = 1 if item at rank i is relevant else 0.

    Rewards both finding relevant items AND ranking them early.
    """
    if not relevant:
        return 0.0

    num_hits = 0
    cumulative_precision = 0.0

    for i, item_id in enumerate(retrieved[:k], start=1):
        if item_id in relevant:
            num_hits += 1
            cumulative_precision += num_hits / i

    normaliser = min(len(relevant), k)
    if normaliser == 0:
        return 0.0
    return cumulative_precision / normaliser


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_query(
    retrieved:  List[str],
    relevant:   Set[str],
    k_values:   List[int],
) -> Dict[str, float]:
    """
    Compute all metrics for a single query.

    Returns dict like:
        {"Recall@5": 1.0, "NDCG@5": 0.82, "mAP@5": 0.67, ...}
    """
    result = {}
    for k in k_values:
        result[f"Recall@{k}"] = recall_at_k(retrieved, relevant, k)
        result[f"NDCG@{k}"]   = ndcg_at_k(retrieved, relevant, k)
        result[f"mAP@{k}"]    = average_precision_at_k(retrieved, relevant, k)
    return result


def evaluate_all(
    query_results: List[Dict],
    k_values:      List[int] = [5, 10, 15],
) -> Dict[str, float]:
    """
    Compute mean retrieval metrics over all queries.

    Parameters
    ----------
    query_results : List of per-query dicts, each containing:
        {
          "query_item_id": str,
          "query_path":    str,
          "retrieved":     [str, ...],   # item_ids of retrieved items, ordered
          "relevant":      {str, ...},   # ground-truth item_ids (excl. query)
        }
    k_values      : Cutoff values to compute.

    Returns
    -------
    Dict of metric_name → mean over queries.
        {"Recall@5": 0.73, "NDCG@5": 0.61, ..., "num_queries": N}
    """
    accumulator: Dict[str, List[float]] = {}

    for qr in query_results:
        retrieved = qr["retrieved"]
        relevant  = qr["relevant"]
        per_query = evaluate_query(retrieved, relevant, k_values)
        for metric, value in per_query.items():
            accumulator.setdefault(metric, []).append(value)

    means = {metric: float(sum(vals) / len(vals))
             for metric, vals in accumulator.items() if vals}
    stds  = {f"{metric}_std": float(
                 (sum((v - means[metric]) ** 2 for v in vals) / len(vals)) ** 0.5
             )
             for metric, vals in accumulator.items() if vals}

    means["num_queries"] = float(len(query_results))
    means.update(stds)
    return means


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_metrics(metrics: Dict[str, float], k_values: List[int] = [5, 10, 15]) -> str:
    """Return a human-readable table of metrics."""
    lines = [
        f"{'Metric':<15} {'Value':>10}",
        "-" * 27,
    ]
    for k in k_values:
        for name in [f"Recall@{k}", f"NDCG@{k}", f"mAP@{k}"]:
            v = metrics.get(name, float("nan"))
            lines.append(f"{name:<15} {v:>10.4f}")
        lines.append("")
    lines.append(f"{'num_queries':<15} {int(metrics.get('num_queries', 0)):>10d}")
    return "\n".join(lines)
