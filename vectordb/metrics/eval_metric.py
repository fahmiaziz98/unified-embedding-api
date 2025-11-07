from typing import List


def calculate_hit_rate(retrieved_ids: List[str], relevant_id: str) -> int:
    """
    Hit rate: Checks if the relevant document is present in the retrieved results.

    Args:
        retrieved_ids: List of retrieved document IDs.
        relevant_id: The ID of the correct/relevant document.

    Returns:
        1 if hit, 0 if miss.
    """
    return 1 if relevant_id in retrieved_ids else 0


def calculate_mrr(retrieved_ids: List[str], relevant_id: str) -> float:
    """
    Mean Reciprocal Rank: The reciprocal of the rank of the first relevant document.
    Higher rank means better performance (max = 1.0).

    Args:
        retrieved_ids: List of retrieved document IDs.
        relevant_id: The ID of the correct/relevant document.

    Returns:
        1/rank if found, 0 if not found.
    """
    try:
        rank = retrieved_ids.index(relevant_id) + 1  # +1 because index starts from 0
        return 1.0 / rank
    except ValueError:
        return 0.0


def calculate_precision_at_k(
    retrieved_ids: List[str], relevant_ids: List[str], k: int = None
) -> float:
    """
    Precision@K: How many relevant documents are in the top-K results.

    Args:
        retrieved_ids: List of retrieved document IDs.
        relevant_ids: List of relevant document IDs (can be multiple).
        k: Cut-off point (if None, use all retrieved_ids).

    Returns:
        Precision score (0.0 - 1.0).
    """
    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    if len(retrieved_ids) == 0:
        return 0.0

    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_ids)

    hits = len(relevant_set.intersection(retrieved_set))
    return hits / len(retrieved_ids)
