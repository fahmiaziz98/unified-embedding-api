import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, List


def convert_sparse_to_csr(sparse_dict: Dict[str, List]) -> csr_matrix:
    """
    Convert sparse embedding to scipy CSR matrix
    
    API format: {"indices": [10, 25, 42], "values": [0.85, 0.62, 0.91]}
    Milvus format: scipy.sparse.csr_matrix with shape (1, max_dimension)
    
    Args:
        sparse_dict: Dictionary with 'indices' and 'values'
    
    Returns:
        scipy CSR matrix
    """
    indices = sparse_dict["indices"]
    values = sparse_dict["values"]
    
    max_dim = max(indices) + 1 if indices else 1
    
    # Create CSR matrix
    # Shape: (1, max_dim) karena ini single vector
    row_indices = [0] * len(indices)  # Semua di row 0
    col_indices = indices
    
    sparse_matrix = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(1, max_dim)
    )
    
    return sparse_matrix


def batch_convert_sparse_to_csr(sparse_list: List[Dict[str, List]]) -> csr_matrix:
    """
    Convert batch of sparse embeddings to single CSR matrix
    
    Args:
        sparse_list: List of sparse dicts
    
    Returns:
        scipy CSR matrix with shape (batch_size, max_dim)
    """
    if not sparse_list:
        return csr_matrix((0, 0))
    
    max_dim = 0
    for sparse_dict in sparse_list:
        if sparse_dict["indices"]:
            max_dim = max(max_dim, max(sparse_dict["indices"]) + 1)
    
    if max_dim == 0:
        max_dim = 30000  # Default vocab size for SPLADE
    
    # Build row indices, column indices, and values
    row_indices = []
    col_indices = []
    values = []
    
    for row_idx, sparse_dict in enumerate(sparse_list):
        indices = sparse_dict["indices"]
        vals = sparse_dict["values"]
        
        row_indices.extend([row_idx] * len(indices))
        col_indices.extend(indices)
        values.extend(vals)
    
    # Create CSR matrix
    sparse_matrix = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(sparse_list), max_dim)
    )
    
    return sparse_matrix