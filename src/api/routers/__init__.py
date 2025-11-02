"""
API routes package.

This package contains all API route modules organized by domain.
"""

from . import embedding, model_list, health, rerank

__all__ = ["embedding", "model_list", "health", "rerank"]
