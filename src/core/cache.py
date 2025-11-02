"""
Simple in-memory caching layer for embeddings.

This module provides an LRU cache for embedding results to reduce
redundant computations for identical requests.
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Union
from collections import OrderedDict
from threading import Lock
from loguru import logger


class EmbeddingCache:
    """
    Thread-safe LRU cache for embedding results.

    This cache stores embedding results with a TTL (time-to-live) and
    implements LRU eviction when the cache is full.

    Attributes:
        max_size: Maximum number of entries in the cache
        ttl: Time-to-live in seconds for cached entries
        _cache: OrderedDict storing cached entries
        _lock: Threading lock for thread-safety
        _hits: Number of cache hits
        _misses: Number of cache misses
    """

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize the embedding cache.

        Args:
            max_size: Maximum number of entries (default: 1000)
            ttl: Time-to-live in seconds (default: 3600 = 1 hour)
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

        logger.info(f"Initialized embedding cache (max_size={max_size}, ttl={ttl}s)")

    def _generate_key(
        self,
        texts: Union[str, List[str]],
        model_id: str,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate a unique cache key for the request.

        Args:
            texts: Single text or list of texts
            model_id: Model identifier
            prompt: Optional prompt
            **kwargs: Additional parameters

        Returns:
            SHA256 hash of the request parameters
        """
        # Normalize texts to list
        if isinstance(texts, str):
            texts = [texts]

        # Create deterministic representation
        cache_dict = {
            "texts": texts,
            "model_id": model_id,
            "prompt": prompt,
            "kwargs": sorted(kwargs.items()) if kwargs else [],
        }

        # Generate hash
        cache_str = json.dumps(cache_dict, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()

    def get(
        self,
        texts: Union[str, List[str]],
        model_id: str,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> Optional[Any]:
        """
        Retrieve a cached embedding result.

        Args:
            texts: Single text or list of texts
            model_id: Model identifier
            prompt: Optional prompt
            **kwargs: Additional parameters

        Returns:
            Cached result if found and not expired, None otherwise
        """
        key = self._generate_key(texts, model_id, prompt, **kwargs)

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check if expired
            if time.time() - entry["timestamp"] > self.ttl:
                del self._cache[key]
                self._misses += 1
                logger.debug(f"Cache entry expired: {key[:8]}...")
                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)
            self._hits += 1

            logger.debug(f"Cache hit: {key[:8]}... (hit_rate={self.hit_rate:.2%})")

            return entry["result"]

    def set(
        self,
        texts: Union[str, List[str]],
        model_id: str,
        result: Any,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Store an embedding result in the cache.

        Args:
            texts: Single text or list of texts
            model_id: Model identifier
            result: Embedding result to cache
            prompt: Optional prompt
            **kwargs: Additional parameters
        """
        key = self._generate_key(texts, model_id, prompt, **kwargs)

        with self._lock:
            # Evict oldest entry if cache is full
            if len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug(f"Cache full, evicted: {oldest_key[:8]}...")

            # Store new entry
            self._cache[key] = {"result": result, "timestamp": time.time()}

            logger.debug(
                f"Cache set: {key[:8]}... (size={len(self._cache)}/{self.max_size})"
            )

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info(f"Cleared {count} cache entries")

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.

        Returns:
            Number of entries removed
        """
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key
                for key, entry in self._cache.items()
                if current_time - entry["timestamp"] > self.ttl
            ]

            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

            return len(expired_keys)

    @property
    def size(self) -> int:
        """Get current number of cached entries."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """
        Calculate cache hit rate.

        Returns:
            Hit rate as a float between 0 and 1
        """
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self.hit_rate:.2%}",
            "ttl": self.ttl,
        }

    def __repr__(self) -> str:
        """String representation of the cache."""
        return (
            f"EmbeddingCache("
            f"size={self.size}/{self.max_size}, "
            f"hits={self._hits}, "
            f"misses={self._misses}, "
            f"hit_rate={self.hit_rate:.2%})"
        )
