"""
API package.

This package contains all API-related modules including
routes, dependencies, and middleware.
"""

from . import routes, dependencies

__all__ = [
    "routes",
    "dependencies",
]