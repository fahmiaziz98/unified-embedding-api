"""
API package.

This package contains all API-related modules including
routes, dependencies, and middleware.
"""

from . import routers, dependencies

__all__ = [
    "routers",
    "dependencies",
]