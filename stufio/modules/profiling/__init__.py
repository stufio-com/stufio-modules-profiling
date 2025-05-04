"""
Query profiling module for the Stufio framework.

This module provides functionality to track and analyze database query performance
across different database types (MongoDB, ClickHouse, Redis).
"""

from .__version__ import __version__
from .config import ProfilingSettings
from .services.query_profiler import query_profiler
from .models.query_profile import ProfileType
from .module import ProfilingModule
from .decorators import (
    profile_clickhouse_query,
    profile_mongodb_operation,
    profile_redis_operation,
    profile_api_external,
    profile_api_internal,
    profile_queue_operation,
    profile_computation,
    profile_cache_operation,
    profile_storage_operation,
    profile_compute_operation,
    profile_generic_operation,
)

# For convenience, expose context manager
profile_query = query_profiler.profile_query

# Export public API
__all__ = [
    "query_profiler",
    "profile_query",
    "ProfileType",
    "profile_clickhouse_query",
    "profile_mongodb_operation",
    "profile_redis_operation",
    "profile_queue_operation",
    "profile_api_external",
    "profile_api_internal",
    "profile_computation",
    "profile_cache_operation",
    "profile_storage_operation",
    "profile_compute_operation",
    "profile_generic_operation",
    "ProfilingModule",
]
