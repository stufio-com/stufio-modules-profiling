import logging
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from typing import List, Optional

from stufio.core.config import get_settings
from stufio.core.module_registry import ModuleInterface
from .api import router as profiling_router
from .config import ProfilingSettings
from .middleware import PrometheusMiddleware, ProfilingMiddleware, QueryProfilingMiddleware
from .decorators import apply_all_patches
from .services.query_profiler import query_profiler
from .__version__ import __version__

settings = get_settings()
logger = logging.getLogger(__name__)


class ProfilingModule(ModuleInterface):
    """Profiling module for Stufio Framework"""

    version = __version__

    async def startup(self, app: FastAPI):
        """Initialize the profiling module on application startup"""
        logger.info("Initializing profiling module")

        # Get module settings
        profiling_settings = settings.get_module_settings("profiling", ProfilingSettings)

        # Register middleware
        self._register_middleware(app, profiling_settings)

        # Register API routes
        app.include_router(profiling_router)

        # Setup query profiling if enabled
        if profiling_settings.ENABLE_QUERY_PROFILING:
            logger.info("Query profiling enabled, patching database clients")

            # Initialize the query profiler service with settings
            query_profiler.initialize(
                sample_rate=getattr(profiling_settings, "QUERY_PROFILING_SAMPLE_RATE", 0.1),
                threshold_ms=getattr(profiling_settings, "QUERY_PROFILING_THRESHOLD_MS", 100),
                log_params=getattr(profiling_settings, "QUERY_PROFILING_LOG_PARAMS", False),
                max_query_length=getattr(profiling_settings, "QUERY_PROFILING_MAX_QUERY_LENGTH", 1000),
                include_stacktrace=getattr(profiling_settings, "QUERY_PROFILING_INCLUDE_STACKTRACE", False),
                max_per_request=getattr(profiling_settings, "QUERY_PROFILING_MAX_PER_REQUEST", 100)
            )

            # Patch database clients for automatic query profiling
            patch_results = apply_all_patches()
            logger.info(f"Database client patching results: {patch_results}")

            # Start periodic cleanup of old profiles
            query_profiler.start_periodic_cleanup()

        logger.info("Profiling module initialized")

    def _register_middleware(self, app: FastAPI, profiling_settings: ProfilingSettings):
        """Register profiling middleware based on settings"""
        middleware_to_add: List[BaseHTTPMiddleware] = []

        # Add Prometheus middleware if enabled
        if profiling_settings.ENABLE_PROMETHEUS:
            middleware_to_add.append(
                PrometheusMiddleware(
                    filter_metrics_from_logs=profiling_settings.FILTER_METRICS_FROM_LOGS,
                    request_duration_buckets=profiling_settings.REQUEST_DURATION_BUCKETS,
                )
            )

        # Add code profiling middleware if enabled
        if profiling_settings.ENABLE_PROFILING:
            middleware_to_add.append(
                ProfilingMiddleware(
                    profile_top_functions=profiling_settings.PROFILE_TOP_FUNCTIONS,
                    profile_sort_by=profiling_settings.PROFILE_SORT_BY,
                    path_filter=profiling_settings.PROFILE_PATH_FILTER,
                    sample_rate=profiling_settings.SAMPLE_RATE,
                )
            )

        # Add query profiling middleware if enabled
        if profiling_settings.ENABLE_QUERY_PROFILING:
            middleware_to_add.append(QueryProfilingMiddleware())

        # Register middleware in reverse order (last added will be outermost)
        for middleware in reversed(middleware_to_add):
            app.add_middleware(type(middleware), **middleware.__dict__)
            logger.info(f"Added middleware: {type(middleware).__name__}")

    async def shutdown(self, app: FastAPI):
        """Clean up resources on application shutdown"""
        logger.info("Shutting down profiling module")

        # Get module settings
        profiling_settings = settings.get_module_settings("profiling", ProfilingSettings)

        # Stop query profiler cleanup task if it was started
        if profiling_settings.ENABLE_QUERY_PROFILING:
            query_profiler.stop_periodic_cleanup()

        logger.info("Profiling module shutdown complete")
