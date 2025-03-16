import logging
from typing import List, Any
from fastapi import FastAPI

from pymongo import settings
from stufio.core.module_registry import ModuleInterface
from stufio.core.config import get_settings

from .middleware import PrometheusMiddleware, ProfilingMiddleware
from .services import setup_otlp
from .api import router as metrics_router

settings = get_settings()
logger = logging.getLogger(__name__)

class ProfilingModule(ModuleInterface):
    """Profiling and metrics module for Stufio framework."""
    
    def register(self, app: FastAPI) -> None:
        """Legacy method for backwards compatibility."""
        super().register(app)
        
        # Set up OpenTelemetry if enabled
        if settings.profiling_ENABLE_OPENTELEMETRY:
            app_settings = get_settings()
            setup_otlp(app, app_settings.PROJECT_NAME)
    
    def register_routes(self, app: FastAPI) -> None:
        """Register metrics endpoint."""
        if settings.profiling_ENABLE_PROMETHEUS:
            logger.info("Registering profiling module routes")
            app.include_router(metrics_router)
        else:
            logger.info("Prometheus metrics disabled, skipping route registration")
    
    def get_middlewares(self) -> List[tuple]:
        """Return middleware classes for profiling and metrics."""
        
        app_settings = get_settings()
        middlewares = []
        
        # Add PrometheusMiddleware if enabled
        if settings.profiling_ENABLE_PROMETHEUS:
            middlewares.append((
                PrometheusMiddleware, 
                [], 
                {"app_name": app_settings.PROJECT_NAME}
            ))
        
        # Add ProfilingMiddleware if enabled
        if settings.profiling_ENABLE_PROFILING:
            middlewares.append((ProfilingMiddleware, [], {}))
            
        return middlewares
    
    def get_models(self) -> List[Any]:
        """Return a list of database models defined by this module."""
        return []
