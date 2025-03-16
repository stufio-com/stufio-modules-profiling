import logging
import random
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from stufio.core.config import get_settings

settings = get_settings()

logger = logging.getLogger(__name__)

class ProfilingMiddleware(BaseHTTPMiddleware):
    """Middleware for profiling request processing."""

    def __init__(self, app):
        super().__init__(app)
        self.sample_rate = settings.profiling_SAMPLE_RATE
        self.path_filter = settings.profiling_PROFILE_PATH_FILTER
        self.sort_by = settings.profiling_PROFILE_SORT_BY
        self.top_functions = settings.profiling_PROFILE_TOP_FUNCTIONS

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Apply sampling if configured
        if self.sample_rate < 1.0 and random.random() > self.sample_rate:
            return await call_next(request)
            
        # Apply path filter if configured
        if self.path_filter and self.path_filter not in request.url.path:
            return await call_next(request)
        
        import cProfile
        import pstats
        
        profiler = cProfile.Profile()
        profiler.enable()
        response = await call_next(request)
        profiler.disable()

        stats = pstats.Stats(profiler)
        stats.sort_stats(self.sort_by)
        logger.info(f"Profiling results for {request.method} {request.url.path}")
        stats.print_stats(self.top_functions)

        return response