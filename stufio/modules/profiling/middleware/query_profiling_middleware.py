import logging
import random
import time
from typing import Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from stufio.core.config import get_settings
from ..config import ProfilingSettings
from ..services.query_profiler import query_profiler

# Import TaskContext from events module for correlation ID management
from stufio.modules.events.utils.context import TaskContext

logger = logging.getLogger(__name__)
settings = get_settings()


class QueryProfilingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for profiling database queries and other operations during HTTP requests.
    
    This middleware:
    1. Uses the correlation ID managed by TaskContext from events module
    2. Decides whether to profile queries for this request based on sampling rate
    3. Sets up the profiling context for the request
    4. Cleans up the profiling data after the request
    5. Adds profiling summary information to response headers (optional)
    """
    
    def __init__(self, 
                 add_headers: bool = False,
                 correlation_id_header: str = "X-Correlation-ID"):
        """
        Initialize the query profiling middleware.
        
        Args:
            add_headers: Whether to add profiling summary headers to the response
            correlation_id_header: Name of the header to use for correlation ID
        """
        super().__init__(None)
        self.add_headers = add_headers
        self.correlation_id_header = correlation_id_header
        self.profiling_settings = settings.get_module_settings("profiling", ProfilingSettings)
        
        logger.info("QueryProfilingMiddleware initialized")
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process the request, setting up and cleaning up profiling.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
        
        Returns:
            The response from the route handler
        """
        # Skip profiling for excluded paths
        if self._should_skip_profiling(request):
            return await call_next(request)
        
        # Get correlation ID from TaskContext (already set by BaseStufioMiddleware)
        correlation_id = str(TaskContext.get_correlation_id())
        
        # Get session ID if available
        session_id = self._extract_session_id(request)
        
        # Determine if we should profile this request
        should_profile = self._should_profile_request(request, session_id)
        
        # Set up profiling for this request if enabled
        if should_profile:
            # No need to set correlation ID in query profiler as it now uses TaskContext directly
            
            # Clear any existing profiles for this context
            query_profiler.clear_profiles()
            
            # Track request start time
            start_time = time.perf_counter()
            
            try:
                # Process the request
                response = await call_next(request)
                
                # Calculate request duration
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                # Add profiling headers if enabled
                if self.add_headers:
                    self._add_profiling_headers(response, duration_ms)
                
                return response
            finally:
                # Log summary of profiles collected
                self._log_profile_summary(correlation_id)
                
                # Clean up profiles (optional - they may be stored for later analysis)
                # query_profiler.clear_profiles()
        else:
            # Process the request without profiling
            response = await call_next(request)
            
            return response
    
    def _should_skip_profiling(self, request: Request) -> bool:
        """
        Determine if profiling should be skipped for this request.
        
        Args:
            request: The incoming request
            
        Returns:
            True if profiling should be skipped, False otherwise
        """
        # Skip profiling for static files, metrics endpoints, etc.
        path = request.url.path
        skip_paths = [
            "/static/", 
            "/favicon.ico",
            "/metrics",
            "/_health",
            "/docs",
            "/redoc",
            "/openapi.json"
        ]
        
        return any(path.startswith(p) for p in skip_paths)
    
    def _extract_session_id(self, request: Request) -> Optional[str]:
        """
        Extract the session ID from the request if available.
        
        Args:
            request: The incoming request
            
        Returns:
            Session ID or None if not found
        """
        # Try to get session ID from cookies
        session_id = request.cookies.get("session", None)
        
        # If not in cookies, try auth header (might contain session token)
        if not session_id:
            auth_header = request.headers.get("Authorization", "")
            if (auth_header.startswith("Bearer ")):
                session_id = auth_header[7:]  # Remove "Bearer " prefix
        
        return session_id
    
    def _should_profile_request(self, request: Request, session_id: Optional[str]) -> bool:
        """
        Determine if this request should be profiled based on sampling rate.
        
        Args:
            request: The incoming request
            session_id: Session ID if available
            
        Returns:
            True if the request should be profiled, False otherwise
        """
        # Check if query profiling is enabled
        if not self.profiling_settings.ENABLE_QUERY_PROFILING:
            return False
        
        # Always profile if session is explicitly enabled for profiling
        if session_id and query_profiler._session_profiling_enabled(session_id):
            return True
        
        # Apply sampling rate - randomly profile some requests
        return random.random() < self.profiling_settings.QUERY_PROFILING_SAMPLE_RATE
    
    def _add_profiling_headers(self, response: Response, duration_ms: float) -> None:
        """
        Add profiling summary headers to the response.
        
        Args:
            response: The outgoing response
            duration_ms: Request duration in milliseconds
        """
        # Get profiling metrics
        metrics = query_profiler.get_all_metrics()
        
        # Add summary headers
        response.headers["X-Profiling-Request-Time-Ms"] = str(round(duration_ms, 2))
        response.headers["X-Profiling-DB-Operations"] = str(metrics["total_database_operations"])
        response.headers["X-Profiling-DB-Time-Ms"] = str(metrics["total_database_time_ms"])
    
    def _log_profile_summary(self, correlation_id: str) -> None:
        """
        Log a summary of the profiling data.
        
        Args:
            correlation_id: The correlation ID for this request
        """
        # Get all profiles
        profiles = query_profiler.get_profiles()
        
        if not profiles:
            return
        
        # Count profiles by type
        profile_counts = {}
        for profile in profiles:
            profile_type = profile.profile_type.value
            profile_counts[profile_type] = profile_counts.get(profile_type, 0) + 1
        
        total_duration = sum(p.duration_ms for p in profiles if p.duration_ms is not None)
        
        logger.debug(
            f"Request {correlation_id} collected {len(profiles)} profiles "
            f"with total duration {total_duration:.2f}ms: {profile_counts}"
        )
