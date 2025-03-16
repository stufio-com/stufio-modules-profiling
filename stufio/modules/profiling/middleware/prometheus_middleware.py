import time
import random
from typing import Tuple

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from starlette.types import ASGIApp

# Prometheus imports
from prometheus_client import Counter, Gauge, Histogram
from opentelemetry import trace

# Import settings
from stufio.core.config import get_settings

settings = get_settings()

# Prometheus metrics
INFO = Gauge("fastapi_app_info", "FastAPI application information.", ["app_name"])
REQUESTS = Counter(
    "fastapi_requests_total",
    "Total count of requests by method and path.",
    ["method", "path", "app_name"],
)
RESPONSES = Counter(
    "fastapi_responses_total",
    "Total count of responses by method, path and status codes.",
    ["method", "path", "status_code", "app_name"],
)
REQUESTS_PROCESSING_TIME = Histogram(
    "fastapi_requests_duration_seconds",
    "Histogram of requests processing time by path (in seconds)",
    ["method", "path", "app_name"],
    buckets=settings.profiling_REQUEST_DURATION_BUCKETS,
)
EXCEPTIONS = Counter(
    "fastapi_exceptions_total",
    "Total count of exceptions raised by path and exception type",
    ["method", "path", "exception_type", "app_name"],
)
REQUESTS_IN_PROGRESS = Gauge(
    "fastapi_requests_in_progress",
    "Gauge of requests by method and path currently being processed",
    ["method", "path", "app_name"],
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting Prometheus metrics."""

    def __init__(self, app: ASGIApp, app_name: str = "fastapi-app") -> None:
        super().__init__(app)
        self.app_name = app_name
        self.sample_rate = settings.profiling_SAMPLE_RATE
        self.enabled = settings.profiling_ENABLE_PROMETHEUS
        self.filter_metrics = settings.profiling_FILTER_METRICS_FROM_LOGS
        self.metrics_endpoint = settings.profiling_METRICS_ENDPOINT
        
        if self.enabled:
            INFO.labels(app_name=self.app_name).inc()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip metrics collection if disabled
        if not self.enabled:
            return await call_next(request)
            
        # Apply sampling if configured
        if self.sample_rate < 1.0 and random.random() > self.sample_rate:
            return await call_next(request)
            
        method = request.method
        path, is_handled_path = self.get_path(request)
        
        # Skip metrics for the metrics endpoint itself if configured
        if self.filter_metrics and path == self.metrics_endpoint:
            return await call_next(request)

        if not is_handled_path:
            return await call_next(request)

        REQUESTS_IN_PROGRESS.labels(
            method=method, path=path, app_name=self.app_name
        ).inc()
        REQUESTS.labels(method=method, path=path, app_name=self.app_name).inc()
        before_time = time.perf_counter()
        try:
            response = await call_next(request)
        except BaseException as e:
            status_code = HTTP_500_INTERNAL_SERVER_ERROR
            EXCEPTIONS.labels(
                method=method,
                path=path,
                exception_type=type(e).__name__,
                app_name=self.app_name,
            ).inc()
            raise e from None
        else:
            status_code = response.status_code
            after_time = time.perf_counter()
            
            # Add trace ID if configured
            exemplar = {}
            if settings.profiling_LOG_TRACE_IDS:
                # retrieve trace id for exemplar
                span = trace.get_current_span()
                trace_id = trace.format_trace_id(span.get_span_context().trace_id)
                exemplar = {"TraceID": trace_id}

            REQUESTS_PROCESSING_TIME.labels(
                method=method, path=path, app_name=self.app_name
            ).observe(after_time - before_time, exemplar=exemplar)
        finally:
            RESPONSES.labels(
                method=method,
                path=path,
                status_code=status_code,
                app_name=self.app_name,
            ).inc()
            REQUESTS_IN_PROGRESS.labels(
                method=method, path=path, app_name=self.app_name
            ).dec()

        return response

    @staticmethod
    def get_path(request: Request) -> Tuple[str, bool]:
        """Get the matched route path for a request."""
        for route in request.app.routes:
            match, child_scope = route.matches(request.scope)
            if match == Match.FULL:
                return route.path, True

        return request.url.path, False
