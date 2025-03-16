import logging
import os

from starlette.types import ASGIApp
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from stufio.core.config import get_settings

settings = get_settings()

def setup_otlp(app: ASGIApp = None, app_name: str = None, endpoint: str = None) -> None:
    """Configure OpenTelemetry for the application."""
    if not settings.profiling_ENABLE_OPENTELEMETRY:
        logging.info("OpenTelemetry is disabled in settings")
        return
        
    # Use settings or parameters or environment variables
    app_name = app_name or os.environ.get("APP_NAME", "app")
    if settings.profiling_OTLP_SERVICE_NAME_PREFIX:
        app_name = f"{settings.profiling_OTLP_SERVICE_NAME_PREFIX}-{app_name}"
        
    endpoint = endpoint or settings.profiling_OTLP_GRPC_ENDPOINT or os.environ.get("OTLP_GRPC_ENDPOINT", "")
    
    if not endpoint:
        logging.warning("OTLP_GRPC_ENDPOINT not set, skipping OpenTelemetry setup")
        return
    
    # Setting OpenTelemetry
    resource = Resource.create(attributes={
        "service.name": app_name,
        "compose_service": app_name
    })

    # set the tracer provider
    tracer = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer)

    # Use configured batch export delay
    batch_processor = BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint=endpoint, 
            insecure=settings.profiling_OTLP_INSECURE
        ),
        schedule_delay_millis=settings.profiling_OTLP_BATCH_EXPORT_SCHEDULE_DELAY_MILLIS
    )
    tracer.add_span_processor(batch_processor)

    # Configure logging instrumentation
    LoggingInstrumentor().instrument(set_logging_format=True)

    # Instrument FastAPI app if provided
    if app:
        FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer)
    
    # Set up logging filters if configured
    if settings.profiling_FILTER_METRICS_FROM_LOGS:
        class EndpointFilter(logging.Filter):
            # Uvicorn endpoint access log filter
            def filter(self, record: logging.LogRecord) -> bool:
                return record.getMessage().find(f"GET {settings.profiling_METRICS_ENDPOINT}") == -1

        # Filter out metrics endpoint from logs
        access_logger = logging.getLogger("uvicorn.access")
        access_logger.addFilter(EndpointFilter())
    
    # Configure formatted handler for trace IDs if enabled
    if settings.profiling_LOG_TRACE_IDS:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] "
            "[trace_id=%(otelTraceID)s span_id=%(otelSpanID)s resource.service.name=%(otelServiceName)s] - %(message)s"
        ))
        access_logger = logging.getLogger("uvicorn.access")
        access_logger.addHandler(handler)
