from pydantic import Field
from stufio.core.settings import ModuleSettings
from stufio.core.config import get_settings

settings = get_settings()

class ProfilingSettings(ModuleSettings):
    # Feature flags
    ENABLE_PROMETHEUS: bool = True
    ENABLE_OPENTELEMETRY: bool = False
    ENABLE_PROFILING: bool = False  # Defaults to off due to performance impact

    # Prometheus settings
    METRICS_ENDPOINT: str = "/metrics"
    METRICS_ROUTE_TAGS: list[str] = ["monitoring"]

    # OpenTelemetry settings
    OTLP_GRPC_ENDPOINT: str = ""  # Default empty, will use env var
    OTLP_SERVICE_NAME_PREFIX: str = ""  # Prefix for service name in telemetry
    OTLP_INSECURE: bool = True  # Whether to use insecure connection
    OTLP_BATCH_EXPORT_SCHEDULE_DELAY_MILLIS: int = 5000  # Batch export delay

    # Profiling settings
    PROFILE_TOP_FUNCTIONS: int = 20  # Number of top functions to print in profile
    PROFILE_SORT_BY: str = "cumulative"  # Sort method for profiling results
    PROFILE_PATH_FILTER: str = ""  # Filter to only profile specific paths

    # Logging settings
    FILTER_METRICS_FROM_LOGS: bool = True  # Filter /metrics calls from logs
    LOG_TRACE_IDS: bool = True  # Include trace IDs in logs

    # Performance settings
    SAMPLE_RATE: float = 1.0  # Sample rate for profiling (1.0 = profile everything)
    REQUEST_DURATION_BUCKETS: list[float] = Field(
        default=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
        description="Histogram buckets for request duration in seconds"
    )

# Register these settings with the core
settings.register_module_settings("profiling", ProfilingSettings)
