# Stufio Framework :: Profiling Module

Advanced observability, metrics, and performance profiling module for the Stufio framework.

## Features

- **Prometheus Integration**: Automatic collection and exposure of application metrics
- **OpenTelemetry Support**: Distributed tracing across services
- **Performance Profiling**: Python cProfile integration for detailed performance analysis
- **Request/Response Monitoring**: Track HTTP requests timing and status
- **Exception Tracking**: Capture and quantify errors by type
- **Adaptive Sampling**: Configure sampling rates to minimize overhead
- **Configurable Metrics**: Customize histogram buckets and collection parameters
- **Dashboard Ready**: Export metrics in formats compatible with Grafana and other visualization tools

## Installation

```bash
pip install stufio-modules-profiling
```

The module automatically registers with the Stufio framework when installed.

## Configuration

The module provides extensive configuration options through environment variables or direct settings:

### Feature Flags

```python
# Enable/disable entire components
ENABLE_PROMETHEUS = True      # Toggle Prometheus metrics collection
ENABLE_OPENTELEMETRY = True   # Toggle distributed tracing
ENABLE_PROFILING = False      # Toggle Python profiler (disabled by default due to overhead)
```

### Prometheus Settings

```python
# Endpoint configuration
METRICS_ENDPOINT = "/metrics"         # URL path for scraping metrics
METRICS_ROUTE_TAGS = ["monitoring"]   # FastAPI tags for the metrics endpoint

# Performance tuning
REQUEST_DURATION_BUCKETS = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]  # Histogram buckets in seconds
SAMPLE_RATE = 1.0            # Sampling percentage (1.0 = 100%, 0.1 = 10%)

# Logging behavior
FILTER_METRICS_FROM_LOGS = True  # Exclude /metrics calls from application logs
```

### OpenTelemetry Settings

```python
# Connection settings
OTLP_GRPC_ENDPOINT = ""             # gRPC endpoint for OpenTelemetry collector
OTLP_SERVICE_NAME_PREFIX = ""       # Optional prefix for service names
OTLP_INSECURE = True                # Whether to use TLS for OTLP connection

# Performance tuning
OTLP_BATCH_EXPORT_SCHEDULE_DELAY_MILLIS = 5000  # Batch export delay in milliseconds

# Logging options
LOG_TRACE_IDS = True  # Include trace IDs in application logs
```

### Profiling Settings

```python
# Profiling behavior
PROFILE_TOP_FUNCTIONS = 20           # Number of top functions to display
PROFILE_SORT_BY = "cumulative"       # Sort method ("cumulative", "calls", "time")
PROFILE_PATH_FILTER = ""             # Only profile specific URL paths (if specified)
```

## Prometheus Metrics

The module collects the following metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `fastapi_app_info` | Gauge | Application information including version |
| `fastapi_requests_total` | Counter | Total number of HTTP requests by method and path |
| `fastapi_responses_total` | Counter | Response count by status code, method, and path |
| `fastapi_requests_duration_seconds` | Histogram | Request processing time distribution |
| `fastapi_exceptions_total` | Counter | Exception count by type, method, and path |
| `fastapi_requests_in_progress` | Gauge | Currently active requests |

Each metric includes labels for `method`, `path`, and `app_name` to allow filtering and aggregation.

### Prometheus Integration Example

The metrics can be easily visualized using Grafana:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'stufio-application'
    scrape_interval: 15s
    static_configs:
      - targets: ['api.example.com']
    metrics_path: /metrics
```

## OpenTelemetry Capabilities

This module provides:

- **Distributed Tracing**: Track requests across multiple services
- **Automatic Instrumentation**: FastAPI routes and handlers are traced automatically
- **Context Propagation**: Trace context is maintained across async operations
- **Log Correlation**: Trace IDs are included in logs for easy correlation
- **Custom Span Attributes**: Add business-specific details to traces

### OpenTelemetry Integration Example

```python
# Manual tracing example
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@app.get("/users/{user_id}")
async def get_user(user_id: str):
    with tracer.start_as_current_span("fetch_user_data"):
        # Your code here
        user = await db.get_user(user_id)
    return user
```

## Python Profiling

The module integrates Python's cProfile for detailed execution analysis:

- **Function Call Tracking**: Measure exact time spent in each function
- **Call Counts**: Number of times each function was called
- **Cumulative Time**: Total time spent in each function including subfunctions
- **Customizable Output**: Control the level of detail and sorting

### Interpreting Profile Output

The profile output shows:

```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
```

Where:
- `ncalls`: Number of calls to the function
- `tottime`: Time spent in the function (excluding subfunctions)
- `percall`: Average time per call
- `cumtime`: Total time spent in function and subfunctions
- `percall`: Average cumulative time per call

## Usage Examples

### Basic Setup

The module automatically registers with the Stufio framework:

```python
# The settings will be loaded from environment variables
# or can be specified in your app's settings
```

### Custom Configuration

```python
# in your settings file
profiling = {
    "ENABLE_PROFILING": True,
    "SAMPLE_RATE": 0.1,         # Profile only 10% of requests
    "PROFILE_PATH_FILTER": "/api/v1",  # Only profile API endpoints
    "PROFILE_TOP_FUNCTIONS": 30        # Show more functions in reports
}
```

### Advanced OpenTelemetry Usage

```python
# Add custom attributes to spans
from opentelemetry import trace
from stufio.modules.profiling.services import setup_otlp

# Get current span and add attributes
current_span = trace.get_current_span()
current_span.set_attribute("user.id", user_id)
current_span.set_attribute("business.process", "payment")
```

## Dashboard Integration

The module works seamlessly with:

- **Prometheus + Grafana**: For metrics visualization
- **Jaeger/Tempo**: For distributed tracing visualization
- **Loki**: For log aggregation with trace correlation
- **OpenTelemetry Collector**: For data processing and forwarding

## Dependencies

- **prometheus-client**: For metrics collection and exposition
- **opentelemetry-api/sdk**: Core OpenTelemetry functionality
- **opentelemetry-exporter-otlp**: For sending traces to collectors
- **opentelemetry-instrumentation-fastapi**: For automatic FastAPI tracing
