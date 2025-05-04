# Query Profiling for Stufio Framework

This feature extends the Stufio profiling module with comprehensive database query profiling capabilities.

## Overview

Query profiling allows you to collect detailed metrics and logs about database queries in your application, including:

- Query execution time
- Query parameters (optional)
- Database type (MongoDB, ClickHouse, Redis, etc.)
- Collection/table being accessed
- Source module/function that triggered the query
- Stack traces (optional)
- Correlation IDs for tracking queries across requests
- Session-based profiling for debugging specific user sessions

## Features

- **Multiple Database Support**: Monitors MongoDB, ClickHouse, and Redis operations
- **Session-Level Profiling**: Enable profiling for specific user sessions
- **Correlation ID Tracking**: Group queries by request or transaction
- **Sampling**: Control profiling overhead with configurable sampling rates
- **Parameter Redaction**: Automatically redact sensitive data in query parameters
- **Retention Management**: Automatic cleanup of old profiling data
- **REST API**: Query and manage profiling data via HTTP endpoints

## Installation

No additional installation is needed beyond the standard Stufio profiling module.

## Configuration

Enable and configure query profiling in your application settings:

```python
# Settings
ENABLE_QUERY_PROFILING = True  # Master switch for query profiling
QUERY_PROFILING_SAMPLE_RATE = 0.1  # Profile 10% of queries by default
QUERY_PROFILING_THRESHOLD_MS = 100  # Log queries taking longer than 100ms
QUERY_PROFILING_MAX_QUERY_LENGTH = 1000  # Truncate long queries in logs
QUERY_PROFILING_LOG_PARAMS = False  # Don't log parameters by default
QUERY_PROFILING_RETENTION_DAYS = 7  # Keep profiled queries for 7 days
QUERY_PROFILING_INCLUDE_STACKTRACE = False  # Whether to include stacktrace info
QUERY_PROFILING_MAX_PER_REQUEST = 100  # Maximum number of queries per request
```

## Database Schema

The feature uses ClickHouse tables to store query profiling data:

1. `query_profiles`: Stores detailed information about each query
2. `query_profiles_summary`: Materialized view that provides aggregated statistics

The tables include TTL (Time To Live) settings to automatically manage data retention.

## Architecture

The query profiling system consists of several components:

1. **QueryProfiler Service**: Core service for collecting and storing query profiles
2. **Database Decorators**: Patch database drivers to automatically collect query metrics
3. **QueryProfilingMiddleware**: FastAPI middleware to capture context for each request
4. **API Endpoints**: REST API for accessing and managing profiling data
5. **ClickHouse Schema**: Database tables and materialized views for storing profiles

## Usage Examples

### Accessing Query Profiles by Correlation ID

```python
import httpx

async def get_query_profiles(correlation_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.example.com/api/v1/profiling/query-profiles/by-correlation/{correlation_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        return response.json()
```

### Enabling Profiling for a Session

```python
import httpx

async def enable_session_profiling(session_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://api.example.com/api/v1/profiling/query-profiles/session/{session_id}/enable",
            headers={"Authorization": f"Bearer {token}"}
        )
        return response.json()
```

### Manually Profiling a Query

```python
from stufio.modules.profiling.services.query_profiler import query_profiler
from stufio.modules.profiling.models.query_profile import DatabaseType

async def perform_complex_query():
    async with query_profiler.profile_query(
        database_type=DatabaseType.MONGODB,
        query="Complex aggregation pipeline",
        operation_type="aggregate",
        collection_or_table="analytics"
    ):
        # Your complex query code here
        result = await collection.aggregate(pipeline)
    return result
```

## Performance Considerations

- Query profiling adds overhead to database operations
- Use a low sample rate (0.1 or lower) for production environments
- Only log parameters when necessary, as this increases memory usage
- Set an appropriate slow query threshold to reduce the volume of collected data
- Enable stack traces only when debugging specific issues

## API Endpoints

The following API endpoints are available for query profiling:

- `GET /api/v1/profiling/query-profiles/by-correlation/{correlation_id}`
- `GET /api/v1/profiling/query-profiles/summary/{correlation_id}`
- `GET /api/v1/profiling/query-profiles/search`
- `POST /api/v1/profiling/query-profiles/session/{session_id}/enable`
- `POST /api/v1/profiling/query-profiles/session/{session_id}/disable`
- `POST /api/v1/profiling/query-profiles/cleanup`

For detailed usage examples, see the [examples/query_profiling_usage.md](examples/query_profiling_usage.md) file.