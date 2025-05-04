# Query Profiling Usage Examples

This document provides examples of how to use the query profiling features in your application.

## 1. Enabling Query Profiling

You can enable query profiling in your application settings:

```python
# In your settings configuration:
profiling_settings = {
    "ENABLE_QUERY_PROFILING": True,  # Master switch for query profiling
    "QUERY_PROFILING_SAMPLE_RATE": 0.1,  # Profile 10% of queries by default
    "QUERY_PROFILING_THRESHOLD_MS": 100,  # Log queries taking longer than 100ms
    "QUERY_PROFILING_LOG_PARAMS": False,  # Don't log parameters by default (may contain sensitive data)
    "QUERY_PROFILING_RETENTION_DAYS": 7,  # Keep profiled queries for 7 days
    "QUERY_PROFILING_INCLUDE_STACKTRACE": False,  # Whether to include stacktrace info
}
```

## 2. Accessing Query Profiling Data via API

### Get Profiles by Correlation ID

```bash
curl -X GET "https://api.example.com/api/v1/profiling/query-profiles/by-correlation/{correlation_id}" \
  -H "Authorization: Bearer {your_token}"
```

### Get Query Profile Summary

```bash
curl -X GET "https://api.example.com/api/v1/profiling/query-profiles/summary/{correlation_id}" \
  -H "Authorization: Bearer {your_token}"
```

### Search Query Profiles

```bash
curl -X GET "https://api.example.com/api/v1/profiling/query-profiles/search?database_type=mongodb&is_slow=true" \
  -H "Authorization: Bearer {your_token}"
```

## 3. Enabling Session-Level Profiling

To enable profiling for a specific user session:

```bash
curl -X POST "https://api.example.com/api/v1/profiling/query-profiles/session/{session_id}/enable" \
  -H "Authorization: Bearer {your_token}"
```

To disable session-level profiling:

```bash
curl -X POST "https://api.example.com/api/v1/profiling/query-profiles/session/{session_id}/disable" \
  -H "Authorization: Bearer {your_token}"
```

## 4. Manual Cleanup

To trigger cleanup of old query profiles manually:

```bash
curl -X POST "https://api.example.com/api/v1/profiling/query-profiles/cleanup" \
  -H "Authorization: Bearer {your_token}"
```

## 5. Programmatically Profiling Queries

If you need to manually profile a query or operation:

```python
from stufio.modules.profiling.services.query_profiler import query_profiler
from stufio.modules.profiling.models.query_profile import DatabaseType

async def my_database_function():
    # Use the profile_query context manager to profile a database operation
    async with query_profiler.profile_query(
        database_type=DatabaseType.MONGODB,
        query="db.users.find_one({'email': 'user@example.com'})",
        operation_type="find_one",
        collection_or_table="users",
        parameters={"email": "user@example.com"}
    ):
        result = await db.users.find_one({"email": "user@example.com"})
    
    return result
```

## 6. Correlation ID and Context

The query profiler automatically captures the correlation ID from request headers (`x-correlation-id`). If none is provided, a new one is generated.

You can access the current correlation ID in your code:

```python
from stufio.modules.profiling.services.query_profiler import current_correlation_id

def my_function():
    correlation_id = current_correlation_id.get()
    # Use correlation ID for logging or other purposes
```

## 7. Troubleshooting

If you're not seeing any query profiles being collected:

1. Ensure `ENABLE_QUERY_PROFILING` is set to `True`
2. Check that your database operations are being called through the patched drivers
3. Consider increasing the `QUERY_PROFILING_SAMPLE_RATE` temporarily to 1.0 (100% of queries)
4. Enable session-level profiling for your session ID

## 8. Performance Impact

Query profiling adds some overhead to database operations. For production systems:

- Use a low sample rate (0.1 or lower) to reduce overhead
- Set an appropriate threshold (e.g., 100ms) to only capture slow queries
- Disable parameter logging unless specifically needed
- Do not enable stacktrace collection in production unless necessary
- Set an appropriate retention period to manage storage use