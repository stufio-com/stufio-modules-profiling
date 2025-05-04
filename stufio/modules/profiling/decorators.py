import functools
import json
from typing import Any, Dict, Optional, Union, Callable, TypeVar, cast

from stufio.core.config import get_settings
from .models.query_profile import DatabaseType, ProfileType
from .services.query_profiler import query_profiler

settings = get_settings()

# Type variables for decorator functions
F = TypeVar('F', bound=Callable[..., Any])

def profile_clickhouse_query(func: F) -> F:
    """
    Decorator to profile ClickHouse queries.
    
    Wraps any async function that executes a ClickHouse query to collect performance metrics.
    
    Usage:
    ```python
    @profile_clickhouse_query
    async def execute_query(client, query, params=None):
        return await client.query(query, parameters=params)
    ```
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract query from arguments - first positional arg after self/client
        query = args[1] if len(args) > 1 else kwargs.get('query', '')
        
        # Extract parameters if available
        parameters = None
        if len(args) > 2:
            parameters = args[2]
        elif 'parameters' in kwargs:
            parameters = kwargs['parameters']
            
        # Determine query type from the query string
        operation_type = "UNKNOWN"
        if isinstance(query, str):
            query_upper = query.strip().upper()
            if query_upper.startswith("SELECT"):
                operation_type = "SELECT"
            elif query_upper.startswith("INSERT"):
                operation_type = "INSERT"
            elif query_upper.startswith("UPDATE"):
                operation_type = "UPDATE"
            elif query_upper.startswith("DELETE"):
                operation_type = "DELETE"
            elif query_upper.startswith("ALTER"):
                operation_type = "ALTER"
            elif query_upper.startswith("CREATE"):
                operation_type = "CREATE"
            elif query_upper.startswith("DROP"):
                operation_type = "DROP"
            else:
                # Try to find first word
                words = query_upper.split()
                if words:
                    operation_type = words[0]
        
        # Profile the query
        async with query_profiler.profile_query(
            database_type=DatabaseType.CLICKHOUSE,
            query=query if isinstance(query, str) else str(query),
            operation_type=operation_type,
            parameters=parameters
        ):
            return await func(*args, **kwargs)
    
    return cast(F, wrapper)

def profile_mongodb_operation(func: F) -> F:
    """
    Decorator to profile MongoDB operations.
    
    Wraps any async function that performs a MongoDB operation to collect performance metrics.
    
    Usage:
    ```python
    @profile_mongodb_operation
    async def find_user(collection, query):
        return await collection.find_one(query)
    ```
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Use the function name as the operation type
        operation_type = func.__name__
        
        # Try to determine collection name if possible
        collection_name = "unknown"
        if len(args) > 0 and hasattr(args[0], "name"):
            collection_name = getattr(args[0], "name", "unknown")
        
        # Extract query parameters - usually the first argument after collection
        parameters = None
        if len(args) > 1 and isinstance(args[1], dict):
            parameters = args[1]
        
        # Profile the operation
        async with query_profiler.profile_query(
            database_type=DatabaseType.MONGODB,
            query=operation_type,  # Using operation name as query
            operation_type=operation_type,
            collection_or_table=collection_name,
            parameters=parameters
        ):
            return await func(*args, **kwargs)
    
    return cast(F, wrapper)

def profile_redis_operation(func: F) -> F:
    """
    Decorator to profile Redis operations.
    
    Wraps any async function that performs a Redis operation to collect performance metrics.
    
    Usage:
    ```python
    @profile_redis_operation
    async def get_value(redis, key):
        return await redis.get(key)
    ```
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Use the function name as the operation type and command
        operation_type = func.__name__
        
        # Try to extract key(s) from arguments
        key = None
        if len(args) > 1:
            key = args[1]  # Usually the key is the first arg after client
        elif 'key' in kwargs:
            key = kwargs['key']
        elif 'name' in kwargs:  # redis.asyncio often uses 'name' for keys
            key = kwargs['name']
        
        # Format parameters for profiling
        parameters = {}
        if key is not None:
            if isinstance(key, (list, tuple, set)):
                # Multiple keys
                parameters = {"keys": key[:10]}  # Limit to first 10 keys
            else:
                parameters = {"key": str(key)}
        
        # Add any other kwargs as parameters
        for k, v in kwargs.items():
            if k not in ('self', 'client'):
                parameters[k] = v
        
        # Profile the operation
        async with query_profiler.profile_query(
            database_type=DatabaseType.REDIS,
            query=operation_type,
            operation_type=operation_type,
            parameters=parameters
        ):
            return await func(*args, **kwargs)
    
    return cast(F, wrapper)

# Convenience function to apply profiling to a whole class
def apply_profiling_to_class(cls, decorator, methods=None):
    """
    Apply a profiling decorator to all methods (or specified methods) of a class.
    
    Args:
        cls: The class to modify
        decorator: The decorator function to apply
        methods: Optional list of method names to decorate (if None, all methods are decorated)
    
    Returns:
        The modified class
    """
    if methods is None:
        # Find all methods that don't start with underscore
        methods = [name for name, func in vars(cls).items() 
                  if callable(func) and not name.startswith('_')]
    
    for name in methods:
        if hasattr(cls, name) and callable(getattr(cls, name)):
            setattr(cls, name, decorator(getattr(cls, name)))
    
    return cls

# Functions to patch existing driver classes
def patch_clickhouse_client():
    """Patch the ClickHouse AsyncClient class with profiling decorators"""
    try:
        from clickhouse_connect.driver.asyncclient import AsyncClient
        
        # List of methods to patch
        methods = [
            'query', 'query_df', 'query_np', 'command', 
            'raw_query', 'query_column_block_stream'
        ]
        
        # Apply profiling decorator to each method
        apply_profiling_to_class(AsyncClient, profile_clickhouse_query, methods)
        
        return True
    except ImportError:
        return False

def patch_mongodb_client():
    """Patch MongoDB motor classes with profiling decorators"""
    try:
        from motor.core import AgnosticCollection
        
        # List of collection methods to patch
        methods = [
            'find_one', 'find', 'insert_one', 'insert_many', 
            'update_one', 'update_many', 'delete_one', 'delete_many',
            'aggregate', 'count_documents', 'distinct'
        ]
        
        # Apply profiling decorator to each method
        apply_profiling_to_class(AgnosticCollection, profile_mongodb_operation, methods)
        
        return True
    except ImportError:
        return False

def patch_redis_client():
    """Patch Redis client with profiling decorators"""
    results = {}
    
    # Try to patch aioredis client
    try:
        import aioredis
        
        # List of Redis methods to patch
        methods = [
            'get', 'set', 'mget', 'mset', 'delete', 
            'incr', 'decr', 'hget', 'hset', 'hmget', 'hmset',
            'lpush', 'rpush', 'lpop', 'rpop', 'lrange',
            'sadd', 'srem', 'smembers', 'zadd', 'zrange'
        ]
        
        # Apply profiling decorator to each method
        apply_profiling_to_class(aioredis.Redis, profile_redis_operation, methods)
        
        results["aioredis"] = True
    except ImportError:
        results["aioredis"] = False
    
    # Try to patch redis.asyncio client
    try:
        import redis.asyncio
        
        # List of Redis methods to patch for redis.asyncio
        methods = [
            'get', 'set', 'mget', 'mset', 'delete', 'unlink',
            'incr', 'decr', 'hincrby', 'hget', 'hset', 'hmget', 'hmset', 'hgetall',
            'lpush', 'rpush', 'lpop', 'rpop', 'lrange', 'llen',
            'sadd', 'srem', 'smembers', 'sismember', 'scard',
            'zadd', 'zrange', 'zrem', 'zscore', 'zrank',
            'expire', 'expireat', 'ttl', 'persist',
            'exists', 'keys', 'scan', 'type', 'rename'
        ]
        
        # Apply profiling decorator to each method - Redis class
        apply_profiling_to_class(redis.asyncio.Redis, profile_redis_operation, methods)
        
        # Also patch the cluster client if available
        try:
            if hasattr(redis.asyncio, 'RedisCluster'):
                apply_profiling_to_class(redis.asyncio.RedisCluster, profile_redis_operation, methods)
                results["redis.asyncio.cluster"] = True
        except (ImportError, AttributeError):
            results["redis.asyncio.cluster"] = False
        
        results["redis.asyncio"] = True
    except ImportError:
        results["redis.asyncio"] = False
    
    # Return True if at least one Redis client was patched
    return any(results.values())

def apply_all_patches():
    """Apply all database client patches"""
    results = {
        "clickhouse": patch_clickhouse_client(),
        "mongodb": patch_mongodb_client(),
        "redis": patch_redis_client()
    }
    return results

import functools
import inspect
from typing import Any, Callable, Dict, Optional, TypeVar, Union, cast

from .services.query_profiler import query_profiler, ProfileType, DatabaseType

# Type variables for function return types
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])

def profile_api_external(
    endpoint: Optional[str] = None,
    operation_type: str = "GET",
) -> Callable[[F], F]:
    """
    Decorator for profiling external API calls.
    
    Args:
        endpoint: API endpoint or URL being called
        operation_type: HTTP method or operation type (GET, POST, etc.)
        
    Example:
        @profile_api_external(endpoint="https://api.example.com/users", operation_type="GET")
        async def get_user_data(user_id: str):
            async with httpx.AsyncClient() as client:
                return await client.get(f"https://api.example.com/users/{user_id}")
    """
    def decorator(func: F) -> F:
        # Get function name for use in profiling if endpoint not provided
        func_name = func.__name__
        
        # Handle both async and sync functions
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Determine actual endpoint if not provided
                nonlocal endpoint
                actual_endpoint = endpoint or f"{func_name}"
                
                # Capture operation with profile context manager
                async with query_profiler.profile_operation(
                    profile_type=ProfileType.API_EXTERNAL,
                    operation=f"External API: {actual_endpoint}",
                    operation_type=operation_type,
                    target=actual_endpoint,
                ):
                    return await func(*args, **kwargs)
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Determine actual endpoint if not provided
                nonlocal endpoint
                actual_endpoint = endpoint or f"{func_name}"
                
                # Note: This will work but won't be async context manager
                # For sync functions, we manually time and record
                import time
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    
                    # Only record if above threshold
                    if duration_ms >= query_profiler.threshold_ms:
                        query_profiler.get_profiles().append({
                            "profile_type": ProfileType.API_EXTERNAL,
                            "operation": f"External API: {actual_endpoint}",
                            "operation_type": operation_type,
                            "target": actual_endpoint,
                            "duration_ms": duration_ms,
                            "status": "success",
                        })
                    
                    return result
                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    query_profiler.get_profiles().append({
                        "profile_type": ProfileType.API_EXTERNAL,
                        "operation": f"External API: {actual_endpoint}",
                        "operation_type": operation_type,
                        "target": actual_endpoint,
                        "duration_ms": duration_ms,
                        "status": "error",
                        "error_message": str(e),
                    })
                    raise
            return cast(F, sync_wrapper)
    return decorator

def profile_api_internal(
    service: Optional[str] = None,
    endpoint: Optional[str] = None,
    operation_type: str = "GET",
) -> Callable[[F], F]:
    """
    Decorator for profiling internal API calls to other microservices.
    
    Args:
        service: Name of the service being called
        endpoint: API endpoint being called
        operation_type: HTTP method or operation type (GET, POST, etc.)
        
    Example:
        @profile_api_internal(service="user-service", endpoint="/users", operation_type="GET")
        async def get_user_from_service(user_id: str):
            return await user_service_client.get_user(user_id)
    """
    def decorator(func: F) -> F:
        # Get function name for use in profiling
        func_name = func.__name__
        
        # Handle both async and sync functions
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Determine actual endpoint if not provided
                nonlocal service, endpoint
                actual_service = service or "internal-service"
                actual_endpoint = endpoint or f"{func_name}"
                target = f"{actual_service}{actual_endpoint}"
                
                # Capture operation with profile context manager
                async with query_profiler.profile_operation(
                    profile_type=ProfileType.API_INTERNAL,
                    operation=f"Internal API: {target}",
                    operation_type=operation_type,
                    target=target,
                ):
                    return await func(*args, **kwargs)
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Similar implementation to profile_api_external for sync functions
                nonlocal service, endpoint
                actual_service = service or "internal-service"
                actual_endpoint = endpoint or f"{func_name}"
                target = f"{actual_service}{actual_endpoint}"
                
                import time
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    
                    if duration_ms >= query_profiler.threshold_ms:
                        query_profiler.get_profiles().append({
                            "profile_type": ProfileType.API_INTERNAL,
                            "operation": f"Internal API: {target}",
                            "operation_type": operation_type,
                            "target": target,
                            "duration_ms": duration_ms,
                            "status": "success",
                        })
                    
                    return result
                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    query_profiler.get_profiles().append({
                        "profile_type": ProfileType.API_INTERNAL,
                        "operation": f"Internal API: {target}",
                        "operation_type": operation_type,
                        "target": target,
                        "duration_ms": duration_ms,
                        "status": "error",
                        "error_message": str(e),
                    })
                    raise
            return cast(F, sync_wrapper)
    return decorator

def profile_queue_operation(
    queue_name: Optional[str] = None,
    operation_type: str = "publish",
) -> Callable[[F], F]:
    """
    Decorator for profiling message queue operations.
    
    Args:
        queue_name: Name of the queue being used
        operation_type: Operation type (publish, consume, etc.)
        
    Example:
        @profile_queue_operation(queue_name="tasks", operation_type="publish")
        async def enqueue_task(task_data: Dict[str, Any]):
            return await queue.publish("tasks", task_data)
    """
    def decorator(func: F) -> F:
        # Get function name for use in profiling
        func_name = func.__name__
        
        # Handle both async and sync functions
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Determine actual queue if not provided
                nonlocal queue_name
                actual_queue = queue_name or f"{func_name}_queue"
                
                # Capture operation with profile context manager
                async with query_profiler.profile_operation(
                    profile_type=ProfileType.QUEUE,
                    operation=f"Queue: {actual_queue}",
                    operation_type=operation_type,
                    target=actual_queue,
                ):
                    return await func(*args, **kwargs)
            return cast(F, async_wrapper)
        else:
            # Similar sync implementation as other decorators
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Implementation similar to other sync decorators
                nonlocal queue_name
                actual_queue = queue_name or f"{func_name}_queue"
                
                import time
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    
                    if duration_ms >= query_profiler.threshold_ms:
                        query_profiler.get_profiles().append({
                            "profile_type": ProfileType.QUEUE,
                            "operation": f"Queue: {actual_queue}",
                            "operation_type": operation_type,
                            "target": actual_queue,
                            "duration_ms": duration_ms,
                            "status": "success",
                        })
                    
                    return result
                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    query_profiler.get_profiles().append({
                        "profile_type": ProfileType.QUEUE,
                        "operation": f"Queue: {actual_queue}",
                        "operation_type": operation_type,
                        "target": actual_queue,
                        "duration_ms": duration_ms,
                        "status": "error",
                        "error_message": str(e),
                    })
                    raise
            return cast(F, sync_wrapper)
    return decorator

def profile_computation(
    name: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator for profiling heavy computational operations.
    
    Args:
        name: Name of the computation being performed
        
    Example:
        @profile_computation(name="generate_recommendations")
        async def generate_user_recommendations(user_id: str):
            # Complex recommendation algorithm
            return recommendations
    """
    def decorator(func: F) -> F:
        # Get function name for use in profiling
        func_name = func.__name__
        
        # Handle both async and sync functions
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Determine actual name if not provided
                nonlocal name
                computation_name = name or func_name
                
                # Capture operation with profile context manager
                async with query_profiler.profile_operation(
                    profile_type=ProfileType.COMPUTATION,
                    operation=f"Computation: {computation_name}",
                    operation_type="compute",
                    target=computation_name,
                ):
                    return await func(*args, **kwargs)
            return cast(F, async_wrapper)
        else:
            # Similar sync implementation as other decorators
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                nonlocal name
                computation_name = name or func_name
                
                import time
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    
                    if duration_ms >= query_profiler.threshold_ms:
                        query_profiler.get_profiles().append({
                            "profile_type": ProfileType.COMPUTATION,
                            "operation": f"Computation: {computation_name}",
                            "operation_type": "compute",
                            "target": computation_name,
                            "duration_ms": duration_ms,
                            "status": "success",
                        })
                    
                    return result
                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    query_profiler.get_profiles().append({
                        "profile_type": ProfileType.COMPUTATION,
                        "operation": f"Computation: {computation_name}",
                        "operation_type": "compute",
                        "target": computation_name,
                        "duration_ms": duration_ms,
                        "status": "error",
                        "error_message": str(e),
                    })
                    raise
            return cast(F, sync_wrapper)
    return decorator

def profile_storage_operation(
    storage_name: Optional[str] = None,
    operation_type: str = "read",
) -> Callable[[F], F]:
    """
    Decorator for profiling file/blob storage operations.
    
    Args:
        storage_name: Name of the storage service or location
        operation_type: Operation type (read, write, delete, etc.)
        
    Example:
        @profile_storage_operation(storage_name="s3", operation_type="write")
        async def upload_file(file_path: str, content: bytes):
            return await s3_client.upload_file(file_path, content)
    """
    def decorator(func: F) -> F:
        # Get function name for use in profiling
        func_name = func.__name__
        
        # Handle both async and sync functions
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Determine storage name if not provided
                nonlocal storage_name
                actual_storage = storage_name or "file-storage"
                
                # Capture operation with profile context manager
                async with query_profiler.profile_operation(
                    profile_type=ProfileType.STORAGE,
                    operation=f"Storage: {actual_storage}",
                    operation_type=operation_type,
                    target=actual_storage,
                ):
                    return await func(*args, **kwargs)
            return cast(F, async_wrapper)
        else:
            # Similar sync implementation as other decorators
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                nonlocal storage_name
                actual_storage = storage_name or "file-storage"
                
                import time
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    
                    if duration_ms >= query_profiler.threshold_ms:
                        query_profiler.get_profiles().append({
                            "profile_type": ProfileType.STORAGE,
                            "operation": f"Storage: {actual_storage}",
                            "operation_type": operation_type,
                            "target": actual_storage,
                            "duration_ms": duration_ms,
                            "status": "success",
                        })
                    
                    return result
                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    query_profiler.get_profiles().append({
                        "profile_type": ProfileType.STORAGE,
                        "operation": f"Storage: {actual_storage}",
                        "operation_type": operation_type,
                        "target": actual_storage,
                        "duration_ms": duration_ms,
                        "status": "error",
                        "error_message": str(e),
                    })
                    raise
            return cast(F, sync_wrapper)
    return decorator

def profile_external_api(
    endpoint: str, 
    method: str = "GET",
    service_name: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator to profile external API calls.
    
    Args:
        endpoint: The API endpoint being called (e.g. "/users/123")
        method: The HTTP method (GET, POST, etc.)
        service_name: Optional name of the service being called
        
    Usage:
    ```python
    @profile_external_api(endpoint="/users", method="GET", service_name="UserService")
    async def get_users(client, params=None):
        return await client.get("/users", params=params)
    ```
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Determine target URL - try to extract from first arg after self/client if it exists
            target = endpoint
            if len(args) > 1 and isinstance(args[1], str):
                # If first positional arg is a string, it might be the URL
                target = args[1]
                
            # Extract parameters if available
            parameters = None
            if len(args) > 2:
                parameters = args[2]
            elif 'params' in kwargs:
                parameters = kwargs['params']
            elif 'json' in kwargs:
                parameters = kwargs['json']
            elif 'data' in kwargs:
                parameters = kwargs['data']
                
            # Profile the API call
            async with query_profiler.profile_operation(
                profile_type=ProfileType.API_EXTERNAL,
                operation=f"{method} {target}",
                operation_type=method,
                target=service_name or target,
                parameters=parameters
            ):
                return await func(*args, **kwargs)
                
        return cast(F, wrapper)
    return decorator

def profile_internal_api(
    endpoint: str, 
    method: str = "GET",
    service_name: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator to profile internal API calls between microservices.
    
    Args:
        endpoint: The API endpoint being called (e.g. "/users/123")
        method: The HTTP method (GET, POST, etc.)
        service_name: Optional name of the service being called
        
    Usage:
    ```python
    @profile_internal_api(endpoint="/users", method="GET", service_name="UserService")
    async def get_users_from_service(client, params=None):
        return await client.get("/users", params=params)
    ```
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Determine target URL - try to extract from first arg after self/client if it exists
            target = endpoint
            if len(args) > 1 and isinstance(args[1], str):
                # If first positional arg is a string, it might be the URL
                target = args[1]
                
            # Extract parameters if available
            parameters = None
            if len(args) > 2:
                parameters = args[2]
            elif 'params' in kwargs:
                parameters = kwargs['params']
            elif 'json' in kwargs:
                parameters = kwargs['json']
            elif 'data' in kwargs:
                parameters = kwargs['data']
                
            # Profile the API call
            async with query_profiler.profile_operation(
                profile_type=ProfileType.API_INTERNAL,
                operation=f"{method} {target}",
                operation_type=method,
                target=service_name or target,
                parameters=parameters
            ):
                return await func(*args, **kwargs)
                
        return cast(F, wrapper)
    return decorator

def profile_queue_operation(
    queue_name: str,
    operation_type: str = "PUBLISH"
) -> Callable[[F], F]:
    """
    Decorator to profile message queue operations (Kafka, RabbitMQ, etc).
    
    Args:
        queue_name: The name of the queue or topic
        operation_type: Type of operation (PUBLISH, CONSUME, etc.)
        
    Usage:
    ```python
    @profile_queue_operation(queue_name="notifications", operation_type="PUBLISH")
    async def publish_message(producer, message):
        return await producer.send("notifications", message)
    ```
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract message/data if available
            parameters = None
            if len(args) > 1:
                parameters = args[1]
            elif 'value' in kwargs:
                parameters = kwargs['value']
            elif 'message' in kwargs:
                parameters = kwargs['message']
                
            # Profile the queue operation
            async with query_profiler.profile_operation(
                profile_type=ProfileType.QUEUE,
                operation=f"{operation_type} {queue_name}",
                operation_type=operation_type,
                target=queue_name,
                parameters=parameters
            ):
                return await func(*args, **kwargs)
                
        return cast(F, wrapper)
    return decorator

def profile_compute_operation(
    operation_name: str,
    compute_type: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator to profile compute-intensive operations.
    
    Args:
        operation_name: Name of the compute operation
        compute_type: Optional category of computation (e.g., "ML_INFERENCE", "DATA_PROCESSING")
        
    Usage:
    ```python
    @profile_compute_operation("image_processing", "IMAGE_TRANSFORM")
    async def process_image(image_data):
        # heavy computation
        return processed_result
    ```
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Profile the compute operation
            async with query_profiler.profile_operation(
                profile_type=ProfileType.COMPUTE,
                operation=operation_name,
                operation_type=compute_type or "COMPUTE",
                target=func.__name__,
                parameters=None  # Usually don't need to log parameters for compute ops
            ):
                return await func(*args, **kwargs)
                
        return cast(F, wrapper)
    return decorator

def profile_cache_operation(
    cache_name: str,
    operation_type: str = "GET"
) -> Callable[[F], F]:
    """
    Decorator to profile cache operations beyond Redis.
    
    Args:
        cache_name: Name of the cache system or region
        operation_type: Type of operation (GET, SET, DELETE, etc.)
        
    Usage:
    ```python
    @profile_cache_operation("user_profiles", "GET")
    async def get_user_from_cache(cache, user_id):
        return await cache.get(f"user:{user_id}")
    ```
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract key if available (usually the first arg after the cache client)
            target = cache_name
            if len(args) > 1 and isinstance(args[1], str):
                target = f"{cache_name}:{args[1]}"
                
            # Extract value for SET operations
            parameters = None
            if operation_type == "SET" and len(args) > 2:
                parameters = args[2]
            elif 'value' in kwargs:
                parameters = kwargs['value']
                
            # Profile the cache operation
            async with query_profiler.profile_operation(
                profile_type=ProfileType.CACHE,
                operation=f"{operation_type} {target}",
                operation_type=operation_type,
                target=target,
                parameters=parameters
            ):
                return await func(*args, **kwargs)
                
        return cast(F, wrapper)
    return decorator

def profile_storage_operation(
    storage_name: str,
    operation_type: str = "READ"
) -> Callable[[F], F]:
    """
    Decorator to profile file storage operations (S3, local files, etc).
    
    Args:
        storage_name: Name of the storage system
        operation_type: Type of operation (READ, WRITE, DELETE, etc.)
        
    Usage:
    ```python
    @profile_storage_operation("s3_bucket", "READ")
    async def read_file_from_s3(s3_client, file_path):
        return await s3_client.get_object(Bucket="my-bucket", Key=file_path)
    ```
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Try to determine the file path
            target = storage_name
            # Check positional args
            if len(args) > 1 and isinstance(args[1], str):
                target = f"{storage_name}:{args[1]}"
            # Check common kwargs patterns
            elif 'path' in kwargs:
                target = f"{storage_name}:{kwargs['path']}"
            elif 'key' in kwargs:
                target = f"{storage_name}:{kwargs['key']}"
            elif 'Key' in kwargs:
                target = f"{storage_name}:{kwargs['Key']}"
            elif 'file_path' in kwargs:
                target = f"{storage_name}:{kwargs['file_path']}"
                
            # Profile the storage operation
            async with query_profiler.profile_operation(
                profile_type=ProfileType.STORAGE,
                operation=f"{operation_type} {target}",
                operation_type=operation_type,
                target=target,
                parameters=None  # Usually don't log parameters for storage ops
            ):
                return await func(*args, **kwargs)
                
        return cast(F, wrapper)
    return decorator

# Generic operation profiler for any other type of operation
def profile_generic_operation(
    profile_type: ProfileType,
    operation_name: str,
    operation_type: Optional[str] = None,
    target: Optional[str] = None
) -> Callable[[F], F]:
    """
    Generic decorator to profile any type of operation.
    
    Args:
        profile_type: Type from ProfileType enum
        operation_name: Name of the operation
        operation_type: Type/category of the operation
        target: Target system or entity
        
    Usage:
    ```python
    @profile_generic_operation(
        profile_type=ProfileType.CUSTOM,
        operation_name="complex_business_logic",
        operation_type="BUSINESS_RULE",
        target="pricing_engine"
    )
    async def calculate_pricing(data):
        # complex business logic
        return result
    ```
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Profile the operation
            async with query_profiler.profile_operation(
                profile_type=profile_type,
                operation=operation_name,
                operation_type=operation_type or "GENERIC",
                target=target or func.__name__,
                parameters=None
            ):
                return await func(*args, **kwargs)
                
        return cast(F, wrapper)
    return decorator