"""
Operation profiler service for tracking database operations, API calls, and other activities performance.
"""
import datetime
import time
import logging
import traceback
import json
import random
import asyncio
from typing import Dict, Any, Optional, AsyncContextManager, List, Set
from contextlib import asynccontextmanager
from contextvars import ContextVar

from ..models.query_profile import QueryProfile, ProfileType, DatabaseType, StackTraceEntry
from stufio.core.config import get_settings
from stufio.modules.events.utils.context import TaskContext

# Initialize settings
settings = get_settings()

# Create logger
logger = logging.getLogger(__name__)

# Context variable to store query profile data for current request/task
current_profiles: ContextVar[List[Dict]] = ContextVar('current_profiles', default=[])

# We're removing the current_correlation_id context var and will use TaskContext directly


class QueryProfiler:
    """
    Service for tracking performance of database queries, API calls, and other operations.
    
    This class provides methods to:
    1. Profile individual operations with timing information
    2. Collect operation profiles for the current request context
    3. Export profiling data in various formats
    
    Usage:
        # As a context manager for database operations
        async with query_profiler.profile_query(
            database_type=DatabaseType.MONGODB,
            query="find_one",
            parameters={"_id": "123"}
        ):
            result = await collection.find_one({"_id": "123"})
            
        # As a context manager for API calls
        async with query_profiler.profile_operation(
            profile_type=ProfileType.API_EXTERNAL,
            operation="GET /users",
            operation_type="GET",
            target="https://api.example.com/users"
        ):
            response = await http_client.get("https://api.example.com/users")
            
        # Using decorators (see decorators.py)
        @profile_mongodb_operation
        async def get_user(collection, user_id):
            return await collection.find_one({"_id": user_id})
    """
    
    def __init__(self):
        """Initialize the QueryProfiler."""
        self.initialized = False
        self.sample_rate = 0.1  # Default: profile 10% of queries
        self.threshold_ms = 100  # Default: only log queries slower than 100ms
        self.log_params = False  # Default: don't log query parameters
        self.max_query_length = 1000  # Default: truncate queries longer than 1000 chars
        self.include_stacktrace = False  # Default: don't include stacktrace
        self.max_per_request = 100  # Default: max 100 queries per request
        
        # Sets to track seen queries (for sampling logic)
        self._seen_queries: Set[str] = set()
        self._sampled_queries: Set[str] = set()
        
        # Track sessions with explicit profiling enabled
        self._profiled_sessions: Set[str] = set()
        
        # Tracking for cleanup task
        self._cleanup_task = None
    
    def initialize(
        self,
        sample_rate: float = 0.1,
        threshold_ms: int = 100,
        log_params: bool = False,
        max_query_length: int = 1000,
        include_stacktrace: bool = False,
        max_per_request: int = 100,
    ):
        """
        Initialize the query profiler with configuration settings.
        
        Args:
            sample_rate: Fraction of queries to profile (0.0 to 1.0)
            threshold_ms: Only log queries that take longer than this (ms)
            log_params: Whether to include query parameters in logs
            max_query_length: Maximum length of query string to store
            include_stacktrace: Whether to include stacktrace in profiles
            max_per_request: Maximum profiles to collect per request
        """
        self.sample_rate = sample_rate
        self.threshold_ms = threshold_ms
        self.log_params = log_params
        self.max_query_length = max_query_length
        self.include_stacktrace = include_stacktrace
        self.max_per_request = max_per_request
        self.initialized = True
        
        logger.info(
            f"QueryProfiler initialized: sample_rate={sample_rate}, "
            f"threshold_ms={threshold_ms}, max_per_request={max_per_request}"
        )
    
    def _should_sample_query(self, query_hash: str) -> bool:
        """
        Determine if a query should be sampled based on sampling rate.
        
        Uses consistent sampling to ensure the same queries are sampled
        consistently across requests.
        
        Args:
            query_hash: A hash string representing the query
            
        Returns:
            bool: True if the query should be sampled
        """
        # If already decided for this query hash, return previous decision
        if query_hash in self._seen_queries:
            return query_hash in self._sampled_queries
            
        # Add to seen queries
        self._seen_queries.add(query_hash)
        
        # Make sampling decision
        if random.random() < self.sample_rate:
            self._sampled_queries.add(query_hash)
            return True
        
        return False
    
    def _get_current_profiles(self) -> List[Dict]:
        """Get the current list of profiles for this context/task."""
        try:
            return current_profiles.get()
        except LookupError:
            # Initialize if not set
            profiles: List[Dict] = []
            current_profiles.set(profiles)
            return profiles
    
    def get_correlation_id(self) -> str:
        """
        Get the current correlation ID for this context/task.
        
        Uses TaskContext from the events module instead of managing our own context variable.
        """
        # Use TaskContext directly instead of our own context variable
        return str(TaskContext.get_correlation_id())
    
    # We're removing set_correlation_id since we'll use TaskContext directly
    
    def enable_session_profiling(self, session_id: str) -> None:
        """
        Enable profiling for a specific session.
        
        This will cause all requests with this session ID to be profiled,
        regardless of the global sampling rate.
        
        Args:
            session_id: The session ID to enable profiling for
        """
        self._profiled_sessions.add(session_id)
        logger.info(f"Enabled profiling for session {session_id}")
    
    def disable_session_profiling(self, session_id: str) -> None:
        """
        Disable profiling for a specific session.
        
        Args:
            session_id: The session ID to disable profiling for
        """
        if session_id in self._profiled_sessions:
            self._profiled_sessions.remove(session_id)
            logger.info(f"Disabled profiling for session {session_id}")
    
    def _session_profiling_enabled(self, session_id: str) -> bool:
        """
        Check if profiling is explicitly enabled for a session.
        
        Args:
            session_id: The session ID to check
            
        Returns:
            bool: True if profiling is enabled for this session
        """
        return session_id in self._profiled_sessions
    
    async def start_periodic_cleanup(self, interval_seconds: int = 3600) -> None:
        """
        Start a periodic task to clean up old profile data.
        
        Args:
            interval_seconds: How often to run the cleanup task (default: hourly)
        """
        if self._cleanup_task is not None:
            logger.warning("Cleanup task already running")
            return
        
        async def _cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    await self.cleanup_old_profiles()
                except asyncio.CancelledError:
                    logger.info("Profile cleanup task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in profile cleanup task: {e}", exc_info=True)
        
        self._cleanup_task = asyncio.create_task(_cleanup_loop())
        logger.info(f"Started periodic profile cleanup task (interval: {interval_seconds}s)")
    
    def stop_periodic_cleanup(self) -> None:
        """Stop the periodic cleanup task if it's running."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            self._cleanup_task = None
            logger.info("Stopped periodic profile cleanup task")
    
    async def cleanup_old_profiles(self) -> bool:
        """
        Clean up old profile data based on retention settings.
        
        Returns:
            bool: True if cleanup was successful
        """
        # This is just a placeholder - in a real implementation,
        # this would delete old profiles from a database or storage system
        logger.info("Cleaning up old profile data")
        
        # In a real implementation, we would delete old data here
        # For example:
        # retention_days = settings.get_module_settings("profiling").QUERY_PROFILING_RETENTION_DAYS
        # cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=retention_days)
        # await db.profiles.delete_many({"timestamp": {"$lt": cutoff_date}})
        
        return True
    
    @asynccontextmanager
    async def profile_query(
        self,
        database_type: DatabaseType,
        query: str,
        operation_type: Optional[str] = None,
        collection_or_table: Optional[str] = None,
        parameters: Optional[Any] = None,
    ) -> AsyncContextManager[None]:
        """
        Context manager for profiling a database operation.
        
        Args:
            database_type: Type of database (MongoDB, ClickHouse, Redis)
            query: The query or operation being executed
            operation_type: Type of operation (e.g., SELECT, INSERT, find_one)
            collection_or_table: Name of collection or table being accessed
            parameters: Query parameters (optional, may be omitted for privacy)
            
        Usage:
            async with query_profiler.profile_query(...):
                # Run your database operation here
        """
        if not self.initialized:
            # If not initialized, just run the operation without profiling
            yield
            return
        
        # Check if we've reached the maximum profiles per request
        profiles = self._get_current_profiles()
        if len(profiles) >= self.max_per_request:
            # Skip profiling if we've reached the limit
            yield
            return
        
        # Create a profile object
        profile = QueryProfile(
            database_type=database_type,
            query=query[:self.max_query_length] if query else "",
            operation_type=operation_type,
            collection_or_table=collection_or_table,
            correlation_id=self.get_correlation_id(),
            start_time=time.time(),
        )
        
        # Sampling decision
        query_hash = f"{database_type}:{operation_type}:{collection_or_table}"
        if not self._should_sample_query(query_hash):
            # Skip profiling if not sampled
            yield
            return
        
        # Add parameters if configured
        if self.log_params and parameters:
            try:
                # Convert parameters to string with limits
                if isinstance(parameters, (dict, list)):
                    profile.parameters = json.dumps(parameters)[:1000]
                else:
                    profile.parameters = str(parameters)[:1000]
            except Exception as e:
                logger.debug(f"Error serializing parameters: {e}")
        
        # Add stacktrace if configured
        if self.include_stacktrace:
            profile.stacktrace = "".join(traceback.format_stack()[:-2])
        
        try:
            # Start timing
            start_time = time.perf_counter()
            
            # Execute the wrapped code
            yield
            
            # End timing
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Update profile with duration
            profile.duration_ms = duration_ms
            profile.end_time = time.time()
            
            # Only record queries that exceed the threshold
            if duration_ms >= self.threshold_ms:
                # Add profile to context
                profiles = self._get_current_profiles()
                profiles.append(profile)
                
                # Log slow query
                logger.debug(
                    f"Slow {database_type.value} query: {duration_ms:.2f}ms - "
                    f"{operation_type} {collection_or_table or ''}"
                )
                
        except Exception as e:
            # Record exception in profile
            profile.error = str(e)
            profile.end_time = time.time()
            profiles = self._get_current_profiles()
            profiles.append(profile)
            
            # Re-raise the exception
            raise
    
    @asynccontextmanager
    async def profile_operation(
        self,
        profile_type: ProfileType,
        operation: str,
        operation_type: str,
        target: Optional[str] = None,
        parameters: Optional[Any] = None,
    ) -> AsyncContextManager[None]:
        """
        Context manager for profiling any type of operation (API calls, computations, etc).
        
        Args:
            profile_type: Type of operation (API_EXTERNAL, API_INTERNAL, QUEUE, etc.)
            operation: Description of the operation being executed
            operation_type: Type of operation (e.g., GET, POST, PUBLISH, COMPUTE)
            target: Target of the operation (e.g., URL, queue name, etc.)
            parameters: Operation parameters (optional, may be omitted for privacy)
            
        Usage:
            async with query_profiler.profile_operation(
                profile_type=ProfileType.API_EXTERNAL,
                operation="Get user data",
                operation_type="GET",
                target="https://api.example.com/users/123"
            ):
                # Run your operation here (API call, computation, etc.)
                response = await http_client.get("https://api.example.com/users/123")
        """
        if not self.initialized:
            # If not initialized, just run the operation without profiling
            yield
            return
        
        # Check if we've reached the maximum profiles per request
        profiles = self._get_current_profiles()
        if len(profiles) >= self.max_per_request:
            # Skip profiling if we've reached the limit
            yield
            return
        
        # Create a profile object
        profile = QueryProfile(
            id=f"{time.time()}-{random.randint(1000, 9999)}",
            profile_type=profile_type,
            operation=operation[:self.max_query_length] if operation else "",
            operation_type=operation_type,
            target=target,
            correlation_id=self.get_correlation_id() or f"auto-{time.time()}",
            duration_ms=0.0,  # Will be updated after execution
            timestamp=datetime.utcnow(),
        )
        
        # Sampling decision
        op_hash = f"{profile_type}:{operation_type}:{target}"
        if not self._should_sample_query(op_hash):
            # Skip profiling if not sampled
            yield
            return
        
        # Add parameters if configured
        if self.log_params and parameters:
            try:
                # Convert parameters to string with limits
                if isinstance(parameters, (dict, list)):
                    profile.parameters = json.dumps(parameters)[:1000]
                else:
                    profile.parameters = str(parameters)[:1000]
            except Exception as e:
                logger.debug(f"Error serializing parameters: {e}")
        
        # Add stacktrace if configured
        if self.include_stacktrace:
            try:
                stack = traceback.extract_stack()
                profile.stacktrace = [
                    StackTraceEntry(
                        file=frame.filename,
                        line=frame.lineno,
                        function=frame.name,
                        code=frame.line
                    )
                    for frame in stack[:-2]  # Skip the last two frames (this function)
                ]
            except Exception as e:
                logger.debug(f"Error capturing stacktrace: {e}")
        
        try:
            # Start timing
            start_time = time.perf_counter()
            
            # Execute the wrapped code
            yield
            
            # End timing
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Update profile with duration
            profile.duration_ms = duration_ms
            profile.is_slow = duration_ms >= self.threshold_ms
            
            # Only record operations that exceed the threshold
            if duration_ms >= self.threshold_ms:
                # Add profile to context
                profiles = self._get_current_profiles()
                profiles.append(profile)
                
                # Log slow operation
                logger.debug(
                    f"Slow {profile_type.value} operation: {duration_ms:.2f}ms - "
                    f"{operation_type} {target or ''}"
                )
                
        except Exception as e:
            # Record exception in profile
            profile.status = "error"
            profile.error_message = str(e)
            profiles = self._get_current_profiles()
            profiles.append(profile)
            
            # Log error
            logger.debug(
                f"Error in {profile_type.value} operation: {str(e)} - "
                f"{operation_type} {target or ''}"
            )
            
            # Re-raise the exception
            raise

    def get_profiles(self) -> List[QueryProfile]:
        """Get all query profiles for the current request/context."""
        return self._get_current_profiles()
    
    def clear_profiles(self) -> None:
        """Clear all query profiles for the current request/context."""
        current_profiles.set([])
    
    def get_mongodb_metrics(self) -> Dict[str, Any]:
        """
        Get MongoDB metrics from current profiles.
        
        Returns a dictionary with key metrics about MongoDB usage:
        - total_queries: Total number of MongoDB operations
        - total_time_ms: Total time spent on MongoDB operations
        - operation_counts: Breakdown of operations by type
        """
        profiles = self._get_current_profiles()
        mongodb_profiles = [p for p in profiles if p.database_type == DatabaseType.MONGODB]
        
        if not mongodb_profiles:
            return {
                "total_queries": 0,
                "total_time_ms": 0,
                "operation_counts": {}
            }
        
        # Calculate metrics
        total_time_ms = sum(p.duration_ms for p in mongodb_profiles if p.duration_ms is not None)
        
        # Count operations by type
        operation_counts = {}
        for profile in mongodb_profiles:
            if profile.operation_type:
                operation_counts[profile.operation_type] = operation_counts.get(profile.operation_type, 0) + 1
        
        return {
            "total_queries": len(mongodb_profiles),
            "total_time_ms": round(total_time_ms, 2),
            "operation_counts": operation_counts
        }
    
    def get_clickhouse_metrics(self) -> Dict[str, Any]:
        """
        Get ClickHouse metrics from current profiles.
        
        Returns a dictionary with key metrics about ClickHouse usage:
        - total_queries: Total number of ClickHouse operations
        - total_time_ms: Total time spent on ClickHouse operations
        - operation_counts: Breakdown of operations by type
        """
        profiles = self._get_current_profiles()
        ch_profiles = [p for p in profiles if p.database_type == DatabaseType.CLICKHOUSE]
        
        if not ch_profiles:
            return {
                "total_queries": 0,
                "total_time_ms": 0,
                "operation_counts": {}
            }
        
        # Calculate metrics
        total_time_ms = sum(p.duration_ms for p in ch_profiles if p.duration_ms is not None)
        
        # Count operations by type
        operation_counts = {}
        for profile in ch_profiles:
            if profile.operation_type:
                operation_counts[profile.operation_type] = operation_counts.get(profile.operation_type, 0) + 1
        
        return {
            "total_queries": len(ch_profiles),
            "total_time_ms": round(total_time_ms, 2),
            "operation_counts": operation_counts
        }
    
    def get_redis_metrics(self) -> Dict[str, Any]:
        """
        Get Redis metrics from current profiles.
        
        Returns a dictionary with key metrics about Redis usage:
        - total_operations: Total number of Redis operations
        - total_time_ms: Total time spent on Redis operations
        - operation_counts: Breakdown of operations by type
        """
        profiles = self._get_current_profiles()
        redis_profiles = [p for p in profiles if p.database_type == DatabaseType.REDIS]
        
        if not redis_profiles:
            return {
                "total_operations": 0,
                "total_time_ms": 0,
                "operation_counts": {}
            }
        
        # Calculate metrics
        total_time_ms = sum(p.duration_ms for p in redis_profiles if p.duration_ms is not None)
        
        # Count operations by type
        operation_counts = {}
        for profile in redis_profiles:
            if profile.operation_type:
                operation_counts[profile.operation_type] = operation_counts.get(profile.operation_type, 0) + 1
        
        return {
            "total_operations": len(redis_profiles),
            "total_time_ms": round(total_time_ms, 2),
            "operation_counts": operation_counts
        }
    
    def get_api_metrics(self) -> Dict[str, Any]:
        """
        Get API call metrics from current profiles.
        
        Returns a dictionary with key metrics about API usage:
        - total_calls: Total number of API calls
        - total_time_ms: Total time spent on API calls
        - external_calls: Number of external API calls
        - internal_calls: Number of internal service calls
        - operation_counts: Breakdown of operations by type (GET, POST, etc.)
        - error_count: Number of failed API calls
        """
        profiles = self._get_current_profiles()
        api_profiles = [
            p for p in profiles 
            if p.profile_type in (ProfileType.API_EXTERNAL, ProfileType.API_INTERNAL)
        ]
        
        if not api_profiles:
            return {
                "total_calls": 0,
                "total_time_ms": 0,
                "external_calls": 0,
                "internal_calls": 0,
                "operation_counts": {},
                "error_count": 0
            }
        
        # Calculate metrics
        total_time_ms = sum(p.duration_ms for p in api_profiles if p.duration_ms is not None)
        external_calls = sum(1 for p in api_profiles if p.profile_type == ProfileType.API_EXTERNAL)
        internal_calls = sum(1 for p in api_profiles if p.profile_type == ProfileType.API_INTERNAL)
        error_count = sum(1 for p in api_profiles if p.status == "error")
        
        # Count operations by type
        operation_counts = {}
        for profile in api_profiles:
            if profile.operation_type:
                operation_counts[profile.operation_type] = operation_counts.get(profile.operation_type, 0) + 1
        
        return {
            "total_calls": len(api_profiles),
            "total_time_ms": round(total_time_ms, 2),
            "external_calls": external_calls,
            "internal_calls": internal_calls,
            "operation_counts": operation_counts,
            "error_count": error_count
        }
    
    def get_other_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for other operation types (queue, cache, computation, etc).
        
        Returns a dictionary with metrics broken down by profile type.
        """
        profiles = self._get_current_profiles()
        other_profiles = [
            p for p in profiles 
            if p.profile_type not in (
                ProfileType.MONGODB, 
                ProfileType.CLICKHOUSE, 
                ProfileType.REDIS, 
                ProfileType.POSTGRESQL,
                ProfileType.API_EXTERNAL, 
                ProfileType.API_INTERNAL
            )
        ]
        
        if not other_profiles:
            return {
                "total_operations": 0,
                "total_time_ms": 0,
                "profile_type_counts": {},
                "error_count": 0
            }
        
        # Calculate metrics
        total_time_ms = sum(p.duration_ms for p in other_profiles if p.duration_ms is not None)
        error_count = sum(1 for p in other_profiles if p.status == "error")
        
        # Count by profile type
        profile_type_counts = {}
        for profile in other_profiles:
            profile_type = profile.profile_type.value
            profile_type_counts[profile_type] = profile_type_counts.get(profile_type, 0) + 1
        
        return {
            "total_operations": len(other_profiles),
            "total_time_ms": round(total_time_ms, 2),
            "profile_type_counts": profile_type_counts,
            "error_count": error_count
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for all database types combined.
        
        Returns a dictionary with nested metrics for each database type
        and overall totals.
        """
        mongodb_metrics = self.get_mongodb_metrics()
        clickhouse_metrics = self.get_clickhouse_metrics()
        redis_metrics = self.get_redis_metrics()
        api_metrics = self.get_api_metrics()
        other_metrics = self.get_other_metrics()
        
        # Calculate overall totals
        total_queries = (
            mongodb_metrics["total_queries"] +
            clickhouse_metrics["total_queries"] +
            redis_metrics["total_operations"] +
            api_metrics["total_calls"] +
            other_metrics["total_operations"]
        )
        
        total_time_ms = (
            mongodb_metrics["total_time_ms"] +
            clickhouse_metrics["total_time_ms"] +
            redis_metrics["total_time_ms"] +
            api_metrics["total_time_ms"] +
            other_metrics["total_time_ms"]
        )
        
        return {
            "total_database_operations": total_queries,
            "total_database_time_ms": round(total_time_ms, 2),
            "mongodb": mongodb_metrics,
            "clickhouse": clickhouse_metrics,
            "redis": redis_metrics,
            "api": api_metrics,
            "other": other_metrics
        }


# Create a singleton instance
query_profiler = QueryProfiler()