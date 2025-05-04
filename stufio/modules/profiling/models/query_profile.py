from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator

class ProfileType(str, Enum):
    """Types of operations supported for profiling"""
    # Database types
    MONGODB = "mongodb"
    CLICKHOUSE = "clickhouse"
    REDIS = "redis"
    POSTGRESQL = "postgresql"
    
    # API and other types
    API_EXTERNAL = "api_external"  # External API calls
    API_INTERNAL = "api_internal"  # Internal microservice calls
    QUEUE = "queue"  # Message queue operations
    CACHE = "cache"  # Cache operations (non-Redis)
    STORAGE = "storage"  # File/blob storage operations
    COMPUTATION = "computation"  # Heavy computational tasks
    
    OTHER = "other"  # Fallback for anything else

# For backward compatibility
DatabaseType = ProfileType

class StackTraceEntry(BaseModel):
    """Represents a single stack trace entry for debugging purposes"""
    file: str
    line: int
    function: str
    code: Optional[str] = None

class QueryProfileBase(BaseModel):
    """Base model for operation profiling data"""
    correlation_id: str = Field(..., description="Correlation ID to link operations within a request")
    session_id: Optional[str] = Field(None, description="Session ID for user-specific profiling")
    profile_type: ProfileType = Field(..., description="Type of operation being profiled")
    operation: str = Field(..., description="The operation or query executed")
    operation_type: str = Field(..., description="Type of operation (e.g., find, insert, GET, POST)")
    duration_ms: float = Field(..., description="Duration of the operation in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the operation was executed")
    
    # Optional fields for extended profiling
    parameters: Optional[Dict[str, Any]] = Field(None, description="Operation parameters (sensitive data may be redacted)")
    target: Optional[str] = Field(None, description="Target of operation (collection, table, endpoint, etc.)")
    is_slow: bool = Field(False, description="Whether this is considered a slow operation")
    source_module: Optional[str] = Field(None, description="Source module that triggered the operation")
    source_function: Optional[str] = Field(None, description="Function that triggered the operation")
    stacktrace: Optional[List[StackTraceEntry]] = Field(None, description="Stacktrace for debugging")
    result_size: Optional[int] = Field(None, description="Size of the operation result (if applicable)")
    status: Optional[str] = Field("success", description="Operation execution status")
    error_message: Optional[str] = Field(None, description="Error message if operation failed")
    
    # For backward compatibility
    @property
    def database_type(self) -> ProfileType:
        return self.profile_type
    
    @property
    def query(self) -> str:
        return self.operation
    
    @property
    def collection_or_table(self) -> Optional[str]:
        return self.target

class QueryProfileCreate(QueryProfileBase):
    """Model for creating a new query profile entry"""
    pass

class QueryProfile(QueryProfileBase):
    """Model for a complete query profile entry"""
    id: str = Field(..., description="Unique identifier for the query profile")
    
    class Config:
        from_attributes = True

class QueryProfileFilter(BaseModel):
    """Filter criteria for querying operation profiles"""
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    profile_type: Optional[ProfileType] = None
    operation_type: Optional[str] = None
    target: Optional[str] = None
    is_slow: Optional[bool] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_duration_ms: Optional[float] = None
    status: Optional[str] = None
    
    # For backward compatibility
    @property
    def database_type(self) -> Optional[ProfileType]:
        return self.profile_type
    
    @database_type.setter
    def database_type(self, value: Optional[ProfileType]):
        self.profile_type = value
    
    @property
    def collection_or_table(self) -> Optional[str]:
        return self.target
    
    @collection_or_table.setter
    def collection_or_table(self, value: Optional[str]):
        self.target = value

class QueryProfileSummary(BaseModel):
    """Summary statistics for a group of operations"""
    count: int = Field(..., description="Total number of operations")
    total_duration_ms: float = Field(..., description="Total duration of all operations in milliseconds")
    avg_duration_ms: float = Field(..., description="Average operation duration in milliseconds")
    max_duration_ms: float = Field(..., description="Maximum operation duration in milliseconds")
    slow_count: int = Field(..., description="Number of slow operations")
    error_count: int = Field(..., description="Number of operations that resulted in errors")
    profile_types: Dict[str, int] = Field(..., description="Distribution of profile types")
    operation_types: Dict[str, int] = Field(..., description="Distribution of operation types")
    
    # For backward compatibility
    @property
    def database_types(self) -> Dict[str, int]:
        return self.profile_types
    
    @property
    def slow_query_count(self) -> int:
        return self.slow_count