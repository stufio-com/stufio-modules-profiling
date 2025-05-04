from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from stufio.api.deps import get_current_user
from stufio.core.config import get_settings
from ...models.query_profile import DatabaseType, QueryProfileFilter, QueryProfileSummary
from ...services.query_profiler import query_profiler

router = APIRouter(
    prefix="/query-profiles",
    tags=["query-profiling"],
    responses={404: {"description": "Not found"}},
)

settings = get_settings()

# Helper function to check if query profiling is enabled
def check_profiling_enabled():
    """Check if query profiling is enabled in settings"""
    if not getattr(settings, "profiling_ENABLE_QUERY_PROFILING", False):
        raise HTTPException(
            status_code=404, 
            detail="Query profiling is not enabled. Enable it in settings."
        )
    return True

@router.get("/by-correlation/{correlation_id}", response_model=List[Dict])
async def get_query_profiles_by_correlation_id(
    correlation_id: str,
    _: bool = Depends(check_profiling_enabled)
):
    """
    Get all query profiles for a specific correlation ID.
    
    This endpoint allows you to see all database queries that were executed
    within a specific request or transaction, identified by its correlation ID.
    """
    profiles = await query_profiler.get_profiles_by_correlation_id(correlation_id)
    return profiles

@router.get("/summary/{correlation_id}", response_model=QueryProfileSummary)
async def get_query_profiles_summary(
    correlation_id: str,
    _: bool = Depends(check_profiling_enabled)
):
    """
    Get a summary of query profiles for a specific correlation ID.
    
    This endpoint provides aggregate statistics about the queries executed
    within a specific request or transaction, identified by its correlation ID.
    """
    summary = await query_profiler.get_profile_summary(correlation_id=correlation_id)
    return summary

@router.get("/search", response_model=List[Dict])
async def search_query_profiles(
    correlation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    database_type: Optional[DatabaseType] = None,
    operation_type: Optional[str] = None,
    collection_or_table: Optional[str] = None,
    is_slow: Optional[bool] = None,
    min_duration_ms: Optional[float] = None,
    status: Optional[str] = None,
    _: bool = Depends(check_profiling_enabled)
):
    """
    Search for query profiles using various filter criteria.
    
    This endpoint allows you to find query profiles matching specific criteria,
    such as database type, operation type, or performance characteristics.
    """
    filter_params = QueryProfileFilter(
        correlation_id=correlation_id,
        session_id=session_id,
        database_type=database_type,
        operation_type=operation_type,
        collection_or_table=collection_or_table,
        is_slow=is_slow,
        min_duration_ms=min_duration_ms,
        status=status
    )
    
    profiles = await query_profiler.get_profiles_by_filter(filter_params)
    return profiles

@router.post("/session/{session_id}/enable")
async def enable_session_profiling(
    session_id: str,
    _: bool = Depends(check_profiling_enabled),
    current_user: Dict = Depends(get_current_user)
):
    """
    Enable query profiling for a specific session.
    
    This endpoint allows you to specifically enable query profiling for
    a particular session, regardless of the global sampling rate.
    """
    query_profiler.enable_session_profiling(session_id)
    return {"status": "success", "message": f"Query profiling enabled for session {session_id}"}

@router.post("/session/{session_id}/disable")
async def disable_session_profiling(
    session_id: str,
    _: bool = Depends(check_profiling_enabled),
    current_user: Dict = Depends(get_current_user)
):
    """
    Disable query profiling for a specific session.
    
    This endpoint allows you to turn off query profiling for a particular session,
    which is useful after debugging to reduce overhead.
    """
    query_profiler.disable_session_profiling(session_id)
    return {"status": "success", "message": f"Query profiling disabled for session {session_id}"}

@router.post("/cleanup")
async def cleanup_old_profiles(
    _: bool = Depends(check_profiling_enabled),
    current_user: Dict = Depends(get_current_user)
):
    """
    Manually trigger cleanup of old query profiles.
    
    This endpoint triggers immediate deletion of query profiles older than
    the retention period specified in settings.
    """
    result = await query_profiler.cleanup_old_profiles()
    if result:
        return {"status": "success", "message": "Old query profiles cleaned up successfully"}
    else:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Failed to clean up old query profiles"}
        )