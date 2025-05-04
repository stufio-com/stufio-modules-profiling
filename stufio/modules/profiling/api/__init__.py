from fastapi import APIRouter
from stufio.core.config import get_settings

router = APIRouter()
settings = get_settings()

if settings.profiling_ENABLE_PROMETHEUS:
    from .metrics import router as metrics_router
    # Include the query profiles router if query profiling is enabled
    router.include_router(
        metrics_router,
        prefix=settings.profiling_METRICS_ENDPOINT,
        tags=settings.profiling_METRICS_ROUTE_TAGS,
    )


if settings.profiling_ENABLE_QUERY_PROFILING:
    from .v1.query_profiles import router as query_profiles_router
    router.include_router(query_profiles_router)


# Export the main router
__all__ = ["router"]
