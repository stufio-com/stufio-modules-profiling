from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import Response
from prometheus_client import REGISTRY, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
from stufio.core.config import get_settings

router = APIRouter()

settings = get_settings()

@router.get(settings.profiling_METRICS_ENDPOINT, tags=settings.profiling_METRICS_ROUTE_TAGS)
async def metrics(request: Request) -> Response:
    """Endpoint for exposing Prometheus metrics."""
    return Response(generate_latest(REGISTRY), headers={"Content-Type": CONTENT_TYPE_LATEST})
