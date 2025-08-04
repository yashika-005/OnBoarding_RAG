from fastapi import APIRouter
from .api import routes as policy_routes

router = APIRouter()
router.include_router(policy_routes.router, prefix="/policies", tags=["policies"])

__all__ = ["router"]