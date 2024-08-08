from fastapi import APIRouter
from src.apps.web.api.v1.endpoints import explains

api_router = APIRouter()
api_router.include_router(explains.router)