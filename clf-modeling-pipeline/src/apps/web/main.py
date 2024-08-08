from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from src.apps.web.utils import AppException
from src.apps.web.api.v1.api import api_router

app = FastAPI(
    title="FastAPI", 
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = [
        "http://localhost",
        "http://localhost:8080",
    ],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)
app.include_router(api_router, prefix="/api/v1")

@app.exception_handler(AppException)
async def application_exception_handler(_: Request, exc: AppException) -> JSONResponse:
    return JSONResponse(
        status_code = exc.status_code,
        content = {
            "detail": exc.detail
        },
    )


@app.get("/healthz")
def health_check() -> str:
    return "OK"


@app.get("/")
def root() -> str:
    return "OK"
    