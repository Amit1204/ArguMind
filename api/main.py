"""
ArguMind FastAPI application entry point.

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Docs available at:
  http://localhost:8000/docs  (Swagger UI)
  http://localhost:8000/redoc (ReDoc)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="ArguMind API",
        description=(
            "Adaptive Multi-Agent Evidence Reasoning System. "
            "Decomposes research queries, searches arXiv + web, resolves conflicts, "
            "clusters evidence semantically, and synthesizes consensus."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow Streamlit frontend and local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    @app.on_event("startup")
    async def startup():
        logger.info(
            f"ArguMind API starting on {settings.api_host}:{settings.api_port}"
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
