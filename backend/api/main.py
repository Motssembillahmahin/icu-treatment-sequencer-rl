"""FastAPI application factory and lifespan."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.dependencies import load_agent_from_path
from backend.api.routers import health, inference, metrics, episodes, training
from backend.config.settings import get_settings
from backend.training.replay_buffer import EpisodeDB


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle."""
    settings = get_settings()
    settings.ensure_dirs()

    # Initialize SQLite DB tables
    db = EpisodeDB(settings.episodes_db)
    await db.initialize()

    # Try to load the most recent trained model
    final_model = settings.models_dir / f"{settings.default_agent}_final.zip"
    if final_model.exists():
        try:
            load_agent_from_path(final_model, settings.default_agent)
        except Exception as exc:
            import logging
            logging.warning(f"Could not load model at startup: {exc}")

    yield
    # Cleanup (nothing needed currently)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="ICU Treatment Sequencer RL API",
        description="REST API for querying and controlling an ICU RL agent",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — allow all origins in development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else ["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    prefix = "/api/v1"
    app.include_router(health.router, prefix=prefix, tags=["health"])
    app.include_router(inference.router, prefix=prefix, tags=["inference"])
    app.include_router(metrics.router, prefix=prefix, tags=["metrics"])
    app.include_router(episodes.router, prefix=prefix, tags=["episodes"])
    app.include_router(training.router, prefix=prefix, tags=["training"])

    return app


app = create_app()


def serve() -> None:
    import uvicorn
    settings = get_settings()
    uvicorn.run("backend.api.main:app", host=settings.api_host, port=settings.api_port, reload=True)


if __name__ == "__main__":
    serve()
