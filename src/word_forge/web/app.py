"""Word Forge Web UI — FastAPI Application Factory.

Creates and configures the FastAPI application with Jinja2 templates,
static file serving, and API route registration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from word_forge.config import config
from word_forge.database.database_manager import DBManager
from word_forge.graph.graph_manager import GraphManager
from word_forge.graph.graph_visualizer import GraphVisualizer

logger = logging.getLogger(__name__)

# ── Package path anchors ─────────────────────────────────────────────
_WEB_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _WEB_DIR / "static"
_TEMPLATE_DIR = _WEB_DIR / "templates"


class WebApp:
    """Encapsulates the FastAPI application and its shared state.

    This class owns the database manager, graph manager, and visualizer
    instances that the API routes need.  It exposes a single ``.app``
    attribute for Uvicorn consumption.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self.host = host
        self.port = port

        # ── Core services ────────────────────────────────────────────
        self.db_manager = DBManager(db_path=db_path)
        self.graph_manager = GraphManager(db_manager=self.db_manager)
        self.graph_manager.build_graph(compute_layout=False)
        self.visualizer = GraphVisualizer(self.graph_manager)

        # ── FastAPI application ──────────────────────────────────────
        self.app = FastAPI(
            title="Word Forge Explorer",
            description="Interactive Multi-Dimensional Lexical Ecosystem",
            version="1.0.0",
        )

        # Templates & static assets
        self.templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
        self.app.mount(
            "/static",
            StaticFiles(directory=str(_STATIC_DIR)),
            name="static",
        )

        # Attach self to app state so routes can access it
        self.app.state.web = self  # type: ignore[attr-defined]

        # ── Register routes ──────────────────────────────────────────
        self._register_routes()

        logger.info(
            "Word Forge Web UI initialized  (db=%s, graph_nodes=%d)",
            self.db_manager.db_path,
            self.graph_manager.graph.number_of_nodes()
            if self.graph_manager.graph
            else 0,
        )

    # ── Route Registration ───────────────────────────────────────────
    def _register_routes(self) -> None:
        """Import and include all API routers."""
        from word_forge.web.api.graph import router as graph_router
        from word_forge.web.api.stats import router as stats_router
        from word_forge.web.api.words import router as words_router

        self.app.include_router(graph_router, prefix="/api", tags=["graph"])
        self.app.include_router(words_router, prefix="/api", tags=["words"])
        self.app.include_router(stats_router, prefix="/api", tags=["stats"])

        # ── Dashboard page ───────────────────────────────────────────
        from fastapi import Request
        from fastapi.responses import HTMLResponse

        @self.app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
        async def dashboard(request: Request) -> HTMLResponse:
            """Serve the main explorer dashboard."""
            return self.templates.TemplateResponse(
                "index.html",
                {"request": request},
            )

    # ── Server Lifecycle ─────────────────────────────────────────────
    def run(self) -> None:
        """Start the Uvicorn server (blocking)."""
        import uvicorn

        logger.info("Starting Word Forge Explorer on http://%s:%d", self.host, self.port)
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")


def create_app(db_path: Optional[str] = None) -> FastAPI:
    """Factory function returning a bare FastAPI instance.

    Useful for ``uvicorn word_forge.web.app:create_app --factory``.
    """
    web = WebApp(db_path=db_path)
    return web.app
