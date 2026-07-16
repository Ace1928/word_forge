"""Word Forge Web API — Graph Endpoints.

Provides subgraph extraction and whole-graph retrieval formatted for
Vis.js and 3D-Force-Graph consumption.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_web(request: Request) -> Any:
    """Retrieve the WebApp instance from FastAPI state."""
    return request.app.state.web


@router.get("/graph")
def get_graph(
    request: Request,
    focus: Optional[str] = Query(None, description="Center word for neighborhood"),
    depth: int = Query(2, ge=1, le=5, description="Hop depth"),
    dimensions: str = Query(
        "lexical,emotional,affective,connotative,contextual",
        description="Comma-separated active dimensions",
    ),
    limit: int = Query(1000, ge=1, le=10000),
    whole_graph: bool = Query(False, description="Load entire database graph"),
) -> Dict[str, List[Dict[str, Any]]]:
    """Return a graph payload formatted for frontend renderers.

    Supports both focused neighborhood extraction and whole-database
    retrieval.  Each node carries metadata for inspector drill-down.
    """
    web = _get_web(request)

    try:
        # Refresh graph from DB
        web.graph_manager.update_graph()

        dims = [d.strip() for d in dimensions.split(",") if d.strip()]

        if whole_graph:
            subgraph = web.visualizer._select_graph(
                dimensions_filter=dims,
                focus_term=None,
                focus_language=None,
                depth=0,
                max_nodes=None,
                max_edges=None,
            )
        else:
            focus_term = focus or "happy"
            from word_forge.parser.parser_refiner import normalize_term

            norm = normalize_term(focus_term)
            focus_id = web.graph_manager.query.get_node_id(focus_term)

            if focus_id is None:
                # Fallback: normalized lookup in DB
                with web.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT id FROM words WHERE normalized_term = ? LIMIT 1",
                        (norm,),
                    )
                    row = cursor.fetchone()
                    if row:
                        focus_id = row[0]
                    else:
                        raise HTTPException(
                            status_code=404,
                            detail=f"Term '{focus_term}' not found.",
                        )

            subgraph = web.visualizer._select_graph(
                dimensions_filter=dims,
                focus_term=focus_term,
                focus_language=None,
                depth=depth,
                max_nodes=limit,
                max_edges=limit * 2,
            )

        # ── Serialize nodes ──────────────────────────────────────────
        nodes_payload: List[Dict[str, Any]] = []
        for raw_id, attrs in subgraph.nodes(data=True):
            node_id = int(raw_id)
            term = str(attrs.get("term", f"ID:{node_id}"))
            label = str(attrs.get("label", term))
            language = str(attrs.get("language", "en"))
            is_stub = bool(attrs.get("is_stub", False))

            node_color = web.visualizer._get_node_color(attrs)
            node_size = web.visualizer._calculate_node_size(node_id, subgraph)

            nodes_payload.append(
                {
                    "id": str(node_id),
                    "label": label,
                    "title": f"Term: {term}\nLang: {language}",
                    "size": node_size,
                    "color": node_color,
                    "wfTerm": term,
                    "wfLanguage": language,
                    "wfStub": is_stub,
                }
            )

        # ── Serialize edges ──────────────────────────────────────────
        edges_payload: List[Dict[str, Any]] = []
        for idx, (u, v, attrs) in enumerate(subgraph.edges(data=True)):
            dim = attrs.get("dimensions", attrs.get("dimension", "lexical"))
            rel_type = attrs.get("relationship_type", "synonym")
            edge_color = web.visualizer._get_edge_color(rel_type, dim, attrs)

            edges_payload.append(
                {
                    "id": f"wf-edge-{idx}",
                    "from": str(u),
                    "to": str(v),
                    "label": rel_type,
                    "title": f"Type: {rel_type}\nDimension: {dim}",
                    "color": edge_color,
                    "width": 1.5,
                    "dashes": rel_type == "translation",
                }
            )

        return {"nodes": nodes_payload, "edges": edges_payload}

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
