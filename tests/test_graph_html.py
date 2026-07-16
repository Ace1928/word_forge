"""Security and durability tests for standalone graph viewer exports."""

from __future__ import annotations

from pathlib import Path

import pytest
from pyvis.network import Network

from word_forge.graph.graph_html import (
    GraphViewerMetadata,
    atomic_write_graph_viewer,
    render_graph_viewer,
)

_PYVIS_DOCUMENT = """<!doctype html>
<html>
<head>
  <title>PyVis export</title>
  <link rel="stylesheet" href="https://cdn.example.invalid/vis.css">
  <script src="https://cdn.example.invalid/vis.js"></script>
</head>
<body>
  <p>This generated shell is replaced.</p>
  <div id="mynetwork"></div>
  <script type="text/javascript">
    const nodes = {get: () => [], update: () => {}};
    const edges = {get: () => [], update: () => {}};
    const network = {};
    function drawGraph() {}
    drawGraph();
  </script>
</body>
</html>
"""


def _metadata(**overrides: object) -> GraphViewerMetadata:
    values: dict[str, object] = {
        "title": "Lexical connection graph",
        "total_nodes": 8,
        "total_edges": 9,
        "rendered_nodes": 5,
        "rendered_edges": 4,
        "height": "720px",
    }
    values.update(overrides)
    return GraphViewerMetadata(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"title": "   "}, "title"),
        ({"total_nodes": -1}, "total_nodes"),
        ({"rendered_nodes": 9}, "rendered_nodes"),
        ({"rendered_edges": 10}, "rendered_edges"),
        ({"height": "720px; color: red"}, "height"),
    ],
)
def test_graph_viewer_metadata_rejects_invalid_values(
    overrides: dict[str, object], message: str
) -> None:
    """Untrusted counts and CSS values cannot reach the generated document."""

    with pytest.raises(ValueError, match=message):
        _metadata(**overrides)


def test_render_graph_viewer_is_standalone_accessible_and_script_safe() -> None:
    """The wrapper removes remote assets and safely embeds viewer metadata."""

    document = render_graph_viewer(
        _PYVIS_DOCUMENT,
        _metadata(title='Words </title><script id="unsafe">alert(1)</script>'),
    )

    assert document.startswith("<!doctype html>")
    assert document.count("<head>") == 1
    assert document.count("<body ") == 1
    assert document.count('id="mynetwork"') == 1
    assert 'data-word-forge-viewer="1"' in document
    assert 'aria-label="Interactive lexical connection graph"' in document
    assert 'id="wf-search"' in document
    assert 'id="wf-language"' in document
    assert 'id="wf-neighborhood"' in document
    assert "Content-Security-Policy" in document
    assert "connect-src 'self'" in document
    assert "cdn.example.invalid" not in document
    assert '<script id="unsafe">' not in document
    assert "\\u003c/script\\u003e" in document
    assert '"rendered_nodes":5' in document
    assert "Showing 5/8 nodes" in document
    assert 'search.value = "";' in document


def test_render_graph_viewer_normalizes_real_pyvis_heading_markup() -> None:
    """Real PyVis output becomes a valid shell with one graph initializer."""

    network = Network(
        heading="Upstream graph heading",
        cdn_resources="in_line",
        select_menu=False,
        filter_menu=False,
    )
    network.add_node(1, label="forge")
    network.add_node(2, label="word")
    network.add_edge(1, 2)

    document = render_graph_viewer(network.generate_html(), _metadata())

    assert "Upstream graph heading" not in document
    assert "<center>" not in document.casefold()
    assert document.index("<head>") < document.index("</head>")
    assert document.index("</head>") < document.index("<body ")
    assert document.count("function drawGraph()") == 1
    assert document.count("drawGraph();") == 1
    assert "new TomSelect" not in document


@pytest.mark.parametrize(
    "document",
    [
        "<html><head></head><body></body></html>",
        '<html><body><div id="mynetwork"></div><script>drawGraph();</script></body></html>',
        '<html><head></head><body><div id="mynetwork"></div>drawGraph();</body></html>',
    ],
)
def test_render_graph_viewer_rejects_unsupported_generated_markup(
    document: str,
) -> None:
    """Unexpected upstream output fails closed instead of yielding broken HTML."""

    with pytest.raises(ValueError):
        render_graph_viewer(document, _metadata())


def test_atomic_write_graph_viewer_replaces_existing_export(tmp_path: Path) -> None:
    """Readers never observe a partially written graph viewer."""

    destination = tmp_path / "nested" / "graph.html"
    destination.parent.mkdir()
    destination.write_text("stale", encoding="utf-8")

    atomic_write_graph_viewer(destination, "fresh\n")

    assert destination.read_text(encoding="utf-8") == "fresh\n"
    assert destination.stat().st_mode & 0o777 == 0o644
    assert list(destination.parent.glob(".*.tmp")) == []
