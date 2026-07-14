"""Self-contained, responsive HTML shell for lexical graph exploration."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Final

_RESOURCE_SCRIPT_RE: Final[re.Pattern[str]] = re.compile(
    r"\s*<script\b(?=[^>]*\bsrc=[\"'])[^>]*>\s*</script>\s*",
    flags=re.IGNORECASE,
)
_RESOURCE_LINK_RE: Final[re.Pattern[str]] = re.compile(
    r"\s*<link\b(?=[^>]*\bhref=[\"'])[^>]*?/?>\s*",
    flags=re.IGNORECASE,
)
_TITLE_RE: Final[re.Pattern[str]] = re.compile(
    r"<title\b[^>]*>.*?</title>", flags=re.IGNORECASE | re.DOTALL
)
_HEAD_RE: Final[re.Pattern[str]] = re.compile(
    r"<head\b[^>]*>(?P<content>.*?)</head>",
    flags=re.IGNORECASE | re.DOTALL,
)
_BODY_RE: Final[re.Pattern[str]] = re.compile(
    r"<body\b[^>]*>(?P<content>.*?)</body>",
    flags=re.IGNORECASE | re.DOTALL,
)
_GRAPH_SCRIPT_RE: Final[re.Pattern[str]] = re.compile(
    r"<script\s+type=[\"']text/javascript[\"'][^>]*>(?P<content>.*?)</script>",
    flags=re.IGNORECASE | re.DOTALL,
)
_CENTER_BLOCK_RE: Final[re.Pattern[str]] = re.compile(
    r"\s*<center\b[^>]*>.*?</center>\s*",
    flags=re.IGNORECASE | re.DOTALL,
)
_CSS_SIZE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:auto|\d+(?:\.\d+)?(?:px|vh|vw|rem|em|%))$"
)


@dataclass(frozen=True, slots=True)
class GraphViewerMetadata:
    """Stable metadata embedded in a generated graph viewer."""

    title: str
    total_nodes: int
    total_edges: int
    rendered_nodes: int
    rendered_edges: int
    height: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        """Validate metadata before it reaches HTML or CSS contexts."""

        if not self.title.strip():
            raise ValueError("viewer title must be non-empty")
        for field_name in (
            "total_nodes",
            "total_edges",
            "rendered_nodes",
            "rendered_edges",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.rendered_nodes > self.total_nodes:
            raise ValueError("rendered_nodes cannot exceed total_nodes")
        if self.rendered_edges > self.total_edges:
            raise ValueError("rendered_edges cannot exceed total_edges")
        if not _CSS_SIZE_RE.fullmatch(self.height.strip()):
            raise ValueError(f"Unsupported viewer height: {self.height!r}")


def render_graph_viewer(
    pyvis_html: str,
    metadata: GraphViewerMetadata,
) -> str:
    """Wrap generated PyVis markup in the Word Forge exploration interface.

    Args:
        pyvis_html: Complete HTML produced with inline PyVis resources.
        metadata: Validated viewer counts, title, and dimensions.

    Returns:
        A self-contained HTML document with responsive controls.

    Raises:
        ValueError: If the generated PyVis document lacks expected anchors.
    """

    if '<div id="mynetwork"' not in pyvis_html or "drawGraph();" not in pyvis_html:
        raise ValueError("Generated PyVis document is missing required graph anchors")

    sanitized = _RESOURCE_SCRIPT_RE.sub("\n", pyvis_html)
    sanitized = _RESOURCE_LINK_RE.sub("\n", sanitized)
    head_match = _HEAD_RE.search(sanitized)
    body_match = _BODY_RE.search(sanitized)
    if head_match is None or body_match is None:
        raise ValueError("Generated PyVis document has an unsupported structure")

    head_content = _TITLE_RE.sub("", head_match.group("content"))
    # PyVis places its optional heading in ``<center>`` blocks inside ``head``.
    # Browsers repair that invalid HTML by closing ``head`` early, moving later
    # security metadata and dependencies into ``body``. Preserve resources only.
    head_content = _CENTER_BLOCK_RE.sub("\n", head_content).strip()
    graph_scripts = [
        match.group("content")
        for match in _GRAPH_SCRIPT_RE.finditer(body_match.group("content"))
        if "drawGraph();" in match.group("content")
    ]
    if len(graph_scripts) != 1:
        raise ValueError("Generated PyVis document has no unique graph initializer")

    initializer = graph_scripts[0]
    marker_index = initializer.rfind("drawGraph();")
    if marker_index < 0:
        raise ValueError("Generated PyVis document has no graph initializer")
    insertion_index = marker_index + len("drawGraph();")
    payload = _safe_script_json(asdict(metadata))
    behavior = f"\nwindow.WORD_FORGE_VIEWER = {payload};\n" f"{_GRAPH_VIEWER_SCRIPT}\n"
    initializer = (
        initializer[:insertion_index] + behavior + initializer[insertion_index:]
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
{_head_markup(metadata)}
{head_content}
</head>
<body data-word-forge-viewer="1">
{_viewer_shell(metadata)}
<script type="text/javascript">
{initializer}
</script>
</body>
</html>
"""
    return document.replace("\r\n", "\n")


def atomic_write_graph_viewer(path: Path, document: str) -> None:
    """Durably replace a viewer without exposing a partially written file.

    Args:
        path: Final HTML destination.
        document: Complete UTF-8 viewer document.

    Raises:
        OSError: If the temporary file cannot be written or replaced.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(document)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _safe_script_json(value: object) -> str:
    """Serialize JSON without permitting an inline-script closing sequence."""

    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _head_markup(metadata: GraphViewerMetadata) -> str:
    """Build CSP, viewport, title, and visual design markup."""

    title = escape(metadata.title)
    return f"""
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="color-scheme" content="dark">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
    <title>{title}</title>
    <style>{_GRAPH_VIEWER_STYLE}</style>
    """


def _viewer_shell(metadata: GraphViewerMetadata) -> str:
    """Build semantic, accessible viewer controls and canvas containers."""

    title = escape(metadata.title)
    node_label = _pluralize(metadata.rendered_nodes, "node")
    edge_label = _pluralize(metadata.rendered_edges, "connection")
    rendered_summary = ""
    if (
        metadata.rendered_nodes < metadata.total_nodes
        or metadata.rendered_edges < metadata.total_edges
    ):
        rendered_summary = (
            '<span class="wf-stat wf-stat--notice" title="Rendering limit applied">'
            f"Showing {metadata.rendered_nodes:,}/{metadata.total_nodes:,} nodes · "
            f"{metadata.rendered_edges:,}/{metadata.total_edges:,} connections"
            "</span>"
        )
    return f"""
    <div class="wf-app" style="--wf-canvas-height: {escape(metadata.height)}">
      <header class="wf-header">
        <div class="wf-brand" aria-label="Word Forge lexical graph">
          <span class="wf-mark" aria-hidden="true">W</span>
          <span><strong>Word Forge</strong><small>{title}</small></span>
        </div>
        <div class="wf-stats" aria-label="Graph totals">
          <span class="wf-stat"><strong>{metadata.rendered_nodes:,}</strong> {node_label}</span>
          <span class="wf-stat"><strong>{metadata.rendered_edges:,}</strong> {edge_label}</span>
          {rendered_summary}
        </div>
        <div class="wf-header-actions">
          <button class="wf-button wf-mobile-only" id="wf-toggle-filters" type="button" aria-controls="wf-sidebar" aria-expanded="false">Filters</button>
          <button class="wf-button" id="wf-fit" type="button">Fit graph</button>
          <button class="wf-button" id="wf-physics" type="button" aria-pressed="false">Live layout</button>
        </div>
      </header>

      <div class="wf-workspace">
        <aside class="wf-sidebar" id="wf-sidebar" aria-label="Graph filters">
          <div class="wf-sidebar-heading">
            <div><span class="wf-eyebrow">Explore</span><h1>Find a lexical path</h1></div>
            <button class="wf-icon-button wf-mobile-only" id="wf-close-filters" type="button" aria-label="Close filters">×</button>
          </div>

          <label class="wf-field">
            <span>Search terms <kbd>/</kbd></span>
            <input id="wf-search" type="search" autocomplete="off" placeholder="Type a word or phrase…" aria-describedby="wf-search-hint">
            <small id="wf-search-hint">Press Enter to focus the closest match.</small>
          </label>

          <div class="wf-field-grid">
            <label class="wf-field"><span>Language</span><select id="wf-language"><option value="">All languages</option></select></label>
            <label class="wf-field"><span>Node source</span><select id="wf-source"><option value="">All sources</option></select></label>
          </div>

          <label class="wf-field">
            <span>Relationship</span>
            <select id="wf-relationship"><option value="">All relationships</option></select>
          </label>

          <fieldset class="wf-fieldset">
            <legend>Dimensions</legend>
            <label><input type="checkbox" name="wf-dimension" value="lexical" checked><span class="wf-swatch wf-swatch--lexical"></span>Lexical</label>
            <label><input type="checkbox" name="wf-dimension" value="emotional" checked><span class="wf-swatch wf-swatch--emotional"></span>Emotional</label>
            <label><input type="checkbox" name="wf-dimension" value="affective" checked><span class="wf-swatch wf-swatch--affective"></span>Affective</label>
            <label><input type="checkbox" name="wf-dimension" value="connotative" checked><span class="wf-swatch wf-swatch--connotative"></span>Connotative</label>
            <label><input type="checkbox" name="wf-dimension" value="contextual" checked><span class="wf-swatch wf-swatch--contextual"></span>Contextual</label>
          </fieldset>

          <div class="wf-toggles">
            <label><input id="wf-hide-stubs" type="checkbox"><span>Hide unenriched stubs</span></label>
            <label><input id="wf-show-isolates" type="checkbox" checked><span>Show isolated nodes</span></label>
            <label><input id="wf-show-labels" type="checkbox" checked><span>Show labels</span></label>
          </div>

          <div class="wf-filter-actions">
            <button class="wf-button wf-button--primary" id="wf-reset" type="button">Reset view</button>
            <button class="wf-button" id="wf-neighborhood" type="button" disabled>Focus neighbors</button>
          </div>

          <p class="wf-results" id="wf-results" role="status" aria-live="polite"></p>
          <footer class="wf-sidebar-footer">Self-contained export · no network connection required</footer>
        </aside>

        <main class="wf-main">
          <div class="wf-canvas-card">
            <div id="mynetwork" role="application" aria-label="Interactive lexical connection graph"></div>
            <div class="wf-canvas-hint">Scroll to zoom · drag to pan · select a node or connection for details</div>
          </div>

          <section class="wf-inspector" id="wf-inspector" aria-live="polite" hidden>
            <div><span class="wf-eyebrow" id="wf-inspector-kind">Selection</span><h2 id="wf-inspector-title">Details</h2></div>
            <dl>
              <div><dt>Language / dimension</dt><dd id="wf-inspector-language">—</dd></div>
              <div><dt>Script / relationship</dt><dd id="wf-inspector-script">—</dd></div>
              <div><dt>Source</dt><dd id="wf-inspector-source">—</dd></div>
              <div><dt>Connections / assertions</dt><dd id="wf-inspector-count">—</dd></div>
            </dl>
            <button class="wf-icon-button" id="wf-close-inspector" type="button" aria-label="Close details">×</button>
          </section>
        </main>
      </div>
    </div>
    """


def _pluralize(count: int, singular: str) -> str:
    """Return a grammatically correct English count noun."""

    return singular if count == 1 else f"{singular}s"


_GRAPH_VIEWER_STYLE: Final[str] = r"""
:root {
  color-scheme: dark;
  --wf-bg: #080b14;
  --wf-panel: rgba(16, 21, 37, 0.92);
  --wf-panel-strong: #12182a;
  --wf-line: rgba(164, 180, 222, 0.16);
  --wf-line-strong: rgba(164, 180, 222, 0.28);
  --wf-text: #f4f7ff;
  --wf-muted: #9aa7c6;
  --wf-accent: #7c8cff;
  --wf-accent-strong: #a778ff;
  --wf-good: #43d7a2;
  --wf-shadow: 0 24px 70px rgba(0, 0, 0, 0.42);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-synthesis: none;
}
* { box-sizing: border-box; }
html, body { width: 100%; min-height: 100%; margin: 0; background: var(--wf-bg); color: var(--wf-text); }
body {
  background:
    radial-gradient(circle at 15% -10%, rgba(124, 140, 255, 0.2), transparent 35rem),
    radial-gradient(circle at 90% 10%, rgba(67, 215, 162, 0.1), transparent 30rem),
    var(--wf-bg);
}
button, input, select { font: inherit; }
button, select { cursor: pointer; }
button:focus-visible, input:focus-visible, select:focus-visible { outline: 2px solid #aeb8ff; outline-offset: 2px; }
.wf-app { min-height: 100vh; padding: 18px; }
.wf-header {
  max-width: 1800px; margin: 0 auto 14px; min-height: 72px; display: flex; align-items: center; gap: 18px;
  padding: 12px 14px 12px 16px; border: 1px solid var(--wf-line); border-radius: 18px;
  background: rgba(12, 16, 29, 0.78); box-shadow: 0 10px 40px rgba(0, 0, 0, 0.24); backdrop-filter: blur(18px);
}
.wf-brand { display: flex; align-items: center; gap: 11px; min-width: max-content; }
.wf-brand > span:last-child { display: grid; gap: 2px; }
.wf-brand strong { font-size: 15px; letter-spacing: 0.02em; }
.wf-brand small { color: var(--wf-muted); font-size: 12px; }
.wf-mark {
  display: grid; place-items: center; width: 40px; height: 40px; border-radius: 13px; font-weight: 850; color: white;
  background: linear-gradient(145deg, #6579ff, #a868ee); box-shadow: 0 8px 25px rgba(124, 140, 255, 0.34);
}
.wf-stats { display: flex; gap: 8px; align-items: center; flex: 1; flex-wrap: wrap; }
.wf-stat { padding: 7px 10px; border: 1px solid var(--wf-line); border-radius: 999px; color: var(--wf-muted); font-size: 12px; }
.wf-stat strong { color: var(--wf-text); font-variant-numeric: tabular-nums; }
.wf-stat--notice { border-color: rgba(255, 194, 102, 0.32); color: #ffd498; }
.wf-header-actions, .wf-filter-actions { display: flex; align-items: center; gap: 8px; }
.wf-button, .wf-icon-button {
  border: 1px solid var(--wf-line-strong); color: var(--wf-text); background: rgba(32, 41, 67, 0.68);
  border-radius: 11px; padding: 9px 12px; transition: border-color .16s ease, background .16s ease, transform .16s ease;
}
.wf-button:hover, .wf-icon-button:hover { border-color: rgba(174, 184, 255, 0.62); background: rgba(52, 64, 101, 0.76); }
.wf-button:active, .wf-icon-button:active { transform: translateY(1px); }
.wf-button[aria-pressed="true"] { border-color: rgba(67, 215, 162, 0.65); color: #9ff1d4; }
.wf-button:disabled { cursor: not-allowed; opacity: .42; }
.wf-button--primary { background: linear-gradient(135deg, #6477f5, #8d62d9); border-color: transparent; }
.wf-icon-button { display: grid; place-items: center; width: 34px; height: 34px; padding: 0; border-radius: 10px; font-size: 20px; }
.wf-workspace { max-width: 1800px; margin: 0 auto; display: grid; grid-template-columns: 310px minmax(0, 1fr); gap: 14px; align-items: start; }
.wf-sidebar {
  position: sticky; top: 18px; max-height: calc(100vh - 36px); overflow: auto; padding: 20px; border: 1px solid var(--wf-line);
  border-radius: 20px; background: var(--wf-panel); box-shadow: var(--wf-shadow); backdrop-filter: blur(20px);
}
.wf-sidebar-heading { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 18px; }
.wf-eyebrow { color: #93a2ff; font-size: 10px; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
.wf-sidebar h1, .wf-inspector h2 { margin: 4px 0 0; font-size: 20px; line-height: 1.15; }
.wf-field { display: grid; gap: 7px; margin: 0 0 14px; color: var(--wf-muted); font-size: 12px; font-weight: 650; }
.wf-field small { font-weight: 450; color: #7683a2; }
.wf-field span { display: flex; justify-content: space-between; align-items: center; }
.wf-field-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
input[type="search"], select {
  width: 100%; min-height: 42px; border: 1px solid var(--wf-line); border-radius: 11px; padding: 9px 11px;
  color: var(--wf-text); background: #0d1221;
}
input[type="search"]::placeholder { color: #687593; }
select { appearance: auto; }
kbd { padding: 2px 6px; border: 1px solid var(--wf-line-strong); border-radius: 5px; color: #b6c0da; font-size: 10px; font-weight: 500; }
.wf-fieldset { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; margin: 2px 0 17px; padding: 13px; border: 1px solid var(--wf-line); border-radius: 13px; }
.wf-fieldset legend { padding: 0 5px; color: var(--wf-muted); font-size: 12px; font-weight: 700; }
.wf-fieldset label, .wf-toggles label { display: flex; align-items: center; gap: 7px; color: #c7d0e8; font-size: 12px; cursor: pointer; }
.wf-fieldset input, .wf-toggles input { accent-color: var(--wf-accent); }
.wf-swatch { width: 7px; height: 7px; border-radius: 50%; }
.wf-swatch--lexical { background: #6689ff; }.wf-swatch--emotional { background: #ffbd5d; }.wf-swatch--affective { background: #ef72bd; }
.wf-swatch--connotative { background: #a778ff; }.wf-swatch--contextual { background: #43d7a2; }
.wf-toggles { display: grid; gap: 10px; margin: 0 0 18px; }
.wf-filter-actions { flex-wrap: wrap; }
.wf-results { min-height: 18px; margin: 15px 0 0; color: #b4bfd8; font-size: 12px; }
.wf-sidebar-footer { margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--wf-line); color: #65718d; font-size: 10px; }
.wf-main { min-width: 0; display: grid; gap: 12px; }
.wf-canvas-card {
  position: relative; height: min(var(--wf-canvas-height), calc(100vh - 122px)); min-height: 520px; overflow: hidden;
  border: 1px solid var(--wf-line); border-radius: 20px; background: rgba(9, 13, 24, 0.88); box-shadow: var(--wf-shadow);
}
#mynetwork { width: 100% !important; height: 100% !important; border: 0 !important; float: none !important; background: transparent !important; }
.wf-canvas-card canvas { border-radius: 20px; }
.wf-canvas-hint {
  position: absolute; left: 14px; bottom: 12px; pointer-events: none; padding: 7px 10px; border: 1px solid var(--wf-line);
  border-radius: 9px; color: #8491af; background: rgba(8, 11, 20, .72); font-size: 10px; backdrop-filter: blur(8px);
}
.wf-inspector {
  position: relative; display: grid; grid-template-columns: minmax(150px, .65fr) minmax(0, 1.35fr); gap: 20px; align-items: center;
  padding: 16px 54px 16px 18px; border: 1px solid var(--wf-line); border-radius: 17px; background: var(--wf-panel-strong);
}
.wf-inspector[hidden] { display: none; }
.wf-inspector dl { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 0; }
.wf-inspector dl div { min-width: 0; }
.wf-inspector dt { color: #7784a3; font-size: 10px; text-transform: uppercase; letter-spacing: .08em; }
.wf-inspector dd { margin: 4px 0 0; overflow-wrap: anywhere; color: #dce3f6; font-size: 12px; }
.wf-inspector > .wf-icon-button { position: absolute; top: 12px; right: 12px; }
.wf-mobile-only { display: none; }
@media (max-width: 980px) {
  .wf-app { padding: 10px; }
  .wf-header { margin-bottom: 10px; border-radius: 15px; }
  .wf-stats { display: none; }
  .wf-header-actions { margin-left: auto; }
  .wf-mobile-only { display: grid; }
  .wf-workspace { grid-template-columns: 1fr; gap: 10px; }
  .wf-sidebar { position: fixed; z-index: 20; inset: 0 auto 0 0; width: min(340px, 92vw); max-height: none; border-radius: 0 20px 20px 0; transform: translateX(-105%); transition: transform .2s ease; }
  .wf-sidebar.wf-sidebar--open { transform: translateX(0); }
  .wf-canvas-card { height: calc(100vh - 104px); min-height: 460px; }
  .wf-inspector { grid-template-columns: 1fr; }
  .wf-inspector dl { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 560px) {
  .wf-brand small, .wf-header-actions #wf-physics { display: none; }
  .wf-header { gap: 8px; }
  .wf-field-grid, .wf-fieldset { grid-template-columns: 1fr; }
  .wf-canvas-hint { display: none; }
  .wf-inspector dl { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; } }
"""


_GRAPH_VIEWER_SCRIPT: Final[str] = r"""
(function initializeWordForgeViewer() {
  "use strict";
  const byId = (id) => document.getElementById(id);
  const search = byId("wf-search");
  const language = byId("wf-language");
  const source = byId("wf-source");
  const relationship = byId("wf-relationship");
  const hideStubs = byId("wf-hide-stubs");
  const showIsolates = byId("wf-show-isolates");
  const showLabels = byId("wf-show-labels");
  const results = byId("wf-results");
  const neighborhood = byId("wf-neighborhood");
  const inspector = byId("wf-inspector");
  const originalNodes = nodes.get().map((node) => Object.assign({}, node));
  const originalEdges = edges.get().map((edge) => Object.assign({}, edge));
  const nodeIndex = new Map(originalNodes.map((node) => [String(node.id), node]));
  let focusedNode = null;
  let neighborhoodIds = null;
  let physicsEnabled = false;
  let filterTimer = null;

  const text = (value) => String(value == null ? "" : value);
  const folded = (value) => text(value).normalize("NFKC").toLocaleLowerCase();
  const values = (value) => Array.isArray(value) ? value.map(text) : text(value).split("|").filter(Boolean);
  const uniqueSorted = (items) => Array.from(new Set(items.filter(Boolean))).sort((a, b) => a.localeCompare(b));
  const countLabel = (count, singular) => `${count.toLocaleString()} ${count === 1 ? singular : `${singular}s`}`;

  function addOptions(select, optionValues) {
    uniqueSorted(optionValues).forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      select.appendChild(option);
    });
  }

  addOptions(language, originalNodes.map((node) => text(node.wfLanguage || "und")));
  addOptions(source, originalNodes.map((node) => text(node.wfSource || "unknown")));
  addOptions(relationship, originalEdges.flatMap((edge) => values(edge.wfTypes)));

  function activeDimensions() {
    return new Set(Array.from(document.querySelectorAll('input[name="wf-dimension"]:checked')).map((input) => input.value));
  }

  function nodePasses(node, query) {
    if (neighborhoodIds && !neighborhoodIds.has(String(node.id))) return false;
    if (query && !folded(node.wfTerm || node.label).includes(query)) return false;
    if (language.value && text(node.wfLanguage) !== language.value) return false;
    if (source.value && text(node.wfSource) !== source.value) return false;
    if (hideStubs.checked && Boolean(node.wfStub)) return false;
    return true;
  }

  function edgePasses(edge, dimensions) {
    const edgeDimensions = values(edge.wfDimensions);
    if (!edgeDimensions.some((value) => dimensions.has(value))) return false;
    if (relationship.value && !values(edge.wfTypes).includes(relationship.value)) return false;
    return true;
  }

  function applyFilters(options) {
    const settings = options || {};
    const query = folded(search.value.trim());
    const dimensions = activeDimensions();
    const candidateNodes = new Set(originalNodes.filter((node) => nodePasses(node, query)).map((node) => String(node.id)));
    const visibleEdges = new Set();
    const incidentNodes = new Set();

    originalEdges.forEach((edge) => {
      const from = String(edge.from);
      const to = String(edge.to);
      const visible = candidateNodes.has(from) && candidateNodes.has(to) && edgePasses(edge, dimensions);
      if (visible) {
        visibleEdges.add(String(edge.id));
        incidentNodes.add(from);
        incidentNodes.add(to);
      }
    });

    const visibleNodes = new Set(Array.from(candidateNodes).filter((id) => showIsolates.checked || incidentNodes.has(id)));
    nodes.update(originalNodes.map((node) => ({
      id: node.id,
      hidden: !visibleNodes.has(String(node.id)),
      label: showLabels.checked ? text(node.wfDisplayLabel || node.label) : ""
    })));
    edges.update(originalEdges.map((edge) => ({
      id: edge.id,
      hidden: !visibleEdges.has(String(edge.id)) || !visibleNodes.has(String(edge.from)) || !visibleNodes.has(String(edge.to))
    })));

    results.textContent = `${countLabel(visibleNodes.size, "node")} · ${countLabel(visibleEdges.size, "connection")} visible`;
    if (settings.fit) window.setTimeout(() => network.fit({ animation: { duration: 260, easingFunction: "easeInOutQuad" } }), 20);
    network.redraw();
    return visibleNodes;
  }

  function scheduleFilters() {
    window.clearTimeout(filterTimer);
    filterTimer = window.setTimeout(() => applyFilters(), 90);
  }

  function setInspectorField(id, value) { byId(id).textContent = text(value) || "—"; }

  function inspectNode(nodeId) {
    const node = nodeIndex.get(String(nodeId));
    if (!node) return;
    focusedNode = String(nodeId);
    neighborhood.disabled = false;
    byId("wf-inspector-kind").textContent = Boolean(node.wfStub) ? "Unenriched node" : "Lexical node";
    byId("wf-inspector-title").textContent = text(node.wfTerm || node.label);
    setInspectorField("wf-inspector-language", node.wfLanguage || "und");
    setInspectorField("wf-inspector-script", node.wfScript || "Zzzz");
    setInspectorField("wf-inspector-source", node.wfSource || "unknown");
    setInspectorField("wf-inspector-count", network.getConnectedEdges(nodeId).length);
    inspector.hidden = false;
  }

  function inspectEdge(edgeId) {
    const edge = originalEdges.find((candidate) => String(candidate.id) === String(edgeId));
    if (!edge) return;
    focusedNode = null;
    neighborhood.disabled = true;
    const from = nodeIndex.get(String(edge.from));
    const to = nodeIndex.get(String(edge.to));
    byId("wf-inspector-kind").textContent = "Connection";
    byId("wf-inspector-title").textContent = `${text(from && (from.wfTerm || from.label))} ↔ ${text(to && (to.wfTerm || to.label))}`;
    setInspectorField("wf-inspector-language", values(edge.wfDimensions).join(", "));
    setInspectorField("wf-inspector-script", values(edge.wfTypes).join(", "));
    setInspectorField("wf-inspector-source", values(edge.wfSources).join(", "));
    setInspectorField("wf-inspector-count", values(edge.wfAssertions).length || edge.wfAssertionCount || 1);
    inspector.hidden = false;
  }

  function focusSearchResult() {
    const query = folded(search.value.trim());
    if (!query) return;
    const visible = applyFilters();
    const candidates = originalNodes.filter((node) => visible.has(String(node.id)));
    candidates.sort((left, right) => {
      const leftTerm = folded(left.wfTerm || left.label);
      const rightTerm = folded(right.wfTerm || right.label);
      const leftRank = leftTerm === query ? 0 : leftTerm.startsWith(query) ? 1 : 2;
      const rightRank = rightTerm === query ? 0 : rightTerm.startsWith(query) ? 1 : 2;
      return leftRank - rightRank || leftTerm.localeCompare(rightTerm);
    });
    if (!candidates.length) return;
    const nodeId = candidates[0].id;
    network.selectNodes([nodeId]);
    network.focus(nodeId, { scale: 1.35, animation: { duration: 320, easingFunction: "easeInOutQuad" } });
    inspectNode(nodeId);
  }

  function resetView() {
    search.value = "";
    language.value = "";
    source.value = "";
    relationship.value = "";
    hideStubs.checked = false;
    showIsolates.checked = true;
    showLabels.checked = true;
    document.querySelectorAll('input[name="wf-dimension"]').forEach((input) => { input.checked = true; });
    neighborhoodIds = null;
    focusedNode = null;
    neighborhood.disabled = true;
    inspector.hidden = true;
    network.unselectAll();
    applyFilters({ fit: true });
  }

  document.querySelectorAll('#wf-sidebar input, #wf-sidebar select').forEach((control) => {
    control.addEventListener(control === search ? "input" : "change", scheduleFilters);
  });
  search.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); focusSearchResult(); } });
  byId("wf-fit").addEventListener("click", () => network.fit({ animation: { duration: 300, easingFunction: "easeInOutQuad" } }));
  byId("wf-reset").addEventListener("click", resetView);
  byId("wf-close-inspector").addEventListener("click", () => { inspector.hidden = true; network.unselectAll(); focusedNode = null; neighborhood.disabled = true; });
  byId("wf-physics").addEventListener("click", (event) => {
    physicsEnabled = !physicsEnabled;
    event.currentTarget.setAttribute("aria-pressed", String(physicsEnabled));
    event.currentTarget.textContent = physicsEnabled ? "Freeze layout" : "Live layout";
    network.setOptions({ physics: { enabled: physicsEnabled, stabilization: { iterations: 120 } } });
    if (physicsEnabled) network.startSimulation(); else network.stopSimulation();
  });
  neighborhood.addEventListener("click", () => {
    if (!focusedNode) return;
    const connected = network.getConnectedNodes(focusedNode).map(String);
    search.value = "";
    neighborhoodIds = new Set([focusedNode].concat(connected));
    applyFilters({ fit: true });
  });
  network.on("selectNode", (params) => { if (params.nodes.length) inspectNode(params.nodes[0]); });
  network.on("selectEdge", (params) => { if (!params.nodes.length && params.edges.length) inspectEdge(params.edges[0]); });
  network.on("deselectNode", () => { focusedNode = null; neighborhood.disabled = true; });

  const sidebar = byId("wf-sidebar");
  function setSidebar(open) {
    sidebar.classList.toggle("wf-sidebar--open", open);
    byId("wf-toggle-filters").setAttribute("aria-expanded", String(open));
  }
  byId("wf-toggle-filters").addEventListener("click", () => setSidebar(!sidebar.classList.contains("wf-sidebar--open")));
  byId("wf-close-filters").addEventListener("click", () => setSidebar(false));
  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement !== search) { event.preventDefault(); search.focus(); }
    if (event.key === "Escape") { setSidebar(false); network.unselectAll(); inspector.hidden = true; }
  });

  applyFilters();
  window.setTimeout(() => network.fit({ animation: false }), 80);
})();
"""


__all__ = [
    "GraphViewerMetadata",
    "atomic_write_graph_viewer",
    "render_graph_viewer",
]
