/* ═══════════════════════════════════════════════════════════════════
   Word Forge — Main Application Controller
   Coordinates: Search, Graph2D, Graph3D, Inspector
   ═══════════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    class WordForgeApp {
        constructor() {
            // ─── State ───
            this.focusTerm = '';
            this.renderMode = '2d';   // '2d' | '3d'
            this.physicsEnabled = true;
            this.wholeGraph = false;
            this.depth = 2;
            this.activeDimensions = ['lexical', 'emotional'];

            // ─── Module instances ───
            this.search = null;
            this.graph2d = null;
            this.graph3d = null;
            this.inspector = null;

            // ─── Cached graph data ───
            this._lastNodes = [];
            this._lastEdges = [];

            // ─── DOM refs ───
            this.els = {};
        }

        /**
         * Boot the application.
         */
        init() {
            this._cacheDom();
            this._initModules();
            this._bindControls();
            this._bindKeyboard();
            this.loadStats();

            console.log('[WordForge] Initialized');
        }

        /* ═══════════════════════════════
           DOM & Modules
           ═══════════════════════════════ */

        _cacheDom() {
            this.els = {
                searchInput:     document.getElementById('wf-search-input'),
                searchResults:   document.getElementById('wf-search-results'),
                depthSlider:     document.getElementById('wf-depth-slider'),
                depthValue:      document.getElementById('wf-depth-value'),
                wholeGraphCb:    document.getElementById('wf-whole-graph'),
                mode3dCb:        document.getElementById('wf-3d-mode'),
                physicsCb:       document.getElementById('wf-physics'),
                computeBtn:      document.getElementById('wf-compute-btn'),
                graphContainer:  document.getElementById('wf-graph-container'),
                emptyState:      document.getElementById('wf-empty-state'),
                loadingOverlay:  document.getElementById('wf-loading-overlay'),
                hudNodes:        document.getElementById('wf-hud-nodes'),
                hudEdges:        document.getElementById('wf-hud-edges'),
                largeWarning:    document.getElementById('wf-large-graph-warning'),
                btnFit:          document.getElementById('wf-btn-fit'),
                btnZoomIn:       document.getElementById('wf-btn-zoom-in'),
                btnZoomOut:      document.getElementById('wf-btn-zoom-out'),
                btnExport:       document.getElementById('wf-btn-export'),
                // Stats
                statWords:       document.getElementById('wf-stat-words'),
                statRels:        document.getElementById('wf-stat-rels'),
                statLangs:       document.getElementById('wf-stat-langs'),
                statStubs:       document.getElementById('wf-stat-stubs'),
                // Dimension checkboxes
                dimCheckboxes:   document.querySelectorAll('.wf-dim-tag input[type="checkbox"]'),
            };
        }

        _initModules() {
            var self = this;

            // Search
            this.search = new Search(this.els.searchInput, this.els.searchResults, {
                onSelect: function (item) {
                    self.focusTerm = item.term;
                    self.loadGraph();
                }
            });

            // 2D graph — init immediately
            this.graph2d = new Graph2D();
            this.graph2d.init(this.els.graphContainer);
            this.graph2d.onNodeSelect(function (node) {
                self._onNodeSelected(node);
            });

            // 3D graph — lazy, created on demand
            this.graph3d = new Graph3D();

            // Inspector
            this.inspector = new Inspector();
            this.inspector.onNavigate(function (term) {
                self.focusTerm = term;
                self.search.setValue(term);
                self.loadGraph();
            });
        }

        /* ═══════════════════════════════
           Control Binding
           ═══════════════════════════════ */

        _bindControls() {
            var self = this;

            // Depth slider
            this.els.depthSlider.addEventListener('input', function () {
                self.depth = parseInt(this.value, 10);
                self.els.depthValue.textContent = self.depth;
            });

            // Dimension checkboxes
            this.els.dimCheckboxes.forEach(function (cb) {
                cb.addEventListener('change', function () {
                    self._syncDimensions();
                });
            });

            // Whole graph toggle
            this.els.wholeGraphCb.addEventListener('change', function () {
                self.wholeGraph = this.checked;
            });

            // 3D mode toggle
            this.els.mode3dCb.addEventListener('change', function () {
                self.renderMode = this.checked ? '3d' : '2d';
                self._switchRenderer();
            });

            // Physics toggle
            this.els.physicsCb.addEventListener('change', function () {
                self.physicsEnabled = this.checked;
                self._activeRenderer().setPhysics(self.physicsEnabled);
            });

            // Compute button
            this.els.computeBtn.addEventListener('click', function () {
                self.loadGraph();
            });

            // Toolbar buttons
            this.els.btnFit.addEventListener('click', function () {
                self._activeRenderer().fitView();
            });

            this.els.btnZoomIn.addEventListener('click', function () {
                if (self.renderMode === '2d' && self.graph2d) {
                    self.graph2d.zoomIn();
                }
                // 3D zoom handled by orbit controls
            });

            this.els.btnZoomOut.addEventListener('click', function () {
                if (self.renderMode === '2d' && self.graph2d) {
                    self.graph2d.zoomOut();
                }
            });

            this.els.btnExport.addEventListener('click', function () {
                self.exportPNG();
            });
        }

        _bindKeyboard() {
            var self = this;

            document.addEventListener('keydown', function (e) {
                // Escape → close inspector
                if (e.key === 'Escape') {
                    if (self.inspector && self.inspector.isOpen()) {
                        self.inspector.close();
                    }
                }
            });
        }

        _syncDimensions() {
            this.activeDimensions = [];
            this.els.dimCheckboxes.forEach(function (cb) {
                if (cb.checked) {
                    this.activeDimensions.push(cb.value);
                }
            }.bind(this));
        }

        /* ═══════════════════════════════
           Graph Loading
           ═══════════════════════════════ */

        async loadGraph() {
            // Build query params
            var params = new URLSearchParams();

            if (this.focusTerm) {
                params.set('focus', this.focusTerm);
            }
            params.set('depth', this.depth);
            params.set('whole_graph', this.wholeGraph);

            if (this.activeDimensions.length > 0) {
                params.set('dimensions', this.activeDimensions.join(','));
            }

            this._showLoading(true);

            try {
                var resp = await fetch('/api/graph?' + params.toString());
                if (!resp.ok) throw new Error('Failed to load graph (HTTP ' + resp.status + ')');

                var data = await resp.json();
                var nodes = data.nodes || [];
                var edges = data.edges || [];

                this._lastNodes = nodes;
                this._lastEdges = edges;

                // Dispatch to active renderer
                var result = this._activeRenderer().setData(nodes, edges);

                // Update HUD
                this.els.hudNodes.textContent = nodes.length + ' node' + (nodes.length !== 1 ? 's' : '');
                this.els.hudEdges.textContent = edges.length + ' edge' + (edges.length !== 1 ? 's' : '');

                // Empty state
                this.els.emptyState.classList.toggle('is-hidden', nodes.length > 0);

                // Large graph warning
                if (result && result.largeGraph) {
                    this.els.largeWarning.classList.add('is-visible');
                    this.els.physicsCb.checked = false;
                    this.physicsEnabled = false;
                } else {
                    this.els.largeWarning.classList.remove('is-visible');
                }

                // Fit after stabilization
                var renderer = this._activeRenderer();
                setTimeout(function () { renderer.fitView(); }, 600);

            } catch (err) {
                console.error('[WordForge] Graph load error:', err);
                this._toast('Error: ' + err.message, true);
            } finally {
                this._showLoading(false);
            }
        }

        /* ═══════════════════════════════
           Stats Loading
           ═══════════════════════════════ */

        async loadStats() {
            try {
                var resp = await fetch('/api/stats');
                if (!resp.ok) return;
                var stats = await resp.json();

                if (stats.total_words != null) {
                    this.els.statWords.textContent = this._formatNum(stats.total_words);
                }
                if (stats.total_relationships != null) {
                    this.els.statRels.textContent = this._formatNum(stats.total_relationships);
                }
                if (stats.total_languages != null) {
                    this.els.statLangs.textContent = stats.total_languages;
                }
                if (stats.total_stubs != null) {
                    this.els.statStubs.textContent = this._formatNum(stats.total_stubs);
                }
            } catch (err) {
                console.warn('[WordForge] Stats load failed:', err);
            }
        }

        /* ═══════════════════════════════
           Renderer Switching
           ═══════════════════════════════ */

        _activeRenderer() {
            return this.renderMode === '3d' ? this.graph3d : this.graph2d;
        }

        async _switchRenderer() {
            if (this.renderMode === '3d') {
                // Destroy 2D
                if (this.graph2d && this.graph2d.isReady()) {
                    this.graph2d.destroy();
                }
                // Clear container
                this.els.graphContainer.innerHTML = '';

                // Init 3D (lazy)
                this._showLoading(true);
                try {
                    var self = this;
                    await this.graph3d.init(this.els.graphContainer);
                    this.graph3d.onNodeSelect(function (node) {
                        self._onNodeSelected(node);
                    });

                    // Re-render existing data
                    if (this._lastNodes.length > 0) {
                        this.graph3d.setData(this._lastNodes, this._lastEdges);
                        setTimeout(function () { self.graph3d.fitView(); }, 800);
                    }

                    this._toast('Switched to 3D mode');
                } catch (err) {
                    console.error('[WordForge] 3D init error:', err);
                    this._toast('Failed to load 3D renderer', true);
                    // Fall back to 2D
                    this.renderMode = '2d';
                    this.els.mode3dCb.checked = false;
                    this._switchRenderer();
                } finally {
                    this._showLoading(false);
                }
            } else {
                // Destroy 3D
                if (this.graph3d && this.graph3d.isReady()) {
                    this.graph3d.destroy();
                }
                // Clear container
                this.els.graphContainer.innerHTML = '';

                // Re-init 2D
                this.graph2d = new Graph2D();
                this.graph2d.init(this.els.graphContainer);

                var self2 = this;
                this.graph2d.onNodeSelect(function (node) {
                    self2._onNodeSelected(node);
                });

                // Re-render existing data
                if (this._lastNodes.length > 0) {
                    this.graph2d.setData(this._lastNodes, this._lastEdges);
                    this.graph2d.setPhysics(this.physicsEnabled);
                    setTimeout(function () { self2.graph2d.fitView(); }, 400);
                }

                this._toast('Switched to 2D mode');
            }
        }

        /* ═══════════════════════════════
           Node Selection
           ═══════════════════════════════ */

        _onNodeSelected(node) {
            if (!node) return;

            // Use wfTerm if present (vis.js custom field), else label
            var term = node.wfTerm || node.label || node.id;
            if (term) {
                this.inspector.open(term);
            }
        }

        /* ═══════════════════════════════
           Export
           ═══════════════════════════════ */

        exportPNG() {
            var dataUrl = this._activeRenderer().exportPNG();
            if (!dataUrl) {
                this._toast('Export not available', true);
                return;
            }

            var link = document.createElement('a');
            link.download = 'word-forge-graph.png';
            link.href = dataUrl;
            link.click();

            this._toast('PNG exported');
        }

        /* ═══════════════════════════════
           Utilities
           ═══════════════════════════════ */

        _showLoading(show) {
            if (this.els.loadingOverlay) {
                this.els.loadingOverlay.classList.toggle('is-visible', show);
                this.els.loadingOverlay.setAttribute('aria-hidden', !show);
            }
        }

        _toast(message, isError) {
            var container = document.getElementById('wf-toast-container');
            if (!container) return;

            var toast = document.createElement('div');
            toast.className = 'wf-toast' + (isError ? ' wf-toast--error' : '');
            toast.textContent = message;
            container.appendChild(toast);

            setTimeout(function () {
                toast.classList.add('is-leaving');
                setTimeout(function () { toast.remove(); }, 300);
            }, 3000);
        }

        _formatNum(n) {
            if (n == null) return '—';
            if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
            if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
            return String(n);
        }
    }

    // ─── Bootstrap ───
    window.WordForgeApp = WordForgeApp;

    document.addEventListener('DOMContentLoaded', function () {
        var app = new WordForgeApp();
        app.init();
        window._wfApp = app; // Expose for debugging
    });
})();
