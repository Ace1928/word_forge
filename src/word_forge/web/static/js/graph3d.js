/* ═══════════════════════════════════════════════════════════════════
   Word Forge — 3D Graph Renderer (3d-force-graph)
   Lazy-loaded: CDN script is injected only when 3D mode is first activated.
   ═══════════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    var CDN_URL = 'https://unpkg.com/3d-force-graph';
    var _scriptLoaded = false;
    var _scriptLoading = false;
    var _loadCallbacks = [];

    /**
     * Lazy-load the 3d-force-graph library from CDN.
     * @returns {Promise<void>}
     */
    function ensureLib() {
        if (_scriptLoaded && window.ForceGraph3D) {
            return Promise.resolve();
        }

        if (_scriptLoading) {
            return new Promise(function (resolve, reject) {
                _loadCallbacks.push({ resolve: resolve, reject: reject });
            });
        }

        _scriptLoading = true;

        return new Promise(function (resolve, reject) {
            var script = document.createElement('script');
            script.src = CDN_URL;
            script.async = true;

            script.onload = function () {
                _scriptLoaded = true;
                _scriptLoading = false;
                resolve();
                _loadCallbacks.forEach(function (cb) { cb.resolve(); });
                _loadCallbacks = [];
            };

            script.onerror = function () {
                _scriptLoading = false;
                var err = new Error('Failed to load 3d-force-graph from CDN');
                reject(err);
                _loadCallbacks.forEach(function (cb) { cb.reject(err); });
                _loadCallbacks = [];
            };

            document.head.appendChild(script);
        });
    }


    class Graph3D {
        constructor() {
            this.graph = null;
            this.container = null;
            this._onNodeSelectCb = null;
            this._currentData = null;
        }

        /**
         * Initialize the 3D force graph.
         * @param {HTMLElement} container
         * @returns {Promise<void>}
         */
        async init(container) {
            this.container = container;

            await ensureLib();

            this.graph = ForceGraph3D()(container)
                .backgroundColor('#0a0e1a')
                .showNavInfo(false)
                .nodeLabel(function (node) { return node.label || node.id; })
                .nodeColor(function (node) {
                    return (node.color && node.color.background) ? node.color.background : '#38bdf8';
                })
                .nodeVal(function (node) {
                    return node.size ? node.size / 5 : 4;
                })
                .nodeOpacity(0.92)
                .linkColor(function (link) {
                    return (link.color && typeof link.color === 'string')
                        ? link.color
                        : 'rgba(148,163,184,0.35)';
                })
                .linkWidth(0.6)
                .linkOpacity(0.5)
                .linkDirectionalArrowLength(3)
                .linkDirectionalArrowRelPos(1)
                .enableNodeDrag(true)
                .enableNavigationControls(true);

            // Bloom post-processing (if Three is available)
            try {
                if (this.graph.postProcessingComposer) {
                    // 3d-force-graph exposes this when Three.js UnrealBloomPass is available
                    // This is a soft attempt — bloom is a nice-to-have
                }
            } catch (_) { /* no-op */ }

            // Node click handler
            var self = this;
            this.graph.onNodeClick(function (node) {
                if (self._onNodeSelectCb && node) {
                    self._onNodeSelectCb(node);
                }
            });
        }

        /**
         * Set graph data.  Converts vis.js format → 3d-force-graph format.
         * vis.js edges use `from`/`to`; 3d-force-graph uses `source`/`target`.
         * @param {Array} nodes
         * @param {Array} edges
         * @returns {{ largeGraph: boolean }}
         */
        setData(nodes, edges) {
            // Deep-clone to avoid mutating caller data
            var g3dNodes = nodes.map(function (n) {
                return Object.assign({}, n);
            });

            var g3dLinks = edges.map(function (e) {
                return {
                    source: e.from,
                    target: e.to,
                    label: e.label || '',
                    color: e.color || undefined
                };
            });

            this._currentData = { nodes: g3dNodes, links: g3dLinks };

            if (this.graph) {
                this.graph.graphData(this._currentData);
            }

            return { largeGraph: nodes.length > 1500 };
        }

        /**
         * Toggle physics simulation.
         * @param {boolean} enabled
         */
        setPhysics(enabled) {
            if (!this.graph) return;

            if (enabled) {
                this.graph.d3ReheatSimulation();
            } else {
                // Cool down immediately
                this.graph.cooldownTicks(0);
                if (this.graph.d3Force) {
                    try { this.graph.d3Force('charge').strength(0); } catch (_) {}
                }
            }
        }

        /**
         * Reposition camera to fit all nodes.
         */
        fitView() {
            if (!this.graph) return;
            this.graph.zoomToFit(600, 50);
        }

        /**
         * Register callback for node click/select.
         * @param {function} callback — receives node data object
         */
        onNodeSelect(callback) {
            this._onNodeSelectCb = callback;
            // Re-wire if graph already exists
            if (this.graph) {
                var self = this;
                this.graph.onNodeClick(function (node) {
                    if (self._onNodeSelectCb && node) {
                        self._onNodeSelectCb(node);
                    }
                });
            }
        }

        /**
         * Export the current view as a PNG data URL.
         * @returns {string|null}
         */
        exportPNG() {
            if (!this.graph) return null;
            try {
                var renderer = this.graph.renderer();
                if (renderer && renderer.domElement) {
                    return renderer.domElement.toDataURL('image/png');
                }
            } catch (_) {}
            return null;
        }

        /**
         * Resize to fit container.
         */
        resize() {
            if (this.graph && this.container) {
                this.graph.width(this.container.clientWidth);
                this.graph.height(this.container.clientHeight);
            }
        }

        /**
         * Clean up resources.
         */
        destroy() {
            if (this.graph) {
                this.graph._destructor && this.graph._destructor();
                this.graph = null;
            }
            if (this.container) {
                this.container.innerHTML = '';
            }
            this._currentData = null;
        }

        /**
         * Check if initialized.
         * @returns {boolean}
         */
        isReady() {
            return this.graph !== null;
        }
    }

    // Export
    window.Graph3D = Graph3D;
})();
