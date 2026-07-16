/* ═══════════════════════════════════════════════════════════════════
   Word Forge — 2D Graph Renderer (vis-network)
   ═══════════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    var LARGE_GRAPH_THRESHOLD = 1500;

    class Graph2D {
        constructor() {
            this.network = null;
            this.container = null;
            this.nodesDS = null;
            this.edgesDS = null;
            this._onNodeSelectCb = null;
            this._onNodeHoverCb = null;
        }

        /**
         * Initialize the vis.Network instance.
         * @param {HTMLElement} container
         */
        init(container) {
            this.container = container;
            this.nodesDS = new vis.DataSet([]);
            this.edgesDS = new vis.DataSet([]);

            var options = {
                autoResize: true,
                nodes: {
                    shape: 'dot',
                    font: {
                        color: '#e2e8f0',
                        face: 'Inter, sans-serif',
                        size: 13,
                        strokeWidth: 3,
                        strokeColor: '#0a0e1a'
                    },
                    borderWidth: 2,
                    borderWidthSelected: 3,
                    scaling: {
                        min: 10,
                        max: 40,
                        label: { enabled: true, min: 11, max: 18 }
                    },
                    shadow: {
                        enabled: true,
                        color: 'rgba(0,0,0,0.35)',
                        size: 8,
                        x: 0,
                        y: 3
                    }
                },
                edges: {
                    smooth: {
                        type: 'continuous',
                        roundness: 0.4
                    },
                    arrows: {
                        to: { enabled: true, scaleFactor: 0.4, type: 'arrow' }
                    },
                    color: {
                        inherit: 'from',
                        opacity: 0.6
                    },
                    font: {
                        color: '#64748b',
                        face: 'Inter, sans-serif',
                        size: 9,
                        strokeWidth: 2,
                        strokeColor: '#0a0e1a',
                        align: 'middle'
                    },
                    width: 1.2,
                    hoverWidth: 2,
                    selectionWidth: 2.5
                },
                physics: {
                    enabled: true,
                    solver: 'barnesHut',
                    barnesHut: {
                        gravitationalConstant: -3500,
                        centralGravity: 0.25,
                        springLength: 120,
                        springConstant: 0.03,
                        damping: 0.12,
                        avoidOverlap: 0.3
                    },
                    stabilization: {
                        enabled: true,
                        iterations: 150,
                        updateInterval: 25
                    }
                },
                interaction: {
                    hover: true,
                    tooltipDelay: 100,
                    zoomView: true,
                    dragView: true,
                    multiselect: false,
                    navigationButtons: false,
                    keyboard: false
                },
                layout: {
                    improvedLayout: true,
                    randomSeed: 42
                }
            };

            this.network = new vis.Network(
                container,
                { nodes: this.nodesDS, edges: this.edgesDS },
                options
            );

            // Wire up internal events
            var self = this;

            this.network.on('selectNode', function (params) {
                if (self._onNodeSelectCb && params.nodes.length > 0) {
                    var nodeId = params.nodes[0];
                    var nodeData = self.nodesDS.get(nodeId);
                    self._onNodeSelectCb(nodeData);
                }
            });

            this.network.on('hoverNode', function () {
                container.style.cursor = 'pointer';
            });

            this.network.on('blurNode', function () {
                container.style.cursor = 'default';
            });

            if (self._onNodeHoverCb) {
                this.network.on('hoverNode', function (params) {
                    var nodeData = self.nodesDS.get(params.node);
                    self._onNodeHoverCb(nodeData, true);
                });
                this.network.on('blurNode', function (params) {
                    var nodeData = self.nodesDS.get(params.node);
                    self._onNodeHoverCb(nodeData, false);
                });
            }
        }

        /**
         * Set graph data. Auto-disables physics for large graphs.
         * @param {Array} nodes
         * @param {Array} edges
         * @returns {{ largeGraph: boolean }}
         */
        setData(nodes, edges) {
            var isLarge = nodes.length > LARGE_GRAPH_THRESHOLD;

            this.nodesDS.clear();
            this.edgesDS.clear();
            this.nodesDS.add(nodes);
            this.edgesDS.add(edges);

            if (isLarge) {
                this.setPhysics(false);
            }

            return { largeGraph: isLarge };
        }

        /**
         * Toggle physics simulation.
         * @param {boolean} enabled
         */
        setPhysics(enabled) {
            if (this.network) {
                this.network.setOptions({ physics: { enabled: enabled } });
            }
        }

        /**
         * Fit the graph into the viewport.
         */
        fitView() {
            if (this.network) {
                this.network.fit({ animation: { duration: 400, easingFunction: 'easeInOutQuad' } });
            }
        }

        /**
         * Zoom in.
         */
        zoomIn() {
            if (!this.network) return;
            var scale = this.network.getScale();
            this.network.moveTo({ scale: scale * 1.3, animation: { duration: 200, easingFunction: 'easeInOutQuad' } });
        }

        /**
         * Zoom out.
         */
        zoomOut() {
            if (!this.network) return;
            var scale = this.network.getScale();
            this.network.moveTo({ scale: scale / 1.3, animation: { duration: 200, easingFunction: 'easeInOutQuad' } });
        }

        /**
         * Register a callback for node selection.
         * @param {function} callback — receives node data object
         */
        onNodeSelect(callback) {
            this._onNodeSelectCb = callback;
        }

        /**
         * Register a callback for node hover.
         * @param {function} callback — receives (nodeData, isHovering)
         */
        onNodeHover(callback) {
            this._onNodeHoverCb = callback;
        }

        /**
         * Export the current view as a PNG data URL.
         * @returns {string|null}
         */
        exportPNG() {
            if (!this.network) return null;
            var canvas = this.container.querySelector('canvas');
            if (!canvas) return null;
            return canvas.toDataURL('image/png');
        }

        /**
         * Clean up resources.
         */
        destroy() {
            if (this.network) {
                this.network.destroy();
                this.network = null;
            }
            this.nodesDS = null;
            this.edgesDS = null;
        }

        /**
         * Check if initialized.
         * @returns {boolean}
         */
        isReady() {
            return this.network !== null;
        }
    }

    // Export
    window.Graph2D = Graph2D;
})();
