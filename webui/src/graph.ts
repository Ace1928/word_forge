import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import FA2Layout from "graphology-layout-forceatlas2/worker";
import Sigma from "sigma";

import type { NetworkNode, NetworkSnapshot } from "./types";

const COLORS = ["#78e8c2", "#8eb8ff", "#f7be6d", "#d69cff", "#ff8b8b", "#7bd8ef"];

function hash(value: string): number {
  let result = 2166136261;
  for (const character of value) {
    result ^= character.codePointAt(0) ?? 0;
    result = Math.imul(result, 16777619);
  }
  return result >>> 0;
}

function languageColor(language: string): string {
  return COLORS[hash(language) % COLORS.length] ?? COLORS[0]!;
}

function initialPosition(id: string): { x: number; y: number } {
  const value = hash(id);
  const angle = ((value % 3600) / 3600) * Math.PI * 2;
  const radius = 0.2 + (((value >>> 12) % 800) / 1000);
  return { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius };
}

export interface GraphController {
  destroy(): void;
  fit(): void;
  zoomIn(): void;
  zoomOut(): void;
}

export function mountNetwork(
  container: HTMLElement,
  snapshot: NetworkSnapshot,
  onSelect: (node: NetworkNode) => void,
): GraphController {
  const graph = new Graph({ multi: true, type: "directed" });
  const nodesById = new Map(snapshot.nodes.map((node) => [node.id, node]));
  for (const node of snapshot.nodes) {
    graph.addNode(node.id, {
      ...initialPosition(node.id),
      label: node.term,
      color: languageColor(node.language),
      size: Math.min(17, 4 + Math.sqrt(node.weight) * 2.2),
      forceLabel: node.weight > 3,
    });
  }
  for (const edge of snapshot.edges) {
    if (graph.hasNode(edge.source) && graph.hasNode(edge.target)) {
      graph.addDirectedEdgeWithKey(edge.id, edge.source, edge.target, {
        label: edge.kind,
        color: "#496074",
        size: 0.8,
      });
    }
  }

  const renderer = new Sigma(graph, container, {
    allowInvalidContainer: false,
    defaultEdgeType: "arrow",
    enableEdgeEvents: false,
    labelColor: { color: "#dce8ed" },
    labelDensity: 0.08,
    labelGridCellSize: 110,
    labelRenderedSizeThreshold: 7,
    renderEdgeLabels: false,
    stagePadding: 44,
    zIndex: true,
  });

  let selected: string | null = null;
  renderer.on("clickNode", ({ node }) => {
    selected = node;
    const lexicalNode = nodesById.get(node);
    if (lexicalNode) onSelect(lexicalNode);
    renderer.refresh();
  });
  renderer.setSetting("nodeReducer", (node, data) => {
    if (!selected || node === selected) return data;
    return { ...data, color: "#344958", label: "", zIndex: 0 };
  });

  let layout: FA2Layout | null = null;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reducedMotion && graph.order > 1 && graph.order <= 1_000) {
    layout = new FA2Layout(graph, {
      settings: {
        ...forceAtlas2.inferSettings(graph),
        barnesHutOptimize: graph.order > 300,
        gravity: 0.08,
        scalingRatio: 8,
        slowDown: 5,
      },
    });
    layout.start();
    window.setTimeout(() => layout?.stop(), Math.min(4_000, 900 + graph.order * 3));
  }

  const camera = renderer.getCamera();
  return {
    destroy() {
      layout?.kill();
      renderer.kill();
      graph.clear();
    },
    fit() {
      camera.animatedReset({ duration: reducedMotion ? 0 : 280 });
    },
    zoomIn() {
      camera.animatedZoom({ duration: reducedMotion ? 0 : 180 });
    },
    zoomOut() {
      camera.animatedUnzoom({ duration: reducedMotion ? 0 : 180 });
    },
  };
}
