export interface Relationship {
  id: string;
  target: string;
  kind: string;
  language: string | null;
  revision: number;
  source: string;
  confidence: number;
}

export interface Sense {
  id: string;
  definition: string;
  part_of_speech: string;
  domain: string;
  register: string;
  source: string;
  confidence: number;
  review_state: string;
  revision: number;
}

export interface LexicalEntry {
  id: string;
  term: string;
  language: string;
  normalized_term: string;
  script: string;
  definition: string;
  part_of_speech: string;
  examples: string[];
  relationships: Relationship[];
  senses: Sense[];
  source: string;
  base_term: string;
  revision: number;
  confidence: number;
  review_state: string;
  created_at: number;
  updated_at: number;
  last_refreshed: number;
}

export interface Statistics {
  words: number;
  lexemes: number;
  languages: number;
  relationships: number;
  translations: number;
}

export interface NetworkNode {
  id: string;
  term: string;
  language: string;
  definition: string;
  part_of_speech: string;
  weight: number;
}

export interface NetworkEdge {
  id: string;
  source: string;
  target: string;
  kind: string;
}

export interface NetworkSnapshot {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  truncated: boolean;
}

export interface ChangeEvent {
  type: "ready" | "entry.changed" | "graph.changed" | "sense.changed" | "database.restored" | "resync.required";
  term?: string;
  language?: string;
  target?: string;
  sense_id?: string;
  version?: string;
  backup?: string;
  sequence?: number;
}

export interface IntegrityReport {
  database: string;
  ok: boolean;
  integrity: string;
  schema_version: number;
  foreign_key_violations: number;
  pending_invalidations: number;
  pending_jobs: number;
  checked_at: number;
}

export interface BackupSnapshot {
  name: string;
  size: number;
  modified_at: number;
}
