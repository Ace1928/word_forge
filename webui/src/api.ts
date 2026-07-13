import type {
  BackupSnapshot,
  ChangeEvent,
  IntegrityReport,
  LexicalEntry,
  NetworkSnapshot,
  Sense,
  Statistics,
} from "./types";

const TOKEN_KEY = "word-forge.session-token";

export function resolveSessionToken(
  locationLike: Pick<Location, "hash" | "pathname" | "search"> = window.location,
  storage: Pick<Storage, "getItem" | "setItem"> = window.sessionStorage,
): string {
  const hash = new URLSearchParams(locationLike.hash.replace(/^#/, ""));
  const supplied = hash.get("token")?.trim() ?? "";
  if (supplied) {
    storage.setItem(TOKEN_KEY, supplied);
    if (typeof window !== "undefined" && window.history) {
      window.history.replaceState({}, "", `${locationLike.pathname}${locationLike.search}`);
    }
    return supplied;
  }
  return storage.getItem(TOKEN_KEY)?.trim() ?? "";
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code = "request_failed",
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export class WordForgeApi {
  constructor(readonly token: string) {}

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers);
    headers.set("Accept", "application/json");
    headers.set("X-Word-Forge-Token", this.token);
    if (init.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
    const response = await fetch(path, { ...init, headers });
    const payload = (await response.json().catch(() => ({}))) as Record<string, unknown>;
    if (!response.ok) {
      throw new ApiError(
        String(payload.message ?? payload.detail ?? `Request failed with HTTP ${response.status}`),
        response.status,
        String(payload.error ?? "request_failed"),
      );
    }
    return payload as T;
  }

  statistics(): Promise<Statistics> {
    return this.request<Statistics>("/api/stats");
  }

  entries(query: string, language = "", limit = 40): Promise<LexicalEntry[]> {
    const parameters = new URLSearchParams({ query, limit: String(limit) });
    if (language) parameters.set("language", language);
    return this.request<LexicalEntry[]>(`/api/entries?${parameters}`);
  }

  entry(term: string, language: string): Promise<LexicalEntry> {
    const parameters = new URLSearchParams({ term, language });
    return this.request<LexicalEntry>(`/api/entry?${parameters}`);
  }

  network(query = "", language = "", limit = 400): Promise<NetworkSnapshot> {
    const parameters = new URLSearchParams({ query, limit: String(limit) });
    if (language) parameters.set("language", language);
    return this.request<NetworkSnapshot>(`/api/network?${parameters}`);
  }

  addEntry(payload: Record<string, unknown>): Promise<LexicalEntry> {
    return this.request<LexicalEntry>("/api/entries", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  addSense(payload: Record<string, unknown>): Promise<Sense> {
    return this.request<Sense>("/api/senses", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  addRelationship(payload: Record<string, unknown>): Promise<{ created: boolean }> {
    return this.request<{ created: boolean }>("/api/relationships", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  addTranslation(payload: Record<string, unknown>): Promise<{ created: boolean }> {
    return this.request<{ created: boolean }>("/api/translations", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  deleteSense(id: string, expectedRevision: number): Promise<{ deleted: boolean }> {
    const parameters = new URLSearchParams({ expected_revision: String(expectedRevision) });
    return this.request<{ deleted: boolean }>(
      `/api/senses/${encodeURIComponent(id)}?${parameters}`,
      { method: "DELETE" },
    );
  }

  integrity(): Promise<IntegrityReport> {
    return this.request<IntegrityReport>("/api/maintenance/integrity");
  }

  backups(): Promise<BackupSnapshot[]> {
    return this.request<BackupSnapshot[]>("/api/maintenance/backups");
  }

  createBackup(label = ""): Promise<BackupSnapshot> {
    return this.request<BackupSnapshot>("/api/maintenance/backups", {
      method: "POST",
      body: JSON.stringify({ label }),
    });
  }

  restoreBackup(backupName: string, confirmation: string): Promise<{ restored: boolean }> {
    return this.request<{ restored: boolean }>("/api/maintenance/restore", {
      method: "POST",
      body: JSON.stringify({ backup_name: backupName, confirmation }),
    });
  }

  connectEvents(
    onEvent: (event: ChangeEvent) => void,
    onState: (state: "connecting" | "live" | "offline") => void,
  ): () => void {
    let stopped = false;
    let socket: WebSocket | null = null;
    let retry = 0;
    let reconnectTimer = 0;
    let lastSequence = 0;

    const connect = () => {
      if (stopped) return;
      onState("connecting");
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(`${protocol}//${window.location.host}/api/events?since=${lastSequence}`, [
        "word-forge",
        this.token,
      ]);
      socket.addEventListener("open", () => {
        retry = 0;
        onState("live");
      });
      socket.addEventListener("message", (message) => {
        try {
          const event = JSON.parse(String(message.data)) as ChangeEvent;
          if (typeof event.sequence === "number") lastSequence = Math.max(lastSequence, event.sequence);
          onEvent(event);
        } catch {
          // Invalid events are ignored; the next valid event still triggers a refetch.
        }
      });
      socket.addEventListener("close", () => {
        onState("offline");
        if (!stopped) {
          const delay = Math.min(10_000, 500 * 2 ** retry++);
          reconnectTimer = window.setTimeout(connect, delay);
        }
      });
    };
    connect();
    return () => {
      stopped = true;
      window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }
}
