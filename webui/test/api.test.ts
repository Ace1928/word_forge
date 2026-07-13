import { afterEach, describe, expect, it, vi } from "vitest";

import { resolveSessionToken, WordForgeApi } from "../src/api";

class MemoryStorage {
  private values = new Map<string, string>();
  getItem(key: string): string | null { return this.values.get(key) ?? null; }
  setItem(key: string, value: string): void { this.values.set(key, value); }
}

describe("resolveSessionToken", () => {
  it("moves a launch token into session storage", () => {
    const storage = new MemoryStorage();
    const token = resolveSessionToken(
      { hash: "#token=secure-token", pathname: "/", search: "" },
      storage,
    );
    expect(token).toBe("secure-token");
    expect(storage.getItem("word-forge.session-token")).toBe("secure-token");
  });

  it("reuses the session token after the fragment is removed", () => {
    const storage = new MemoryStorage();
    storage.setItem("word-forge.session-token", "remembered-token");
    expect(
      resolveSessionToken({ hash: "", pathname: "/", search: "" }, storage),
    ).toBe("remembered-token");
  });
});

describe("WordForgeApi", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("sends authenticated maintenance requests", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 201,
      json: async () => ({ name: "snapshot.sqlite", size: 1024, modified_at: 1 }),
    });
    vi.stubGlobal("fetch", fetchMock);
    const api = new WordForgeApi("secret");
    await expect(api.createBackup("before-import")).resolves.toMatchObject({
      name: "snapshot.sqlite",
    });
    const [path, request] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(path).toBe("/api/maintenance/backups");
    expect(request.method).toBe("POST");
    expect(new Headers(request.headers).get("X-Word-Forge-Token")).toBe("secret");
    expect(JSON.parse(String(request.body))).toEqual({ label: "before-import" });
  });

  it("surfaces FastAPI detail messages", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: "Type RESTORE to confirm replacement" }),
    }));
    const api = new WordForgeApi("secret");
    await expect(api.restoreBackup("snapshot.sqlite", "no")).rejects.toEqual(
      expect.objectContaining({
        message: "Type RESTORE to confirm replacement",
        name: "ApiError",
        status: 400,
        code: "request_failed",
      }),
    );
  });
});
