import "./styles.css";

import { ApiError, resolveSessionToken, WordForgeApi } from "./api";
import { mountNetwork, type GraphController } from "./graph";
import type {
  BackupSnapshot,
  IntegrityReport,
  LexicalEntry,
  NetworkNode,
  NetworkSnapshot,
  Statistics,
} from "./types";

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("Word Forge application root is missing");

app.innerHTML = `
  <div class="app-shell">
    <aside class="sidebar" aria-label="Primary navigation">
      <div class="brand-block">
        <span class="brand-mark" aria-hidden="true">W</span>
        <div><strong>Word Forge</strong><span>Living lexicon</span></div>
      </div>
      <nav class="nav-list">
        <button class="nav-item is-active" data-target="overview"><span aria-hidden="true">01</span> Overview</button>
        <button class="nav-item" data-target="network"><span aria-hidden="true">02</span> Network</button>
        <button class="nav-item" data-target="lexicon"><span aria-hidden="true">03</span> Lexicon</button>
        <button class="nav-item" data-target="operations"><span aria-hidden="true">04</span> Operations</button>
      </nav>
      <div class="sidebar-footer">
        <span id="live-indicator" class="status-dot is-connecting" aria-hidden="true"></span>
        <span id="live-label">Connecting</span>
        <button id="theme-toggle" class="icon-button" type="button" aria-label="Toggle color theme">A</button>
      </div>
    </aside>

    <main id="workspace-main" class="workspace" tabindex="-1">
      <header class="topbar">
        <form id="global-search" class="search-box" role="search">
          <label class="sr-only" for="search-query">Search the lexicon</label>
          <span aria-hidden="true">/</span>
          <input id="search-query" name="query" autocomplete="off" placeholder="Search terms and definitions" />
          <kbd aria-hidden="true">Enter</kbd>
          <button class="search-submit" type="submit">Search</button>
        </form>
        <button id="add-entry" class="primary-button" type="button">+ New entry</button>
      </header>

      <section id="overview" class="hero-section" aria-labelledby="workspace-title">
        <div>
          <p class="eyebrow">Local knowledge, visibly connected</p>
          <h1 id="workspace-title">Explore meaning as a living system.</h1>
          <p class="hero-copy">Search, edit, translate, and trace relationships without surrendering your data to a hosted service.</p>
        </div>
        <div id="stats-grid" class="stats-grid" aria-label="Lexicon statistics"></div>
      </section>

      <section id="network" class="network-section" aria-labelledby="network-title">
        <div class="section-heading">
          <div><p class="eyebrow">Coordinated view</p><h2 id="network-title">Semantic network</h2></div>
          <div class="network-filters">
            <label>Language <input id="network-language" inputmode="text" maxlength="35" placeholder="all" /></label>
            <label>Nodes <select id="network-limit"><option>100</option><option selected>400</option><option>1000</option><option>2000</option></select></label>
            <button id="refresh-network" class="secondary-button" type="button">Refresh</button>
          </div>
        </div>
        <div class="network-workbench">
          <div class="graph-panel">
            <div id="graph-canvas" role="img" aria-label="Interactive semantic relationship graph"></div>
            <div class="graph-toolbar" aria-label="Graph view controls">
              <button id="zoom-in" class="icon-button" type="button" aria-label="Zoom in">+</button>
              <button id="zoom-out" class="icon-button" type="button" aria-label="Zoom out">-</button>
              <button id="zoom-fit" class="icon-button" type="button" aria-label="Fit graph">Fit</button>
            </div>
            <p id="graph-status" class="graph-status" aria-live="polite">Loading network...</p>
          </div>
          <aside id="node-inspector" class="inspector" aria-labelledby="inspector-title">
            <p class="eyebrow">Inspector</p>
            <h3 id="inspector-title">Select a term</h3>
            <p id="inspector-definition">Choose a node or a list entry to inspect its lexical record.</p>
            <div class="inspector-actions" aria-label="Selected entry actions">
              <button id="edit-entry" class="secondary-button" type="button" disabled>Edit</button>
              <button id="add-sense" class="secondary-button" type="button" disabled>+ Sense</button>
              <button id="add-relationship" class="secondary-button" type="button" disabled>+ Relation</button>
              <button id="add-translation" class="secondary-button" type="button" disabled>+ Translation</button>
            </div>
            <dl id="inspector-meta" class="metadata-list"></dl>
            <section class="sense-section" aria-labelledby="inspector-senses-title">
              <h4 id="inspector-senses-title">Senses</h4>
              <div id="inspector-senses" class="sense-list"></div>
            </section>
            <div id="inspector-relations" class="relationship-list"></div>
          </aside>
        </div>
        <details class="network-list-panel">
          <summary>Accessible network list <span id="network-list-count"></span></summary>
          <div id="network-list" class="result-list"></div>
        </details>
      </section>

      <section id="lexicon" class="lexicon-section" aria-labelledby="lexicon-title">
        <div class="section-heading">
          <div><p class="eyebrow">Structured view</p><h2 id="lexicon-title">Lexicon results</h2></div>
          <p id="result-summary" aria-live="polite">Search to inspect exact records.</p>
        </div>
        <div id="search-results" class="result-list result-grid"></div>
      </section>

      <section id="operations" class="operations-section" aria-labelledby="operations-title">
        <div class="section-heading"><div><p class="eyebrow">System</p><h2 id="operations-title">Operations</h2></div><button id="refresh-operations" class="secondary-button" type="button">Refresh health</button></div>
        <div class="operation-grid">
          <article class="operation-health"><span>Integrity</span><h3 id="integrity-title">Checking local database...</h3><dl id="integrity-details" class="metadata-list"></dl></article>
          <article><span>Snapshot</span><h3>Create a safe backup</h3><p>Uses SQLite's online backup API, validates the copy, then publishes it atomically.</p><label>Optional label <input id="backup-label" maxlength="80" placeholder="before-import" /></label><button id="create-backup" class="primary-button" type="button">Create snapshot</button></article>
          <article><span>Restore</span><h3>Return to a snapshot</h3><p>This replaces the active lexicon. Select a managed snapshot and type RESTORE.</p><label>Snapshot <select id="restore-backup"><option value="">No snapshots yet</option></select></label><label>Confirmation <input id="restore-confirmation" autocomplete="off" placeholder="RESTORE" /></label><button id="restore-backup-button" class="danger-button" type="button">Restore selected snapshot</button></article>
        </div>
        <div class="backup-panel"><h3>Managed snapshots</h3><div id="backup-list" class="backup-list"></div></div>
      </section>
    </main>
  </div>

  <dialog id="entry-dialog" aria-labelledby="entry-dialog-title">
    <form id="entry-form" method="dialog">
      <div class="dialog-heading"><div><p class="eyebrow">Lexical record</p><h2 id="entry-dialog-title">Add an entry</h2></div><button id="close-dialog" class="icon-button" type="button" aria-label="Close">x</button></div>
      <input name="expected_revision" type="hidden" />
      <label>Term <input name="term" required maxlength="512" autofocus /></label>
      <div class="form-row"><label>Language <input name="language" value="en" required maxlength="35" /></label><label>Part of speech <input name="part_of_speech" maxlength="100" /></label></div>
      <label>Definition <textarea name="definition" rows="4" maxlength="100000"></textarea></label>
      <label>English base term <input name="base_term" maxlength="512" /></label>
      <label>Source <input name="source" maxlength="500" value="manual" /></label>
      <label>Example <textarea name="example" rows="2"></textarea></label>
      <p id="form-error" class="form-error" role="alert"></p>
      <div class="dialog-actions"><button class="secondary-button" id="cancel-entry" type="button">Cancel</button><button class="primary-button" type="submit">Save entry</button></div>
    </form>
  </dialog>

  <dialog id="sense-dialog" aria-labelledby="sense-dialog-title">
    <form id="sense-form" method="dialog">
      <div class="dialog-heading"><div><p class="eyebrow">Meaning</p><h2 id="sense-dialog-title">Add a sense</h2></div><button class="icon-button dialog-close" type="button" aria-label="Close">x</button></div>
      <input name="term" type="hidden" /><input name="language" type="hidden" />
      <label>Definition <textarea name="definition" rows="4" maxlength="20000" required></textarea></label>
      <div class="form-row"><label>Part of speech <input name="part_of_speech" maxlength="100" /></label><label>Domain <input name="domain" maxlength="200" placeholder="finance, botany..." /></label></div>
      <div class="form-row"><label>Register <input name="register" maxlength="100" placeholder="formal, slang..." /></label><label>Source <input name="source" maxlength="500" value="manual" /></label></div>
      <div class="form-row"><label>Confidence <input name="confidence" type="number" min="0" max="1" step="0.01" value="1" /></label><label>Review state <select name="review_state"><option>unreviewed</option><option>reviewed</option><option>rejected</option></select></label></div>
      <p class="form-error" role="alert"></p><div class="dialog-actions"><button class="secondary-button dialog-close" type="button">Cancel</button><button class="primary-button" type="submit">Save sense</button></div>
    </form>
  </dialog>

  <dialog id="relationship-dialog" aria-labelledby="relationship-dialog-title">
    <form id="relationship-form" method="dialog">
      <div class="dialog-heading"><div><p class="eyebrow">Connection</p><h2 id="relationship-dialog-title">Add a relationship</h2></div><button class="icon-button dialog-close" type="button" aria-label="Close">x</button></div>
      <input name="term" type="hidden" />
      <label>Target term <input name="target" required maxlength="512" /></label><label>Relationship kind <input name="kind" value="related" required maxlength="100" /></label>
      <p class="form-error" role="alert"></p><div class="dialog-actions"><button class="secondary-button dialog-close" type="button">Cancel</button><button class="primary-button" type="submit">Connect terms</button></div>
    </form>
  </dialog>

  <dialog id="translation-dialog" aria-labelledby="translation-dialog-title">
    <form id="translation-form" method="dialog">
      <div class="dialog-heading"><div><p class="eyebrow">Multilingual link</p><h2 id="translation-dialog-title">Add a translation</h2></div><button class="icon-button dialog-close" type="button" aria-label="Close">x</button></div>
      <input name="term" type="hidden" /><input name="source_language" type="hidden" />
      <label>Translated term <input name="target_term" required maxlength="512" /></label><label>Target language <input name="target_language" required maxlength="35" placeholder="es, de, zh-Hans..." /></label><label>Source <input name="source" maxlength="500" value="manual" /></label><label class="checkbox-label"><input name="bidirectional" type="checkbox" checked /> Store the reverse translation too</label>
      <p class="form-error" role="alert"></p><div class="dialog-actions"><button class="secondary-button dialog-close" type="button">Cancel</button><button class="primary-button" type="submit">Save translation</button></div>
    </form>
  </dialog>

  <div id="toast" class="toast" role="status" aria-live="polite"></div>
`;

function element<T extends Element>(selector: string): T {
  const found = document.querySelector<T>(selector);
  if (!found) throw new Error(`Required interface element is missing: ${selector}`);
  return found;
}

const token = resolveSessionToken();
const api = new WordForgeApi(token);
let graphController: GraphController | null = null;
let graphQuery = "";
let refreshTimer = 0;
let selectedEntry: LexicalEntry | null = null;

const graphStatus = element<HTMLParagraphElement>("#graph-status");
const resultSummary = element<HTMLParagraphElement>("#result-summary");
const toast = element<HTMLDivElement>("#toast");
const dialog = element<HTMLDialogElement>("#entry-dialog");
const formError = element<HTMLParagraphElement>("#form-error");

function setInspectorActions(enabled: boolean): void {
  for (const selector of ["#edit-entry", "#add-sense", "#add-relationship", "#add-translation"]) {
    element<HTMLButtonElement>(selector).disabled = !enabled;
  }
}

function clearInspector(): void {
  selectedEntry = null;
  setInspectorActions(false);
  element<HTMLHeadingElement>("#inspector-title").textContent = "Select a term";
  element<HTMLParagraphElement>("#inspector-definition").textContent =
    "Choose a node or a list entry to inspect its lexical record.";
  element<HTMLDListElement>("#inspector-meta").replaceChildren();
  element<HTMLDivElement>("#inspector-senses").replaceChildren();
  element<HTMLDivElement>("#inspector-relations").replaceChildren();
}

function showToast(message: string): void {
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.setTimeout(() => toast.classList.remove("is-visible"), 2800);
}

function showError(error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  showToast(message);
  console.error(error);
}

function renderStats(statistics: Statistics): void {
  const grid = element<HTMLDivElement>("#stats-grid");
  grid.replaceChildren();
  const values: Array<[string, number]> = [
    ["Words", statistics.words],
    ["Lexemes", statistics.lexemes],
    ["Languages", statistics.languages],
    ["Connections", statistics.relationships + statistics.translations],
  ];
  for (const [label, value] of values) {
    const card = document.createElement("article");
    const number = document.createElement("strong");
    number.textContent = new Intl.NumberFormat().format(value);
    const caption = document.createElement("span");
    caption.textContent = label;
    card.append(number, caption);
    grid.append(card);
  }
}

async function refreshStats(): Promise<void> {
  renderStats(await api.statistics());
}

function resultButton(entry: Pick<LexicalEntry, "term" | "language" | "definition" | "part_of_speech">): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "result-card";
  const title = document.createElement("strong");
  title.textContent = entry.term;
  const badge = document.createElement("span");
  badge.className = "language-badge";
  badge.textContent = entry.language;
  const definition = document.createElement("p");
  definition.textContent = entry.definition || "Definition not recorded";
  const meta = document.createElement("small");
  meta.textContent = entry.part_of_speech || "Unclassified";
  button.append(title, badge, definition, meta);
  button.addEventListener("click", () => void inspectEntry(entry.term, entry.language));
  return button;
}

async function inspectEntry(term: string, language: string): Promise<void> {
  const title = element<HTMLHeadingElement>("#inspector-title");
  const definition = element<HTMLParagraphElement>("#inspector-definition");
  const metadata = element<HTMLDListElement>("#inspector-meta");
  const senses = element<HTMLDivElement>("#inspector-senses");
  const relations = element<HTMLDivElement>("#inspector-relations");
  title.textContent = term;
  definition.textContent = "Loading record...";
  selectedEntry = null;
  setInspectorActions(false);
  metadata.replaceChildren();
  senses.replaceChildren();
  relations.replaceChildren();
  try {
    const entry = await api.entry(term, language);
    selectedEntry = entry;
    setInspectorActions(true);
    const relationshipAction = element<HTMLButtonElement>("#add-relationship");
    relationshipAction.disabled = entry.language !== "en";
    relationshipAction.title = entry.language === "en"
      ? "Connect this English entry to another term"
      : "Typed semantic relationships currently use the English base layer";
    title.textContent = `${entry.term} [${entry.language}]`;
    definition.textContent = entry.definition || "Definition not recorded.";
    const values: Array<[string, string]> = [
      ["Part of speech", entry.part_of_speech || "Unclassified"],
      ["Source", entry.source || "Not recorded"],
      ["Base term", entry.base_term || "None"],
      ["Script", entry.script || "Unknown"],
      ["Revision", String(entry.revision)],
      ["Review state", entry.review_state || "Unreviewed"],
    ];
    for (const [label, value] of values) {
      const termElement = document.createElement("dt");
      termElement.textContent = label;
      const valueElement = document.createElement("dd");
      valueElement.textContent = value;
      metadata.append(termElement, valueElement);
    }
    if (entry.senses.length === 0) {
      const empty = document.createElement("p");
      empty.className = "empty-state";
      empty.textContent = "No independently reviewed senses recorded.";
      senses.append(empty);
    }
    for (const sense of entry.senses) {
      const item = document.createElement("article");
      const heading = document.createElement("strong");
      heading.textContent = [sense.part_of_speech, sense.domain].filter(Boolean).join(" - ") || "Sense";
      const senseDefinition = document.createElement("p");
      senseDefinition.textContent = sense.definition;
      const provenance = document.createElement("small");
      provenance.textContent = `${sense.review_state} - revision ${sense.revision}${sense.source ? ` - ${sense.source}` : ""}`;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "text-button danger-text";
      remove.textContent = "Remove";
      remove.addEventListener("click", async () => {
        if (!window.confirm(`Remove this sense?\n\n${sense.definition}`)) return;
        try {
          await api.deleteSense(sense.id, sense.revision);
          showToast("Sense removed");
          await inspectEntry(entry.term, entry.language);
        } catch (error) {
          showError(error);
        }
      });
      item.append(heading, senseDefinition, provenance, remove);
      senses.append(item);
    }
    for (const relation of entry.relationships) {
      const item = document.createElement("button");
      item.type = "button";
      item.textContent = `${relation.kind}: ${relation.target}${relation.language ? ` [${relation.language}]` : ""}`;
      item.addEventListener("click", () => void inspectEntry(relation.target, relation.language ?? "en"));
      relations.append(item);
    }
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      definition.textContent = "This connected term does not have a full record yet.";
      return;
    }
    definition.textContent = "Unable to load this record.";
    showError(error);
  }
}

function renderNetworkList(snapshot: NetworkSnapshot): void {
  const list = element<HTMLDivElement>("#network-list");
  list.replaceChildren();
  for (const node of snapshot.nodes) list.append(resultButton(node));
  element<HTMLSpanElement>("#network-list-count").textContent = `(${snapshot.nodes.length})`;
}

async function refreshNetwork(): Promise<void> {
  graphStatus.textContent = "Loading network...";
  const language = element<HTMLInputElement>("#network-language").value.trim();
  const limit = Number(element<HTMLSelectElement>("#network-limit").value);
  try {
    const snapshot = await api.network(graphQuery, language, limit);
    graphController?.destroy();
    graphController = mountNetwork(
      element<HTMLDivElement>("#graph-canvas"),
      snapshot,
      (node: NetworkNode) => void inspectEntry(node.term, node.language),
    );
    renderNetworkList(snapshot);
    graphStatus.textContent = `${snapshot.nodes.length} nodes, ${snapshot.edges.length} connections${snapshot.truncated ? " - bounded view" : ""}`;
  } catch (error) {
    graphStatus.textContent = "Network unavailable.";
    showError(error);
  }
}

async function search(query: string): Promise<void> {
  const normalized = query.trim();
  if (!normalized) return;
  resultSummary.textContent = `Searching for ${normalized}...`;
  try {
    const entries = await api.entries(normalized);
    const results = element<HTMLDivElement>("#search-results");
    results.replaceChildren(...entries.map(resultButton));
    resultSummary.textContent = entries.length
      ? `${entries.length} matching ${entries.length === 1 ? "entry" : "entries"}`
      : `No entries matched ${normalized}.`;
    graphQuery = normalized;
    await refreshNetwork();
    element<HTMLElement>("#lexicon").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    resultSummary.textContent = "Search failed.";
    showError(error);
  }
}

function scheduleRefresh(): void {
  window.clearTimeout(refreshTimer);
  refreshTimer = window.setTimeout(() => {
    void Promise.all([refreshStats(), refreshNetwork()]);
  }, 180);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(value >= 10 ? 1 : 2)} ${units[index]}`;
}

function renderIntegrity(report: IntegrityReport): void {
  const title = element<HTMLHeadingElement>("#integrity-title");
  title.textContent = report.ok ? "Database healthy" : "Attention required";
  title.classList.toggle("danger-text", !report.ok);
  const details = element<HTMLDListElement>("#integrity-details");
  details.replaceChildren();
  const values: Array<[string, string]> = [
    ["SQLite", report.integrity],
    ["Schema", String(report.schema_version)],
    ["Foreign keys", String(report.foreign_key_violations)],
    ["Pending derived work", String(report.pending_invalidations)],
    ["Pending jobs", String(report.pending_jobs)],
  ];
  for (const [label, value] of values) {
    const key = document.createElement("dt");
    key.textContent = label;
    const content = document.createElement("dd");
    content.textContent = value;
    details.append(key, content);
  }
}

function renderBackups(backups: BackupSnapshot[]): void {
  const list = element<HTMLDivElement>("#backup-list");
  const select = element<HTMLSelectElement>("#restore-backup");
  list.replaceChildren();
  select.replaceChildren();
  if (backups.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No managed snapshots yet.";
    list.append(empty);
    select.append(new Option("No snapshots yet", ""));
    return;
  }
  for (const backup of backups) {
    select.append(new Option(backup.name, backup.name));
    const row = document.createElement("div");
    row.className = "backup-item";
    const name = document.createElement("strong");
    name.textContent = backup.name;
    const metadata = document.createElement("span");
    metadata.textContent = `${formatBytes(backup.size)} - ${new Date(backup.modified_at * 1000).toLocaleString()}`;
    row.append(name, metadata);
    list.append(row);
  }
}

async function refreshOperations(): Promise<void> {
  try {
    const [integrity, backups] = await Promise.all([api.integrity(), api.backups()]);
    renderIntegrity(integrity);
    renderBackups(backups);
  } catch (error) {
    element<HTMLHeadingElement>("#integrity-title").textContent = "Health check unavailable";
    showError(error);
  }
}

function namedControl<T extends HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>(
  form: HTMLFormElement,
  name: string,
): T {
  const control = form.elements.namedItem(name);
  if (!(control instanceof HTMLElement)) throw new Error(`Form control is missing: ${name}`);
  return control as T;
}

function openEntryDialog(entry: LexicalEntry | null): void {
  const form = element<HTMLFormElement>("#entry-form");
  form.reset();
  const term = namedControl<HTMLInputElement>(form, "term");
  const language = namedControl<HTMLInputElement>(form, "language");
  term.readOnly = Boolean(entry);
  language.readOnly = Boolean(entry);
  namedControl<HTMLInputElement>(form, "expected_revision").value = String(entry?.revision ?? 0);
  namedControl<HTMLInputElement>(form, "source").value = entry?.source || "manual";
  if (entry) {
    element<HTMLHeadingElement>("#entry-dialog-title").textContent = "Edit entry";
    term.value = entry.term;
    language.value = entry.language;
    namedControl<HTMLInputElement>(form, "part_of_speech").value = entry.part_of_speech;
    namedControl<HTMLTextAreaElement>(form, "definition").value = entry.definition;
    namedControl<HTMLInputElement>(form, "base_term").value = entry.base_term;
    namedControl<HTMLTextAreaElement>(form, "example").value = entry.examples.join("\n");
  } else {
    element<HTMLHeadingElement>("#entry-dialog-title").textContent = "Add an entry";
    language.value = "en";
  }
  dialog.showModal();
}

element<HTMLFormElement>("#global-search").addEventListener("submit", (event) => {
  event.preventDefault();
  void search(element<HTMLInputElement>("#search-query").value);
});
element<HTMLButtonElement>("#refresh-network").addEventListener("click", () => void refreshNetwork());
element<HTMLSelectElement>("#network-limit").addEventListener("change", () => void refreshNetwork());
element<HTMLInputElement>("#network-language").addEventListener("change", () => void refreshNetwork());
element<HTMLButtonElement>("#zoom-in").addEventListener("click", () => graphController?.zoomIn());
element<HTMLButtonElement>("#zoom-out").addEventListener("click", () => graphController?.zoomOut());
element<HTMLButtonElement>("#zoom-fit").addEventListener("click", () => graphController?.fit());

for (const navigation of document.querySelectorAll<HTMLButtonElement>(".nav-item")) {
  navigation.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("is-active"));
    navigation.classList.add("is-active");
    const target = navigation.dataset.target;
    if (target) element<HTMLElement>(`#${target}`).scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

element<HTMLButtonElement>("#theme-toggle").addEventListener("click", () => {
  const root = document.documentElement;
  const next = root.dataset.theme === "light" ? "dark" : "light";
  root.dataset.theme = next;
  window.localStorage.setItem("word-forge.theme", next);
});
document.documentElement.dataset.theme = window.localStorage.getItem("word-forge.theme") ?? "dark";

element<HTMLButtonElement>("#add-entry").addEventListener("click", () => openEntryDialog(null));
element<HTMLButtonElement>("#edit-entry").addEventListener("click", () => {
  if (selectedEntry) openEntryDialog(selectedEntry);
});
const closeDialog = () => {
  formError.textContent = "";
  dialog.close();
};
element<HTMLButtonElement>("#close-dialog").addEventListener("click", closeDialog);
element<HTMLButtonElement>("#cancel-entry").addEventListener("click", closeDialog);
const entryForm = element<HTMLFormElement>("#entry-form");
entryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  formError.textContent = "";
  const form = event.currentTarget as HTMLFormElement;
  const data = new FormData(form);
  try {
    const entry = await api.addEntry({
      term: String(data.get("term") ?? ""),
      language: String(data.get("language") ?? "en"),
      part_of_speech: String(data.get("part_of_speech") ?? ""),
      definition: String(data.get("definition") ?? ""),
      base_term: String(data.get("base_term") ?? ""),
      source: String(data.get("source") ?? "manual"),
      examples: String(data.get("example") ?? "").split("\n").map((line) => line.trim()).filter(Boolean),
      expected_revision: Number(data.get("expected_revision") ?? 0),
    });
    form.reset();
    closeDialog();
    showToast(`${entry.term} saved`);
    await inspectEntry(entry.term, entry.language);
    scheduleRefresh();
  } catch (error) {
    formError.textContent = error instanceof Error ? error.message : String(error);
  }
});

for (const close of document.querySelectorAll<HTMLButtonElement>(".dialog-close")) {
  close.addEventListener("click", () => close.closest("dialog")?.close());
}

function openSelectedDialog(dialogSelector: string, formSelector: string): void {
  if (!selectedEntry) return;
  const targetDialog = element<HTMLDialogElement>(dialogSelector);
  const form = element<HTMLFormElement>(formSelector);
  form.reset();
  namedControl<HTMLInputElement>(form, "term").value = selectedEntry.term;
  const language = form.elements.namedItem("language");
  if (language instanceof HTMLInputElement) language.value = selectedEntry.language;
  const sourceLanguage = form.elements.namedItem("source_language");
  if (sourceLanguage instanceof HTMLInputElement) sourceLanguage.value = selectedEntry.language;
  const source = form.elements.namedItem("source");
  if (source instanceof HTMLInputElement) source.value = "manual";
  targetDialog.showModal();
}

element<HTMLButtonElement>("#add-sense").addEventListener("click", () => {
  openSelectedDialog("#sense-dialog", "#sense-form");
});
element<HTMLButtonElement>("#add-relationship").addEventListener("click", () => {
  openSelectedDialog("#relationship-dialog", "#relationship-form");
});
element<HTMLButtonElement>("#add-translation").addEventListener("click", () => {
  openSelectedDialog("#translation-dialog", "#translation-form");
});

element<HTMLFormElement>("#sense-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget as HTMLFormElement;
  const error = form.querySelector<HTMLParagraphElement>(".form-error");
  if (error) error.textContent = "";
  const data = new FormData(form);
  try {
    await api.addSense({
      term: String(data.get("term") ?? ""),
      language: String(data.get("language") ?? "en"),
      definition: String(data.get("definition") ?? ""),
      part_of_speech: String(data.get("part_of_speech") ?? ""),
      domain: String(data.get("domain") ?? ""),
      register: String(data.get("register") ?? ""),
      source: String(data.get("source") ?? "manual"),
      confidence: Number(data.get("confidence") ?? 1),
      review_state: String(data.get("review_state") ?? "unreviewed"),
      expected_revision: 0,
    });
    element<HTMLDialogElement>("#sense-dialog").close();
    showToast("Sense saved");
    if (selectedEntry) await inspectEntry(selectedEntry.term, selectedEntry.language);
  } catch (caught) {
    if (error) error.textContent = caught instanceof Error ? caught.message : String(caught);
  }
});

element<HTMLFormElement>("#relationship-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget as HTMLFormElement;
  const error = form.querySelector<HTMLParagraphElement>(".form-error");
  if (error) error.textContent = "";
  const data = new FormData(form);
  try {
    const result = await api.addRelationship({
      term: String(data.get("term") ?? ""),
      target: String(data.get("target") ?? ""),
      kind: String(data.get("kind") ?? "related"),
    });
    element<HTMLDialogElement>("#relationship-dialog").close();
    showToast(result.created ? "Relationship created" : "Relationship already exists");
    if (selectedEntry) await inspectEntry(selectedEntry.term, selectedEntry.language);
    scheduleRefresh();
  } catch (caught) {
    if (error) error.textContent = caught instanceof Error ? caught.message : String(caught);
  }
});

element<HTMLFormElement>("#translation-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget as HTMLFormElement;
  const error = form.querySelector<HTMLParagraphElement>(".form-error");
  if (error) error.textContent = "";
  const data = new FormData(form);
  try {
    await api.addTranslation({
      term: String(data.get("term") ?? ""),
      source_language: String(data.get("source_language") ?? "en"),
      target_term: String(data.get("target_term") ?? ""),
      target_language: String(data.get("target_language") ?? ""),
      source: String(data.get("source") ?? "manual"),
      bidirectional: data.get("bidirectional") === "on",
    });
    element<HTMLDialogElement>("#translation-dialog").close();
    showToast("Translation saved");
    if (selectedEntry) await inspectEntry(selectedEntry.term, selectedEntry.language);
    scheduleRefresh();
  } catch (caught) {
    if (error) error.textContent = caught instanceof Error ? caught.message : String(caught);
  }
});

element<HTMLButtonElement>("#refresh-operations").addEventListener("click", () => {
  void refreshOperations();
});
element<HTMLButtonElement>("#create-backup").addEventListener("click", async () => {
  const button = element<HTMLButtonElement>("#create-backup");
  button.disabled = true;
  try {
    const snapshot = await api.createBackup(element<HTMLInputElement>("#backup-label").value);
    showToast(`Snapshot created: ${snapshot.name}`);
    element<HTMLInputElement>("#backup-label").value = "";
    await refreshOperations();
  } catch (error) {
    showError(error);
  } finally {
    button.disabled = false;
  }
});
element<HTMLButtonElement>("#restore-backup-button").addEventListener("click", async () => {
  const backupName = element<HTMLSelectElement>("#restore-backup").value;
  const confirmation = element<HTMLInputElement>("#restore-confirmation").value;
  if (!backupName) {
    showToast("Select a snapshot first");
    return;
  }
  if (confirmation !== "RESTORE") {
    showToast("Type RESTORE to confirm replacement");
    return;
  }
  const button = element<HTMLButtonElement>("#restore-backup-button");
  button.disabled = true;
  try {
    await api.restoreBackup(backupName, confirmation);
    clearInspector();
    element<HTMLInputElement>("#restore-confirmation").value = "";
    showToast(`Restored ${backupName}`);
    await Promise.all([refreshStats(), refreshNetwork(), refreshOperations()]);
  } catch (error) {
    showError(error);
  } finally {
    button.disabled = false;
  }
});

if (!token) {
  graphStatus.textContent = "Session token missing. Launch this workspace with `word-forge gui`.";
  showToast("Authenticated session URL required");
} else {
  api.connectEvents(
    (event) => {
      if (event.type !== "ready") scheduleRefresh();
      if (event.type === "database.restored") {
        clearInspector();
        void refreshOperations();
      } else if (
        selectedEntry
        && event.term === selectedEntry.term
        && (!event.language || event.language === selectedEntry.language)
      ) {
        void inspectEntry(selectedEntry.term, selectedEntry.language);
      }
    },
    (state) => {
      const dot = element<HTMLSpanElement>("#live-indicator");
      dot.className = `status-dot is-${state}`;
      element<HTMLSpanElement>("#live-label").textContent =
        state === "live" ? "Live" : state === "connecting" ? "Connecting" : "Reconnecting";
    },
  );
  void Promise.all([refreshStats(), refreshNetwork(), refreshOperations()]).catch(showError);
}
