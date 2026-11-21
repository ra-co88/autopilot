import { describe, it } from "node:test";
import assert from "node:assert";
import { readFile, readdir, stat } from "node:fs/promises";
import type { ChatMessage } from "./agent/messages.js";
import { addCheckpoint, listSessions, loadSession, makeSessionId, saveSession, sessionsDir, SESSION_ID_MAX_CHARS, type SessionFile } from "./sessions.js";

function session(id: string, messages: ChatMessage[]): SessionFile {
  const now = new Date().toISOString();
  return { id, cwd: "/repo", model: "m", createdAt: now, updatedAt: now, messages };
}

function messages(n: number, size: number): ChatMessage[] {
  return Array.from({ length: n }, (_, i) => ({ role: i % 2 ? "assistant" : "user", content: `${i} ${"x".repeat(size)}` }) as ChatMessage);
}

describe("makeSessionId (session-id-64-cap)", () => {
  // `new Date().toISOString().replace(/[:.]/g, "-")` — e.g. 2026-11-21T19-04-05-123Z.
  const TS_LEN = 24;
  const SEP = "_";

  it("caps a long prompt's id at the prompt_cache_key limit (criterion 1)", () => {
    const id = makeSessionId("x".repeat(100));
    assert.ok(id.length <= SESSION_ID_MAX_CHARS, `${id.length} > ${SESSION_ID_MAX_CHARS}`);
    assert.equal(id.length, SESSION_ID_MAX_CHARS, "a 40-char slug is capped to exactly the limit");
  });

  it("keeps the <timestamp>_<slug> structure with the slug a capped prefix (criterion 2)", () => {
    const prompt = "Fix the flaky test ".repeat(10);
    const id = makeSessionId(prompt);
    const sep = id.indexOf(SEP);
    assert.equal(sep, TS_LEN, "the separator follows the 24-char timestamp");
    assert.match(id.slice(0, TS_LEN), /^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z$/);
    const slug = id.slice(sep + 1);
    assert.equal(slug, "fix-the-flaky-test-fix-the-flaky-test-f", "the capped prefix of the sanitized prompt");
    assert.ok(prompt.toLowerCase().replace(/[^a-z0-9]+/g, "-").startsWith(slug), "the slug is a prefix of the sanitized prompt");
  });

  it("leaves a short prompt's id unchanged (criterion 3)", () => {
    const id = makeSessionId("Fix the bug");
    assert.equal(id.slice(TS_LEN + 1), "fix-the-bug");
    assert.equal(id.length, TS_LEN + 1 + "fix-the-bug".length);
    assert.equal(makeSessionId("").slice(TS_LEN + 1), "session", "an empty prompt keeps the fallback slug");
  });

  it("produces only characters the uncapped construction could produce (criterion 4)", () => {
    const id = makeSessionId("Fix the flaky test ".repeat(10));
    // interactive-resume.ts:27 SESSION_ID — the id is a file name in sessionsDir().
    assert.match(id, /^[A-Za-z0-9][A-Za-z0-9._-]*$/);
    assert.ok(!id.includes("..") && !id.includes("/") && !id.includes("\\"), "no path traversal characters");
  });

  it("holds the invariant across prompt lengths, with the boundary at 39 slug chars", () => {
    for (const n of [0, 1, 38, 39, 40, 41, 100, 1000]) {
      const id = makeSessionId("a".repeat(n));
      assert.ok(id.length <= SESSION_ID_MAX_CHARS, `prompt ${n} → ${id.length}`);
    }
    assert.equal(makeSessionId("a".repeat(39)).length, SESSION_ID_MAX_CHARS, "a 39-char slug fills the id exactly");
    assert.equal(makeSessionId("a".repeat(38)).length, SESSION_ID_MAX_CHARS - 1, "a 38-char slug is not capped");
  });
});

describe("saveSession (T4-01)", () => {
  it("never leaves a torn file when large and small saves race, and the last save wins", async () => {
    const id = "race-session";
    const big = session(id, messages(200, 8_000)); // ~1.6 MB
    const small = session(id, messages(2, 10));
    const writes: Promise<string>[] = [];
    for (let i = 0; i < 10; i++) writes.push(saveSession(i % 2 ? small : big));
    writes.push(saveSession(small));
    const paths = await Promise.all(writes);
    const loaded = JSON.parse(await readFile(paths[0]!, "utf8")) as SessionFile;
    assert.equal(loaded.messages.length, 2, "the save issued last is the one on disk");
    const listed = await listSessions(30);
    assert.ok(listed.some((s) => s.id === id), "still listed for /resume");
    assert.ok(!(await readdir(sessionsDir())).some((f) => f.endsWith(".tmp")), "no temp files left behind");
  });

  it("writes session files owner-only", async () => {
    if (process.platform === "win32") return;
    const path = await saveSession(session("mode-session", messages(1, 1)));
    assert.equal((await stat(path)).mode & 0o777, 0o600);
  });

  it("snapshots the session at call time, not when its queued write runs", async () => {
    const file = session("snapshot-session", messages(1, 1));
    const pending = saveSession(file);
    file.messages.push({ role: "assistant", content: "later" } as ChatMessage);
    const path = await pending;
    const loaded = JSON.parse(await readFile(path, "utf8")) as SessionFile;
    assert.equal(loaded.messages.length, 1);
  });

  it("concurrent checkpoints are all kept (no lost read-modify-write)", async () => {
    const file = session("checkpoint-session", messages(4, 100));
    const path = await saveSession(file);
    const at = new Date().toISOString();
    await Promise.all(
      ["cp1", "cp2", "cp3", "cp4"].map((id, i) => addCheckpoint(path, { id, label: id, turnIndex: i, timestamp: at } as never)),
    );
    const after = await loadSession(path);
    assert.deepStrictEqual(after.checkpoints?.map((c) => c.id), ["cp1", "cp2", "cp3", "cp4"]);
    assert.equal(after.messages.length, 4);
  });
});
