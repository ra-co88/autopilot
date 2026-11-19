/**
 * Thinking models that exhaust the completion budget: a reply that is only
 * reasoning and was cut at `finish_reason: "length"` is retried once — with a
 * lower reasoning effort where the provider has the knob, else with the budget
 * raised to the model-aware ceiling (MR-24) — before the abort message
 * surfaces. At most one escalation per turn.
 */

import { describe, it, before, after, afterEach } from "node:test";
import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { reasoningOnlyLine, runAgentTurn, type AgentTurnOpts } from "./loop.js";
import type { LlmAuth } from "./llm-auth.js";
import type { ChatMessage } from "./messages.js";
import type { ModelEntry } from "../models/registry.js";
import { registerUserModels } from "../models/registry.js";
import { dlFreeModelEntry } from "../providers/directionally-local.js";
import type { ToolExecutor } from "../tools/executor.js";

const DL_MODEL = "deepseek/deepseek-v4.1-flash-1017";
const EFFORT_MODEL = "test/effort-model";

/** A fake OpenAI-compatible server: /models 404s, chat completions stream the
 *  next queued reply, and every request body is recorded. */
async function fakeServer(): Promise<{
  url: string;
  replies: unknown[][];
  bodies: Array<Record<string, unknown>>;
  close: () => Promise<void>;
}> {
  const replies: unknown[][] = [];
  const bodies: Array<Record<string, unknown>> = [];
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      const path = (req.url ?? "").split("?")[0];
      if (path !== "/v1/chat/completions") {
        res.writeHead(404, { "content-type": "application/json" });
        res.end("{}");
        return;
      }
      bodies.push(raw ? (JSON.parse(raw) as Record<string, unknown>) : {});
      const chunks = replies.shift() ?? [];
      res.writeHead(200, { "content-type": "text/event-stream" });
      for (const chunk of chunks) res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      res.end("data: [DONE]\n\n");
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}/v1`,
    replies,
    bodies,
    close: () =>
      new Promise<void>((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}

const delta = (d: Record<string, unknown>, finish: string | null = null) => ({
  id: "chatcmpl-1",
  choices: [{ index: 0, delta: d, finish_reason: finish }],
});

/** A reply that is all reasoning and was cut at the completion budget. */
const reasoningOnlyLength = () => [
  delta({ role: "assistant", reasoning_content: "Let me think about this carefully." }),
  delta({ reasoning_content: " And some more." }, "length"),
];

function effortModel(): ModelEntry {
  return {
    id: EFFORT_MODEL,
    contextWindow: 1_048_576,
    maxOutputTokens: 65_536,
    pricing: { inputPerMtok: 0, outputPerMtok: 0 },
    supports: { tools: true, reasoning: true, streaming: true },
    reasoningEfforts: ["low", "medium", "high"],
  };
}

const dlAuth = (url: string): LlmAuth => ({
  providerId: "directionally-local",
  customEndpoint: { baseUrl: url, apiKey: "test-token", providerId: "directionally-local" },
});

const effortAuth = (url: string): LlmAuth => ({
  providerId: "custom",
  customEndpoint: { baseUrl: url, apiKey: "test-token" },
});

describe("thinking completion budget: the one-shot escalation", () => {
  let server: Awaited<ReturnType<typeof fakeServer>>;
  before(async () => {
    server = await fakeServer();
  });
  after(() => server.close());
  afterEach(() => {
    registerUserModels([]);
    server.replies.length = 0;
    server.bodies.length = 0;
  });

  async function turn(auth: LlmAuth, model: string, extra: Partial<AgentTurnOpts> = {}) {
    const messages: ChatMessage[] = [
      { role: "system", content: "test" },
      { role: "user", content: "go" },
    ];
    const warnings: string[] = [];
    const text: string[] = [];
    await runAgentTurn({
      ...auth,
      model,
      messages,
      tools: [],
      executor: { list: () => [], run: async () => ({ tool_call_id: "", ok: true, content: "" }) } as unknown as ToolExecutor,
      cwd: "/tmp",
      signal: new AbortController().signal,
      retrySleep: async () => {},
      ...extra,
      callbacks: {
        askPermission: async () => "allow",
        onWarning: (w) => warnings.push(w),
        onTextDelta: (d) => text.push(d),
      },
    });
    return { warnings, text: text.join(""), messages };
  }

  it("retries once with a lower reasoning effort where the provider has the knob (criterion 2)", async () => {
    registerUserModels([effortModel()]);
    server.replies.push(reasoningOnlyLength(), [delta({ content: "Done." }, "stop")]);
    const t = await turn(effortAuth(server.url), EFFORT_MODEL, { reasoningEffort: "high" });
    assert.equal(server.bodies.length, 2, "one retry");
    assert.equal(server.bodies[0]?.reasoning_effort, "high");
    assert.equal(server.bodies[1]?.reasoning_effort, "medium", "the retry thinks less");
    assert.equal(t.text, "Done.", "the escalated attempt's outcome surfaces");
    assert.deepEqual(t.warnings, []);
  });

  it("retries once with the budget raised to the ceiling where there is no effort knob (criterion 2)", async () => {
    registerUserModels([dlFreeModelEntry(DL_MODEL)]);
    server.replies.push(reasoningOnlyLength(), [delta({ content: "Done." }, "stop")]);
    const t = await turn(dlAuth(server.url), DL_MODEL);
    assert.equal(server.bodies.length, 2, "one retry");
    assert.equal(server.bodies[0]?.max_tokens, 32_768, "the model-aware budget");
    assert.equal(server.bodies[1]?.max_tokens, 32_768, "the raised budget is the ceiling");
    assert.equal(t.text, "Done.");
    assert.deepEqual(t.warnings, []);
  });

  it("escalates at most once and surfaces the message noting what was tried (criteria 3, 6)", async () => {
    registerUserModels([dlFreeModelEntry(DL_MODEL)]);
    server.replies.push(reasoningOnlyLength(), reasoningOnlyLength(), reasoningOnlyLength());
    const t = await turn(dlAuth(server.url), DL_MODEL);
    assert.equal(server.bodies.length, 2, "a second length failure surfaces, no loop");
    assert.equal(t.warnings.length, 1);
    assert.match(t.warnings[0]!, /retried once/);
    assert.match(t.warnings[0]!, /32,768-token budget/);
    assert.equal(t.warnings[0], reasoningOnlyLine("length", { maxTokens: 32_768 }));
  });

  it("leaves a reasoning-only reply that was not cut at length alone", async () => {
    registerUserModels([dlFreeModelEntry(DL_MODEL)]);
    server.replies.push([delta({ role: "assistant", reasoning_content: "Hmm." }, "stop")]);
    const t = await turn(dlAuth(server.url), DL_MODEL);
    assert.equal(server.bodies.length, 1, "no escalation");
    assert.deepEqual(t.warnings, ["The model only thought and gave no answer. Ask again."]);
  });
});
