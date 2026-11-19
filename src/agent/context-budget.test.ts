import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  BUDGET_SAFETY_MARGIN_TOKENS,
  compactToFit,
  completionBudgetFor,
  DEFAULT_COMPLETION_CEILING_TOKENS,
  DEFAULT_MAX_COMPLETION_TOKENS,
  effectiveInputBudget,
  hardCompactionThreshold,
  proactiveCompactionThreshold,
  sessionFileCompactionTarget,
  softCompactionThreshold,
} from "./context-budget.js";
import { registerProviderModels, registerUserModels } from "../models/registry.js";
import { registerDlModels } from "../providers/directionally-local.js";
import { estimatePromptTokens } from "./artifact-compaction.js";
import type { ChatMessage } from "./messages.js";
import type { SerializedArtifact, SessionState } from "./session-state.js";

/** A complete turn: user → assistant tool call → tool result → assistant answer. */
function turn(i: number, payloadChars: number): ChatMessage[] {
  return [
    { role: "user", content: `question ${i}` },
    { role: "assistant", content: null, tool_calls: [{ id: `c${i}`, type: "function", function: { name: "read", arguments: JSON.stringify({ path: `src/f${i}.ts` }) } }] },
    { role: "tool", tool_call_id: `c${i}`, name: "read", content: "x".repeat(payloadChars) },
    { role: "assistant", content: `answer ${i}` },
  ];
}

function history(turns: number, payloadChars: number): ChatMessage[] {
  const out: ChatMessage[] = [{ role: "system", content: "system prompt" }];
  for (let i = 0; i < turns; i++) out.push(...turn(i, payloadChars));
  return out;
}

function memoryTarget() {
  const file: { sessionState?: SessionState; artifactStore?: SerializedArtifact[] } = {};
  return { file, target: sessionFileCompactionTarget(file) };
}

describe("context budget", () => {
  it("derives the input budget and proactive threshold from the model", () => {
    // Unknown models are inferred at a 128k window.
    assert.equal(effectiveInputBudget("unknown/model"), 128_000 - 16_384 - 8_192);
    assert.equal(effectiveInputBudget("unknown/model", 4_096), 128_000 - 4_096 - 8_192);
    const budget = 128_000 - 16_384 - 8_192;
    assert.equal(softCompactionThreshold("unknown/model"), Math.floor(budget * 0.6));
    assert.equal(hardCompactionThreshold("unknown/model"), Math.floor(budget * 0.85));
    // The deprecated alias now follows the budget instead of the 80k cap.
    assert.equal(proactiveCompactionThreshold("unknown/model"), softCompactionThreshold("unknown/model"));
  });

  it("scales thresholds with large windows and honors an opt-in absolute cap", () => {
    const large = effectiveInputBudget("moonshotai/kimi-k3");
    assert.equal(softCompactionThreshold("moonshotai/kimi-k3"), Math.floor(large * 0.6));
    assert.ok(softCompactionThreshold("moonshotai/kimi-k3") > 80_000, "no 80k cap by default");
    assert.equal(softCompactionThreshold("moonshotai/kimi-k3", { absoluteCapTokens: 80_000 }), 80_000);
    assert.equal(hardCompactionThreshold("moonshotai/kimi-k3", { absoluteCapTokens: 80_000 }), 80_000);
  });

  it("compacts older complete turns until the history fits, keeping the active turn", () => {
    const messages = history(8, 35_000); // ~10k tokens per turn
    const active = messages.slice(-4);
    const { file, target } = memoryTarget();
    const budget = 25_000;
    const outcome = compactToFit(messages, budget, target)!;
    assert.ok(outcome, "compaction ran");
    assert.ok(outcome.tokensAfter <= budget, `fits: ${outcome.tokensAfter}`);
    assert.equal(outcome.tokensAfter, estimatePromptTokens(outcome.messages));
    assert.deepEqual(outcome.messages.slice(-4), active, "active turn kept intact");
    assert.equal(outcome.messages[0]!.content, "system prompt");
    // Every kept tool result still has its assistant tool call.
    const callIds = new Set(outcome.messages.flatMap((m) => m.tool_calls?.map((c) => c.id) ?? []));
    for (const m of outcome.messages.filter((m) => m.role === "tool")) assert.ok(callIds.has(m.tool_call_id!));
    // Archived raw content is persisted on the session file, not dropped.
    assert.ok((file.artifactStore?.length ?? 0) > 0);
    assert.ok(Object.keys(file.sessionState?.artifact_index ?? {}).length > 0);
  });

  it("keeps the larger working set when it already fits", () => {
    const messages = history(8, 35_000);
    const { target } = memoryTarget();
    const outcome = compactToFit(messages, 60_000, target)!;
    assert.equal(outcome.turnsRemoved, 4, "kept the last four turns");
  });

  it("returns null when only the active turn exists", () => {
    const { target } = memoryTarget();
    assert.equal(compactToFit(history(1, 500_000), 1_000, target), null);
  });

  it("does not stack compiled state messages across repeated compactions", () => {
    const { target } = memoryTarget();
    const first = compactToFit(history(8, 35_000), 25_000, target)!;
    const grown = [...first.messages, ...turn(100, 35_000), ...turn(101, 35_000), ...turn(102, 35_000)];
    const second = compactToFit(grown, 25_000, target)!;
    const stateMessages = second.messages.filter((m) => m.role === "system" && String(m.content).startsWith("[compiled session state]"));
    assert.equal(stateMessages.length, 1);
  });
});

describe("context budget: a local server's reported context", () => {
  it("uses the served context, the capped reply and a margin that fits the window", async () => {
    const { localContextLength, resetLocalContextForTesting } = await import("../models/local-context.js");
    const { effectiveInputBudget } = await import("./context-budget.js");
    resetLocalContextForTesting();
    const fetchImpl = (async () => new Response(JSON.stringify({ data: [{ id: "served-model", max_model_len: 32768 }] }))) as unknown as typeof fetch;
    const before = effectiveInputBudget("served-model");
    await localContextLength({ baseUrl: "http://127.0.0.1:8000/v1", providerId: "vllm" }, "served-model", { fetchImpl });
    // 32768 - min(16384, 32768 / 2) - 32768 / 16
    assert.equal(effectiveInputBudget("served-model"), 32768 - 16384 - 2048);
    assert.notEqual(before, effectiveInputBudget("served-model"));
    resetLocalContextForTesting();
  });
});

describe("completion budget: model-aware, ceiling-clamped", () => {
  const DL_MODEL = "deepseek/deepseek-v4.1-flash-1017";

  function withDlModels<T>(fn: () => T): T {
    registerDlModels();
    try {
      return fn();
    } finally {
      registerProviderModels("directionally-local", []);
    }
  }

  /** Synthetic entries for the resolution logic; the DL tests use the
   *  production registration above, not a test-installed copy. */
  function withModels<T>(entries: Parameters<typeof registerUserModels>[0], fn: () => T): T {
    registerUserModels(entries);
    try {
      return fn();
    } finally {
      registerUserModels([]);
    }
  }

  it("resolves the free tier's budget against the documented ceiling (criterion 1)", () => {
    withDlModels(() => {
      // min(65,536 declared, 32,768 ceiling, 1,048,576 - 8,192 headroom)
      assert.equal(completionBudgetFor(DL_MODEL, { freeTier: true }), 32_768);
      assert.equal(DEFAULT_COMPLETION_CEILING_TOKENS, 32_768);
      // MR-22's raise to the declared limit: the ceiling is a parameter, not a hardcode.
      assert.equal(completionBudgetFor(DL_MODEL, { freeTier: true, ceiling: 65_536 }), 65_536);
      // MR-23's allowance-dependence: a smaller remaining allowance lowers it.
      assert.equal(completionBudgetFor(DL_MODEL, { freeTier: true, ceiling: 20_000 }), 20_000);
      // Off the free tier the declared limit is the ceiling.
      assert.equal(completionBudgetFor(DL_MODEL), 65_536);
    });
  });

  it("keeps the flat default for a non-thinking model or one with no declared output limit (criterion 4)", () => {
    assert.equal(completionBudgetFor("unknown/model"), DEFAULT_MAX_COMPLETION_TOKENS);
    withModels(
      [
        {
          id: "test/non-thinking",
          contextWindow: 1_000_000,
          maxOutputTokens: 131_072,
          pricing: { inputPerMtok: 0, outputPerMtok: 0 },
          supports: { tools: true, reasoning: false, streaming: true },
        },
        {
          id: "test/no-output-limit",
          contextWindow: 1_000_000,
          maxOutputTokens: 0,
          pricing: { inputPerMtok: 0, outputPerMtok: 0 },
          supports: { tools: true, reasoning: true, streaming: true },
        },
      ],
      () => {
        assert.equal(completionBudgetFor("test/non-thinking"), DEFAULT_MAX_COMPLETION_TOKENS);
        assert.equal(completionBudgetFor("test/no-output-limit"), DEFAULT_MAX_COMPLETION_TOKENS);
      },
    );
  });

  it("lets an explicit maxCompletionTokens override the model-aware default and the ceiling (criterion 5)", () => {
    withDlModels(() => {
      assert.equal(completionBudgetFor(DL_MODEL, { freeTier: true, maxCompletionTokens: 4_096 }), 4_096);
      assert.equal(completionBudgetFor(DL_MODEL, { freeTier: true, maxCompletionTokens: 100_000 }), 100_000);
    });
  });

  it("clamps to the context-window headroom", () => {
    withModels(
      [
        {
          id: "test/small-window",
          contextWindow: 40_000,
          maxOutputTokens: 65_536,
          pricing: { inputPerMtok: 0, outputPerMtok: 0 },
          supports: { tools: true, reasoning: true, streaming: true },
        },
      ],
      () => {
        assert.equal(completionBudgetFor("test/small-window"), 40_000 - BUDGET_SAFETY_MARGIN_TOKENS);
      },
    );
  });

  it("moves the input budget and compaction thresholds with the completion budget", () => {
    withDlModels(() => {
      const budget = 1_048_576 - 32_768 - BUDGET_SAFETY_MARGIN_TOKENS;
      assert.equal(effectiveInputBudget(DL_MODEL, undefined, { freeTier: true }), budget);
      assert.equal(softCompactionThreshold(DL_MODEL, { freeTier: true }), Math.floor(budget * 0.6));
      assert.equal(hardCompactionThreshold(DL_MODEL, { freeTier: true }), Math.floor(budget * 0.85));
    });
  });

  it("registers the DL models through the production catalog init (the probe matrix)", async () => {
    const { ensureCatalogForProvider } = await import("../models/catalog.js");
    const { llmAuthFromConfig } = await import("../agent/llm-auth.js");
    const { getModelOrInfer } = await import("../models/registry.js");
    try {
      await ensureCatalogForProvider(
        llmAuthFromConfig({
          provider: "directionally-local",
          providers: { "directionally-local": { apiKey: "at" } },
        } as never),
      );
      // The real registry answers with the provider's declared metadata, not the
      // unknown-model fallback (128k window, 16k output, reasoning: false).
      const entry = getModelOrInfer(DL_MODEL);
      assert.equal(entry.contextWindow, 1_048_576);
      assert.equal(entry.maxOutputTokens, 65_536);
      assert.equal(entry.supports.reasoning, true);
      // The budget and the compaction thresholds read that metadata.
      assert.equal(completionBudgetFor(DL_MODEL, { freeTier: true }), 32_768);
      assert.equal(
        effectiveInputBudget(DL_MODEL, undefined, { freeTier: true }),
        1_048_576 - 32_768 - BUDGET_SAFETY_MARGIN_TOKENS,
      );
    } finally {
      registerProviderModels("directionally-local", []);
    }
  });

  it("documents the ceiling in the maintainer's wording (criterion 7)", async () => {
    const source = await readFile(new URL("./context-budget.ts", import.meta.url), "utf8");
    // The comment wraps; compare the wording with its line breaks flattened.
    const prose = source.replace(/\s*\*\s*/g, " ").replace(/\s+/g, " ");
    assert.match(
      prose,
      /up to the model's maxOutputTokens, and on the free tier also limited by the remaining allowance/,
    );
  });
});
