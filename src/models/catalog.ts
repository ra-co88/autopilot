/**
 * Load the model catalog of the active provider (cache-first; public
 * endpoints, no key needed for hosted catalogs). One entry point for the CLI,
 * SDK and server so the picker, capabilities and pricing always describe the
 * provider in use.
 */

import { providerIdOf, type LlmAuth } from "../agent/llm-auth.js";
import { registerProviderModels } from "./registry.js";

export async function ensureCatalogForProvider(auth: LlmAuth, fetchImpl: typeof fetch = fetch): Promise<void> {
  const id = providerIdOf(auth);
  switch (id) {
    case "openrouter": {
      registerProviderModels(id, []);
      const { ensureOpenRouterCatalog } = await import("./openrouter-catalog.js");
      await ensureOpenRouterCatalog({ fetchImpl });
      return;
    }
    case "requesty": {
      registerProviderModels(id, []);
      const { ensureRequestyCatalog } = await import("./requesty-catalog.js");
      await ensureRequestyCatalog({ fetchImpl });
      return;
    }
    case "directionally-local": {
      // The free tier's models are declared in the provider, not fetched: register
      // them so every metadata consumer (budget, compaction, capabilities) reads
      // the real 1M window and 65k output instead of the unknown-model fallback.
      const { registerDlModels } = await import("../providers/directionally-local.js");
      registerDlModels();
      return;
    }
    default: {
      const { ensureProviderCatalog } = await import("./provider-catalog.js");
      await ensureProviderCatalog(id, auth.customEndpoint ?? {}, fetchImpl);
    }
  }
}
