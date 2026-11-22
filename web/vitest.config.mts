import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  // Pure-math tests run in node; component tests opt into jsdom with a file comment.
  test: { environment: "node" },
});
