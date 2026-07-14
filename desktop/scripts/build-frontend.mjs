import { cp, mkdir, rm } from "node:fs/promises";
import { resolve } from "node:path";
import { build } from "esbuild";

import { verifyBundledDesktopApi } from "./verify-frontend-api.mjs";

const desktopRoot = resolve(import.meta.dirname, "..");
const sourceRoot = resolve(desktopRoot, "..", "frontend");
const outputRoot = resolve(desktopRoot, "dist");

await rm(outputRoot, { recursive: true, force: true });
await mkdir(outputRoot, { recursive: true });
await cp(resolve(sourceRoot, "index.html"), resolve(outputRoot, "index.html"));
await cp(resolve(sourceRoot, "styles"), resolve(outputRoot, "styles"), { recursive: true });

await build({
  entryPoints: {
    main: resolve(sourceRoot, "js", "main.js"),
    "api-client": resolve(sourceRoot, "js", "api.js"),
    "desktop-bridge": resolve(sourceRoot, "js", "desktop.js"),
  },
  outdir: resolve(outputRoot, "js"),
  bundle: true,
  format: "esm",
  splitting: true,
  target: "es2022",
  nodePaths: [resolve(desktopRoot, "node_modules")],
  sourcemap: process.env.NODE_ENV === "production" ? false : "linked",
  logLevel: "info"
});

await verifyBundledDesktopApi(outputRoot);
