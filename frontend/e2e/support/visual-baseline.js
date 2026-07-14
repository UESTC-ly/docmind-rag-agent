import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const sourceRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../visual-baselines");

export function materializeVisualBaseline(testInfo, name) {
  if (process.env.UPDATE_VISUAL_BASELINES === "1") return;

  const source = resolve(sourceRoot, "chromium", `${name}.base64`);
  if (!existsSync(source)) {
    throw new Error(`缺少 Chromium 视觉基线：${source}`);
  }
  const expected = testInfo.snapshotPath(name);
  mkdirSync(dirname(expected), { recursive: true });
  const encoded = readFileSync(source, "utf8").replace(/\s+/g, "");
  writeFileSync(expected, Buffer.from(encoded, "base64"));
}
