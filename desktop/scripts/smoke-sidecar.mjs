import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import process from "node:process";

const desktopRoot = resolve(import.meta.dirname, "..");
const defaultConfig = join(desktopRoot, "sidecar", "default.env");

function targetTriple() {
  if (process.env.TAURI_ENV_TARGET_TRIPLE) return process.env.TAURI_ENV_TARGET_TRIPLE;
  const result = spawnSync("rustc", ["-vV"], { encoding: "utf8" });
  if (result.error || result.status !== 0) throw new Error("rustc -vV failed");
  const host = result.stdout.split(/\r?\n/).find((line) => line.startsWith("host: "));
  if (!host) throw new Error("rustc did not report a target triple");
  return host.slice(6).trim();
}

function defaultBinary() {
  const suffix = process.platform === "win32" ? ".exe" : "";
  return join(
    desktopRoot,
    "src-tauri",
    "binaries",
    `docmind-sidecar-${targetTriple()}${suffix}`,
  );
}

async function reservePort() {
  const server = createServer();
  await new Promise((resolvePromise, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolvePromise);
  });
  const address = server.address();
  const port = typeof address === "object" && address ? address.port : 0;
  await new Promise((resolvePromise) => server.close(resolvePromise));
  if (!port) throw new Error("failed to reserve a loopback port");
  return port;
}

const delay = (milliseconds) => new Promise((resolvePromise) => {
  setTimeout(resolvePromise, milliseconds);
});

async function stop(child) {
  if (child.exitCode !== null) return;
  child.kill();
  for (let attempt = 0; attempt < 20 && child.exitCode === null; attempt += 1) {
    await delay(100);
  }
  if (child.exitCode === null) child.kill("SIGKILL");
}

async function smoke(binary) {
  if (!existsSync(binary)) throw new Error(`sidecar binary not found: ${binary}`);
  const runtimeRoot = mkdtempSync(join(tmpdir(), "docmind-sidecar-health-"));
  const dataDir = join(runtimeRoot, "data");
  const port = await reservePort();
  const child = spawn(binary, [
    "serve",
    "--host", "127.0.0.1",
    "--port", String(port),
    "--data-dir", dataDir,
    "--config", defaultConfig,
  ], { stdio: ["ignore", "pipe", "pipe"] });
  let output = "";
  child.stdout.on("data", (chunk) => { output += chunk.toString(); });
  child.stderr.on("data", (chunk) => { output += chunk.toString(); });

  try {
    let healthy = false;
    for (let attempt = 0; attempt < 120; attempt += 1) {
      if (child.exitCode !== null) {
        throw new Error(`sidecar exited before health check (${child.exitCode})\n${output}`);
      }
      try {
        const response = await fetch(`http://127.0.0.1:${port}/health`, {
          signal: AbortSignal.timeout(1000),
        });
        const body = await response.json();
        if (response.status === 200 && body.status === "ok") {
          healthy = true;
          break;
        }
      } catch {
        // One-file extraction and first-run migrations can take several seconds.
      }
      await delay(500);
    }
    if (!healthy) throw new Error(`sidecar /health did not become ready\n${output}`);
    if (!existsSync(join(dataDir, "docmind.db"))) {
      throw new Error("health succeeded without creating the migrated SQLite database");
    }

    for (const origin of ["tauri://localhost", "http://tauri.localhost"]) {
      const response = await fetch(`http://127.0.0.1:${port}/auth/me`, {
        method: "OPTIONS",
        headers: {
          Origin: origin,
          "Access-Control-Request-Method": "GET",
        },
      });
      if (response.status !== 200) {
        throw new Error(`sidecar rejected CORS preflight from ${origin}: ${response.status}`);
      }
      if (response.headers.get("access-control-allow-origin") !== origin) {
        throw new Error(`sidecar did not allow the desktop origin ${origin}`);
      }
    }
    console.log(`Sidecar health smoke passed on 127.0.0.1:${port}`);
  } finally {
    await stop(child);
    rmSync(runtimeRoot, { recursive: true, force: true });
  }
}

try {
  await smoke(process.argv[2] ? resolve(process.argv[2]) : defaultBinary());
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
