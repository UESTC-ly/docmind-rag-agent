import { spawnSync } from "node:child_process";
import { resolve } from "node:path";
import process from "node:process";

const desktopRoot = resolve(import.meta.dirname, "..");
const configured = process.env.DOCMIND_BUILD_PYTHON;
const candidates = configured
  ? [[configured, []]]
  : process.platform === "win32"
    ? [["py", ["-3.12"]], ["python", []]]
    : [["python3", []], ["python", []]];

for (const [command, prefix] of candidates) {
  const probe = spawnSync(command, [...prefix, "--version"], { stdio: "ignore" });
  if (probe.error || probe.status !== 0) continue;
  const result = spawnSync(
    command,
    [
      ...prefix,
      "-m", "pytest", "-p", "no:cacheprovider",
      "-c", "sidecar/pytest.ini", "sidecar/tests", "-q",
    ],
    { cwd: desktopRoot, stdio: "inherit" },
  );
  process.exitCode = result.status ?? 1;
  process.exit();
}

console.error("Python 3.12+ with the desktop build dependencies was not found");
process.exitCode = 1;
