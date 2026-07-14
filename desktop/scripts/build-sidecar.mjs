import { createHash } from "node:crypto";
import {
  chmodSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, delimiter, dirname, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import process from "node:process";

const desktopRoot = resolve(import.meta.dirname, "..");
const projectRoot = resolve(desktopRoot, "..");
const buildRoot = join(desktopRoot, ".build");
const venvRoot = join(buildRoot, "sidecar-venv");
const requirements = join(desktopRoot, "sidecar", "requirements-build.txt");
const runtimeRequirements = join(desktopRoot, "sidecar", "requirements-runtime.txt");
const specFile = join(desktopRoot, "sidecar", "docmind-sidecar.spec");
const defaultConfig = join(desktopRoot, "sidecar", "default.env");

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd || projectRoot,
    env: options.env || process.env,
    encoding: "utf8",
    stdio: options.capture ? "pipe" : "inherit",
  });
  if (result.error || result.status !== 0) {
    const detail = result.error?.message || result.stderr || `exit ${result.status}`;
    throw new Error(`${command} ${args.join(" ")} failed: ${detail}`);
  }
  return (result.stdout || "").trim();
}

function targetTriple() {
  if (process.env.TAURI_ENV_TARGET_TRIPLE) return process.env.TAURI_ENV_TARGET_TRIPLE;
  const verbose = run("rustc", ["-vV"], { capture: true });
  const host = verbose.split(/\r?\n/).find((line) => line.startsWith("host: "));
  if (!host) throw new Error("rustc -vV did not report a host target triple");
  return host.slice("host: ".length).trim();
}

function pythonCandidate() {
  const configured = process.env.DOCMIND_BUILD_PYTHON;
  const candidates = configured
    ? [[configured, []]]
    : process.platform === "win32"
      ? [["py", ["-3.12"]], ["python", []]]
      : [["python3", []], ["python", []]];
  for (const [command, prefix] of candidates) {
    const result = spawnSync(command, [...prefix, "--version"], {
      encoding: "utf8",
      stdio: "pipe",
    });
    if (!result.error && result.status === 0) return { command, prefix };
  }
  throw new Error("Python 3.12+ is required on the build runner (not on end-user hosts)");
}

function venvPython() {
  return process.platform === "win32"
    ? join(venvRoot, "Scripts", "python.exe")
    : join(venvRoot, "bin", "python");
}

function dependencyFingerprint() {
  const hash = createHash("sha256");
  hash.update(readFileSync(requirements));
  hash.update(readFileSync(runtimeRequirements));
  return hash.digest("hex");
}

function ensureBuildEnvironment() {
  mkdirSync(buildRoot, { recursive: true });
  const python = pythonCandidate();
  if (!existsSync(venvPython())) {
    run(python.command, [...python.prefix, "-m", "venv", venvRoot]);
  }

  const version = run(venvPython(), [
    "-c",
    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
  ], { capture: true });
  if (!/^3\.(1[2-9]|[2-9]\d)$/.test(version)) {
    throw new Error(`sidecar builds require Python 3.12 or newer, found ${version}`);
  }

  const fingerprint = dependencyFingerprint();
  const marker = join(venvRoot, ".docmind-requirements.sha256");
  if (!existsSync(marker) || readFileSync(marker, "utf8").trim() !== fingerprint) {
    run(venvPython(), [
      "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
      "-r", requirements,
    ]);
    writeFileSync(marker, `${fingerprint}\n`, "utf8");
  }
  return venvPython();
}

function assertNativeTarget(python, target) {
  const machine = run(python, ["-c", "import platform; print(platform.machine().lower())"], {
    capture: true,
  });
  const pythonArch = machine === "arm64" ? "aarch64" : machine === "amd64" ? "x86_64" : machine;
  const targetArch = target.split("-")[0];
  if (pythonArch !== targetArch) {
    throw new Error(
      `PyInstaller cannot cross-compile: Python is ${pythonArch}, target is ${targetArch}. ` +
      "Use a native release runner for each desktop architecture.",
    );
  }
}

function verifyBinary(binary) {
  const checkRoot = mkdtempSync(join(tmpdir(), "docmind-sidecar-check-"));
  try {
    const output = run(binary, [
      "check", "--data-dir", join(checkRoot, "data"), "--config", defaultConfig,
    ], { capture: true });
    const lines = output.split(/\r?\n/).filter(Boolean);
    const manifest = JSON.parse(lines.at(-1));
    const expected = {
      status: "ok",
      frozen: true,
      database: "sqlite",
      vector_store: "qdrant-local",
      task_executor: "local",
      package_script_confinement: "macos-sandbox-exec",
      external_runtime_required: false,
    };
    for (const [key, value] of Object.entries(expected)) {
      if (manifest[key] !== value) {
        throw new Error(`sidecar self-check ${key}: expected ${value}, got ${manifest[key]}`);
      }
    }

    const scriptOutput = run(binary, [
      "run-script",
      "/app/skills/packages/security-ownership-map/scripts/query_ownership.py",
      "--",
      "--help",
    ], { capture: true });
    if (!scriptOutput.toLowerCase().includes("usage:")) {
      throw new Error("frozen package-script runner did not execute its packaged probe");
    }

    const forbiddenScript = join(checkRoot, "outside.py");
    writeFileSync(forbiddenScript, "raise SystemExit(0)\n", "utf8");
    const forbidden = spawnSync(binary, ["run-script", forbiddenScript], {
      encoding: "utf8",
      stdio: "pipe",
    });
    if (!forbidden.error && forbidden.status === 0) {
      throw new Error("frozen package-script runner accepted a path outside signed resources");
    }
  } finally {
    rmSync(checkRoot, { recursive: true, force: true });
  }
}

function build() {
  const target = targetTriple();
  if (process.argv.includes("--print-target")) {
    console.log(target);
    return;
  }

  const python = ensureBuildEnvironment();
  assertNativeTarget(python, target);
  const distRoot = join(buildRoot, "sidecar-dist");
  const workRoot = join(buildRoot, "sidecar-work");
  const pyinstallerConfig = join(buildRoot, "pyinstaller-config");
  rmSync(distRoot, { recursive: true, force: true });
  rmSync(workRoot, { recursive: true, force: true });
  mkdirSync(pyinstallerConfig, { recursive: true });
  run(python, [
    "-m", "PyInstaller",
    "--clean",
    "--noconfirm",
    "--distpath", distRoot,
    "--workpath", workRoot,
    specFile,
  ], {
    env: { ...process.env, PYINSTALLER_CONFIG_DIR: pyinstallerConfig },
  });

  const suffix = process.platform === "win32" ? ".exe" : "";
  const built = join(distRoot, `docmind-sidecar${suffix}`);
  if (!existsSync(built)) throw new Error(`PyInstaller did not create ${built}`);
  verifyBinary(built);
  run(process.execPath, [join(desktopRoot, "scripts", "smoke-sidecar.mjs"), built]);

  const binariesRoot = join(desktopRoot, "src-tauri", "binaries");
  mkdirSync(binariesRoot, { recursive: true });
  const bundled = join(binariesRoot, `docmind-sidecar-${target}${suffix}`);
  copyFileSync(built, bundled);
  if (process.platform !== "win32") chmodSync(bundled, 0o755);
  console.log(`Verified frozen sidecar: ${bundled}`);
}

try {
  build();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
