import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";

function memoryStorage() {
  const values = new Map();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
  };
}

export async function verifyBundledDesktopApi(outputRoot) {
  const apiEntry = resolve(outputRoot, "js", "api-client.js");
  const bridgeEntry = resolve(outputRoot, "js", "desktop-bridge.js");
  const mainEntry = resolve(outputRoot, "js", "main.js");
  const originalWindow = globalThis.window;
  const originalStorage = globalThis.localStorage;
  const originalFetch = globalThis.fetch;
  const commands = [];
  const requests = [];
  const expectedOrigin = "http://127.0.0.1:8000";

  globalThis.window = {
    __TAURI_INTERNALS__: {
      invoke: async (command) => {
        commands.push(command);
        if (command === "backend_status") {
          return { state: "ready", message: "ready" };
        }
        if (command === "backend_origin") return expectedOrigin;
        throw new Error(`unexpected Tauri command: ${command}`);
      },
    },
  };
  globalThis.localStorage = memoryStorage();
  globalThis.fetch = async (input) => {
    const url = String(input);
    requests.push(url);
    if (url.endsWith("/chat/stream")) {
      return new Response('event: token\ndata: {"text":"ok"}\n\n', {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      });
    }
    return new Response(JSON.stringify({ access_token: "desktop-test" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  try {
    const cacheKey = `?verify=${Date.now()}`;
    const bridge = await import(`${pathToFileURL(bridgeEntry).href}${cacheKey}`);
    const client = await import(`${pathToFileURL(apiEntry).href}${cacheKey}`);
    const origin = await bridge.waitForDesktopBackend();
    client.configureApiOrigin(origin);

    await client.api.login("desktop@example.com", "not-a-real-password");
    const events = [];
    await client.api.streamChat({ question: "probe" }, (event, data) => {
      events.push({ event, data });
    });

    assert.deepEqual(commands, ["backend_status", "backend_origin"]);
    assert.deepEqual(requests, [
      `${expectedOrigin}/auth/login`,
      `${expectedOrigin}/chat/stream`,
    ]);
    assert.deepEqual(events, [{ event: "token", data: { text: "ok" } }]);
    assert.throws(
      () => client.configureApiOrigin("https://api.example.com"),
      /127\.0\.0\.1/,
    );

    client.configureApiOrigin();
    await client.api.me();
    assert.equal(requests.at(-1), "/auth/me");

    const mainBundle = await readFile(mainEntry, "utf8");
    assert.match(mainBundle, /configureApiOrigin/);
    assert.match(mainBundle, /waitForDesktopBackend/);
  } finally {
    if (originalWindow === undefined) delete globalThis.window;
    else globalThis.window = originalWindow;
    if (originalStorage === undefined) delete globalThis.localStorage;
    else globalThis.localStorage = originalStorage;
    if (originalFetch === undefined) delete globalThis.fetch;
    else globalThis.fetch = originalFetch;
  }

  process.stdout.write("Bundled desktop REST/SSE origin smoke passed.\n");
}
