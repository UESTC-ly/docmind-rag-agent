import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, normalize, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..", "dist");
const port = Number(process.env.DOCMIND_DESKTOP_DEV_PORT || 1420);
const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".svg": "image/svg+xml"
};

function fileFor(url) {
  const requestPath = decodeURIComponent(new URL(url, "http://localhost").pathname);
  const normalized = normalize(requestPath).replace(/^([/\\])+/, "");
  const candidate = resolve(root, normalized || "index.html");
  return candidate.startsWith(root) ? candidate : resolve(root, "index.html");
}

const server = createServer(async (request, response) => {
  let file = fileFor(request.url || "/");
  try {
    if ((await stat(file)).isDirectory()) file = resolve(file, "index.html");
    const type = mimeTypes[extname(file)] || "application/octet-stream";
    response.writeHead(200, { "Content-Type": type, "Cache-Control": "no-store" });
    createReadStream(file).pipe(response);
  } catch {
    response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    response.end("Not found");
  }
});

server.listen(port, "127.0.0.1", () => {
  console.log(`DocMind desktop frontend: http://127.0.0.1:${port}`);
});
