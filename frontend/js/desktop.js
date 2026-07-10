// 桌面端桥接层：浏览器保持原有行为，Tauri WebView 才动态加载原生能力。

export function isDesktopApp() {
  return Boolean(window.__TAURI_INTERNALS__);
}

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function invoke(command, args) {
  const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
  return tauriInvoke(command, args);
}

export async function waitForDesktopBackend(onStatus) {
  if (!isDesktopApp()) return;

  for (let attempt = 0; attempt < 100; attempt += 1) {
    const status = await invoke("backend_status");
    onStatus?.(status);
    if (status.state === "ready") return;
    if (status.state === "error") throw new Error(status.message);
    await delay(1000);
  }
  throw new Error("DocMind 本地服务启动超时，请重启应用后重试。");
}

export async function retryDesktopBackend() {
  if (isDesktopApp()) await invoke("retry_backend");
}

export async function selectNativeDocument() {
  if (!isDesktopApp()) return null;

  const [{ open }, { readFile }] = await Promise.all([
    import("@tauri-apps/plugin-dialog"),
    import("@tauri-apps/plugin-fs"),
  ]);
  const selected = await open({
    multiple: false,
    title: "选择要导入 DocMind 的文档",
    filters: [
      { name: "支持的文档", extensions: ["pdf", "docx", "doc", "txt", "md"] },
    ],
  });
  if (!selected || Array.isArray(selected)) return null;

  const bytes = await readFile(selected);
  const filename = selected.split(/[\\/]/).pop() || "document";
  return new File([bytes], filename);
}

export async function saveNativeArtifact(download) {
  if (!isDesktopApp()) return false;

  const [{ save }, { writeFile }] = await Promise.all([
    import("@tauri-apps/plugin-dialog"),
    import("@tauri-apps/plugin-fs"),
  ]);
  const filename = download.filename || "docmind-artifact";
  const path = await save({
    title: "保存 DocMind 产出",
    defaultPath: filename,
  });
  if (!path) return true;

  const content = download.encoding === "base64"
    ? decodeBase64(download.content)
    : new TextEncoder().encode(download.content || "");
  await writeFile(path, content);
  await notifyDesktop("产出已保存", filename);
  return true;
}

export async function confirmDesktopAction(message, options = {}) {
  if (!isDesktopApp()) return window.confirm(message);
  const { confirm } = await import("@tauri-apps/plugin-dialog");
  return confirm(message, { title: "DocMind", kind: "warning", ...options });
}

export async function notifyDesktop(title, body) {
  if (!isDesktopApp()) return;
  const notification = await import("@tauri-apps/plugin-notification");
  let granted = await notification.isPermissionGranted();
  if (!granted) granted = (await notification.requestPermission()) === "granted";
  if (granted) notification.sendNotification({ title, body });
}

function decodeBase64(content) {
  const binary = atob(content);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}
