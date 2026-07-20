// API 客户端：统一鉴权、错误处理、SSE 流式解析。

const TOKEN_KEY = "docmind_token";
let apiOrigin = "";

// Browser deployments keep relative URLs. The Tauri host provides its loopback
// sidecar origin only after that backend reports ready.
export function configureApiOrigin(origin = "") {
  if (!origin) {
    apiOrigin = "";
    return;
  }

  const parsed = new URL(origin);
  if (
    parsed.protocol !== "http:" ||
    parsed.hostname !== "127.0.0.1" ||
    parsed.username ||
    parsed.password ||
    parsed.pathname !== "/" ||
    parsed.search ||
    parsed.hash
  ) {
    throw new Error("桌面 API 地址必须是无凭据的 127.0.0.1 HTTP origin");
  }
  apiOrigin = parsed.origin;
}

export function resolveApiUrl(path) {
  if (!path.startsWith("/") || path.startsWith("//")) {
    throw new Error(`API 路径必须以单个 / 开头：${path}`);
  }
  return apiOrigin ? `${apiOrigin}${path}` : path;
}

export const auth = {
  get token() {
    return localStorage.getItem(TOKEN_KEY);
  },
  set(token) {
    localStorage.setItem(TOKEN_KEY, token);
  },
  clear() {
    localStorage.removeItem(TOKEN_KEY);
  },
  get isLoggedIn() {
    return Boolean(this.token);
  },
};

function headers(extra = {}) {
  const h = { ...extra };
  if (auth.token) h["Authorization"] = `Bearer ${auth.token}`;
  return h;
}

function httpError(message, status) {
  const error = new Error(message);
  error.status = status;
  return error;
}

// 统一请求：非 2xx 抛带 message 的错误；401 触发登出事件。
async function request(path, { method = "GET", body, json = true } = {}) {
  const opts = { method, headers: headers() };
  if (body !== undefined) {
    if (json) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    } else {
      opts.body = body; // FormData
    }
  }

  const resp = await fetch(resolveApiUrl(path), opts);
  if (resp.status === 401) {
    window.dispatchEvent(new CustomEvent("auth:expired"));
    throw httpError("登录已过期，请重新登录", resp.status);
  }
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw httpError(detail.detail || `请求失败 (${resp.status})`, resp.status);
  }
  return resp.status === 204 ? null : resp.json();
}

export const api = {
  register: (email, password) =>
    request("/auth/register", { method: "POST", body: { email, password } }),
  login: (email, password) =>
    request("/auth/login", { method: "POST", body: { email, password } }),
  me: () => request("/auth/me"),

  listDocuments: () => request("/documents/"),
  getDocument: (id) => request(`/documents/${id}`),
  deleteDocument: (id) => request(`/documents/${id}`, { method: "DELETE" }),
  uploadDocument(file) {
    const fd = new FormData();
    fd.append("file", file);
    return request("/documents/upload", { method: "POST", body: fd, json: false });
  },

  listConversations: () => request("/chat/conversations"),
  getHistory: (id) => request(`/chat/conversations/${id}`),

  listSkills: () => request("/agent/skills"),
  agentChat: ({ message, conversationId, documentId, skillName, runId = null }) =>
    request("/agent/chat", {
      method: "POST",
      body: {
        message,
        conversation_id: conversationId ?? null,
        document_id: documentId ?? null,
        skill_name: skillName ?? null,
        run_id: runId,
      },
    }),
  resumeAgent: ({ runId, approved, comment = null, editedArgs = null }) =>
    request("/agent/resume", {
      method: "POST",
      body: {
        run_id: runId,
        approved,
        comment,
        edited_args: editedArgs,
      },
    }),
  getAgentRun: (runId) => request(`/agent/runs/${encodeURIComponent(runId)}`),
  recoverAgent: (runId) =>
    request(`/agent/runs/${encodeURIComponent(runId)}/recover`, { method: "POST" }),

  listDatasets: () => request("/eval/datasets"),
  createRun: (datasetId) =>
    request("/eval/runs", { method: "POST", body: { dataset_id: datasetId } }),
  getRun: (runId) => request(`/eval/runs/${runId}`),
  getRunDetails: (runId) => request(`/eval/runs/${runId}/details`),

  // SSE 流式问答：用 fetch + ReadableStream（EventSource 不能带 Authorization）。
  // 回调 onEvent(eventName, data) 逐事件触发。
  async streamChat({ question, conversationId, documentId }, onEvent) {
    const resp = await fetch(resolveApiUrl("/chat/stream"), {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        question,
        conversation_id: conversationId ?? null,
        document_id: documentId ?? null,
      }),
    });
    if (resp.status === 401) {
      window.dispatchEvent(new CustomEvent("auth:expired"));
      throw httpError("登录已过期", resp.status);
    }
    if (!resp.ok) throw httpError(`流式请求失败 (${resp.status})`, resp.status);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE 事件以空行分隔
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop(); // 末尾可能是半条，留到下轮
      for (const chunk of chunks) {
        const parsed = parseSSE(chunk);
        if (parsed) onEvent(parsed.event, parsed.data);
      }
    }
  },
};

function parseSSE(chunk) {
  let event = "message";
  let data = "";
  for (const line of chunk.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return null;
  try {
    return { event, data: JSON.parse(data) };
  } catch {
    return { event, data: {} };
  }
}
