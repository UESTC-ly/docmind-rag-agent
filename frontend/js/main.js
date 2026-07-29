// 入口：鉴权流转、视图路由、各模块装配。

import { api, auth, configureApiOrigin } from "./api.js";
import { $, $$, el, toast } from "./ui.js";
import { initChat, loadConversation, newConversation } from "./chat.js";
import { initDocs, refreshDocs } from "./docs.js";
import { refreshEval } from "./eval.js";
import { initSkills, refreshSkills } from "./skills.js";
import { isDesktopApp, retryDesktopBackend, waitForDesktopBackend } from "./desktop.js";

const VIEW_META = {
  chat: { title: "证据问答", sub: "带引用、质量门与拒答的文档回答" },
  skills: { title: "Agent 工作台", sub: "计划 · 工具 · 自适应检索 · 证据门" },
  docs: { title: "知识库", sub: "文档、来源版本与可定位证据" },
  eval: { title: "EvalOps 实验室", sub: "公开基准 · 回归门禁 · Badcase" },
};

let registerMode = false;

// ── 视图切换 ──────────────────────────────
function switchView(name) {
  $$("[data-view-panel]").forEach((p) => (p.hidden = p.dataset.viewPanel !== name));
  $$(".nav__item[data-view]").forEach((b) =>
    b.setAttribute("aria-current", b.dataset.view === name ? "page" : "false")
  );
  $("#view-title").textContent = VIEW_META[name].title;
  $("#view-sub").textContent = VIEW_META[name].sub;
  if (name === "skills") refreshSkills();
  if (name === "docs") refreshDocs();
  if (name === "eval") refreshEval();
}

// ── 会话列表 ──────────────────────────────
async function refreshConversations(activeId = null) {
  try {
    const convos = await api.listConversations();
    const list = $("#conversation-list");
    list.replaceChildren(
      el("button", { class: "nav__item", text: "＋ 新对话", onClick: () => {
        newConversation();
        switchView("chat");
      } }),
      ...convos.map((c) =>
        el("button", {
          class: "nav__item",
          text: c.title || "未命名对话",
          "aria-current": c.id === activeId ? "page" : "false",
          onClick: () => {
            loadConversation(c.id);
            switchView("chat");
          },
        })
      )
    );
  } catch {
    /* 会话列表失败不阻塞主流程 */
  }
}

// ── 鉴权 ─────────────────────────────────
function showAuth() {
  $("#auth-screen").hidden = false;
  $("#app").hidden = true;
  document.body.classList.remove("is-app");
  window.scrollTo({ top: 0, left: 0 });
}

async function enterApp() {
  $("#auth-screen").hidden = true;
  $("#app").hidden = false;
  document.body.classList.add("is-app");
  window.scrollTo({ top: 0, left: 0 });
  try {
    const me = await api.me();
    $("#user-email").textContent = me.email;
  } catch {
    return showAuth();
  }
  refreshConversations();
  switchView("skills");
}

function initAuthScreen() {
  const form = $("#auth-form");
  const errorEl = $("#auth-error");

  $("#auth-switch").addEventListener("click", () => {
    registerMode = !registerMode;
    $("#auth-submit").textContent = registerMode ? "注册" : "登录";
    $("#auth-hint").textContent = registerMode ? "已有账号？" : "还没有账号？";
    $("#auth-switch").textContent = registerMode ? "登录" : "注册";
    errorEl.textContent = "";
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorEl.textContent = "";
    const email = $("#email").value.trim();
    const password = $("#password").value;
    const submit = $("#auth-submit");
    submit.disabled = true;
    try {
      if (registerMode) await api.register(email, password);
      const { access_token } = await api.login(email, password);
      auth.set(access_token);
      await enterApp();
    } catch (err) {
      errorEl.textContent = err.message;
    } finally {
      submit.disabled = false;
    }
  });
}

// ── 装配 ─────────────────────────────────
async function waitForDesktopServices() {
  if (!isDesktopApp()) return true;

  const status = $("#desktop-status");
  const retry = $("#desktop-retry");
  status.hidden = false;
  retry.hidden = true;
  try {
    const backendOrigin = await waitForDesktopBackend((backend) => {
      status.textContent = backend.message;
    });
    configureApiOrigin(backendOrigin);
    status.hidden = true;
    return true;
  } catch (error) {
    status.textContent = error.message;
    retry.hidden = false;
    retry.addEventListener("click", async () => {
      retry.disabled = true;
      retry.textContent = "重试中…";
      await retryDesktopBackend();
      location.reload();
    }, { once: true });
    return false;
  }
}

async function init() {
  if (!(await waitForDesktopServices())) return;
  initAuthScreen();
  initChat();
  initDocs();
  initSkills();

  $$(".nav__item[data-view]").forEach((b) =>
    b.addEventListener("click", () => switchView(b.dataset.view))
  );
  $("#logout-btn").addEventListener("click", () => {
    auth.clear();
    location.reload();
  });

  // 新会话产生后刷新侧栏列表
  window.addEventListener("chat:conversation", (e) =>
    refreshConversations(e.detail.conversation_id)
  );
  // token 过期统一处理
  window.addEventListener("auth:expired", () => {
    auth.clear();
    toast("登录已过期");
    showAuth();
  });

  if (auth.isLoggedIn) enterApp();
  else showAuth();
}

void init();
