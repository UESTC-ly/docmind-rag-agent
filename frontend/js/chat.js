// 流式对话：发送问题 → SSE 逐字渲染 → 来源卡片 → 落地。

import { api } from "./api.js";
import { $, el, toast } from "./ui.js";

let currentConversationId = null;
let activeDocumentId = null; // 可选：限定某文档
let streaming = false;

const stream = () => $("#chat-stream");

export function setActiveDocument(id) {
  activeDocumentId = id;
}

export function newConversation() {
  currentConversationId = null;
  stream().replaceChildren();
  addPlaceholder();
}

function addPlaceholder() {
  stream().append(
    el("div", { class: "empty", text: "问点什么，答案会实时逐字浮现。" })
  );
}

function clearPlaceholder() {
  const ph = stream().querySelector(".empty");
  if (ph) ph.remove();
}

function addMessage(role, text = "") {
  clearPlaceholder();
  const body = el("div", { class: "msg__body", text });
  const msg = el("div", { class: `msg msg--${role}` }, [
    el("div", { class: "msg__role", text: role === "user" ? "你" : "DocMind" }),
    body,
  ]);
  stream().append(msg);
  stream().scrollTop = stream().scrollHeight;
  return { msg, body };
}

function renderSources(container, sources) {
  if (!sources?.length) return;
  const box = el("div", { class: "sources" });
  for (const s of sources) {
    box.append(
      el("div", { class: "source" }, [
        el("div", {
          class: "source__meta",
          text: `文档 ${s.document_id} · 片段 ${s.chunk_index} · ${s.score.toFixed(3)}`,
        }),
        el("div", { class: "source__text", text: s.content }),
      ])
    );
  }
  container.append(box);
}

export async function loadConversation(id) {
  currentConversationId = id;
  stream().replaceChildren();
  try {
    const history = await api.getHistory(id);
    if (!history.length) return addPlaceholder();
    for (const m of history) addMessage(m.role, m.content);
  } catch (e) {
    toast(e.message);
  }
}

async function send(question) {
  if (streaming || !question.trim()) return;
  streaming = true;
  $("#send-btn").disabled = true;

  addMessage("user", question);
  const { msg, body } = addMessage("assistant", "");
  msg.classList.add("msg--streaming");

  try {
    await api.streamChat(
      { question, conversationId: currentConversationId, documentId: activeDocumentId },
      (event, data) => {
        if (event === "meta") {
          currentConversationId = data.conversation_id;
          renderSources(msg, data.sources);
          window.dispatchEvent(new CustomEvent("chat:conversation", { detail: data }));
        } else if (event === "token") {
          body.textContent += data.text;
          stream().scrollTop = stream().scrollHeight;
        }
      }
    );
  } catch (e) {
    body.textContent = body.textContent || `⚠ ${e.message}`;
    toast(e.message);
  } finally {
    msg.classList.remove("msg--streaming");
    streaming = false;
    $("#send-btn").disabled = false;
  }
}

export function initChat() {
  const form = $("#composer");
  const input = $("#composer-input");

  // 自适应高度
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
  });

  // Enter 发送，Shift+Enter 换行
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    input.style.height = "auto";
    send(q);
  });

  addPlaceholder();
}
