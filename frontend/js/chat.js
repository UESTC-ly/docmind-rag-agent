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

function sourceLocation(source) {
  const ranges = [];
  if (Number.isInteger(source.page_start)) {
    ranges.push(
      source.page_end && source.page_end !== source.page_start
        ? `第 ${source.page_start}-${source.page_end} 页`
        : `第 ${source.page_start} 页`,
    );
  }
  if (Number.isInteger(source.paragraph_start)) {
    ranges.push(
      source.paragraph_end && source.paragraph_end !== source.paragraph_start
        ? `第 ${source.paragraph_start}-${source.paragraph_end} 段`
        : `第 ${source.paragraph_start} 段`,
    );
  }
  if (Number.isInteger(source.char_start) && Number.isInteger(source.char_end)) {
    ranges.push(`字符 ${source.char_start}-${source.char_end}`);
  }
  return ranges.join(" · ");
}

function renderSourceExcerpt(container, source) {
  const text = String(source.source_excerpt ?? source.content ?? "");
  const start = Number(source.highlight_start);
  const end = Number(source.highlight_end);
  container.replaceChildren();
  if (
    Number.isInteger(start)
    && Number.isInteger(end)
    && start >= 0
    && end > start
    && end <= text.length
  ) {
    container.append(
      document.createTextNode(text.slice(0, start)),
      el("mark", { class: "source__highlight", text: text.slice(start, end) }),
      document.createTextNode(text.slice(end)),
    );
    container.classList.add("source__text--expanded");
    return;
  }
  container.textContent = text;
}

function renderSources(container, sources) {
  if (!sources?.length) return;
  const box = el("div", { class: "sources" });
  for (const s of sources) {
    const sourceText = el("div", { class: "source__text", text: s.content });
    const location = sourceLocation(s);
    const meta = el("button", {
      class: "source__meta source__jump",
      type: "button",
      text: `${s.citation_id ? `[${s.citation_id}] · ` : ""}${s.document_name || `文档 ${s.document_id}`} · 片段 ${s.chunk_index}${location ? ` · ${location}` : ""} · ${s.score.toFixed(3)}`,
      title: "跳转并高亮原文证据",
    });
    meta.addEventListener("click", async () => {
      meta.disabled = true;
      try {
        const original = await api.getDocumentChunk(
          s.document_id,
          s.chunk_index,
        );
        renderSourceExcerpt(sourceText, original);
        const originalLocation = sourceLocation(original);
        meta.textContent = `[${original.citation_id}] · ${original.document_name} · 片段 ${original.chunk_index}${originalLocation ? ` · ${originalLocation}` : ""} · ${original.source_version || "版本未知"} · ${original.source_status}`;
      } catch (error) {
        toast(error.message);
      } finally {
        meta.disabled = false;
      }
    });
    box.append(el("div", { class: "source" }, [meta, sourceText]));
  }
  container.append(box);
}

function renderCitationReport(container, report) {
  if (!report) return;
  const passed = Boolean(report.passed);
  const semantic = Boolean(report.semantic_entailment_checked);
  const summary = passed
    ? `${semantic ? "依据性" : "引用结构"}校验通过：${report.supported_claim_count}/${report.claim_count} 条结论有依据`
    : `${semantic ? "依据性" : "引用结构"}校验未通过：${report.unsupported_claim_count} 条结论未获支持`;
  const rows = [
    el("strong", { text: summary }),
    el("span", {
      text: semantic
        ? `Groundedness ${fmt(report.groundedness)} · Faithfulness ${fmt(report.faithfulness)} · Citation Correctness ${fmt(report.citation_correctness)}`
        : `引用精确率 ${fmt(report.citation_precision)} · 引用召回率 ${fmt(report.citation_recall)}`,
    }),
  ];
  if (report.delivery_action === "refused_after_quality_gate") {
    rows.push(el("span", { text: "质量门已拒绝输出无依据结论。" }));
  }
  if (report.delivery_action === "refused_due_to_conflicting_evidence") {
    rows.push(el("span", { text: "来源冲突未被答案充分披露，质量门已拒绝给出唯一结论。" }));
  }
  if (report.conflict_count) {
    rows.push(el("span", { text: `检测到 ${report.conflict_count} 组来源冲突。` }));
  }
  const interventions = report.quality_interventions || [];
  if (interventions.length) {
    const labels = {
      rewrite_query: "改写检索问题",
      expand_retrieval: "扩大召回范围",
      switch_pipeline: "切换已评测管线",
      re_retrieve_after_unsupported_claims: "无依据后再次检索",
      remove_unsupported_or_refuse: "删除无依据结论或拒答",
      refuse_insufficient_evidence: "证据不足拒答",
      refuse_stale_evidence: "当前有效证据不足",
      refuse_conflicting_evidence: "冲突证据拒答",
      refuse_judge_unavailable: "质量 Judge 不可用，拒绝交付",
    };
    const text = interventions
      .map((step) => labels[step.action] || step.action)
      .join(" · ");
    rows.push(el("span", { text: `Agent 质量策略：${text}` }));
  }
  if (semantic && report.claim_state_counts) {
    const labels = {
      supported: "原文支持",
      contradicted: "原文反驳",
      mixed: "证据混合",
      insufficient: "证据不足",
      inference: "模型推测",
    };
    const states = Object.entries(report.claim_state_counts)
      .filter(([, count]) => count)
      .map(([state, count]) => `${labels[state] || state} ${count}`)
      .join(" · ");
    if (states) rows.push(el("span", { text: states }));
  }
  const unsupported = (report.claims || []).filter((claim) =>
    semantic ? !claim.semantically_supported : !claim.supported
  );
  if (unsupported.length) {
    rows.push(
      el("ul", {}, unsupported.slice(0, 3).map((claim) =>
        el("li", { text: claim.text })
      ))
    );
  }
  container.append(
    el("div", {
      class: passed
        ? "citation-verification citation-verification--passed"
        : "citation-verification citation-verification--failed",
      role: passed ? "status" : "alert",
    }, rows)
  );
}

function fmt(value) {
  return typeof value === "number" ? value.toFixed(3) : "N/A";
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
        } else if (event === "sources") {
          const current = msg.querySelector(".sources");
          if (current) current.remove();
          renderSources(msg, data.sources);
        } else if (event === "verification") {
          renderCitationReport(msg, data);
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
