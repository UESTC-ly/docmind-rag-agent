// Agent Skills 入口：展示已注册技能，并通过 /agent/chat 让 Agent 调用工具。

import { api } from "./api.js";
import { $, el, toast } from "./ui.js";
import { saveNativeArtifact } from "./desktop.js";

let skillsLoaded = false;
let agentConversationId = null;
let mermaidPromise = null;
let mermaidRenderSeq = 0;
let selectedSkillName = null;
let pendingRestoreAttempted = false;

const PENDING_AGENT_RUN_KEY = "docmind_pending_agent_run";

const MERMAID_CDN =
  "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";

const SKILL_HINTS = {
  search_knowledge_base: "根据已上传文档回答问题",
  generate_mindmap: "从文档内容生成 Mermaid 思维导图",
  generate_relation_graph: "抽取概念实体与关系图谱",
  generate_report: "检索资料并生成结构化报告",
  generate_weekly_report: "基于材料生成可下载 Markdown 周报",
  generate_presentation: "基于材料生成可下载 PowerPoint 文件",
  web_search: "知识库不足时联网搜索补充信息",
};

const SKILL_PROMPTS = {
  search_knowledge_base: "请检索知识库并回答：",
  generate_mindmap: "请基于当前文档生成思维导图：",
  generate_relation_graph: "请基于当前文档生成关系图谱：",
  generate_report: "请围绕以下主题生成报告：",
  generate_weekly_report: "请根据材料写一份本周周报，主题是：",
  generate_presentation: "请根据材料制作一份 6 页 PPT，主题是：",
  web_search: "请联网搜索并总结：",
};

function skillCard(skill) {
  const prompt = SKILL_PROMPTS[skill.name] || `请使用 ${skill.name} 完成：`;
  const packageText = skill.package
    ? `Package ${skill.package.slug} · ${skill.package.source || "skill_json"} · ${skill.package.templates.length} 模板 · ${skill.package.references.length} 参考`
    : SKILL_HINTS[skill.name] || "由 Agent 通过 Function Calling 调用";
  const mode = !skill.available
    ? "暂不可用"
    : skill.execution_mode === "generic_package"
      ? "通用包"
      : (skill.package ? "Package" : "可用");
  const grounding = skill.grounding_mode === "hybrid_rag"
    ? "RAG 检索"
    : skill.grounding_mode === "document_prefix"
      ? "顺序正文（前 8000 字）"
      : skill.grounding_mode === "rag_when_document_selected"
        ? "选中文档时走 RAG"
        : skill.grounding_mode === "web"
          ? "联网"
          : "无需文档";
  return el("article", { class: "skill-card" }, [
    el("div", { class: "skill-card__top" }, [
      el("h3", { text: skill.name }),
      el("span", { class: "status", text: mode }),
    ]),
    el("p", { text: skill.description }),
    !skill.available
      ? el("p", { class: "empty", text: skill.unavailable_reason || "尚未完成运行时适配。" })
      : null,
    el("div", {
      class: "doc__meta",
      text: `${packageText} · ${grounding}${skill.produces_download ? " · 可下载" : ""}`,
    }),
    el("button", {
      class: "btn btn--ghost",
      type: "button",
      text: skill.available ? "选择并填入模板" : "等待适配",
      disabled: skill.available ? null : "",
      onClick: () => {
        $("#skill-agent-input").value = prompt;
        selectedSkillName = skill.name;
        $("#selected-skill-name").textContent = `已锁定技能：${skill.name}`;
        $("#skill-agent-input").focus();
      },
    }),
  ].filter(Boolean));
}

function renderSkills(skills) {
  const list = $("#skill-list");
  if (!skills.length) {
    list.replaceChildren(el("p", { class: "empty", text: "当前没有注册的技能。" }));
    return;
  }
  list.replaceChildren(...skills.map(skillCard));
}

export async function refreshSkills() {
  const list = $("#skill-list");
  if (!skillsLoaded) {
    list.replaceChildren(el("p", { class: "empty", text: "正在加载技能列表…" }));
  }
  try {
    const skills = await api.listSkills();
    renderSkills(skills);
    skillsLoaded = true;
    await restorePendingAgentRun();
  } catch (e) {
    list.replaceChildren(el("p", { class: "empty", text: e.message }));
    toast(e.message);
  }
}

async function restorePendingAgentRun() {
  if (pendingRestoreAttempted) return;
  pendingRestoreAttempted = true;
  const runId = localStorage.getItem(PENDING_AGENT_RUN_KEY);
  if (!runId) return;
  try {
    await renderCheckpoint(runId);
  } catch (error) {
    // Run 可能已经清理、属于上一个登录用户或不再可访问；不让恢复失败阻塞技能列表。
    // 只在权威 404 时丢弃 ID；401/网络/服务端暂时失败都保留，便于重新登录或刷新后再试。
    if (error.status === 404) localStorage.removeItem(PENDING_AGENT_RUN_KEY);
  }
}

async function renderCheckpoint(runId) {
  const result = await api.getAgentRun(runId);
  agentConversationId = result.conversation_id;
  renderAgentResult(result);
  return result;
}

function renderTrace(trace) {
  if (!trace?.length) return null;
  return el("div", { class: "trace" }, [
    el("div", { class: "section-kicker", text: "调用轨迹" }),
    ...trace.map((step) =>
      el("div", { class: "trace__item" }, [
        el("span", { text: `#${step.step + 1}` }),
        el("strong", { text: step.skill }),
        el("code", { text: JSON.stringify(step.args ?? {}) }),
        step.approval
          ? el("span", {
              class: "status",
              text: step.approval.approved ? "已审批" : "已拒绝",
            })
          : null,
      ].filter(Boolean))
    ),
  ]);
}

function stripMermaidFence(code) {
  return String(code ?? "")
    .trim()
    .replace(/^```(?:mermaid)?\s*/i, "")
    .replace(/```$/i, "")
    .trim();
}

function mermaidCodeFor(artifact) {
  if (artifact.type === "mindmap") return stripMermaidFence(artifact.content);
  if (artifact.type === "relation_graph") return stripMermaidFence(artifact.mermaid);
  return "";
}

function decodeBase64(content) {
  const binary = atob(content);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function downloadArtifact(download) {
  if (await saveNativeArtifact(download)) {
    toast("已保存到系统选择的位置");
    return;
  }
  const payload =
    download.encoding === "base64"
      ? decodeBase64(download.content)
      : download.content || "";
  const blob = new Blob([payload], {
    type: download.mime_type || "application/octet-stream",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = download.filename || "docmind-artifact";
  document.body.append(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function renderDownload(download) {
  if (!download?.content) return null;
  return el("div", { class: "artifact-download" }, [
    el("div", {}, [
      el("div", { class: "artifact-download__name", text: download.filename || "产出文件" }),
      el("div", {
        class: "doc__meta",
        text: `${download.mime_type || "application/octet-stream"} · ${download.encoding || "text"}`,
      }),
    ]),
    el("button", {
      class: "btn btn--accent",
      type: "button",
      text: "下载产出",
      onClick: () => downloadArtifact(download),
    }),
  ]);
}

function loadMermaid() {
  if (!mermaidPromise) {
    mermaidPromise = import(MERMAID_CDN).then((mod) => {
      const mermaid = mod.default;
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "strict",
        theme: window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "neutral",
      });
      return mermaid;
    });
  }
  return mermaidPromise;
}

async function renderMermaid(container, code) {
  const source = stripMermaidFence(code);
  if (!source) return;

  container.replaceChildren(
    el("p", { class: "mermaid-status", text: "正在渲染 Mermaid 图…" })
  );

  try {
    const mermaid = await loadMermaid();
    const id = `docmind-mermaid-${Date.now()}-${mermaidRenderSeq++}`;
    const { svg } = await mermaid.render(id, source);
    // Mermaid 在 securityLevel=strict 下生成 SVG；这里必须插入 SVG 字符串才能显示图。
    container.innerHTML = svg;
  } catch (err) {
    container.replaceChildren(
      el("p", {
        class: "mermaid-status mermaid-status--error",
        text: `Mermaid 渲染失败，已显示源码：${err.message || err}`,
      }),
      el("pre", { text: source })
    );
  }
}

function renderArtifactBody(artifact) {
  const download = renderDownload(artifact.download);
  const mermaidCode = mermaidCodeFor(artifact);
  if (mermaidCode) {
    const preview = el("div", { class: "mermaid-preview" });
    queueMicrotask(() => renderMermaid(preview, mermaidCode));
    return el("div", { class: "artifact__body" }, [
      download,
      preview,
      el("details", { class: "artifact-source" }, [
        el("summary", { text: "查看 Mermaid 源码" }),
        el("pre", { text: mermaidCode }),
      ]),
    ].filter(Boolean));
  }

  if (artifact.type === "weekly_report") {
    return el("div", { class: "artifact__body" }, [
      download,
      el("pre", { text: artifact.content || "周报内容为空。" }),
    ].filter(Boolean));
  }

  if (artifact.type === "presentation") {
    return el("div", { class: "artifact__body" }, [
      download,
      el("div", { class: "slide-preview-list" }, [
        ...(artifact.slides || []).map((slide, index) =>
          el("article", { class: "slide-preview" }, [
            el("div", { class: "doc__meta", text: `Slide ${index + 1}` }),
            el("h3", { text: slide.title || "未命名页面" }),
            el("ul", {}, (slide.bullets || []).map((b) => el("li", { text: b }))),
          ])
        ),
      ]),
    ].filter(Boolean));
  }

  return el("div", { class: "artifact__body" }, [
    download,
    el("pre", { text: JSON.stringify(artifact, null, 2) }),
  ].filter(Boolean));
}

function renderArtifacts(artifacts) {
  if (!artifacts?.length) return null;
  return el("div", { class: "artifact-list" }, [
    el("div", { class: "section-kicker", text: "结构化产出" }),
    ...artifacts.map((artifact, index) =>
      el("details", { class: "artifact", open: index === 0 ? "" : null }, [
        el("summary", { text: artifact.type || `artifact-${index + 1}` }),
        renderArtifactBody(artifact),
      ])
    ),
  ]);
}

function approvalCard(result) {
  const approval = result.approval;
  if (result.status !== "waiting_approval" || !approval || !result.run_id) return null;

  const approve = el("button", {
    class: "btn btn--accent",
    type: "button",
    text: "批准并继续",
  });
  const reject = el("button", {
    class: "btn btn--ghost",
    type: "button",
    text: "拒绝本次调用",
  });
  const controls = [approve, reject];

  async function decide(approved) {
    for (const button of controls) button.disabled = true;
    approve.textContent = approved ? "正在恢复…" : "批准并继续";
    reject.textContent = approved ? "拒绝本次调用" : "正在拒绝…";
    try {
      const resumed = await api.resumeAgent({ runId: result.run_id, approved });
      agentConversationId = resumed.conversation_id;
      renderAgentResult(resumed);
      window.dispatchEvent(new CustomEvent("chat:conversation", { detail: resumed }));
    } catch (e) {
      try {
        await renderCheckpoint(result.run_id);
      } catch {
        toast(e.message);
        for (const button of controls) button.disabled = false;
        approve.textContent = "批准并继续";
        reject.textContent = "拒绝本次调用";
      }
    }
  }

  approve.addEventListener("click", () => decide(true));
  reject.addEventListener("click", () => decide(false));
  const riskCapabilities = approval.risk_capabilities?.length
    ? `高风险能力：${approval.risk_capabilities.join("、")}`
    : "高风险外层工具调用";
  return el("article", { class: "approval-card", role: "alert" }, [
    el("div", { class: "section-kicker", text: "需要人工审批" }),
    el("h3", { text: approval.tool || "未命名工具" }),
    el("p", { text: approval.reason || riskCapabilities }),
    el("div", { class: "doc__meta", text: riskCapabilities }),
    el("pre", { text: JSON.stringify(approval.args ?? {}, null, 2) }),
    approval.scope === "outer_skill"
      ? el("p", {
          class: "approval-card__scope",
          text: "审批范围：本次外层 Skill 调用。Generic Skill 内部逐动作审批将在后续子图迁移中实现。",
        })
      : null,
    el("div", { class: "approval-card__actions" }, controls),
  ].filter(Boolean));
}

function recoveryCard(result) {
  if (result.status !== "running" || !result.recoverable || !result.run_id) return null;
  const recover = el("button", {
    class: "btn btn--accent",
    type: "button",
    text: "从 checkpoint 继续",
  });
  recover.addEventListener("click", async () => {
    recover.disabled = true;
    recover.textContent = "正在恢复…";
    try {
      const resumed = await api.recoverAgent(result.run_id);
      agentConversationId = resumed.conversation_id;
      renderAgentResult(resumed);
      window.dispatchEvent(new CustomEvent("chat:conversation", { detail: resumed }));
    } catch (e) {
      try {
        await renderCheckpoint(result.run_id);
      } catch {
        recover.disabled = false;
        recover.textContent = "从 checkpoint 继续";
        toast(e.message);
      }
    }
  });
  return el("article", { class: "approval-card recovery-card", role: "status" }, [
    el("div", { class: "section-kicker", text: "发现未完成的 Agent run" }),
    el("p", {
      text: "确认原执行已停止后，可从该节点继续。恢复会复用工具执行回执；结果不确定的副作用会再次请求人工决定，绝不会静默重试。",
    }),
    recover,
  ]);
}

function renderAgentResult(result) {
  if (
    (result.status === "waiting_approval" || result.recoverable) &&
    result.run_id
  ) {
    localStorage.setItem(PENDING_AGENT_RUN_KEY, result.run_id);
  } else if (
    result.run_id &&
    localStorage.getItem(PENDING_AGENT_RUN_KEY) === result.run_id
  ) {
    localStorage.removeItem(PENDING_AGENT_RUN_KEY);
  }
  const root = $("#skill-agent-result");
  const children = [
    el("article", { class: "agent-answer" }, [
      el("div", { class: "msg__role", text: "DocMind Agent" }),
      el("div", { class: "msg__body", text: result.answer || "Agent 未返回文本答案。" }),
    ]),
  ];
  const trace = renderTrace(result.trace);
  const artifacts = renderArtifacts(result.artifacts);
  const approval = approvalCard(result);
  const recovery = recoveryCard(result);
  if (approval) children.push(approval);
  if (recovery) children.push(recovery);
  if (trace) children.push(trace);
  if (artifacts) children.push(artifacts);
  root.replaceChildren(...children);
}

async function submitAgentTask() {
  const input = $("#skill-agent-input");
  const docInput = $("#skill-document-id");
  const message = input.value.trim();
  if (!message) return;

  const submit = $("#skill-agent-submit");
  submit.disabled = true;
  submit.textContent = "调用中…";
  $("#skill-agent-result").replaceChildren(
    el("p", { class: "empty", text: "Agent 正在选择技能并执行…" })
  );

  const documentId = docInput.value ? Number(docInput.value) : null;
  const runId = crypto.randomUUID();
  localStorage.setItem(PENDING_AGENT_RUN_KEY, runId);
  try {
    const result = await api.agentChat({
      message,
      conversationId: agentConversationId,
      documentId,
      skillName: selectedSkillName,
      runId,
    });
    agentConversationId = result.conversation_id;
    renderAgentResult(result);
    selectedSkillName = null;
    $("#selected-skill-name").textContent = "未锁定技能：Agent 自动选择";
    window.dispatchEvent(new CustomEvent("chat:conversation", { detail: result }));
  } catch (e) {
    try {
      await renderCheckpoint(runId);
    } catch (checkpointError) {
      if (checkpointError.status === 404) {
        localStorage.removeItem(PENDING_AGENT_RUN_KEY);
      }
      $("#skill-agent-result").replaceChildren(el("p", { class: "empty", text: e.message }));
      toast(e.message);
    }
  } finally {
    submit.disabled = false;
    submit.textContent = "调用 Agent";
  }
}

export function initSkills() {
  window.addEventListener("auth:expired", () => {
    pendingRestoreAttempted = false;
    agentConversationId = null;
  });
  $("#refresh-skills-btn").addEventListener("click", () => {
    skillsLoaded = false;
    refreshSkills();
  });
  $("#skill-agent-form").addEventListener("submit", (e) => {
    e.preventDefault();
    submitAgentTask();
  });
  $("#skill-agent-input").addEventListener("input", () => {
    if (selectedSkillName) {
      selectedSkillName = null;
      $("#selected-skill-name").textContent = "未锁定技能：Agent 自动选择";
    }
  });
}
