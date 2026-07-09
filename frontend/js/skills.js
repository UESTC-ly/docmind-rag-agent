// Agent Skills 入口：展示已注册技能，并通过 /agent/chat 让 Agent 调用工具。

import { api } from "./api.js";
import { $, el, toast } from "./ui.js";

let skillsLoaded = false;
let agentConversationId = null;

const SKILL_HINTS = {
  search_knowledge_base: "根据已上传文档回答问题",
  generate_mindmap: "从文档内容生成 Mermaid 思维导图",
  generate_relation_graph: "抽取概念实体与关系图谱",
  generate_report: "检索资料并生成结构化报告",
  web_search: "知识库不足时联网搜索补充信息",
};

const SKILL_PROMPTS = {
  search_knowledge_base: "请检索知识库并回答：",
  generate_mindmap: "请基于当前文档生成思维导图：",
  generate_relation_graph: "请基于当前文档生成关系图谱：",
  generate_report: "请围绕以下主题生成报告：",
  web_search: "请联网搜索并总结：",
};

function skillCard(skill) {
  const prompt = SKILL_PROMPTS[skill.name] || `请使用 ${skill.name} 完成：`;
  return el("article", { class: "skill-card" }, [
    el("div", { class: "skill-card__top" }, [
      el("h3", { text: skill.name }),
      el("span", { class: "status", text: "可用" }),
    ]),
    el("p", { text: skill.description }),
    el("div", {
      class: "doc__meta",
      text: SKILL_HINTS[skill.name] || "由 Agent 通过 Function Calling 调用",
    }),
    el("button", {
      class: "btn btn--ghost",
      type: "button",
      text: "填入调用模板",
      onClick: () => {
        $("#skill-agent-input").value = prompt;
        $("#skill-agent-input").focus();
      },
    }),
  ]);
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
  } catch (e) {
    list.replaceChildren(el("p", { class: "empty", text: e.message }));
    toast(e.message);
  }
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
      ])
    ),
  ]);
}

function renderArtifacts(artifacts) {
  if (!artifacts?.length) return null;
  return el("div", { class: "artifact-list" }, [
    el("div", { class: "section-kicker", text: "结构化产出" }),
    ...artifacts.map((artifact, index) =>
      el("details", { class: "artifact", open: index === 0 ? "" : null }, [
        el("summary", { text: artifact.type || `artifact-${index + 1}` }),
        el("pre", { text: JSON.stringify(artifact, null, 2) }),
      ])
    ),
  ]);
}

function renderAgentResult(result) {
  const root = $("#skill-agent-result");
  const children = [
    el("article", { class: "agent-answer" }, [
      el("div", { class: "msg__role", text: "DocMind Agent" }),
      el("div", { class: "msg__body", text: result.answer || "Agent 未返回文本答案。" }),
    ]),
  ];
  const trace = renderTrace(result.trace);
  const artifacts = renderArtifacts(result.artifacts);
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
  try {
    const result = await api.agentChat({
      message,
      conversationId: agentConversationId,
      documentId,
    });
    agentConversationId = result.conversation_id;
    renderAgentResult(result);
    window.dispatchEvent(new CustomEvent("chat:conversation", { detail: result }));
  } catch (e) {
    $("#skill-agent-result").replaceChildren(el("p", { class: "empty", text: e.message }));
    toast(e.message);
  } finally {
    submit.disabled = false;
    submit.textContent = "调用 Agent";
  }
}

export function initSkills() {
  $("#refresh-skills-btn").addEventListener("click", () => {
    skillsLoaded = false;
    refreshSkills();
  });
  $("#skill-agent-form").addEventListener("submit", (e) => {
    e.preventDefault();
    submitAgentTask();
  });
}
