// 文档管理：上传、列表、解析状态轮询、删除。

import { api } from "./api.js";
import { $, el, toast } from "./ui.js";
import {
  confirmDesktopAction,
  isDesktopApp,
  notifyDesktop,
  selectNativeDocument,
} from "./desktop.js";

const STATUS_LABEL = {
  pending: "待解析",
  processing: "解析中",
  completed: "已完成",
  failed: "失败",
};

let pollTimer = null;
const knownStatuses = new Map();

function docRow(doc) {
  const status = el("span", {
    class: `status status--${doc.status}`,
    text: STATUS_LABEL[doc.status] || doc.status,
  });
  const del = el("button", {
    class: "btn btn--ghost",
    text: "删除",
    onClick: () => removeDoc(doc.id, doc.filename),
  });
  return el("li", { class: "doc", "data-id": doc.id }, [
    el("div", {}, [
      el("div", { class: "doc__name", text: doc.filename }),
      el("div", {
        class: "doc__meta",
        text: `#${doc.id} · ${doc.chunk_count} 块 · ${new Date(doc.created_at).toLocaleString("zh-CN")}`,
      }),
    ]),
    status,
    del,
  ]);
}

export async function refreshDocs() {
  try {
    const docs = await api.listDocuments();
    const list = $("#doc-list");
    if (!docs.length) {
      list.replaceChildren(
        el("li", { class: "empty", text: "还没有文档，上传一个开始。" })
      );
    } else {
      list.replaceChildren(...docs.map(docRow));
    }
    // 有正在解析的就继续轮询
    const pending = docs.some((d) => d.status === "pending" || d.status === "processing");
    docs.forEach((doc) => {
      const previous = knownStatuses.get(doc.id);
      if (previous && previous !== "completed" && doc.status === "completed") {
        notifyDesktop("文档解析完成", `${doc.filename} 已可用于问答和技能。`);
      }
      knownStatuses.set(doc.id, doc.status);
    });
    schedulePoll(pending);
  } catch (e) {
    toast(e.message);
  }
}

function schedulePoll(shouldPoll) {
  clearTimeout(pollTimer);
  if (shouldPoll) pollTimer = setTimeout(refreshDocs, 2500);
}

async function upload(file) {
  if (!file) return;
  try {
    await api.uploadDocument(file);
    toast(`已上传 ${file.name}，正在后台解析`);
    refreshDocs();
  } catch (e) {
    toast(e.message);
  }
}

async function removeDoc(id, name) {
  const confirmed = await confirmDesktopAction(`删除文档「${name}」？其向量与分块会一并移除。`);
  if (!confirmed) return;
  try {
    await api.deleteDocument(id);
    toast("已删除");
    refreshDocs();
  } catch (e) {
    toast(e.message);
  }
}

export function initDocs() {
  const uploader = $("#uploader");
  const fileInput = $("#file-input");
  const nativePicker = $("#native-file-picker");

  fileInput.addEventListener("change", () => upload(fileInput.files[0]));
  if (isDesktopApp()) {
    nativePicker.hidden = false;
    nativePicker.addEventListener("click", async () => upload(await selectNativeDocument()));
  }

  // 拖拽上传
  ["dragover", "dragenter"].forEach((ev) =>
    uploader.addEventListener(ev, (e) => {
      e.preventDefault();
      uploader.classList.add("is-drag");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    uploader.addEventListener(ev, (e) => {
      e.preventDefault();
      uploader.classList.remove("is-drag");
    })
  );
  uploader.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));
}
