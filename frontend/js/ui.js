// 共享 UI 小工具：DOM 选择、转义、toast。

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on") && typeof v === "function")
      node.addEventListener(k.slice(2).toLowerCase(), v);
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c) node.append(c.nodeType ? c : document.createTextNode(c));
  }
  return node;
}

// 防 XSS：所有动态文本都走 textContent，不用 innerHTML 注入。
export function escapeText(s) {
  return String(s ?? "");
}

let toastTimer;
export function toast(message) {
  const t = $("#toast");
  t.textContent = message;
  t.classList.add("is-show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("is-show"), 2600);
}

export function fmt(n, digits = 3) {
  return typeof n === "number" ? n.toFixed(digits) : "—";
}
