// 评估看板：列数据集、触发运行、可视化检索+生成指标。

import { api } from "./api.js";
import { $, el, fmt, toast } from "./ui.js";

const METRICS = [
  { key: "hit_rate", label: "命中率" },
  { key: "mrr", label: "MRR" },
  { key: "recall", label: "召回率" },
  { key: "precision", label: "精确率" },
  { key: "faithfulness", label: "忠实度" },
  { key: "answer_relevancy", label: "答案相关性" },
];

function metricCard(label, value) {
  const pct = typeof value === "number" ? Math.max(0, Math.min(1, value)) : 0;
  const bar = el("div", { class: "metric__bar" }, [el("span", {})]);
  const card = el("div", { class: "metric" }, [
    el("div", { class: "metric__label", text: label }),
    el("div", { class: "metric__value", text: fmt(value) }),
    bar,
  ]);
  // 进场后再设 transform，触发条形动画
  requestAnimationFrame(() => {
    bar.firstChild.style.transform = `scaleX(${pct})`;
  });
  return card;
}

function renderRun(run) {
  const grid = el("div", { class: "metrics" });
  for (const m of METRICS) grid.append(metricCard(m.label, run[m.key]));
  return grid;
}

function datasetBlock(ds) {
  const runBtn = el("button", {
    class: "btn btn--accent",
    text: "运行评估",
  });
  const result = el("div", {});

  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    runBtn.textContent = "评估中…（逐样本跑 RAG + LLM 打分）";
    try {
      const run = await api.createRun(ds.id);
      if (run.status === "completed") {
        result.replaceChildren(renderRun(run));
        toast("评估完成");
      } else {
        result.replaceChildren(
          el("p", { class: "empty", text: run.error_message || "运行失败" })
        );
      }
    } catch (e) {
      toast(e.message);
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = "重新运行";
    }
  });

  return el("div", { style: "margin-bottom: var(--space-12)" }, [
    el("div", {
      style: "display:flex; justify-content:space-between; align-items:baseline; border-bottom:1px solid var(--color-line); padding-bottom:var(--space-3)",
    }, [
      el("div", {}, [
        el("h2", { style: "font-size:var(--text-lg)", text: ds.name }),
        el("div", {
          class: "doc__meta",
          text: `数据集 #${ds.id} · 文档 ${ds.document_id} · ${ds.sample_count} 条样本`,
        }),
      ]),
      runBtn,
    ]),
    result,
  ]);
}

export async function refreshEval() {
  const root = $("#eval-content");
  try {
    const datasets = await api.listDatasets();
    if (!datasets.length) {
      root.replaceChildren(
        el("p", { class: "empty", text: "还没有评估数据集。可在后端生成或导入基准后在此运行。" })
      );
      return;
    }
    root.replaceChildren(...datasets.map(datasetBlock));
  } catch (e) {
    toast(e.message);
  }
}
