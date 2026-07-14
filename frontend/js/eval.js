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

const DEFAULT_POLL_INTERVAL_MS = 1500;
const POLL_TIMEOUT_MS = 30 * 60 * 1000;

function pollIntervalMs() {
  const override = Number(globalThis.__DOCMIND_EVAL_POLL_MS__);
  return Number.isFinite(override) && override >= 10
    ? override
    : DEFAULT_POLL_INTERVAL_MS;
}

const wait = (milliseconds) =>
  new Promise((resolve) => setTimeout(resolve, milliseconds));

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

function runState(run) {
  if (run.status === "completed") return renderRun(run);

  const failed = run.status === "failed";
  const labels = {
    pending: "评估任务已进入队列，正在等待 Worker…",
    running: "正在逐样本执行检索、生成与评分…",
    failed: run.error_message || "评估运行失败",
  };
  return el("p", {
    class: failed ? "empty eval-status eval-status--failed" : "empty eval-status",
    role: failed ? "alert" : "status",
    text: labels[run.status] || `评估状态：${run.status}`,
  });
}

async function waitForRun(run, onUpdate) {
  const startedAt = Date.now();
  let current = run;

  while (current.status === "pending" || current.status === "running") {
    if (Date.now() - startedAt >= POLL_TIMEOUT_MS) {
      throw new Error("评估仍在后台运行，请稍后重新打开评估页查看结果");
    }
    await wait(pollIntervalMs());
    current = await api.getRun(current.id);
    onUpdate(current);
  }

  return current;
}

function datasetBlock(ds) {
  const runBtn = el("button", {
    class: "btn btn--accent",
    text: "运行评估",
  });
  const result = el("div", {});

  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    runBtn.textContent = "提交中…";
    try {
      let run = await api.createRun(ds.id);
      result.replaceChildren(runState(run));
      runBtn.textContent = run.status === "pending" ? "排队中…" : "评估中…";

      run = await waitForRun(run, (updated) => {
        result.replaceChildren(runState(updated));
        runBtn.textContent = updated.status === "pending" ? "排队中…" : "评估中…";
      });

      if (run.status === "completed") {
        result.replaceChildren(renderRun(run));
        toast("评估完成");
      } else {
        result.replaceChildren(runState(run));
        toast(run.error_message || "评估运行失败");
      }
    } catch (e) {
      result.replaceChildren(
        el("p", {
          class: "empty eval-status eval-status--failed",
          role: "alert",
          text: e.message,
        })
      );
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
