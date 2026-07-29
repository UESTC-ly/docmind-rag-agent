// 评估看板：列数据集、触发运行、可视化检索+生成指标。

import { api } from "./api.js";
import { $, el, fmt, toast } from "./ui.js";

const METRICS = [
  { key: "hit_rate", label: "命中率" },
  { key: "mrr", label: "MRR" },
  { key: "recall", label: "召回率" },
  { key: "precision", label: "精确率" },
  { key: "map_score", label: "MAP@K" },
  { key: "ndcg", label: "nDCG@K" },
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

function pipelineTitle(run) {
  const id = run.pipeline_id || "configured";
  const fingerprint = run.pipeline_fingerprint
    ? ` · ${run.pipeline_fingerprint.slice(0, 12)}`
    : "";
  return `${id}${fingerprint}`;
}

function renderRun(run) {
  const grid = el("div", { class: "metrics" });
  for (const m of METRICS) grid.append(metricCard(m.label, run[m.key]));
  return el("section", { class: "eval-run", "data-run-id": run.id }, [
    el("div", { class: "eval-run__heading" }, [
      el("h3", { class: "eval-run__title", text: pipelineTitle(run) }),
      el("span", {
        class: `eval-role eval-role--${run.comparison_role || "standalone"}`,
        text: run.comparison_role === "baseline"
          ? "基线"
          : run.comparison_role === "candidate"
            ? "候选"
            : "单次运行",
      }),
    ]),
    grid,
    el("div", { class: "eval-run__citation" }, [
      el("span", { text: `引用精确率 ${fmt(run.citation_precision)}` }),
      el("span", { text: `引用召回率 ${fmt(run.citation_recall)}` }),
      el("span", { text: `无证据断言率 ${fmt(run.unsupported_claim_rate)}` }),
    ]),
    el("div", {
      class: "eval-diagnostics",
      text: "正在加载版本化指标、回归门禁与 Badcase…",
    }),
  ]);
}

function aggregateMetrics(rows) {
  return new Map(
    rows
      .filter((row) => row.subject_type === "run")
      .map((row) => [row.metric_name, row]),
  );
}

function metricStrip(metrics) {
  const definitions = [
    ["document_recall_at_k", "Document Recall@K"],
    ["document_ndcg_at_k", "Document nDCG@K"],
    ["groundedness", "Groundedness"],
    ["citation_correctness", "Citation Correctness"],
    ["refusal_correctness", "拒答正确率"],
    ["conflict_detection_accuracy", "冲突识别"],
  ];
  const available = definitions.filter(([key]) =>
    typeof metrics.get(key)?.score === "number");
  if (!available.length) return null;
  return el(
    "div",
    { class: "metrics metrics--diagnostic" },
    available.map(([key, label]) => {
      const metric = metrics.get(key);
      const card = metricCard(label, metric.score);
      card.title = `${metric.metric_version} · ${metric.evaluator_kind}`;
      return card;
    }),
  );
}

function regressionSummary(rows) {
  if (!rows.length) {
    return el("p", {
      class: "eval-note",
      text: "本次运行没有已生效的回归门禁。",
    });
  }
  const failed = rows.filter((row) => !row.passed);
  return el(
    "div",
    {
      class: failed.length
        ? "regression regression--failed"
        : "regression regression--passed",
    },
    [
      el("strong", {
        text: failed.length
          ? `回归门禁失败 ${failed.length}/${rows.length}`
          : `回归门禁通过 ${rows.length}/${rows.length}`,
      }),
      ...failed.map((row) =>
        el("div", {
          class: "eval-note",
          text: `${row.metric_name}: ${row.reason}`,
        })),
    ],
  );
}

function badcaseDiffSummary(diff) {
  if (!diff) return null;
  const introduced = diff.newly_introduced?.length || 0;
  const fixed = diff.fixed?.length || 0;
  const persistent = diff.persistent?.length || 0;
  return el("div", {
    class: introduced
      ? "regression regression--failed"
      : "regression regression--passed",
  }, [
    el("strong", {
      text: `相对基线：新增 ${introduced} · 修复 ${fixed} · 持续 ${persistent}`,
    }),
    ...((diff.newly_introduced || []).slice(0, 5).map((row) =>
      el("div", {
        class: "eval-note",
        text: `${row.external_id || `样本 ${row.sample_id}`}：${row.introduced_categories.join(", ")}`,
      }))),
  ]);
}

function badcaseCard(badcase, run) {
  const sourceBox = el("div", { class: "badcase__sources" });
  for (const source of badcase.source_links || []) {
    const output = el("pre", { class: "badcase__source-text" });
    const button = el("button", {
      class: "btn btn--ghost btn--small",
      type: "button",
      text: `[${source.citation_id}] 核对原文`,
    });
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        const chunk = await api.getDocumentChunk(
          source.document_id,
          source.chunk_index,
        );
        const page = Number.isInteger(chunk.page_start)
          ? chunk.page_end && chunk.page_end !== chunk.page_start
            ? ` · 第 ${chunk.page_start}-${chunk.page_end} 页`
            : ` · 第 ${chunk.page_start} 页`
          : "";
        output.replaceChildren(
          document.createTextNode(`${chunk.document_name} · ${chunk.source_version || "版本未知"} · ${chunk.source_status}${page}\n`),
        );
        const excerpt = String(chunk.source_excerpt ?? chunk.content ?? "");
        const start = Number(chunk.highlight_start);
        const end = Number(chunk.highlight_end);
        if (Number.isInteger(start) && Number.isInteger(end) && start >= 0 && end > start && end <= excerpt.length) {
          output.append(
            document.createTextNode(excerpt.slice(0, start)),
            el("mark", { class: "source__highlight", text: excerpt.slice(start, end) }),
            document.createTextNode(excerpt.slice(end)),
          );
        } else {
          output.append(document.createTextNode(excerpt));
        }
      } catch (error) {
        toast(error.message);
      } finally {
        button.disabled = false;
      }
    });
    sourceBox.append(button, output);
  }
  const rerunState = el("span", {
    class: "badcase__rerun-state",
    role: "status",
  });
  const rerunButton = el("button", {
    class: "btn btn--ghost btn--small",
    type: "button",
    text: "仅重跑此 Badcase",
  });
  rerunButton.addEventListener("click", async () => {
    rerunButton.disabled = true;
    rerunState.textContent = "正在创建诊断重跑…";
    try {
      let rerun = await api.rerunBadcases(run.id, [badcase.sample_id]);
      rerunState.textContent = `诊断重跑 #${rerun.id}：${rerun.status}`;
      if (rerun.status === "pending" || rerun.status === "running") {
        rerun = await waitForRun(rerun, (updated) => {
          rerunState.textContent = `诊断重跑 #${updated.id}：${updated.status}`;
        });
      }
      if (rerun.status === "completed") {
        const [row] = await api.getRunBadcases(rerun.id, true);
        rerunState.textContent = row?.categories?.length
          ? `诊断重跑 #${rerun.id} 完成：仍有 ${row.categories.join(", ")}`
          : `诊断重跑 #${rerun.id} 完成：该样本已通过`;
      } else if (rerun.status === "failed") {
        rerunState.textContent = `诊断重跑失败：${rerun.error_message || "未知错误"}`;
      }
    } catch (error) {
      rerunState.textContent = `诊断重跑失败：${error.message}`;
    } finally {
      rerunButton.disabled = false;
    }
  });
  return el("details", { class: "badcase" }, [
    el("summary", {}, [
      el("strong", { text: badcase.external_id || `样本 ${badcase.sample_id}` }),
      el("span", { text: ` · ${badcase.question}` }),
    ]),
    el(
      "div",
      { class: "badcase__tags" },
      badcase.categories.map((category) =>
        el("span", { class: "badcase-tag", text: category })),
    ),
    ...((badcase.diagnosis || []).map((item) =>
      el("div", { class: "badcase__diagnosis" }, [
        el("strong", { text: `${item.layer} · ${item.category}` }),
        el("span", { text: item.explanation }),
        el("small", {
          text: `建议：${(item.suggested_actions || []).join("；")}`,
        }),
      ]))),
    el("dl", { class: "badcase__content" }, [
      el("dt", { text: "标准答案" }),
      el("dd", {
        text: badcase.ground_truth_answer || "（检索任务无标准生成答案）",
      }),
      el("dt", { text: "系统答案" }),
      el("dd", { text: badcase.generated_answer || "（未生成）" }),
      el("dt", { text: "相关 / 召回片段" }),
      el("dd", {
        text: `${badcase.relevant_chunk_ids.join(", ") || "—"} / ${badcase.retrieved_chunk_ids.join(", ") || "—"}`,
      }),
    ]),
    el("div", { class: "badcase__rerun" }, [rerunButton, rerunState]),
    sourceBox,
  ]);
}

async function loadDiagnostics(run, section) {
  const target = $(".eval-diagnostics", section);
  try {
    const [metricRows, badcases, regressions, diff] = await Promise.all([
      api.getRunMetrics(run.id),
      api.getRunBadcases(run.id),
      api.getRunRegression(run.id),
      run.baseline_run_id
        ? api.getRunBadcaseDiff(run.id)
        : Promise.resolve(null),
    ]);
    const grounding = metricStrip(aggregateMetrics(metricRows));
    target.replaceChildren(
      el("h4", { text: "回归与 Badcase 诊断" }),
      ...(grounding ? [grounding] : []),
      regressionSummary(regressions),
      ...(diff ? [badcaseDiffSummary(diff)] : []),
      el("div", { class: "badcase-list" }, [
        el("strong", { text: `Badcase ${badcases.length} 条` }),
        ...badcases.map((badcase) => badcaseCard(badcase, run)),
      ]),
    );
  } catch (error) {
    target.replaceChildren(
      el("p", {
        class: "eval-note",
        text: `诊断数据加载失败：${error.message}`,
      }),
    );
  }
}

function runState(run) {
  if (run.status === "completed") return renderRun(run);

  const failed = run.status === "failed";
  const labels = {
    pending: "评估任务已进入队列，正在等待 Worker…",
    running: "正在逐样本执行检索、生成与评分…",
    failed: run.error_message || "评估运行失败",
  };
  return el("section", { class: "eval-run" }, [
    el("h3", { class: "eval-run__title", text: pipelineTitle(run) }),
    el("p", {
      class: failed ? "empty eval-status eval-status--failed" : "empty eval-status",
      role: failed ? "alert" : "status",
      text: labels[run.status] || `评估状态：${run.status}`,
    }),
  ]);
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

function pipelinePicker(pipelines) {
  return el("fieldset", { class: "pipeline-picker" }, [
    el("legend", { text: "选择要比较的 RAG 管线" }),
    ...pipelines.map((pipeline) =>
      el("label", { class: "pipeline-option" }, [
        el("input", {
          type: "checkbox",
          value: pipeline.id,
          checked: pipeline.id === "configured" ? "" : null,
        }),
        el("span", {}, [
          el("strong", { text: pipeline.label }),
          el("small", {
            text: `${pipeline.retriever} → ${pipeline.fusion} → ${pipeline.reranker}`,
          }),
        ]),
      ])
    ),
  ]);
}

function datasetBlock(ds, pipelines) {
  const runBtn = el("button", {
    class: "btn btn--accent",
    text: "运行评估",
  });
  const picker = pipelinePicker(pipelines);
  const result = el("div", { class: "eval-runs" });

  runBtn.addEventListener("click", async () => {
    const selected = [...picker.querySelectorAll("input:checked")]
      .map((input) => input.value);
    if (!selected.length) {
      toast("请至少选择一条 RAG 管线");
      return;
    }
    runBtn.disabled = true;
    runBtn.textContent = "提交中…";
    try {
      const initialRuns = selected.length === 1
        ? [await api.createRun(ds.id, selected[0])]
        : await api.createExperiment(ds.id, selected);
      const states = new Map(initialRuns.map((run) => [run.id, run]));
      const renderStates = () => {
        result.replaceChildren(...[...states.values()].map(runState));
      };
      renderStates();
      runBtn.textContent = "评估中…";

      const completedRuns = await Promise.all(initialRuns.map((run) =>
        waitForRun(run, (updated) => {
          states.set(updated.id, updated);
          renderStates();
        })
      ));
      completedRuns.forEach((run) => states.set(run.id, run));
      renderStates();
      await Promise.all(
        completedRuns
          .filter((run) => run.status === "completed")
          .map((run) => {
            const section = result.querySelector(`[data-run-id="${run.id}"]`);
            return section ? loadDiagnostics(run, section) : Promise.resolve();
          }),
      );

      if (completedRuns.every((run) => run.status === "completed")) {
        toast("评估完成");
      } else {
        const failed = completedRuns.find((run) => run.status === "failed");
        toast(failed?.error_message || "部分评估运行失败");
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
      runBtn.textContent = "重新运行选中管线";
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
          text: ds.release_eligible
            ? `${ds.source_name} ${ds.source_version || ""} · ${ds.split || "未标注 split"} · ${ds.sample_count} 条公开标注样本 · 语料 ${String(ds.corpus_fingerprint || "").slice(0, 12)}`
            : `数据集 #${ds.id} · ${ds.sample_count} 条样本 · 非发布回归基准`,
        }),
      ]),
      runBtn,
    ]),
    picker,
    result,
  ]);
}

export async function refreshEval() {
  const root = $("#eval-content");
  try {
    const [datasets, pipelines] = await Promise.all([
      api.listDatasets(),
      api.listPipelines(),
    ]);
    if (!datasets.length) {
      root.replaceChildren(
        el("p", { class: "empty", text: "还没有评估数据集。可在后端生成或导入基准后在此运行。" })
      );
      return;
    }
    root.replaceChildren(...datasets.map((dataset) => datasetBlock(dataset, pipelines)));
  } catch (e) {
    toast(e.message);
  }
}
