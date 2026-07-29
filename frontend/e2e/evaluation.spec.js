import { expect, test } from "@playwright/test";

import { DocMindPage } from "./pages/docmind.page.js";
import { installApiMocks } from "./support/api-mocks.js";

const dataset = {
  id: 31,
  name: "混合检索基准",
  document_id: 5,
  sample_count: 12,
  source_name: "CMRC 2018",
  source_version: "2018",
  split: "dev",
  corpus_fingerprint: "c".repeat(64),
  source_snapshot_fingerprint: "d".repeat(64),
  label_source: "public_ground_truth",
  release_eligible: true,
  created_at: "2026-07-13T08:00:00Z",
};

const baseRun = {
  id: 71,
  dataset_id: 31,
  created_at: "2026-07-13T08:01:00Z",
  completed_at: null,
};

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    globalThis.__DOCMIND_EVAL_POLL_MS__ = 20;
  });
});

test("异步评估从 pending、running 轮询到 completed", async ({ page }) => {
  const api = await installApiMocks(page, {
    evaluation: {
      datasets: [dataset],
      createdRun: { ...baseRun, status: "pending" },
      pollDelayMs: 80,
      pollResponses: [
        { ...baseRun, status: "running" },
        {
          ...baseRun,
          status: "completed",
          completed_at: "2026-07-13T08:02:00Z",
          hit_rate: 0.92,
          mrr: 0.81,
          recall: 0.88,
          precision: 0.73,
          faithfulness: 0.95,
          answer_relevancy: 0.9,
          map_score: 0.79,
          ndcg: 0.84,
        },
      ],
      metricRows: [
        {
          id: 1,
          run_id: 71,
          subject_type: "run",
          subject_id: 0,
          metric_name: "groundedness",
          metric_version: "claim_grounding_v1",
          evaluator_kind: "model",
          score: 0.86,
          passed: false,
          created_at: "2026-07-13T08:02:00Z",
        },
      ],
      regressions: [
        {
          id: 1,
          candidate_run_id: 71,
          gate_id: 3,
          metric_name: "groundedness",
          candidate_score: 0.86,
          passed: false,
          reason: "0.860000 >= 0.900000",
          created_at: "2026-07-13T08:02:00Z",
        },
      ],
      badcases: [
        {
          result_id: 9,
          sample_id: 4,
          external_id: "cmrc-dev-4",
          question: "该结论的依据是什么？",
          ground_truth_answer: "公开答案",
          answerable: true,
          categories: ["unsupported_generation"],
          diagnosis: [{
            category: "unsupported_generation",
            layer: "grounding",
            explanation: "至少一个结论没有被证据支持。",
            suggested_actions: ["补充检索"],
          }],
          metrics: { groundedness: 0 },
          relevant_chunk_ids: [2],
          retrieved_chunk_ids: [2],
          generated_answer: "无依据回答",
          retrieval_trace: [],
          source_links: [{
            citation_id: "D5:C2",
            document_id: 5,
            chunk_index: 2,
            source_status: "current",
            jump_url: "/documents/5/chunks/2",
          }],
        },
      ],
      createdRerun: {
        ...baseRun,
        id: 91,
        status: "completed",
        completed_at: "2026-07-13T08:03:00Z",
        evaluation_scope: "subset",
        sample_filter: "[4]",
        source_run_id: 71,
        comparison_role: "diagnostic",
      },
      rerunBadcases: [{
        result_id: 10,
        sample_id: 4,
        external_id: "cmrc-dev-4",
        question: "该结论的依据是什么？",
        ground_truth_answer: "公开答案",
        answerable: true,
        categories: [],
        diagnosis: [],
        metrics: { groundedness: 1 },
        relevant_chunk_ids: [2],
        retrieved_chunk_ids: [2],
        generated_answer: "有依据回答",
        retrieval_trace: [],
        source_links: [],
      }],
    },
  });
  const app = new DocMindPage(page);

  await app.goto();
  await app.login();
  await app.openView("EvalOps 实验室");
  await expect(page.getByRole("heading", { name: dataset.name })).toBeVisible();

  await page.getByRole("button", { name: "运行评估" }).click();
  await expect(page.locator(".eval-status[role=status]")).toContainText("已进入队列");
  await expect(page.getByText("正在逐样本执行检索、生成与评分…")).toBeVisible();
  await expect(page.locator(".eval-run > .metrics")).toBeVisible();
  await expect(page.locator(".metric", { hasText: "命中率" })).toContainText("0.920");
  await expect(page.getByText("Groundedness", { exact: true })).toBeVisible();
  await expect(page.getByText("回归门禁失败 1/1")).toBeVisible();
  await expect(page.getByText("Badcase 1 条")).toBeVisible();
  await page.getByText("cmrc-dev-4").click();
  await page.getByRole("button", { name: "[D5:C2] 核对原文" }).click();
  await expect(page.getByText("公开数据集中的可核验原文。")).toBeVisible();
  await page.getByRole("button", { name: "仅重跑此 Badcase" }).click();
  await expect(page.getByText("诊断重跑 #91 完成：该样本已通过")).toBeVisible();
  await expect(page.getByRole("button", { name: "重新运行" })).toBeEnabled();

  expect(api.requests.filter((request) => request.path === "/eval/runs/71"))
    .toHaveLength(2);
  expect(api.requests.find((request) => request.path === "/eval/runs/71/rerun")?.body)
    .toEqual({ sample_ids: [4], pipeline_id: null });
});

test("异步评估失败时停止轮询并展示原因", async ({ page }) => {
  const api = await installApiMocks(page, {
    evaluation: {
      datasets: [dataset],
      createdRun: { ...baseRun, status: "pending" },
      pollResponses: [
        { ...baseRun, status: "failed", error_message: "评分服务暂不可用" },
      ],
    },
  });
  const app = new DocMindPage(page);

  await app.goto();
  await app.login();
  await app.openView("EvalOps 实验室");
  await page.getByRole("button", { name: "运行评估" }).click();

  await expect(page.getByRole("alert")).toHaveText("评分服务暂不可用");
  await expect(page.getByRole("button", { name: "重新运行" })).toBeEnabled();
  expect(api.requests.filter((request) => request.path === "/eval/runs/71"))
    .toHaveLength(1);
});

test("可选择多条 RAG 管线并创建同数据集对比实验", async ({ page }) => {
  const completed = {
    ...baseRun,
    status: "completed",
    completed_at: "2026-07-13T08:02:00Z",
    hit_rate: 0.9,
    mrr: 0.8,
    recall: 0.85,
    precision: 0.7,
    faithfulness: 0.95,
    answer_relevancy: 0.9,
    citation_precision: 1,
    citation_recall: 0.8,
    unsupported_claim_rate: 0.2,
  };
  const api = await installApiMocks(page, {
    evaluation: {
      datasets: [dataset],
      createdRuns: [
        {
          ...completed,
          id: 81,
          pipeline_id: "configured",
          pipeline_fingerprint: "a".repeat(64),
          comparison_role: "baseline",
        },
        {
          ...completed,
          id: 82,
          pipeline_id: "dense",
          pipeline_fingerprint: "b".repeat(64),
          comparison_role: "candidate",
          baseline_run_id: 81,
        },
      ],
      badcaseDiff: {
        baseline_run_id: 81,
        candidate_run_id: 82,
        newly_introduced: [{
          sample_id: 7,
          external_id: "public-7",
          baseline_categories: [],
          candidate_categories: ["retrieval_miss"],
          introduced_categories: ["retrieval_miss"],
          fixed_categories: [],
        }],
        fixed: [],
        persistent: [],
        unchanged_passed_count: 11,
      },
    },
  });
  const app = new DocMindPage(page);

  await app.goto();
  await app.login();
  await app.openView("EvalOps 实验室");
  await page.getByLabel("Dense 基线").check();
  await page.getByRole("button", { name: "运行评估" }).click();

  await expect(page.locator(".eval-run")).toHaveCount(2);
  await expect(page.getByText("configured · aaaaaaaaaaaa")).toBeVisible();
  await expect(page.getByText("dense · bbbbbbbbbbbb")).toBeVisible();
  await expect(page.getByText("相对基线：新增 1 · 修复 0 · 持续 0")).toBeVisible();
  expect(api.requests.find((request) => request.path === "/eval/experiments")?.body)
    .toEqual({
      dataset_id: dataset.id,
      pipeline_ids: ["configured", "dense"],
    });
});
