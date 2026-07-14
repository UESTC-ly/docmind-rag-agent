import { expect, test } from "@playwright/test";

import { DocMindPage } from "./pages/docmind.page.js";
import { installApiMocks } from "./support/api-mocks.js";

const dataset = {
  id: 31,
  name: "混合检索基准",
  document_id: 5,
  sample_count: 12,
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
        },
      ],
    },
  });
  const app = new DocMindPage(page);

  await app.goto();
  await app.login();
  await app.openView("评估");
  await expect(page.getByRole("heading", { name: dataset.name })).toBeVisible();

  await page.getByRole("button", { name: "运行评估" }).click();
  await expect(page.locator(".eval-status[role=status]")).toContainText("已进入队列");
  await expect(page.getByText("正在逐样本执行检索、生成与评分…")).toBeVisible();
  await expect(page.locator(".metrics")).toBeVisible();
  await expect(page.locator(".metric", { hasText: "命中率" })).toContainText("0.920");
  await expect(page.getByRole("button", { name: "重新运行" })).toBeEnabled();

  expect(api.requests.filter((request) => request.path === "/eval/runs/71"))
    .toHaveLength(2);
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
  await app.openView("评估");
  await page.getByRole("button", { name: "运行评估" }).click();

  await expect(page.getByRole("alert")).toHaveText("评分服务暂不可用");
  await expect(page.getByRole("button", { name: "重新运行" })).toBeEnabled();
  expect(api.requests.filter((request) => request.path === "/eval/runs/71"))
    .toHaveLength(1);
});
