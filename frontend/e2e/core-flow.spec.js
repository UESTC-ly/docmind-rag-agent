import { expect, test } from "@playwright/test";

import { DocMindPage } from "./pages/docmind.page.js";
import { installApiMocks } from "./support/api-mocks.js";

test.describe("核心用户流程", () => {
  test("注册后自动登录并进入工作台", async ({ page }) => {
    const api = await installApiMocks(page);
    const app = new DocMindPage(page);

    await app.goto();
    await app.register("new-reader@example.com", "correct-horse");

    expect(api.requests.filter((request) => request.path === "/auth/register"))
      .toHaveLength(1);
    expect(api.requests.find((request) => request.path === "/auth/login")?.body)
      .toEqual({ email: "new-reader@example.com", password: "correct-horse" });
  });

  test("登录后可进行带来源的流式文档问答", async ({ page }) => {
    const api = await installApiMocks(page);
    const app = new DocMindPage(page);

    await app.goto();
    await app.login();
    await app.ask("DocMind 如何回答文档问题？");

    await expect(app.chatStream.getByText("DocMind 如何回答文档问题？")).toBeVisible();
    await expect(app.chatStream.getByText("DocMind 会结合文档证据回答问题。")).toBeVisible();
    await expect(app.chatStream.getByText("文档 5 · 片段 2 · 0.927")).toBeVisible();
    expect(api.requests.find((request) => request.path === "/chat/stream")?.body.question)
      .toBe("DocMind 如何回答文档问题？");
  });

  test("上传文档后刷新为已完成状态", async ({ page }) => {
    await installApiMocks(page);
    const app = new DocMindPage(page);

    await app.goto();
    await app.login();
    await app.openView("文档");
    await expect(page.getByText("还没有文档，上传一个开始。")).toBeVisible();

    await app.uploadMarkdown();

    const row = page.locator(".doc", { hasText: "architecture.md" });
    await expect(row).toBeVisible();
    await expect(row.getByText("已完成")).toBeVisible();
    await expect(page.locator("#toast")).toContainText("正在后台解析");
  });

  test("Skills 执行后展示 artifact 并触发浏览器下载", async ({ page }) => {
    const api = await installApiMocks(page, {
      agentResult: {
        conversation_id: 24,
        answer: "周报已生成。",
        trace: [{ step: 0, skill: "search_knowledge_base", args: { query: "本周进展" } }],
        artifacts: [
          {
            type: "weekly_report",
            content: "# 本周进展\n\n- 完成混合检索联调",
            download: {
              filename: "docmind-weekly.md",
              mime_type: "text/markdown",
              encoding: "text",
              content: "# 本周进展\n\n- 完成混合检索联调",
            },
          },
        ],
      },
    });
    const app = new DocMindPage(page);

    await app.goto();
    await app.login();
    await app.openView("技能");
    await expect(page.locator(".skill-card", { hasText: "search_knowledge_base" })).toBeVisible();
    await page.getByRole("button", { name: "选择并填入模板" }).click();
    await expect(page.locator("#selected-skill-name")).toContainText("search_knowledge_base");

    await page.locator("#skill-agent-submit").click();
    await expect(page.locator(".agent-answer")).toContainText("周报已生成");
    await expect(page.locator(".artifact")).toContainText("weekly_report");
    await expect(page.locator(".artifact-download__name")).toHaveText("docmind-weekly.md");

    const downloadEvent = page.waitForEvent("download");
    await page.getByRole("button", { name: "下载产出" }).click();
    const download = await downloadEvent;
    expect(download.suggestedFilename()).toBe("docmind-weekly.md");
    expect(api.requests.find((request) => request.path === "/agent/chat")?.body.skill_name)
      .toBe("search_knowledge_base");
  });
});
