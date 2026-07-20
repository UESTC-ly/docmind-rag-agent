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
        run_id: "run-weekly",
        thread_id: "run-weekly",
        status: "completed",
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

  test("高风险 Skill 必须人工批准后才继续", async ({ page }) => {
    const api = await installApiMocks(page, {
      agentResult: {
        conversation_id: 24,
        run_id: "run-waiting-approval",
        thread_id: "run-waiting-approval",
        status: "waiting_approval",
        answer: "检测到高风险工具调用，等待人工审批后继续。",
        trace: [],
        artifacts: [],
        approval: {
          kind: "tool_approval",
          scope: "outer_skill",
          tool: "repository_update",
          args: { path: "README.md" },
          reason: "该工具会修改仓库文件。",
          risk_capabilities: ["repository"],
        },
      },
      resumeResult: {
        conversation_id: 24,
        run_id: "run-waiting-approval",
        thread_id: "run-waiting-approval",
        status: "completed",
        answer: "审批后已完成仓库更新。",
        trace: [{
          step: 0,
          skill: "repository_update",
          args: { path: "README.md" },
          approval: { required: true, approved: true },
        }],
        artifacts: [],
        approval: null,
      },
    });
    const app = new DocMindPage(page);

    await app.goto();
    await app.login();
    await app.openView("技能");
    await page.locator("#skill-agent-input").fill("修改仓库说明");
    await page.locator("#skill-agent-submit").click();

    const approval = page.locator(".approval-card");
    await expect(approval).toContainText("repository_update");
    await expect(approval).toContainText("本次外层 Skill 调用");
    await approval.getByRole("button", { name: "批准并继续" }).click();
    await expect(page.locator(".agent-answer")).toContainText("审批后已完成仓库更新");
    expect(api.requests.find((request) => request.path === "/agent/resume")?.body)
      .toEqual({
        run_id: "run-waiting-approval",
        approved: true,
        comment: null,
        edited_args: null,
      });
  });

  test("刷新页面后可从 checkpoint 找回待审批 run", async ({ page }) => {
    const waiting = {
      conversation_id: 24,
      run_id: "run-after-reload",
      thread_id: "run-after-reload",
      status: "waiting_approval",
      answer: "等待恢复。",
      trace: [],
      artifacts: [],
      approval: {
        kind: "tool_approval",
        scope: "outer_skill",
        tool: "browser_task",
        args: { url: "https://example.com" },
        reason: "需要浏览器能力。",
        risk_capabilities: ["browser"],
      },
    };
    const api = await installApiMocks(page, { agentRunResult: waiting });
    const app = new DocMindPage(page);

    await app.goto();
    await app.login();
    await page.evaluate(() => {
      localStorage.setItem("docmind_pending_agent_run", "run-after-reload");
    });
    await page.reload();
    await expect(app.app).toBeVisible();
    await app.openView("技能");

    await expect(page.locator(".approval-card")).toContainText("browser_task");
    expect(api.requests.some((request) => request.path === "/agent/runs/run-after-reload"))
      .toBe(true);
  });
});
