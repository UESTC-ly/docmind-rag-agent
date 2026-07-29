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
    await expect(
      app.chatStream.locator(".msg__body", {
        hasText: "DocMind 会结合文档证据回答问题。",
      })
    ).toBeVisible();
    await expect(app.chatStream.getByText("[D5:C2] · 文档 5 · 片段 2 · 0.927")).toBeVisible();
    await expect(app.chatStream.getByRole("alert")).toContainText("引用结构校验未通过");
    await app.chatStream.getByRole("button", { name: /D5:C2.*片段 2/ }).click();
    await expect(app.chatStream.locator(".source__highlight")).toHaveText("公开数据集中的可核验原文。");
    await expect(app.chatStream.getByRole("button", { name: /D5:C2.*第 3 页.*第 4 段/ })).toBeVisible();
    expect(api.requests.find((request) => request.path === "/chat/stream")?.body.question)
      .toBe("DocMind 如何回答文档问题？");
  });

  test("上传文档后刷新为已完成状态", async ({ page }) => {
    await installApiMocks(page);
    const app = new DocMindPage(page);

    await app.goto();
    await app.login();
    await app.openView("知识库");
    await expect(page.getByText("还没有文档，上传一个开始。")).toBeVisible();

    await app.uploadMarkdown();

    const row = page.locator(".doc", { hasText: "architecture.md" });
    await expect(row).toBeVisible();
    await expect(row.getByText("已完成")).toBeVisible();
    await expect(page.locator("#toast")).toContainText("正在后台解析");
  });

  test("Agent 工作台展示计划、质量策略和可验证产物", async ({ page }) => {
    const api = await installApiMocks(page, {
      agentResult: {
        conversation_id: 24,
        run_id: "run-weekly",
        thread_id: "run-weekly",
        status: "completed",
        answer: "周报已生成。",
        plan: {
          contract: "agent_plan_v1",
          objective: "生成经过验证的项目周报",
          status: "completed",
          steps: [{
            id: "step-1",
            title: "生成周报",
            skill: "generate_weekly_report",
            success_criteria: "周报通过质量门",
            status: "completed",
          }],
        },
        trace: [{
          step: 0,
          plan_step_id: "step-1",
          skill: "search_knowledge_base",
          args: { query: "PRIVATE_QUERY" },
          grounding: {
            quality_interventions: [{
              contract: "quality_adaptive_retrieval_v1",
              action: "switch_pipeline",
              reason: "weak_retrieval",
              status: "applied",
              attempt: 2,
              max_interventions: 4,
              query: "PRIVATE_QUERY",
              source_text: "PRIVATE_SOURCE_TEXT",
              model_output: "PRIVATE_MODEL_OUTPUT",
              quality_before: {
                status: "weak_retrieval",
                hit_count: 0,
              },
              quality_after: {
                status: "sufficient",
                hit_count: 3,
              },
            }],
          },
        }],
        artifacts: [
          {
            type: "weekly_report",
            content: "# 本周进展\n\n- 完成混合检索联调",
            verification: {
              contract: "artifact_quality_v1",
              passed: true,
              score: 1,
              claim_count: 1,
              supported_claim_count: 1,
              citation_recall: 1,
              semantic_entailment_checked: false,
              checks: [{
                id: "citation_presence",
                passed: true,
                detail: "1/1 条内容有有效引用",
              }],
            },
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
    await app.openView("Agent 工作台");
    await page.getByText("可用能力", { exact: true }).click();
    const searchSkill = page.locator(".skill-card", {
      hasText: "search_knowledge_base",
    });
    await expect(searchSkill).toBeVisible();
    await searchSkill.getByRole("button", { name: "选择并填入模板" }).click();
    await expect(page.locator("#selected-skill-name")).toContainText("search_knowledge_base");

    await page.locator("#skill-agent-submit").click();
    await expect(page.locator(".agent-answer")).toContainText("周报已生成");
    await expect(page.locator(".agent-plan")).toContainText("生成经过验证的项目周报");
    await expect(page.locator(".trace__quality")).toContainText("切换已评测管线");
    await expect(page.locator(".trace__quality")).toContainText("弱检索");
    await expect(page.locator(".trace__quality")).toContainText("结果：sufficient · 3 条证据");
    await expect(page.locator("body")).not.toContainText("PRIVATE_QUERY");
    await expect(page.locator("body")).not.toContainText("PRIVATE_SOURCE_TEXT");
    await expect(page.locator("body")).not.toContainText("PRIVATE_MODEL_OUTPUT");
    await expect(page.locator(".artifact")).toContainText("weekly_report");
    await expect(page.locator(".artifact-verification")).toContainText("产物质量门通过");
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
    await app.openView("Agent 工作台");
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
    await app.openView("Agent 工作台");

    await expect(page.locator(".approval-card")).toContainText("browser_task");
    expect(api.requests.some((request) => request.path === "/agent/runs/run-after-reload"))
      .toBe(true);
  });

  test("另一个 worker 持有租约时保留 run ID 供稍后恢复", async ({ page }) => {
    await installApiMocks(page, {
      agentError: { status: 423, detail: "Agent run 正由另一个 worker 执行，请稍后重试" },
      // 首个 worker 可能尚未写下第一个 checkpoint，此刻 404 只是暂态。
      agentRunError: { status: 404, detail: "Agent run 不存在" },
    });
    const app = new DocMindPage(page);

    await app.goto();
    await app.login();
    await app.openView("Agent 工作台");
    await page.locator("#skill-agent-input").fill("执行并发任务");
    await page.locator("#skill-agent-submit").click();

    await expect(page.locator("#skill-agent-result")).toContainText("另一个 worker");
    const pendingRunId = await page.evaluate(() =>
      localStorage.getItem("docmind_pending_agent_run")
    );
    expect(pendingRunId).toBeTruthy();
  });

  test("模型服务不可用时显示终止失败而非伪造完成", async ({ page }) => {
    await installApiMocks(page, {
      agentResult: {
        conversation_id: 24,
        run_id: "run-provider-failed",
        thread_id: "run-provider-failed",
        status: "failed",
        answer: "模型服务暂不可用，本次任务未完成。请稍后重试。",
        plan: {
          contract: "agent_plan_v1",
          objective: "生成逐结论有依据的报告",
          status: "failed",
          steps: [{
            id: "step-1",
            title: "调用模型规划报告",
            status: "failed",
          }],
        },
        trace: [{
          step: 0,
          skill: "model_provider",
          ok: false,
          graph_node: "supervisor",
          failure_code: "provider_unavailable",
        }],
        artifacts: [],
      },
    });
    const app = new DocMindPage(page);

    await app.goto();
    await app.login();
    await app.openView("Agent 工作台");
    await page.locator("#skill-agent-input").fill("生成研究报告");
    await page.locator("#skill-agent-submit").click();

    await expect(page.locator(".agent-answer")).toContainText("模型服务暂不可用");
    await expect(page.locator(".agent-plan")).toContainText("失败");
    await expect(page.locator(".trace")).toContainText("model_provider");
    const pendingRunId = await page.evaluate(() =>
      localStorage.getItem("docmind_pending_agent_run")
    );
    expect(pendingRunId).toBeNull();
  });
});
