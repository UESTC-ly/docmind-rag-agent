import { expect, test } from "@playwright/test";

import { DocMindPage } from "../pages/docmind.page.js";

test("真实 FastAPI 支持健康检查、注册鉴权和登录态重载", async ({ page, request }) => {
  const health = await request.get("/health");
  expect(health.status()).toBe(200);
  expect(await health.json()).toEqual({ status: "ok" });

  const anonymousMe = await request.get("/auth/me");
  expect(anonymousMe.status()).toBe(401);

  const app = new DocMindPage(page);
  const email = `fullstack-${Date.now()}@example.com`;

  await app.goto();
  await app.register(email, "correct-horse");

  const token = await page.evaluate(() => localStorage.getItem("docmind_token"));
  expect(token).toMatch(/^eyJ/);
  const headers = { Authorization: `Bearer ${token}` };

  const me = await request.get("/auth/me", { headers });
  expect(me.status()).toBe(200);
  expect((await me.json()).email).toBe(email);

  const documents = await request.get("/documents/", { headers });
  expect(documents.status()).toBe(200);
  expect(await documents.json()).toEqual([]);

  const conversations = await request.get("/chat/conversations", { headers });
  expect(conversations.status()).toBe(200);
  expect(await conversations.json()).toEqual([]);

  const skills = await request.get("/agent/skills", { headers });
  expect(skills.status()).toBe(200);
  expect(await skills.json()).toEqual(
    expect.arrayContaining([
      expect.objectContaining({ name: "search_knowledge_base", available: true }),
    ]),
  );

  const reloadedMe = page.waitForResponse(
    (response) => response.url().endsWith("/auth/me") && response.status() === 200,
  );
  await page.reload();
  await reloadedMe;
  await expect(app.app).toBeVisible();
  await expect(app.authScreen).toBeHidden();
  await expect(page.locator("#user-email")).toHaveText(email);

  await app.openView("知识库");
  await expect(page.getByText("还没有文档，上传一个开始。")).toBeVisible();
});
