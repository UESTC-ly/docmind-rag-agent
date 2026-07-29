import { expect, test } from "@playwright/test";

import { DocMindPage } from "./pages/docmind.page.js";
import { installApiMocks } from "./support/api-mocks.js";
import { materializeVisualBaseline } from "./support/visual-baseline.js";

test.describe("视觉回归", () => {
  test("登录卡片保持视觉基线", async ({ page }, testInfo) => {
    materializeVisualBaseline(testInfo, "auth-card.png");
    await installApiMocks(page);
    const app = new DocMindPage(page);

    await app.goto();
    await expect(page.locator(".auth__card")).toHaveScreenshot("auth-card.png");
  });

  test("登录后的 Agent 工作台保持视觉基线", async ({ page }, testInfo) => {
    materializeVisualBaseline(testInfo, "chat-shell.png");
    await installApiMocks(page);
    const app = new DocMindPage(page);

    await app.goto();
    await app.login();
    await expect(page.locator("#app")).toHaveScreenshot("chat-shell.png");
  });

  test("有意 CSS 变化会被视觉门禁拒绝", async ({ page }, testInfo) => {
    materializeVisualBaseline(testInfo, "auth-card.png");
    await installApiMocks(page);
    const app = new DocMindPage(page);

    await app.goto();
    await page.addStyleTag({
      content: ".auth__card { background: rgb(255, 0, 255) !important; }",
    });

    let mismatchDetected = false;
    try {
      await expect(page.locator(".auth__card")).toHaveScreenshot("auth-card.png");
    } catch {
      mismatchDetected = true;
    }
    expect(mismatchDetected).toBe(true);
  });
});
