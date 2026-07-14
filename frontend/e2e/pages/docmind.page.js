import { expect } from "@playwright/test";

export class DocMindPage {
  constructor(page) {
    this.page = page;
    this.authScreen = page.locator("#auth-screen");
    this.app = page.locator("#app");
    this.email = page.getByLabel("邮箱");
    this.password = page.getByLabel("密码");
    this.composer = page.locator("#composer-input");
    this.chatStream = page.locator("#chat-stream");
  }

  async goto() {
    await this.page.goto("/");
    await expect(this.authScreen).toBeVisible();
  }

  async login(email = "reader@example.com", password = "correct-horse") {
    await this.email.fill(email);
    await this.password.fill(password);
    const loginResponse = this.page.waitForResponse(
      (response) => response.url().endsWith("/auth/login") && response.status() === 200
    );
    await this.page.getByRole("button", { name: "登录", exact: true }).click();
    await loginResponse;
    await expect(this.app).toBeVisible();
    await expect(this.page.locator("#user-email")).toHaveText(email);
  }

  async register(email = "new-reader@example.com", password = "correct-horse") {
    await this.page.locator("#auth-switch").click();
    await expect(this.page.locator("#auth-submit")).toHaveText("注册");
    await this.email.fill(email);
    await this.password.fill(password);
    const registerResponse = this.page.waitForResponse(
      (response) => response.url().endsWith("/auth/register") && response.status() === 201
    );
    const loginResponse = this.page.waitForResponse(
      (response) => response.url().endsWith("/auth/login") && response.status() === 200
    );
    await this.page.locator("#auth-submit").click();
    await Promise.all([registerResponse, loginResponse]);
    await expect(this.app).toBeVisible();
    await expect(this.page.locator("#user-email")).toHaveText(email);
  }

  async openView(name) {
    await this.page.getByRole("button", { name, exact: true }).click();
    await expect(this.page.locator("#view-title")).toHaveText(name);
  }

  async ask(question) {
    await this.composer.fill(question);
    const response = this.page.waitForResponse(
      (candidate) => candidate.url().endsWith("/chat/stream") && candidate.status() === 200
    );
    await this.page.locator("#send-btn").click();
    await response;
    await expect(this.page.locator("#send-btn")).toBeEnabled();
  }

  async uploadMarkdown(name = "architecture.md") {
    const response = this.page.waitForResponse(
      (candidate) => candidate.url().endsWith("/documents/upload") && candidate.status() === 201
    );
    await this.page.locator("#file-input").setInputFiles({
      name,
      mimeType: "text/markdown",
      buffer: Buffer.from("# DocMind architecture\nHybrid retrieval and agent skills."),
    });
    await response;
  }
}
