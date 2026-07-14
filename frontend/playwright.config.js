import { defineConfig, devices } from "@playwright/test";

const ci = Boolean(process.env.CI);

export default defineConfig({
  testDir: "./e2e",
  testIgnore: "**/fullstack/**",
  outputDir: "./test-results",
  fullyParallel: true,
  forbidOnly: ci,
  retries: ci ? 2 : 0,
  workers: ci ? 1 : undefined,
  timeout: 30_000,
  expect: {
    timeout: 5_000,
    toHaveScreenshot: {
      animations: "disabled",
      caret: "hide",
      // v2.2 发布门禁固定在 macOS Chromium；仅保留 0.1% 抗锯齿余量。
      maxDiffPixelRatio: 0.001,
    },
  },
  reporter: ci
    ? [
        ["line"],
        ["html", { outputFolder: "playwright-report", open: "never" }],
        ["junit", { outputFile: "test-results/e2e-junit.xml" }],
      ]
    : [["list"], ["html", { outputFolder: "playwright-report", open: "never" }]],
  snapshotPathTemplate:
    "{testDir}/../test-results/visual-baselines/{projectName}-{platform}/{arg}{ext}",
  use: {
    baseURL: process.env.DOCMIND_E2E_BASE_URL || "http://127.0.0.1:4173",
    actionTimeout: 10_000,
    navigationTimeout: 15_000,
    locale: "zh-CN",
    timezoneId: "Asia/Shanghai",
    colorScheme: "light",
    reducedMotion: "reduce",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1440, height: 900 },
      },
    },
  ],
  webServer: {
    command: "node e2e/support/static-server.mjs",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: !ci,
    timeout: 30_000,
  },
});
