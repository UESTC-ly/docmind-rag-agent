import { defineConfig, devices } from "@playwright/test";

const ci = Boolean(process.env.CI);
const port = Number(process.env.DOCMIND_E2E_FULLSTACK_PORT || 4180);
const baseURL = `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: "./e2e/fullstack",
  outputDir: "./test-results/fullstack",
  fullyParallel: false,
  forbidOnly: ci,
  retries: ci ? 2 : 0,
  workers: 1,
  timeout: 30_000,
  expect: { timeout: 5_000 },
  reporter: ci
    ? [
        ["line"],
        ["html", { outputFolder: "playwright-report-fullstack", open: "never" }],
        ["junit", { outputFile: "test-results/fullstack-junit.xml" }],
      ]
    : [
        ["list"],
        ["html", { outputFolder: "playwright-report-fullstack", open: "never" }],
      ],
  use: {
    baseURL,
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
    command: "node e2e/support/fullstack-server.mjs",
    url: `${baseURL}/health`,
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
