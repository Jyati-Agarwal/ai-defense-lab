import { defineConfig } from "@playwright/test"

const baseURL = "http://127.0.0.1:8010"

export default defineConfig({
  testDir: "./tests-real",
  globalSetup: "./tests-real/global-setup.ts",
  outputDir: "./test-results-real",
  // The evidence test performs 24 screen-specific 36-frame motion samples plus axe,
  // keyboard traversal, live API interactions, and full-page capture at each viewport.
  timeout: 180_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"], ["html", { outputFolder: "playwright-report-real", open: "never" }]],
  use: {
    baseURL,
    colorScheme: "dark",
    locale: "en-IN",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "real-desktop", use: { viewport: { width: 1440, height: 900 } } },
    { name: "real-tablet", use: { viewport: { width: 768, height: 1024 }, hasTouch: true } },
    { name: "real-mobile", use: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } },
  ],
  webServer: {
    command: "npm run build && cd ../.. && .venv/bin/uvicorn app.backend.main:app --host 127.0.0.1 --port 8010",
    url: `${baseURL}/health`,
    reuseExistingServer: false,
    timeout: 120_000,
    stdout: "ignore",
    stderr: "pipe",
  },
})
