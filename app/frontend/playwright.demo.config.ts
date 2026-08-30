import { defineConfig } from "@playwright/test"

const baseURL = "http://127.0.0.1:8020"

export default defineConfig({
  testDir: "./tests-demo",
  outputDir: "./test-results-real/demo-video",
  timeout: 330_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: "list",
  use: {
    baseURL,
    colorScheme: "dark",
    locale: "en-IN",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    video: { mode: "on", size: { width: 1440, height: 900 } },
  },
  projects: [{
    name: "demo-chromium",
    use: {
      browserName: "chromium",
      viewport: { width: 1440, height: 900 },
      contextOptions: { reducedMotion: "no-preference" },
    },
  }],
  webServer: {
    command: "cd ../.. && .venv/bin/uvicorn app.backend.main:app --host 127.0.0.1 --port 8020",
    url: `${baseURL}/health`,
    reuseExistingServer: false,
    timeout: 120_000,
    stdout: "ignore",
    stderr: "pipe",
  },
})
