import { defineConfig } from "@playwright/test"

export default defineConfig({
  testDir: "./tests",
  outputDir: "./test-results",
  timeout: 90_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"], ["html", { outputFolder: "playwright-report", open: "never" }]],
  use: {
    baseURL: "http://127.0.0.1:4173",
    colorScheme: "dark",
    locale: "en-IN",
    screenshot: "on",
    video: "on",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "desktop", use: { viewport: { width: 1440, height: 900 }, contextOptions: { reducedMotion: "no-preference" } } },
    { name: "mobile", use: { viewport: { width: 390, height: 844 }, contextOptions: { reducedMotion: "no-preference" }, isMobile: true, hasTouch: true } },
    { name: "reduced-motion", use: { viewport: { width: 1440, height: 900 }, contextOptions: { reducedMotion: "reduce" } } },
    { name: "performance", use: { viewport: { width: 1440, height: 900 }, contextOptions: { reducedMotion: "no-preference" }, video: "off", screenshot: "only-on-failure" } },
  ],
  webServer: {
    command: "VITE_API_BASE_URL=http://127.0.0.1:8000 VITE_ACCEPT_FIXTURE_HEADER=true VITE_OUT_DIR=test-dist npm run build && npm run preview -- --host 127.0.0.1 --port 4173 --outDir test-dist",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: false,
    timeout: 120_000,
    stdout: "ignore",
    stderr: "pipe",
  },
})
