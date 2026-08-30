import { defineConfig } from "@playwright/test"

// Same suite as playwright.real.config.ts, but pointed at the deployed Vercel
// origin instead of a locally spawned server (no webServer block).
const baseURL = process.env.PROD_BASE_URL ?? "https://ai-defense-lab-lac.vercel.app"

export default defineConfig({
  testDir: "./tests-real",
  globalSetup: "./tests-real/global-setup.ts",
  outputDir: "./test-results-prod",
  timeout: 240_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"], ["html", { outputFolder: "playwright-report-prod", open: "never" }]],
  use: {
    baseURL,
    colorScheme: "dark",
    locale: "en-IN",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "prod-desktop", use: { viewport: { width: 1440, height: 900 } } },
    { name: "prod-mobile", use: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } },
  ],
})
