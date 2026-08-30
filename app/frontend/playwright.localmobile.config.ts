import { defineConfig } from "@playwright/test"
// Diagnostic: identical suite, local backend, mobile only — isolates deployment
// network latency from app-side rendering cost.
const baseURL = "http://127.0.0.1:8010"
export default defineConfig({
  testDir: "./tests-real",
  globalSetup: "./tests-real/global-setup.ts",
  outputDir: "./test-results-localmobile",
  timeout: 240_000,
  expect: { timeout: 20_000 },
  fullyParallel: false, workers: 1, retries: 0,
  reporter: [["list"]],
  use: { baseURL, colorScheme: "dark", locale: "en-IN", screenshot: "off", trace: "off" },
  projects: [{ name: "local-mobile", use: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } }],
})
