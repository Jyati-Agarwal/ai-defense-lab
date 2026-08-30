import { expect, test, type Page, type TestInfo } from "@playwright/test"
import AxeBuilder from "@axe-core/playwright"
import { composeFixture, fixtureResponses } from "../src/data/fixtures"
import type { ComposeInput } from "../src/types"

const endpointNames = [
  "/atlas",
  "/stream",
  "/metrics",
  "/generalization",
  "/arena",
  "/phylogeny",
  "/case/evt_flagship",
  "/precedent/evt_borderline",
  "/vaccination",
]

async function mockApi(page: Page, requested: Set<string>, failurePath?: string) {
  await page.route("http://127.0.0.1:8000/**", async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    requested.add(url.pathname)
    if (url.pathname === failurePath) {
      await route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: "test outage" }) })
      return
    }
    let body: unknown
    if (url.pathname === "/atlas") body = fixtureResponses.atlas
    else if (url.pathname === "/stream") body = fixtureResponses.stream
    else if (url.pathname === "/metrics") body = fixtureResponses.metrics
    else if (url.pathname === "/generalization") body = fixtureResponses.generalization
    else if (url.pathname === "/arena") body = fixtureResponses.arena
    else if (url.pathname === "/phylogeny") body = fixtureResponses.phylogeny
    else if (url.pathname.startsWith("/case/")) body = fixtureResponses.case
    else if (url.pathname.startsWith("/precedent/")) body = fixtureResponses.precedent
    else if (url.pathname === "/vaccination") body = fixtureResponses.vaccination
    else if (url.pathname === "/compose") body = composeFixture(request.postDataJSON() as ComposeInput)
    else return route.fulfill({ status: 404, body: "not found" })
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      headers: {
        "access-control-allow-origin": "*",
        "access-control-expose-headers": "x-ai-defense-fixture",
        "x-ai-defense-fixture": "1",
      },
      body: JSON.stringify(body),
    })
  })
}

function collectBrowserErrors(page: Page) {
  const errors: string[] = []
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text())
  })
  page.on("pageerror", (error) => errors.push(error.message))
  return errors
}

async function expectNoDocumentOverflow(page: Page) {
  const overflow = await page.evaluate(() => ({
    viewport: document.documentElement.clientWidth,
    document: document.documentElement.scrollWidth,
  }))
  expect(overflow.document, JSON.stringify(overflow)).toBeLessThanOrEqual(overflow.viewport + 1)
}

async function captureScreens(page: Page, testInfo: TestInfo) {
  await page.screenshot({ path: testInfo.outputPath("full-narrative.png"), fullPage: true, animations: "disabled" })
  for (const id of ["thesis", "atlas", "live-wire", "mandate", "injection", "precedent", "arena", "vault", "metrics", "economics", "composer", "unknowns"]) {
    const screen = page.locator(`#${id}`)
    await screen.scrollIntoViewIfNeeded()
    await page.waitForTimeout(500)
    await screen.screenshot({
      path: testInfo.outputPath(`screen-${id}.png`),
      animations: "disabled",
      style: "[data-testid=fixture-rail]{position:absolute!important}@media(min-width:768px){header{position:absolute!important}}",
    })
  }
}

test("@smoke twelve-screen narrative, APIs, interactions, accessibility controls, and frame timing", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name === "reduced-motion" || testInfo.project.name === "performance", "covered by a dedicated project")
  const requested = new Set<string>()
  const browserErrors = collectBrowserErrors(page)
  await mockApi(page, requested)

  await page.goto("/", { waitUntil: "domcontentloaded" })
  await expect(page.getByTestId("hero-title")).toContainText("The signature is valid. The intent was stolen.")
  if (testInfo.project.name === "desktop") {
    const heroLines = await page.getByTestId("hero-line").evaluateAll((lines) => lines.map((line) => {
      const box = line.getBoundingClientRect()
      return { height: box.height, lineHeight: Number.parseFloat(getComputedStyle(line).lineHeight) }
    }))
    expect(heroLines).toHaveLength(2)
    expect(heroLines.every(({ height, lineHeight }) => height <= lineHeight * 1.1), JSON.stringify(heroLines)).toBe(true)
  }
  await expect(page.locator("main > section")).toHaveCount(12)
  await expect(page.getByTestId("fixture-rail")).toBeVisible()
  await expect(page.getByText("Development fixture · not run evidence").first()).toBeVisible()

  await expect.poll(() => endpointNames.every((endpoint) => requested.has(endpoint))).toBe(true)
  await expectNoDocumentOverflow(page)

  await page.locator("#atlas").scrollIntoViewIfNeeded()
  const classicFilter = page.getByRole("button", { name: "classic", exact: true })
  await classicFilter.click()
  await expect(classicFilter).toHaveAttribute("aria-pressed", "true")
  await page.locator("#atlas-stage").selectOption({ index: 1 })
  await page.locator("#atlas-stage").selectOption("all")

  await page.getByRole("button", { name: "Pause transaction replay" }).click()
  await expect(page.getByRole("button", { name: "Resume transaction replay" })).toBeVisible()
  await page.getByRole("button", { name: /Synthetic Trail Supply/ }).click()

  const reveal = page.getByRole("switch", { name: "Reveal hidden injection" })
  await reveal.click()
  await expect(page.getByTestId("injection-layer")).toBeVisible()
  await reveal.press("Space")
  await expect(page.getByTestId("injection-layer")).toBeHidden()

  await page.locator("#vault").scrollIntoViewIfNeeded()
  const vault = page.getByTestId("vault-toggle")
  const vaultDoor = page.getByTestId("vault-door-panel")
  const doorStart = await vaultDoor.evaluate((element) => ({ opacity: getComputedStyle(element).opacity, transform: getComputedStyle(element).transform }))
  await page.screenshot({ path: testInfo.outputPath("vault-animation-start.png") })
  await vault.click()
  await page.waitForTimeout(150)
  const doorMid = await vaultDoor.evaluate((element) => ({ opacity: getComputedStyle(element).opacity, transform: getComputedStyle(element).transform }))
  await page.screenshot({ path: testInfo.outputPath("vault-animation-mid.png") })
  expect(doorMid, "Vault door must animate after the real toggle interaction").not.toEqual(doorStart)
  await expect(page.getByTestId("heldout-recall")).toHaveText("58%")
  await expect(page.getByTestId("evaluation-scope-disclosure")).toContainText("Fresh-row post-remediation regression · independent_holdout=false · prior families disclosed")
  await expect(page.getByTestId("evaluation-scope-disclosure")).toContainText("GPF-A20 / ttl_dormancy · GPF-C04 / synthetic_identity_bust_out")
  await expect(vault).toHaveAttribute("aria-expanded", "true")

  await page.locator("#metrics").scrollIntoViewIfNeeded()
  await expect(page.getByText(/A\+B benchmark · p50/)).toBeVisible()

  await page.locator("#compose-channel").selectOption("upi_collect")
  await page.locator("#compose-injection").selectOption("instruction_shift")
  const amountSlider = page.locator("#compose-amount [role=slider]")
  await amountSlider.focus()
  await amountSlider.press("ArrowRight")
  const presence = page.getByRole("switch", { name: "Lie about human presence" })
  await presence.focus()
  await presence.press("Space")
  await page.getByTestId("launch-composer").press("Enter")
  await expect(page.getByTestId("composer-result")).toBeVisible()
  await expect.poll(() => requested.has("/compose")).toBe(true)

  await expectNoDocumentOverflow(page)
  await captureScreens(page, testInfo)
  expect(browserErrors, browserErrors.join("\n")).toEqual([])
})

test("API errors fail closed instead of silently substituting fixture data", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "one browser is sufficient for loader semantics")
  const requested = new Set<string>()
  await mockApi(page, requested, "/atlas")
  await page.goto("/")
  const atlas = page.locator("#atlas")
  await expect(atlas.getByRole("alert")).toContainText("attack atlas unavailable")
  await expect(atlas.getByRole("alert")).toContainText("503 Service Unavailable")
  await expect(atlas.getByRole("button", { name: "agentic", exact: true })).toHaveCount(0)
})

test("cold flagship loading cannot displace an already-painted Atlas", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile", "mobile is the constrained layout-shift viewport")
  const requested = new Set<string>()
  await mockApi(page, requested)
  let releaseCase!: () => void
  const caseReady = new Promise<void>((resolve) => { releaseCase = resolve })
  await page.route("http://127.0.0.1:8000/case/evt_flagship", async (route) => {
    await caseReady
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      headers: {
        "access-control-allow-origin": "*",
        "access-control-expose-headers": "x-ai-defense-fixture",
        "x-ai-defense-fixture": "1",
      },
      body: JSON.stringify(fixtureResponses.case),
    })
  })

  await page.goto("/", { waitUntil: "domcontentloaded" })
  await expect(page.locator("#thesis")).toBeVisible()
  await expect(page.locator("#atlas")).toHaveCount(0)
  releaseCase()
  await expect(page.locator("main > section")).toHaveCount(12)
  await expect(page.locator("#atlas")).toBeVisible()
})

test("keyboard skip path and rendered UI meet automated WCAG checks", async ({ page }, testInfo) => {
  test.skip(!["desktop", "mobile"].includes(testInfo.project.name), "covered at representative pointer layouts")
  const requested = new Set<string>()
  await mockApi(page, requested)
  await page.goto("/")
  await expect(page.locator("main > section")).toHaveCount(12)
  const skipLink = page.getByRole("link", { name: "Skip to evidence" })
  await page.keyboard.press("Tab")
  await expect(skipLink).toBeFocused()
  await skipLink.press("Enter")
  await expect(page.locator("#main-content")).toBeFocused()
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"])
    .analyze()
  expect(results.violations, JSON.stringify(results.violations, null, 2)).toEqual([])
})

test("required stream values and vaccination statuses fail visibly", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "one browser is sufficient for normalization failures")
  const requested = new Set<string>()
  await mockApi(page, requested)
  const stream = structuredClone(fixtureResponses.stream) as unknown as { items: Array<Record<string, unknown>> }
  delete stream.items[0]?.amountMinor
  const vaccination = structuredClone(fixtureResponses.vaccination) as unknown as { stages: Array<Record<string, unknown>> }
  vaccination.stages[0]!.status = "unknown"
  await page.route("http://127.0.0.1:8000/stream?n=8", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(stream) }))
  await page.route("http://127.0.0.1:8000/vaccination", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(vaccination) }))

  await page.goto("/")
  await expect(page.locator("#live-wire").getByRole("alert")).toContainText("API artifact is missing stream event amount")
  await expect(page.locator("#arena").getByRole("alert")).toContainText("API artifact has invalid vaccination gate status: unknown")
})

test("break-even conclusion follows measured post-defense ROI", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "one browser is sufficient for evidence wording")
  const requested = new Set<string>()
  await mockApi(page, requested)
  const metrics = structuredClone(fixtureResponses.metrics)
  metrics.economics.after.roi = 0.25
  await page.route("http://127.0.0.1:8000/metrics", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(metrics) }))

  await page.goto("/")
  const economics = page.locator("#economics")
  await expect(economics).toContainText("Measured post-defense ROI: 0.25×.")
  await expect(economics).not.toContainText("below break-even")
  await expect(economics).not.toContainText("crosses below zero")
})

test("atlas and precedent enums accept only schema evidence", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "one browser is sufficient for enum validation")
  const requested = new Set<string>()
  await mockApi(page, requested)
  const atlasUrl = "http://127.0.0.1:8000/atlas"
  const precedentUrl = "http://127.0.0.1:8000/precedent/evt_borderline"

  const lowAtlas = structuredClone(fixtureResponses.atlas)
  lowAtlas.vectors[12]!.severity = "low"
  await page.route(atlasUrl, (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(lowAtlas) }))
  await page.goto("/")
  await expect(page.locator("#atlas").getByText("low", { exact: true })).toBeVisible()
  await page.unroute(atlasUrl)

  const badTier = structuredClone(fixtureResponses.atlas) as unknown as { vectors: Array<Record<string, unknown>> }
  badTier.vectors[0]!.tier = "unknown"
  await page.route(atlasUrl, (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(badTier) }))
  await page.reload()
  await expect(page.locator("#atlas").getByRole("alert")).toContainText("API artifact has invalid atlas tier: unknown")
  await page.unroute(atlasUrl)

  const badSeverity = structuredClone(fixtureResponses.atlas) as unknown as { vectors: Array<Record<string, unknown>> }
  badSeverity.vectors[0]!.severity = "unknown"
  await page.route(atlasUrl, (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(badSeverity) }))
  await page.reload()
  await expect(page.locator("#atlas").getByRole("alert")).toContainText("API artifact has invalid atlas severity: unknown")
  await page.unroute(atlasUrl)

  const badPrecedent = structuredClone(fixtureResponses.precedent) as unknown as Record<string, unknown>
  badPrecedent.verdict = "unknown"
  await page.route(precedentUrl, (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(badPrecedent) }))
  await page.reload()
  await expect(page.locator("#precedent").getByRole("alert")).toContainText("API artifact has invalid precedent verdict: unknown")
})

test("fixed navigation leaves anchored evidence unobscured", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "fixed navigation starts at the desktop breakpoint")
  const requested = new Set<string>()
  await mockApi(page, requested)
  await page.goto("/")
  await page.locator('header nav a[href="#arena"]').click()
  await expect(page).toHaveURL(/#arena$/)
  await expect.poll(() => page.locator("#arena").evaluate((target) => {
    const obstructionBottom = [...document.querySelectorAll("header, [data-testid=fixture-rail]")]
      .filter((element) => getComputedStyle(element).position === "fixed")
      .reduce((bottom, element) => Math.max(bottom, element.getBoundingClientRect().bottom), 0)
    return target.getBoundingClientRect().top - obstructionBottom
  })).toBeGreaterThanOrEqual(-1)
})

test("arena reveals only the six clearest survivors until requested", async ({ page }, testInfo) => {
  const requested = new Set<string>()
  const survivors = ["artifact-softening", "fanout-reduction", "graph-signal-reduction", "known-payee-reuse", "paced-velocity", "topic-blending", "artifact-softening+fanout-reduction", "paced-velocity+topic-blending"]
  await mockApi(page, requested)
  await page.route("http://127.0.0.1:8000/arena", (route) => route.fulfill({
    status: 200,
    contentType: "application/json",
    headers: { "access-control-allow-origin": "*" },
    body: JSON.stringify({ ...fixtureResponses.arena, survivors }),
  }))
  await page.goto("/")
  await expect(page.getByTestId("survivor-highlights").getByRole("listitem")).toHaveCount(6)
  const disclosure = page.getByTestId("additional-survivors")
  await expect(disclosure).not.toHaveAttribute("open", "")
  await page.locator('header nav a[href="#arena"]').click()
  await page.waitForTimeout(700)
  await page.screenshot({ path: testInfo.outputPath("arena-collapsed.png") })
  await disclosure.getByText("Show 2 additional survivors").press("Enter")
  await expect(disclosure).toHaveAttribute("open", "")
  await expect(disclosure.getByRole("listitem")).toHaveCount(2)
})

test("reduced motion preserves the complete story and interactions", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "reduced-motion", "reduced-motion project only")
  const requested = new Set<string>()
  const browserErrors = collectBrowserErrors(page)
  await mockApi(page, requested)
  await page.goto("/")
  await expect(page.getByTestId("hero-title")).toBeVisible()
  await expect(page.locator("main > section")).toHaveCount(12)
  expect(await page.evaluate(() => matchMedia("(prefers-reduced-motion: reduce)").matches)).toBe(true)
  const heroTransform = await page.locator(".hero-node").first().evaluate((element) => getComputedStyle(element).transform)
  await page.waitForTimeout(350)
  expect(await page.locator(".hero-node").first().evaluate((element) => getComputedStyle(element).transform)).toBe(heroTransform)
  await page.getByTestId("vault-toggle").click()
  await expect(page.getByTestId("vault-result")).toBeVisible()
  await expectNoDocumentOverflow(page)
  await page.screenshot({ path: testInfo.outputPath("reduced-motion.png"), fullPage: true, animations: "disabled" })
  expect(browserErrors, browserErrors.join("\n")).toEqual([])
})

test("basic scroll frame timing stays within the long-frame budget", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "performance", "performance project only")
  const requested = new Set<string>()
  const browserErrors = collectBrowserErrors(page)
  await mockApi(page, requested)
  await page.goto("/")
  await expect(page.getByText("Development fixture · not run evidence").first()).toBeVisible()
  await page.waitForTimeout(1_000)
  const frameSample = await page.evaluate(async () => {
    const deltas: number[] = []
    let previous = performance.now()
    for (let frame = 0; frame < 120; frame += 1) {
      await new Promise<void>((resolve) => requestAnimationFrame((time) => {
        deltas.push(time - previous)
        previous = time
        window.scrollBy(0, 10)
        resolve()
      }))
    }
    const usable = deltas.slice(8)
    return {
      averageMs: usable.reduce((sum, value) => sum + value, 0) / usable.length,
      longFrameRatio: usable.filter((delta) => delta > 34).length / usable.length,
      worstMs: Math.max(...usable),
    }
  })
  await testInfo.attach("frame-timing.json", { body: JSON.stringify(frameSample, null, 2), contentType: "application/json" })
  expect(frameSample.longFrameRatio, JSON.stringify(frameSample)).toBeLessThan(0.3)
  expect(browserErrors, browserErrors.join("\n")).toEqual([])
})
