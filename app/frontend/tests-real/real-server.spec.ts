import { expect, test, type Page, type TestInfo } from "@playwright/test"
import AxeBuilder from "@axe-core/playwright"
import { mkdir, readFile, rename, writeFile } from "node:fs/promises"
import { resolve } from "node:path"

const screenIds = [
  "thesis",
  "atlas",
  "live-wire",
  "mandate",
  "injection",
  "precedent",
  "arena",
  "vault",
  "metrics",
  "economics",
  "composer",
  "unknowns",
] as const

const initialApiPaths = [
  "/atlas",
  "/stream",
  "/metrics",
  "/generalization",
  "/arena",
  "/phylogeny",
  "/case/evt_flagship",
  "/precedent/evt_borderline",
  "/vaccination",
] as const

const axeTags = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]
const incompleteNodeSampleLimit = 12
const transientLifecycleExclusion = {
  selector: ".react-flow__edges > *",
  reason: "React Flow replaces its rendered edge subtree for one frame during layout; the stable edge container, viewport, and app nodes remain sampled.",
} as const

type SeenResponse = { status: number; fixtureHeader: string | null; method: string }
type LayoutShiftRect = { x: number; y: number; width: number; height: number }
type LayoutShiftSourceEvidence = {
  target: string
  previousRect: LayoutShiftRect
  currentRect: LayoutShiftRect
}
type LayoutShiftEvidence = {
  startTime: number
  value: number
  sources: LayoutShiftSourceEvidence[]
}
type ClsSessionWindow = {
  startTime: number
  endTime: number
  value: number
  shifts: LayoutShiftEvidence[]
}
type QaPerformance = {
  lcpMs: number
  cls: number
  clsWindows: ClsSessionWindow[]
  recentInputShifts: LayoutShiftEvidence[]
}
type AxeAnalysis = Awaited<ReturnType<AxeBuilder["analyze"]>>
type SerializedAxe = ReturnType<typeof serializeAxe>
type MotionSample = Awaited<ReturnType<typeof measureScreenMotion>>
type KeyboardEvidence = Awaited<ReturnType<typeof measureKeyboardEvidence>>[string]
type ScreenEvidence = {
  accessibility: SerializedAxe
  touchTargetViolations: string[]
  documentWidths: Awaited<ReturnType<typeof measureScreenWidths>>
  capture: {
    sectionMode: "isolated-section"
    fixedHeaderVisibility: "hidden-during-section-capture"
    screenshot: string
    headerScreenshot: string
  }
  keyboard: KeyboardEvidence
  motion: {
    standard: MotionSample
    reduced?: MotionSample
    interaction?: Record<string, unknown>
  }
}
type MetricsArtifact = { ablation_table: Array<{ heldout_recall: number }> }
type CaseArtifact = {
  mandate_chain: {
    intent: { signature_valid: boolean }
    cart: { items: Array<{ name?: string; sku?: string }>; total_minor: number; payee_did: string; signature_valid: boolean }
    payment: { amount_minor: number; signature_valid: boolean; hash_matches_cart: boolean; agent_registered: boolean }
  }
}
type PrecedentArtifact = { verdict: "suppressed" | "confirmed" }
type VaccinationArtifact = { live_model_unchanged: boolean; stages: Array<{ name: string; status: string }> }

function collectBrowserFailures(page: Page) {
  const failures = { console: [] as string[], page: [] as string[], request: [] as string[] }
  page.on("console", (message) => {
    if (message.type() === "error") failures.console.push(message.text())
  })
  page.on("pageerror", (error) => failures.page.push(error.message))
  page.on("requestfailed", (request) => failures.request.push(`${request.method()} ${request.url()} · ${request.failure()?.errorText ?? "failed"}`))
  return failures
}

async function measureDocumentWidths(page: Page) {
  const widths = await page.evaluate(() => ({
    viewport: document.documentElement.clientWidth,
    document: document.documentElement.scrollWidth,
  }))
  return widths
}

function serializeAxe(results: AxeAnalysis, scope: string) {
  const serialize = (findings: AxeAnalysis["violations"], sampleNodes: boolean) => findings.map(({ id, impact, help, nodes }) => {
    const selectedNodes = sampleNodes ? nodes.slice(0, incompleteNodeSampleLimit) : nodes
    return {
      id,
      impact,
      help,
      nodeCount: nodes.length,
      nodesTruncated: selectedNodes.length < nodes.length,
      nodes: selectedNodes.map(({ target, html, failureSummary }) => ({ target, html, failureSummary })),
    }
  })
  return {
    scope,
    violations: serialize(results.violations, false),
    incomplete: serialize(results.incomplete, true),
  }
}

async function analyzeAccessibility(page: Page, include?: string) {
  let builder = new AxeBuilder({ page }).withTags(axeTags)
  if (include) builder = builder.include(include)
  return serializeAxe(await builder.analyze(), include ?? "document")
}

async function measureTouchTargets(page: Page, scope = "body") {
  return page.locator(scope).locator("a[href], button, input:not([type=hidden]), select, textarea, [role=switch], [role=slider]").evaluateAll((elements) => elements.flatMap((element) => {
    const style = getComputedStyle(element)
    const box = element.getBoundingClientRect()
    if (style.display === "none" || style.visibility === "hidden" || box.width === 0 || box.height === 0) return []
    if (box.width >= 43.5 && box.height >= 43.5) return []
    const label = element.getAttribute("aria-label") || element.textContent?.trim().replace(/\s+/g, " ").slice(0, 80) || element.tagName
    return [`${label}: ${box.width.toFixed(1)}×${box.height.toFixed(1)}`]
  }))
}

async function measureScreenWidths(page: Page, selector: string) {
  return page.locator(selector).evaluate((element) => {
    const describe = (candidate: Element) => {
      const testId = candidate.getAttribute("data-testid")
      if (testId) {
        const matches = [element, ...Array.from(element.querySelectorAll("[data-testid]"))]
          .filter((item) => item.getAttribute("data-testid") === testId)
        return `[data-testid=${JSON.stringify(testId)}]@${matches.indexOf(candidate) + 1}`
      }
      if (candidate.id) return `#${candidate.id}`
      const classes = Array.from(candidate.classList).slice(0, 2)
      return `${candidate.tagName.toLowerCase()}${classes.length ? `.${classes.join(".")}` : ""}`
    }
    const box = element.getBoundingClientRect()
    const visibleDescendants = [element, ...Array.from(element.querySelectorAll("*"))].filter((candidate) => {
      const style = getComputedStyle(candidate)
      const candidateBox = candidate.getBoundingClientRect()
      const isInvisibleRadixBubbleInput = candidate instanceof HTMLInputElement
        && candidate.getAttribute("aria-hidden") === "true"
        && candidate.tabIndex === -1
        && Number.parseFloat(style.opacity) === 0
        && style.position === "absolute"
        && style.pointerEvents === "none"
        && Boolean(candidate.parentElement?.querySelector(":scope > [role=slider], :scope > [role=switch]"))
      if (isInvisibleRadixBubbleInput) return false
      return style.display !== "none" && style.visibility !== "hidden" && candidateBox.width > 0 && candidateBox.height > 0
    })
    const left = Math.min(...visibleDescendants.map((candidate) => candidate.getBoundingClientRect().left))
    const right = Math.max(...visibleDescendants.map((candidate) => candidate.getBoundingClientRect().right))
    const viewport = document.documentElement.clientWidth
    const overflowCandidates = visibleDescendants.flatMap((candidate) => {
      const candidateBox = candidate.getBoundingClientRect()
      if (candidateBox.left >= -1 && candidateBox.right <= viewport + 1) return []
      let ancestor = candidate.parentElement
      let clippedBy: string | null = null
      while (ancestor) {
        const ancestorBox = ancestor.getBoundingClientRect()
        const overflowX = getComputedStyle(ancestor).overflowX
        if (["auto", "scroll", "hidden", "clip"].includes(overflowX)
          && (candidateBox.left < ancestorBox.left - 1 || candidateBox.right > ancestorBox.right + 1)) {
          clippedBy = describe(ancestor)
          break
        }
        ancestor = ancestor.parentElement
      }
      return [{ target: describe(candidate), left: candidateBox.left, right: candidateBox.right, clippedBy }]
    })
    const uncontainedOverflow = overflowCandidates.filter(({ clippedBy }) => clippedBy === null)
    return {
      viewport,
      sectionClientWidth: element.clientWidth,
      sectionScrollWidth: element.scrollWidth,
      contentLeft: left,
      contentRight: right,
      overflowsViewport: uncontainedOverflow.length > 0,
      overflowCandidateCount: overflowCandidates.length,
      overflowCandidatesTruncated: overflowCandidates.length > 12,
      overflowCandidates: overflowCandidates.slice(0, 12),
      uncontainedOverflow,
      box: { left: box.left, right: box.right, width: box.width },
    }
  })
}

async function measureScreenMotion(
  page: Page,
  selector: string,
  options: { sampleFrames?: number; revealOnStart?: boolean; trigger?: string } = {},
) {
  return page.locator(selector).evaluate(async (element, measurement) => {
    const resolveRoot = () => {
      const roots = document.querySelectorAll(measurement.selector)
      return roots.length === 1 ? roots.item(0) : null
    }
    const within = (root: Element) => [root, ...Array.from(root.querySelectorAll("*"))]
    const visibleNodes = [element, ...Array.from(element.querySelectorAll("*"))]
      .filter((candidate) => {
        const style = getComputedStyle(candidate)
        const box = candidate.getBoundingClientRect()
        return style.display !== "none"
          && style.visibility !== "hidden"
          && box.width > 0
          && box.height > 0
      })
    const transientLifecycleNodes = visibleNodes.filter((candidate) => {
      const edgeContainer = candidate.closest(".react-flow__edges")
      return edgeContainer !== null && candidate !== edgeContainer
    })
    const transientLifecycleNodeSet = new Set(transientLifecycleNodes)
    const candidateNodes = visibleNodes
      .filter((candidate) => {
        const hasExplicitKey = Boolean(candidate.id || candidate.getAttribute("data-testid") || candidate.getAttribute("data-id"))
        const hasStableIdentity = candidate === element || hasExplicitKey || candidate.classList.length > 0
        return hasStableIdentity && !transientLifecycleNodeSet.has(candidate)
      })
      .slice(0, 300)
    const makeCandidate = (
      candidate: Element,
      target: string,
      matchesIdentity: (item: Element) => boolean,
    ) => {
      const initialMatches = within(element).filter(matchesIdentity)
      const occurrence = initialMatches.indexOf(candidate)
      const expectedMatchCount = initialMatches.length
      let liveCandidate = candidate
      return {
        target,
        resolve: () => {
          if (liveCandidate.isConnected && element.isConnected && element.contains(liveCandidate)) return liveCandidate
          const root = resolveRoot()
          if (!root) return null
          const matches = within(root).filter(matchesIdentity)
          if (occurrence < 0 || matches.length !== expectedMatchCount) return null
          const replacement = matches[occurrence]
          if (!replacement?.isConnected) return null
          liveCandidate = replacement
          return replacement
        },
      }
    }
    const candidates = candidateNodes.map((candidate) => {
      if (candidate === element) {
        return { target: measurement.selector, resolve: resolveRoot }
      }
      const testId = candidate.getAttribute("data-testid")
      if (testId) {
        const matchesIdentity = (item: Element) => item.tagName === candidate.tagName
          && item.getAttribute("data-testid") === testId
        const occurrence = within(element).filter(matchesIdentity).indexOf(candidate) + 1
        return makeCandidate(candidate, `[data-testid=${JSON.stringify(testId)}]@${occurrence}`, matchesIdentity)
      }
      if (candidate.id) {
        const id = candidate.id
        return makeCandidate(candidate, `#${id}`, (item) => item.id === id)
      }
      const dataId = candidate.getAttribute("data-id")
      if (dataId) {
        const tagName = candidate.tagName
        const matchesIdentity = (item: Element) => item.tagName === tagName && item.getAttribute("data-id") === dataId
        const occurrence = within(element).filter(matchesIdentity).indexOf(candidate) + 1
        return makeCandidate(candidate, `[data-id=${JSON.stringify(dataId)}]@${occurrence}`, matchesIdentity)
      }
      const classNames = Array.from(candidate.classList).sort()
      if (classNames.length > 0) {
        const tagName = candidate.tagName
        const matchesIdentity = (item: Element) => item.tagName === tagName
          && item.classList.length === classNames.length
          && classNames.every((className) => item.classList.contains(className))
        const occurrence = within(element).filter(matchesIdentity).indexOf(candidate) + 1
        const target = `${tagName.toLowerCase()}.${classNames.join(".")}@${occurrence}`
        return makeCandidate(candidate, target, matchesIdentity)
      }
      throw new Error(`Visible motion candidate lacks a stable identity: ${candidate.tagName.toLowerCase()}`)
    })
    const resolveCandidate = (candidate: (typeof candidates)[number]) => {
      const live = candidate.resolve()
      if (!live?.isConnected) throw new Error(`Motion target could not be reacquired: ${candidate.target}`)
      return live
    }
    const snapshot = (record: (typeof candidates)[number]) => {
      const candidate = resolveCandidate(record)
      const style = getComputedStyle(candidate)
      const box = candidate.getBoundingClientRect()
      return {
        opacity: style.opacity,
        transform: style.transform,
        box: { x: box.x + scrollX, y: box.y + scrollY, width: box.width, height: box.height },
      }
    }
    const stateChanged = (first: ReturnType<typeof snapshot>, next: ReturnType<typeof snapshot>) => {
      const boxChanged = (Object.keys(first.box) as Array<keyof typeof first.box>)
        .some((key) => Math.abs(first.box[key] - next.box[key]) > 0.5)
      return first.opacity !== next.opacity || first.transform !== next.transform || boxChanged
    }
    const scrollStart = { x: scrollX, y: scrollY }
    const before = candidates.map(snapshot)
    const transitionElementCount = candidates.filter((candidate) => {
      const live = resolveCandidate(candidate)
      const style = getComputedStyle(live)
      const hasDuration = (value: string) => value.split(",").some((part) => Number.parseFloat(part) > 0)
      return hasDuration(style.transitionDuration) || hasDuration(style.animationDuration)
    }).length
    const sampleFrames = measurement.options.sampleFrames ?? 12
    const checkpointFrames = new Set([0, 1, 3, 6, 12, 24, sampleFrames].filter((frame) => frame <= sampleFrames))
    const timeline = [{ frame: 0, elapsedMs: 0, states: before }]
    let activeAnimationCount = candidates.filter((candidate) => resolveCandidate(candidate).getAnimations().some((animation) => animation.playState === "running")).length
    let registeredAnimationCount = new Set(candidates.flatMap((candidate) => resolveCandidate(candidate).getAnimations())).size
    const frameDeltas: number[] = []
    let previous = performance.now()
    const started = previous
    if (measurement.options.revealOnStart) {
      element.scrollIntoView({ block: "center", behavior: "instant" })
    }
    for (let frame = 1; frame <= sampleFrames; frame += 1) {
      await new Promise<void>((resolveFrame) => requestAnimationFrame((time) => {
        frameDeltas.push(time - previous)
        previous = time
        resolveFrame()
      }))
      activeAnimationCount = Math.max(
        activeAnimationCount,
        candidates.filter((candidate) => resolveCandidate(candidate).getAnimations().some((animation) => animation.playState === "running")).length,
      )
      registeredAnimationCount = Math.max(registeredAnimationCount, new Set(candidates.flatMap((candidate) => resolveCandidate(candidate).getAnimations())).size)
      if (checkpointFrames.has(frame)) timeline.push({ frame, elapsedMs: performance.now() - started, states: candidates.map(snapshot) })
    }
    const after = candidates.map(snapshot)
    const scrollEnd = { x: scrollX, y: scrollY }
    const changedIndexes = candidates.flatMap((_, index) => (
      timeline.some(({ states }) => stateChanged(before[index]!, states[index]!)) || stateChanged(before[index]!, after[index]!)
        ? [index]
        : []
    ))
    const changed = changedIndexes.map((index) => ({
      target: candidates[index]!.target,
      before: before[index]!,
      after: after[index]!,
    }))
    const trajectories = changedIndexes.slice(0, 12).map((index) => ({
      target: candidates[index]!.target,
      frames: timeline.map(({ frame, elapsedMs, states }) => ({ frame, elapsedMs, state: states[index]! })),
    }))
    return {
      selector: measurement.selector,
      trigger: measurement.options.trigger ?? "observation-window",
      mediaMatches: matchMedia("(prefers-reduced-motion: reduce)").matches,
      scrollStart,
      scrollEnd,
      sampledElements: candidates.length,
      transientLifecycleExclusions: {
        selector: measurement.transientLifecycleExclusion.selector,
        reason: measurement.transientLifecycleExclusion.reason,
        excludedElementCount: transientLifecycleNodes.length,
      },
      sampleFrames: frameDeltas.length,
      sampleDurationMs: performance.now() - started,
      averageFrameMs: frameDeltas.reduce((sum, value) => sum + value, 0) / frameDeltas.length,
      worstFrameMs: Math.max(...frameDeltas),
      transitionElementCount,
      transitionCapableElementCount: transitionElementCount,
      activeAnimationCount,
      registeredAnimationCount,
      changedElementCount: changed.length,
      changesTruncated: changed.length > 12,
      changes: changed.slice(0, 12),
      trajectoriesTruncated: changedIndexes.length > trajectories.length,
      trajectories,
    }
  }, { selector, options, transientLifecycleExclusion })
}

async function measureKeyboardEvidence(page: Page) {
  const prepared = await page.evaluate((ids) => {
    const bubbleInput = (candidate: Element) => {
      const style = getComputedStyle(candidate)
      return candidate instanceof HTMLInputElement
        && candidate.getAttribute("aria-hidden") === "true"
        && candidate.tabIndex === -1
        && Number.parseFloat(style.opacity) === 0
        && style.position === "absolute"
        && style.pointerEvents === "none"
        && Boolean(candidate.parentElement?.querySelector(":scope > [role=slider], :scope > [role=switch]"))
    }
    const candidates = Array.from(document.querySelectorAll("a[href], area[href], button, input:not([type=hidden]), select, textarea, summary, [contenteditable]:not([contenteditable=false]), [tabindex]"))
      .filter((candidate): candidate is HTMLElement | SVGElement => {
        if (!(candidate instanceof HTMLElement || candidate instanceof SVGElement)) return false
        if (candidate.matches(":disabled") || candidate.closest("[inert]") || candidate.tabIndex < 0 || bubbleInput(candidate)) return false
        const style = getComputedStyle(candidate)
        const box = candidate.getBoundingClientRect()
        const renderedSvgFocusable = candidate instanceof SVGElement && candidate.hasAttribute("tabindex")
        return style.display !== "none" && style.visibility !== "hidden"
          && (renderedSvgFocusable || (box.width > 0 && box.height > 0))
      })
    const expected: Record<string, number[]> = Object.fromEntries(["header", ...ids.map((id) => `#${id}`)].map((scope) => [scope, []]))
    candidates.forEach((candidate, index) => {
      candidate.dataset.qaKeyboardIndex = String(index)
      const section = candidate.closest("main > section")
      const scope = section?.id ? `#${section.id}` : candidate.closest("header") ? "header" : "document"
      if (scope in expected) expected[scope]!.push(index)
    })
    const bodyHadTabIndex = document.body.hasAttribute("tabindex")
    const bodyTabIndex = document.body.getAttribute("tabindex")
    document.body.tabIndex = -1
    document.body.focus()
    return { total: candidates.length, expected, bodyHadTabIndex, bodyTabIndex }
  }, [...screenIds])

  const steps: Array<{
    index: number
    target: string
    name: string
    role: string
    focusVisible: boolean
    hasVisibleFocusIndicator: boolean
  }> = []
  for (let index = 0; index < prepared.total; index += 1) {
    await page.keyboard.press("Tab")
    const step = await page.evaluate(() => {
      const active = document.activeElement as HTMLElement | null
      const style = active ? getComputedStyle(active) : null
      const outlineVisible = Boolean(style && style.outlineStyle !== "none" && Number.parseFloat(style.outlineWidth) > 0)
      const shadowVisible = Boolean(style && style.boxShadow !== "none" && style.boxShadow !== "rgba(0, 0, 0, 0) 0px 0px 0px 0px")
      const labelledText = (active?.getAttribute("aria-labelledby")?.trim().split(/\s+/) ?? [])
        .map((id) => document.getElementById(id)?.textContent?.trim())
        .filter(Boolean)
        .join(" ")
      const associatedLabel = active?.id
        ? document.querySelector(`label[for="${CSS.escape(active.id)}"]`)?.textContent?.trim()
        : undefined
      const label = active?.getAttribute("aria-label")?.trim()
        || labelledText
        || associatedLabel
        || active?.textContent?.trim().replace(/\s+/g, " ").slice(0, 120)
        || ""
      return {
        index: Number(active?.dataset.qaKeyboardIndex ?? -1),
        target: active?.getAttribute("data-testid") ? `[data-testid=${JSON.stringify(active.getAttribute("data-testid"))}]` : active?.id ? `#${active.id}` : active?.tagName.toLowerCase() ?? "unknown",
        name: label,
        role: active?.getAttribute("role") ?? active?.tagName.toLowerCase() ?? "unknown",
        focusVisible: active?.matches(":focus-visible") ?? false,
        hasVisibleFocusIndicator: outlineVisible || shadowVisible,
      }
    })
    steps.push(step)
  }

  const evidence: Record<string, {
    scope: string
    expectedFocusableCount: number
    visitedFocusableCount: number
    allExpectedVisited: boolean
    allFocusVisible: boolean
    allHaveFocusIndicator: boolean
    steps: typeof steps
  }> = {}
  for (const [scope, expectedIndexes] of Object.entries(prepared.expected)) {
    const scopedSteps = steps.filter((step) => expectedIndexes.includes(step.index))
    const visited = new Set(scopedSteps.map((step) => step.index))
    evidence[scope] = {
      scope,
      expectedFocusableCount: expectedIndexes.length,
      visitedFocusableCount: visited.size,
      allExpectedVisited: visited.size === expectedIndexes.length,
      allFocusVisible: scopedSteps.every((step) => step.focusVisible),
      allHaveFocusIndicator: scopedSteps.every((step) => step.hasVisibleFocusIndicator),
      steps: scopedSteps,
    }
  }
  await page.evaluate(({ bodyHadTabIndex, bodyTabIndex }) => {
    document.querySelectorAll("[data-qa-keyboard-index]").forEach((element) => element.removeAttribute("data-qa-keyboard-index"))
    if (bodyHadTabIndex && bodyTabIndex !== null) document.body.setAttribute("tabindex", bodyTabIndex)
    else document.body.removeAttribute("tabindex")
  }, prepared)
  return evidence
}

async function placeScreenBelowViewport(page: Page, selector: string) {
  await page.locator(selector).evaluate((element) => {
    const top = element.getBoundingClientRect().top + scrollY
    window.scrollTo({ top: Math.max(0, top - innerHeight - 80), behavior: "auto" })
  })
  await page.evaluate(() => new Promise<void>((resolveFrames) => requestAnimationFrame(() => requestAnimationFrame(() => resolveFrames()))))
}

async function sampleFrames(page: Page) {
  return page.evaluate(async () => {
    window.scrollTo(0, 0)
    const deltas: number[] = []
    let previous = performance.now()
    for (let frame = 0; frame < 90; frame += 1) {
      await new Promise<void>((resolveFrame) => requestAnimationFrame((time) => {
        deltas.push(time - previous)
        previous = time
        window.scrollBy(0, 8)
        resolveFrame()
      }))
    }
    const usable = deltas.slice(8)
    return {
      samples: usable.length,
      averageMs: usable.reduce((sum, value) => sum + value, 0) / usable.length,
      worstMs: Math.max(...usable),
      longFrameRatio: usable.filter((value) => value > 34).length / usable.length,
    }
  })
}

async function measurePagePerformance(page: Page) {
  return page.evaluate(() => {
    const navigation = performance.getEntriesByType("navigation")[0] as PerformanceNavigationTiming
    const resources = performance.getEntriesByType("resource") as PerformanceResourceTiming[]
    const paint = performance.getEntriesByName("first-contentful-paint")[0]
    const observed = (window as unknown as { __qaPerformance: QaPerformance }).__qaPerformance
    const transferred = (entry: PerformanceResourceTiming) => entry.transferSize || 0
    return {
      navigation: {
        ttfbMs: navigation.responseStart - navigation.requestStart,
        domContentLoadedMs: navigation.domContentLoadedEventEnd - navigation.startTime,
        loadMs: navigation.loadEventEnd - navigation.startTime,
      },
      fcpMs: paint?.startTime ?? 0,
      lcpMs: observed.lcpMs,
      cls: observed.cls,
      clsAttribution: {
        algorithm: "maximum-session-window",
        sessionGapLimitMs: 1_000,
        sessionDurationLimitMs: 5_000,
        windows: observed.clsWindows,
        recentInputShifts: observed.recentInputShifts,
      },
      requestCount: resources.length + 1,
      transferBytes: navigation.transferSize + resources.reduce((sum, entry) => sum + transferred(entry), 0),
      jsBytes: resources.filter((entry) => entry.name.includes(".js")).reduce((sum, entry) => sum + transferred(entry), 0),
      cssBytes: resources.filter((entry) => entry.name.includes(".css")).reduce((sum, entry) => sum + transferred(entry), 0),
    }
  })
}

async function measureNavKeyboardPath(page: Page) {
  const links = page.locator("header a:visible")
  const expectedHrefs = await links.evaluateAll((items) => items.map((item) => item.getAttribute("href")))
  const steps: Array<{ href: string | null; activeHref: string | null; withinHorizontalScroller: boolean }> = []
  for (const href of expectedHrefs) {
    await page.keyboard.press("Tab")
    const step = await page.evaluate(() => {
      const active = document.activeElement as HTMLElement | null
      const nav = active?.closest("nav")
      const activeBox = active?.getBoundingClientRect()
      const navBox = nav?.getBoundingClientRect()
      return {
        activeHref: active?.getAttribute("href") ?? null,
        withinHorizontalScroller: !navBox || !activeBox || (activeBox.left >= navBox.left - 1 && activeBox.right <= navBox.right + 1),
      }
    })
    steps.push({ href, ...step })
  }
  return steps
}

async function stageMeasuredEvidence(viewport: string, measurement: unknown) {
  const evidenceDirectory = resolve("jury-screens", ".evidence-staging")
  await mkdir(evidenceDirectory, { recursive: true })
  await writeFile(resolve(evidenceDirectory, `${viewport}.json`), `${JSON.stringify(measurement, null, 2)}\n`)
  if (viewport !== "mobile") return

  const labels = ["desktop", "tablet", "mobile"] as const
  const viewports: Record<string, unknown> = {}
  for (const label of labels) {
    try {
      viewports[label] = JSON.parse(await readFile(resolve(evidenceDirectory, `${label}.json`), "utf8")) as unknown
    } catch {
      return
    }
  }
  const finalPath = resolve("jury-screens", "evidence.json")
  const temporaryPath = resolve("jury-screens", ".evidence.json.tmp")
  await writeFile(temporaryPath, `${JSON.stringify({ viewports }, null, 2)}\n`)
  await rename(temporaryPath, finalPath)
}

test("real server serves all API-backed screens and performs live compose", async ({ page }, testInfo: TestInfo) => {
  const browserFailures = collectBrowserFailures(page)
  const apiResponses = new Map<string, SeenResponse>()
  await page.addInitScript(() => {
    type ObservedLayoutShift = PerformanceEntry & {
      hadRecentInput: boolean
      value: number
      sources?: Array<{
        node: Node | null
        previousRect: DOMRectReadOnly
        currentRect: DOMRectReadOnly
      }>
    }
    const sessionGapLimitMs = 1_000
    const sessionDurationLimitMs = 5_000
    const metrics: QaPerformance = { lcpMs: 0, cls: 0, clsWindows: [], recentInputShifts: [] }
    const targetFor = (node: Node | null) => {
      if (!(node instanceof Element)) return node?.nodeName.toLowerCase() ?? "unknown"
      const testId = node.getAttribute("data-testid")
      if (testId) return `[data-testid=${JSON.stringify(testId)}]`
      if (node.id) return `#${node.id}`
      const classes = Array.from(node.classList).slice(0, 2)
      return `${node.tagName.toLowerCase()}${classes.length > 0 ? `.${classes.join(".")}` : ""}`
    }
    const rectEvidence = (rect: DOMRectReadOnly): LayoutShiftRect => ({
      x: rect.x,
      y: rect.y,
      width: rect.width,
      height: rect.height,
    })
    const shiftEvidence = (entry: ObservedLayoutShift): LayoutShiftEvidence => ({
      startTime: entry.startTime,
      value: entry.value,
      sources: (entry.sources ?? []).map((source) => ({
        target: targetFor(source.node),
        previousRect: rectEvidence(source.previousRect),
        currentRect: rectEvidence(source.currentRect),
      })),
    })
    ;(window as unknown as { __qaPerformance: typeof metrics }).__qaPerformance = metrics
    new PerformanceObserver((list) => {
      const latest = list.getEntries().at(-1)
      if (latest) metrics.lcpMs = latest.startTime
    }).observe({ type: "largest-contentful-paint", buffered: true })
    new PerformanceObserver((list) => {
      const entries = list.getEntries() as ObservedLayoutShift[]
      for (const entry of entries) {
        const shift = shiftEvidence(entry)
        if (entry.hadRecentInput) {
          metrics.recentInputShifts.push(shift)
          continue
        }
        let session = metrics.clsWindows.at(-1)
        const startsNewSession = !session
          || entry.startTime - session.endTime > sessionGapLimitMs
          || entry.startTime - session.startTime > sessionDurationLimitMs
        if (startsNewSession) {
          session = {
            startTime: entry.startTime,
            endTime: entry.startTime,
            value: entry.value,
            shifts: [shift],
          }
          metrics.clsWindows.push(session)
        } else {
          session.endTime = entry.startTime
          session.value += entry.value
          session.shifts.push(shift)
        }
        metrics.cls = Math.max(metrics.cls, session.value)
      }
    }).observe({ type: "layout-shift", buffered: true })
  })
  page.on("response", (response) => {
    const url = new URL(response.url())
    if (initialApiPaths.includes(url.pathname as (typeof initialApiPaths)[number]) || url.pathname === "/compose") {
      apiResponses.set(url.pathname, {
        status: response.status(),
        fixtureHeader: response.headers()["x-ai-defense-fixture"] ?? null,
        method: response.request().method(),
      })
    }
  })

  await page.goto("/", { waitUntil: "domcontentloaded" })
  await page.waitForLoadState("networkidle")
  await expect(page.getByTestId("hero-title")).toContainText("The signature is valid. The intent was stolen.")
  await expect(page.locator("main > section")).toHaveCount(screenIds.length)
  await expect(page.getByRole("alert")).toHaveCount(0)
  await expect(page.getByTestId("fixture-rail")).toHaveCount(0)
  await expect(page.getByTestId("fixture-badge")).toHaveCount(0)
  await expect.poll(() => initialApiPaths.every((path) => apiResponses.has(path))).toBe(true)
  for (const path of initialApiPaths) {
    expect(apiResponses.get(path), path).toEqual({ status: 200, fixtureHeader: null, method: "GET" })
  }
  const skipLink = page.getByRole("link", { name: "Skip to evidence" })
  await page.keyboard.press("Tab")
  await expect(skipLink).toBeFocused()
  const skipLinkTarget = await skipLink.evaluate((element) => {
    const box = element.getBoundingClientRect()
    return { width: box.width, height: box.height }
  })
  expect(skipLinkTarget.width).toBeGreaterThanOrEqual(44)
  expect(skipLinkTarget.height).toBeGreaterThanOrEqual(44)
  const navKeyboardPath = await measureNavKeyboardPath(page)
  for (const step of navKeyboardPath) {
    expect(step.activeHref, JSON.stringify(step)).toBe(step.href)
    expect(step.withinHorizontalScroller, JSON.stringify(step)).toBe(true)
  }
  await skipLink.focus()
  await skipLink.press("Enter")
  await expect(page.locator("#main-content")).toBeFocused()
  const navLineCount = await page.locator("header nav li").evaluateAll((items) => new Set(items.map((item) => Math.round(item.getBoundingClientRect().top))).size)
  expect(navLineCount).toBe(1)
  await page.locator("header nav").evaluate((nav) => { nav.scrollLeft = 0 })

  const primaryCta = page.getByTestId("primary-cta")
  await primaryCta.scrollIntoViewIfNeeded()
  const primaryCtaEvidence = await primaryCta.evaluate((element) => {
    const box = element.getBoundingClientRect()
    const header = document.querySelector("header")
    const headerBox = header?.getBoundingClientRect()
    const headerPosition = header ? getComputedStyle(header).position : ""
    const obstructedByHeader = Boolean(headerBox && (headerPosition === "fixed" || headerPosition === "sticky") && box.top < headerBox.bottom && box.bottom > headerBox.top)
    const centerTarget = document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2)
    return {
      box: { top: box.top, right: box.right, bottom: box.bottom, left: box.left, width: box.width, height: box.height },
      headerPosition,
      obstructedByHeader,
      centerHitTarget: element === centerTarget || element.contains(centerTarget),
      fullyInViewport: box.top >= 0 && box.left >= 0 && box.bottom <= innerHeight && box.right <= innerWidth,
    }
  })
  expect(primaryCtaEvidence.obstructedByHeader, JSON.stringify(primaryCtaEvidence)).toBe(false)
  expect(primaryCtaEvidence.centerHitTarget, JSON.stringify(primaryCtaEvidence)).toBe(true)
  expect(primaryCtaEvidence.fullyInViewport, JSON.stringify(primaryCtaEvidence)).toBe(true)
  const artifacts = await page.evaluate(async () => {
    const paths = ["/metrics", "/case/evt_flagship", "/precedent/evt_borderline", "/vaccination"] as const
    const responses = await Promise.all(paths.map(async (path) => {
      const response = await fetch(path)
      if (!response.ok) throw new Error(`${path} ${response.status}`)
      return response.json()
    }))
    return { metrics: responses[0], caseData: responses[1], precedent: responses[2], vaccination: responses[3] }
  }) as { metrics: MetricsArtifact; caseData: CaseArtifact; precedent: PrecedentArtifact; vaccination: VaccinationArtifact }
  const metricsArtifact = artifacts.metrics

  const formatInrMinor = (value: number) => new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(value / 100)
  const caseChain = artifacts.caseData.mandate_chain
  const expectedCaseValues = {
    item: caseChain.cart.items[0]?.name ?? caseChain.cart.items[0]?.sku,
    amount: formatInrMinor(caseChain.payment.amount_minor),
    payee: caseChain.cart.payee_did,
    verified: [caseChain.intent.signature_valid, caseChain.cart.signature_valid, caseChain.payment.signature_valid, caseChain.payment.hash_matches_cart].filter(Boolean).length,
    failed: [caseChain.intent.signature_valid, caseChain.cart.signature_valid, caseChain.payment.signature_valid, caseChain.payment.hash_matches_cart].filter((value) => !value).length,
  }
  const mandate = page.locator("#mandate")
  await mandate.scrollIntoViewIfNeeded()
  if (expectedCaseValues.item) await expect(mandate.getByText(expectedCaseValues.item, { exact: false }).filter({ visible: true }).first()).toBeVisible()
  await expect(mandate.getByText(expectedCaseValues.amount, { exact: false }).filter({ visible: true }).first()).toBeVisible()
  await expect(mandate.getByText(expectedCaseValues.payee, { exact: false }).filter({ visible: true }).first()).toBeVisible()
  await expect(mandate.getByText("verified", { exact: true })).toHaveCount(expectedCaseValues.verified)
  await expect(mandate.getByText("failed", { exact: true })).toHaveCount(expectedCaseValues.failed)

  const vault = page.locator("#vault")
  await vault.scrollIntoViewIfNeeded()
  const vaultDoor = page.getByTestId("vault-door-panel")
  const vaultStyle = () => vaultDoor.evaluate((element) => ({ opacity: getComputedStyle(element).opacity, transform: getComputedStyle(element).transform }))
  const animationStart = await vaultStyle()
  await vault.screenshot({ path: testInfo.outputPath("vault-animation-start.png") })
  await page.getByTestId("vault-toggle").focus()
  await page.getByTestId("vault-toggle").press("Enter")
  await expect(page.getByTestId("vault-toggle")).toHaveAttribute("aria-expanded", "true")
  await expect(page.getByTestId("evaluation-scope-disclosure")).toContainText("Fresh-row post-remediation regression · independent_holdout=false · prior families disclosed")
  await page.waitForTimeout(150)
  const animationMid = await vaultStyle()
  await vault.screenshot({ path: testInfo.outputPath("vault-animation-mid.png") })
  await page.waitForTimeout(700)
  const animationEnd = await vaultStyle()
  expect(animationMid, "Vault door animation must advance after the real toggle interaction").not.toEqual(animationStart)
  expect(animationEnd, "Vault door animation must finish at a different state").not.toEqual(animationStart)
  await testInfo.attach("vault-animation-frames.json", { body: JSON.stringify({ animationStart, animationMid, animationEnd }, null, 2), contentType: "application/json" })

  const juryViewport = testInfo.project.name.replace(/^real-/, "")
  const screenEvidence = {} as Record<(typeof screenIds)[number], ScreenEvidence>
  await mkdir(resolve("jury-screens"), { recursive: true })

  const atlas = page.locator("#atlas")
  await atlas.scrollIntoViewIfNeeded()
  const classicAtlas = atlas.getByRole("button", { name: "classic", exact: true })
  const agenticAtlas = atlas.getByRole("button", { name: "agentic", exact: true })
  await classicAtlas.focus()
  await classicAtlas.press("Enter")
  await expect(page.getByTestId("atlas-selected")).toHaveAttribute("data-tier", "classic")
  await agenticAtlas.focus()
  await agenticAtlas.press("Enter")
  await expect(page.getByTestId("atlas-selected")).toHaveAttribute("data-tier", "agentic")

  const liveWire = page.locator("#live-wire")
  await liveWire.scrollIntoViewIfNeeded()
  const selectedEvent = page.getByTestId("selected-event-id")
  const firstEvent = await selectedEvent.textContent()
  await expect.poll(() => selectedEvent.textContent(), { timeout: 5_000 }).not.toBe(firstEvent)
  const pauseReplay = liveWire.getByRole("button", { name: "Pause transaction replay" })
  await pauseReplay.focus()
  await pauseReplay.press("Enter")
  const pausedEvent = await selectedEvent.textContent()
  await page.waitForTimeout(2_100)
  await expect(selectedEvent).toHaveText(pausedEvent ?? "")
  const resumeReplay = liveWire.getByRole("button", { name: "Resume transaction replay" })
  await resumeReplay.focus()
  await resumeReplay.press("Enter")
  await expect.poll(() => selectedEvent.textContent(), { timeout: 5_000 }).not.toBe(pausedEvent)
  const flagshipRow = liveWire.locator("button").filter({ hasText: "evt_flagship" })
  await expect(flagshipRow).toHaveCount(1)
  await flagshipRow.focus()
  await flagshipRow.press("Enter")
  await expect(selectedEvent).toHaveText("evt_flagship")
  await expect(liveWire.getByRole("button", { name: "Resume transaction replay" })).toBeVisible()
  const selectedTrace = liveWire.locator("aside")
  await expect(selectedTrace.getByText("No intent playback recorded in this event")).toHaveCount(0)
  await expect(selectedTrace.getByText("No structured cart recorded in this event")).toHaveCount(0)
  await expect(selectedTrace.locator(".policy-divergence")).toContainText("Legacy ALLOW. Intent Integrity BLOCK")

  await page.locator("#injection").scrollIntoViewIfNeeded()
  await expect(page.getByRole("button", { name: "Add to mandate" })).toBeDisabled()
  await expect(page.getByText("Read-only recorded evidence; this control cannot alter the mandate.")).toBeVisible()
  const injectionReveal = page.getByRole("switch", { name: "Reveal hidden injection" })
  const injectionRevealBefore = await injectionReveal.getAttribute("aria-checked")
  await injectionReveal.focus()
  await injectionReveal.press("Space")
  await expect(injectionReveal).toHaveAttribute("aria-checked", "true")
  await expect(page.getByTestId("injection-layer")).toBeVisible()
  const injectionRevealEvidence = {
    before: injectionRevealBefore,
    after: await injectionReveal.getAttribute("aria-checked"),
    layerVisible: await page.getByTestId("injection-layer").isVisible(),
  }
  await page.locator("#precedent").scrollIntoViewIfNeeded()
  await expect(page.getByText(artifacts.precedent.verdict === "suppressed" ? "False positive suppressed" : "Anomaly confirmed")).toBeVisible()
  await page.locator("#arena").scrollIntoViewIfNeeded()
  const humanBoundary = artifacts.vaccination.stages.find((stage) => /human promotion/i.test(stage.name))
  if (artifacts.vaccination.live_model_unchanged && humanBoundary && (humanBoundary.status === "promoted" || humanBoundary.status.includes("pending"))) {
    await expect(page.getByText("pending review", { exact: true })).toBeVisible()
    await expect(page.getByText("promoted", { exact: true })).toHaveCount(0)
  }
  await page.locator("#metrics").scrollIntoViewIfNeeded()
  await expect(page.getByText(/A\+B benchmark · p50/)).toBeVisible()
  const baselineRecall = metricsArtifact.ablation_table[0]?.heldout_recall
  const recallDeltas = metricsArtifact.ablation_table.slice(1).map((row) => (baselineRecall ?? row.heldout_recall) - row.heldout_recall)
  const hasRecallDrop = recallDeltas.some((delta) => delta > 0.000_001)
  await expect(page.getByText(hasRecallDrop ? "Ablation · measured fresh-row D recall drop" : "Ablation · fresh-row D recall comparison")).toBeVisible()
  await expect(page.getByTestId("ablation-delta").filter({ hasText: "no change" })).toHaveCount(recallDeltas.filter((delta) => Math.abs(delta) <= 0.000_001).length)

  await page.locator("#composer").scrollIntoViewIfNeeded()
  const composeChannel = page.locator("#compose-channel")
  const composeInjection = page.locator("#compose-injection")
  const channelBefore = await composeChannel.inputValue()
  const injectionBefore = await composeInjection.inputValue()
  await composeChannel.selectOption("upi_collect")
  const channelExercised = await composeChannel.inputValue()
  await composeChannel.selectOption("agentic_checkout")
  await composeInjection.selectOption("instruction_shift")
  const injectionExercised = await composeInjection.inputValue()
  await composeInjection.selectOption("hidden_text")
  const composerControlEvidence: Record<string, unknown> = {
    channel: { before: channelBefore, exercised: channelExercised, after: await composeChannel.inputValue() },
    injection: { before: injectionBefore, exercised: injectionExercised, after: await composeInjection.inputValue() },
  }
  const sliderControls = [
    ["compose-amount", "amount_minor", "Home"],
    ["compose-hops", "hops", "End"],
    ["compose-depth", "delegation_depth", "End"],
    ["compose-ttl", "ttl_delay_seconds", "End"],
    ["compose-fanout", "fanout", "End"],
    ["compose-evasion", "evasion_aggressiveness", "End"],
  ] as const
  const composerSliderPayload: Record<string, number> = {}
  for (const [id, field, key] of sliderControls) {
    const slider = page.locator(`#${id} [role=slider]`)
    const before = await slider.getAttribute("aria-valuenow")
    await slider.focus()
    await slider.press(key)
    const after = await slider.getAttribute("aria-valuenow")
    expect(after, `${id} must expose its measured value`).not.toBeNull()
    expect(after, `${id} must respond to keyboard input`).not.toBe(before)
    composerControlEvidence[id] = { before, key, after }
    const numericAfter = Number(after)
    expect(Number.isFinite(numericAfter), `${id} must expose a numeric value`).toBe(true)
    composerSliderPayload[field] = numericAfter
  }
  const presence = page.getByRole("switch", { name: "Lie about human presence" })
  const presenceBefore = await presence.getAttribute("aria-checked")
  await presence.focus()
  await presence.press("Space")
  await expect(presence).toHaveAttribute("aria-checked", "true")
  composerControlEvidence.presence = { before: presenceBefore, after: await presence.getAttribute("aria-checked") }
  const composeResponsePromise = page.waitForResponse((response) => new URL(response.url()).pathname === "/compose" && response.request().method() === "POST")
  const launchComposer = page.getByTestId("launch-composer")
  await launchComposer.focus()
  await launchComposer.press("Enter")
  const composeResponse = await composeResponsePromise
  expect(composeResponse.status()).toBe(200)
  expect(composeResponse.headers()["x-ai-defense-fixture"]).toBeUndefined()
  expect(composeResponse.request().postDataJSON()).toMatchObject({
    channel: "agentic_checkout",
    injection_style: "hidden_text",
    ...composerSliderPayload,
    lie_about_presence: true,
  })
  const composeBody = await composeResponse.json()
  expect(composeBody.event_id).toMatch(/^evt_composer_[a-f0-9]{12}$/)
  expect(composeBody.decision.reason_codes.length).toBeGreaterThan(0)
  expect(composeBody.decision.legacy_rules_action).toBe("ALLOW")
  expect(composeBody.decision.action).toBe("BLOCK")
  expect(composeBody.scores.fused).toBeGreaterThanOrEqual(0.85)
  await expect(page.getByTestId("composer-result")).toBeVisible()
  await expect(page.getByTestId("composer-result")).toContainText("Legacy rules")
  await expect(page.getByTestId("composer-result")).toContainText("ALLOW")
  await expect(page.getByTestId("composer-result")).toContainText("Intent integrity")
  await expect(page.getByTestId("composer-result")).toContainText("BLOCK")
  expect(apiResponses.get("/compose")).toEqual({ status: 200, fixtureHeader: null, method: "POST" })

  const standardMotionEvidence = {} as Record<(typeof screenIds)[number], MotionSample>
  const motionPage = await page.context().newPage()
  const standardMotionFailures = collectBrowserFailures(motionPage)
  await motionPage.goto("/", { waitUntil: "domcontentloaded" })
  await expect(motionPage.getByTestId("hero-title")).toBeVisible()
  standardMotionEvidence.thesis = await measureScreenMotion(motionPage, "#thesis", {
    sampleFrames: 36,
    trigger: "initial-page-entrance",
  })
  await motionPage.waitForLoadState("networkidle")
  for (const id of screenIds.slice(1)) {
    await placeScreenBelowViewport(motionPage, `#${id}`)
    standardMotionEvidence[id] = await measureScreenMotion(motionPage, `#${id}`, {
      sampleFrames: 36,
      revealOnStart: true,
      trigger: "programmatic-scroll-entry",
    })
  }
  const allStandardMotionFailures = [...standardMotionFailures.console, ...standardMotionFailures.page, ...standardMotionFailures.request]
  expect(allStandardMotionFailures, allStandardMotionFailures.join("\n")).toEqual([])
  await motionPage.close()

  const keyboardEvidence = await measureKeyboardEvidence(page)
  for (const [scope, evidence] of Object.entries(keyboardEvidence)) {
    expect(evidence.allExpectedVisited, `${scope} keyboard path missed a focusable control`).toBe(true)
    expect(evidence.allFocusVisible, `${scope} keyboard path did not use :focus-visible`).toBe(true)
    expect(evidence.allHaveFocusIndicator, `${scope} has a keyboard target without a visible focus indicator`).toBe(true)
  }

  const headerScreenshot = `${juryViewport}-header.png`
  await page.locator("header nav").evaluate((nav) => { nav.scrollLeft = 0 })
  await page.locator("header").screenshot({
    path: resolve("jury-screens", headerScreenshot),
    animations: "disabled",
  })
  const headerEvidence = {
    scope: "header",
    captureMode: "actual-global-header",
    screenshot: headerScreenshot,
    accessibility: await analyzeAccessibility(page, "header"),
    keyboard: keyboardEvidence.header!,
    widths: await measureScreenWidths(page, "header"),
  }
  expect(headerEvidence.accessibility.violations, "axe violations in global header").toEqual([])
  expect(headerEvidence.widths.overflowsViewport, "global header overflow").toBe(false)

  for (const id of screenIds) {
    const screen = page.locator(`#${id}`)
    await screen.scrollIntoViewIfNeeded()
    await expect(screen, `standard motion ${id}`).toBeVisible()
    const screenshot = `${juryViewport}-${id}.png`
    screenEvidence[id] = {
      accessibility: await analyzeAccessibility(page, `#${id}`),
      touchTargetViolations: await measureTouchTargets(page, `#${id}`),
      documentWidths: await measureScreenWidths(page, `#${id}`),
      capture: {
        sectionMode: "isolated-section",
        fixedHeaderVisibility: "hidden-during-section-capture",
        screenshot,
        headerScreenshot,
      },
      keyboard: keyboardEvidence[`#${id}`]!,
      motion: { standard: standardMotionEvidence[id] },
    }
    expect(screenEvidence[id].accessibility.violations, `axe violations in ${id}`).toEqual([])
    expect(screenEvidence[id].touchTargetViolations, `touch targets in ${id}`).toEqual([])
    expect(screenEvidence[id].documentWidths.overflowsViewport, `viewport overflow in ${id}`).toBe(false)
    await screen.screenshot({
      path: resolve("jury-screens", screenshot),
      animations: "disabled",
      style: "header { visibility: hidden !important; }",
    })
  }
  screenEvidence.atlas.motion.interaction = {
    activation: "keyboard Enter",
    exercisedState: "classic",
    capturedState: "agentic",
  }
  screenEvidence["live-wire"].motion.interaction = {
    activation: "keyboard Enter",
    replayExercised: ["pause", "resume"],
    capturedEventId: "evt_flagship",
    capturedStreamingState: "paused",
    policyDivergence: { legacy: "ALLOW", intentIntegrity: "BLOCK" },
  }
  screenEvidence.injection.motion.interaction = {
    activation: "keyboard Space",
    ...injectionRevealEvidence,
  }
  screenEvidence.vault.motion.interaction = {
    activation: "keyboard Enter",
    standard: { start: animationStart, mid: animationMid, end: animationEnd },
  }
  screenEvidence.composer.motion.interaction = {
    activation: "keyboard Enter",
    controls: composerControlEvidence,
    result: {
      legacy: composeBody.decision.legacy_rules_action,
      intentIntegrity: composeBody.decision.action,
      fused: composeBody.scores.fused,
    },
  }

  const reducedPage = await page.context().newPage()
  const reducedMotionFailures = collectBrowserFailures(reducedPage)
  await reducedPage.emulateMedia({ reducedMotion: "reduce" })
  await reducedPage.goto("/", { waitUntil: "domcontentloaded" })
  const reducedMotionMatches = await reducedPage.evaluate(() => matchMedia("(prefers-reduced-motion: reduce)").matches)
  const reducedHero = reducedPage.locator(".hero-node").first()
  await expect(reducedHero).toBeVisible()
  const reducedStart = await reducedHero.evaluate((element) => ({ opacity: getComputedStyle(element).opacity, transform: getComputedStyle(element).transform }))
  screenEvidence.thesis.motion.reduced = await measureScreenMotion(reducedPage, "#thesis", {
    sampleFrames: 36,
    trigger: "reduced-initial-page-entrance",
  })
  const reducedEnd = await reducedHero.evaluate((element) => ({ opacity: getComputedStyle(element).opacity, transform: getComputedStyle(element).transform }))
  await reducedPage.waitForLoadState("networkidle")
  for (const id of screenIds.slice(1)) {
    const reducedScreen = reducedPage.locator(`#${id}`)
    await placeScreenBelowViewport(reducedPage, `#${id}`)
    await expect(reducedScreen, `reduced motion ${id}`).toBeVisible()
    screenEvidence[id].motion.reduced = await measureScreenMotion(reducedPage, `#${id}`, {
      sampleFrames: 36,
      revealOnStart: true,
      trigger: "reduced-programmatic-scroll-entry",
    })
  }
  const reducedVault = reducedPage.locator("#vault")
  await reducedVault.scrollIntoViewIfNeeded()
  const reducedVaultDoor = reducedPage.getByTestId("vault-door-panel")
  const reducedVaultStyle = () => reducedVaultDoor.evaluate((element) => ({ opacity: getComputedStyle(element).opacity, transform: getComputedStyle(element).transform }))
  const reducedVaultBefore = await reducedVaultStyle()
  await reducedPage.getByTestId("vault-toggle").focus()
  await reducedPage.getByTestId("vault-toggle").press("Enter")
  await expect(reducedPage.getByTestId("vault-result")).toBeVisible()
  await reducedPage.evaluate(() => new Promise<void>((resolveFrames) => requestAnimationFrame(() => requestAnimationFrame(() => resolveFrames()))))
  const reducedVaultAfter = await reducedVaultStyle()
  await reducedPage.waitForTimeout(350)
  const reducedVaultSettled = await reducedVaultStyle()
  screenEvidence.vault.motion.interaction = {
    ...screenEvidence.vault.motion.interaction,
    reduced: { before: reducedVaultBefore, after: reducedVaultAfter, settled: reducedVaultSettled },
  }
  const reducedDocumentWidths = await measureDocumentWidths(reducedPage)
  expect(reducedMotionMatches).toBe(true)
  expect(reducedEnd, "Reduced-motion media must suppress the hero reveal animation").toEqual(reducedStart)
  expect(reducedVaultSettled, "Reduced motion must not leave the vault transition running").toEqual(reducedVaultAfter)
  const movingReducedScreens = screenIds.filter((id) => screenEvidence[id].motion.reduced?.changedElementCount !== 0)
  await testInfo.attach(`reduced-motion-${juryViewport}.json`, {
    body: JSON.stringify(Object.fromEntries(screenIds.map((id) => [id, screenEvidence[id].motion.reduced])), null, 2),
    contentType: "application/json",
  })
  expect(movingReducedScreens, `Reduced-motion screens changed during sampling: ${movingReducedScreens.join(", ")}`).toEqual([])
  expect(reducedDocumentWidths.document, JSON.stringify(reducedDocumentWidths)).toBeLessThanOrEqual(reducedDocumentWidths.viewport + 1)
  const allReducedMotionFailures = [...reducedMotionFailures.console, ...reducedMotionFailures.page, ...reducedMotionFailures.request]
  expect(allReducedMotionFailures, allReducedMotionFailures.join("\n")).toEqual([])
  await reducedPage.close()

  const visibleStringAudit = await page.locator("body").evaluate((body) => {
    const text = (body as HTMLElement).innerText
    return {
      unicodeDashLines: text.split("\n").map((line) => line.trim()).filter((line) => /[\u2013\u2014]/.test(line)),
      unsupportedFlagshipLines: text.split("\n").map((line) => line.trim()).filter((line) => /₹\s?78(?:,?000|k)\b/i.test(line)),
    }
  })
  expect(visibleStringAudit.unicodeDashLines, visibleStringAudit.unicodeDashLines.join("\n")).toEqual([])
  expect(visibleStringAudit.unsupportedFlagshipLines, visibleStringAudit.unsupportedFlagshipLines.join("\n")).toEqual([])

  const accessibility = await analyzeAccessibility(page)
  expect(accessibility.violations, JSON.stringify(accessibility.violations, null, 2)).toEqual([])

  const documentWidths = await measureDocumentWidths(page)
  const touchTargetViolations = await measureTouchTargets(page)
  const frameTiming = await sampleFrames(page)
  const pagePerformance = await measurePagePerformance(page)
  await expect(page.getByTestId("fixture-rail")).toHaveCount(0)
  await expect(page.getByTestId("fixture-badge")).toHaveCount(0)
  await page.screenshot({ path: testInfo.outputPath("real-full-page.png"), fullPage: true, animations: "disabled" })
  const viewport = await page.evaluate(() => ({ width: innerWidth, height: innerHeight, devicePixelRatio }))
  const viewportLabel = testInfo.project.name.replace(/^real-/, "")
  const measurement = {
    viewport,
    screenCount: await page.locator("main > section").count(),
    capturedScreens: screenIds,
    apiResponses: Object.fromEntries(apiResponses),
    fixtureRailCount: await page.getByTestId("fixture-rail").count(),
    browserFailures,
    documentWidths,
    skipLinkTarget,
    navigation: { lineCount: navLineCount, keyboardPath: navKeyboardPath },
    primaryCta: primaryCtaEvidence,
    touchTargetViolations,
    accessibility,
    screenEvidence,
    headerEvidence,
    visibleStringAudit,
    reducedMotion: {
      mediaMatches: reducedMotionMatches,
      heroStart: reducedStart,
      heroEnd: reducedEnd,
      browserFailures: reducedMotionFailures,
      documentWidths: reducedDocumentWidths,
    },
    animationFrames: { start: animationStart, mid: animationMid, end: animationEnd },
    frameTiming,
    pagePerformance,
    ablations: metricsArtifact.ablation_table,
    normalizedCaseEvidence: expectedCaseValues,
    vaccination: { liveModelUnchanged: artifacts.vaccination.live_model_unchanged, humanBoundary },
    composeEventId: composeBody.event_id,
    composerControls: composerControlEvidence,
    injectionReveal: injectionRevealEvidence,
  }

  expect(documentWidths.document, JSON.stringify(documentWidths)).toBeLessThanOrEqual(documentWidths.viewport + 1)
  expect(touchTargetViolations, touchTargetViolations.join("\n")).toEqual([])
  expect(frameTiming.longFrameRatio, JSON.stringify(frameTiming)).toBeLessThan(0.3)
  expect(pagePerformance.fcpMs, JSON.stringify(pagePerformance)).toBeGreaterThan(0)
  expect(pagePerformance.lcpMs, JSON.stringify(pagePerformance)).toBeGreaterThan(0)
  expect(pagePerformance.lcpMs, JSON.stringify(pagePerformance)).toBeLessThan(2_500)
  expect(pagePerformance.cls, JSON.stringify(pagePerformance)).toBeLessThan(0.1)
  const allBrowserFailures = [...browserFailures.console, ...browserFailures.page, ...browserFailures.request]
  expect(allBrowserFailures, allBrowserFailures.join("\n")).toEqual([])
  await stageMeasuredEvidence(viewportLabel, measurement)
})
