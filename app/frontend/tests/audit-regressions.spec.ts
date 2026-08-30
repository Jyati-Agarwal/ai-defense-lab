import { readFileSync } from "node:fs"
import { expect, test } from "@playwright/test"

const appSource = readFileSync(new URL("../src/App.tsx", import.meta.url), "utf8")

test("run evidence drives emergent, economics, limitation, and hero states", () => {
  expect(appSource).toContain("phylogeny.nodes.filter((node) => node.emergent).length")
  expect(appSource).not.toContain("1 emergent")
  expect(appSource).toContain("No observed successes; censored estimate omitted")
  expect(appSource).toContain('["Observed successes"')
  expect(appSource).not.toContain("What still gets through.")
  expect(appSource).not.toContain(">evades<")
  expect(appSource).not.toContain("Loading case evidence")
  expect(appSource).not.toContain("Loading registry evidence")
  expect(appSource).not.toContain("Loading retrieval evidence")
  expect(appSource).not.toContain("Loading payment evidence")
  expect(appSource).toContain('state.status === "error"')
  expect(appSource).toContain('title="Flagship case unavailable"')
})

test("evaluation copy discloses the post-remediation scope without overclaiming", () => {
  expect(appSource).toContain("Fresh-row post-remediation regression · independent_holdout=false · prior families disclosed")
  expect(appSource).toContain("GPF-A20 / ttl_dormancy · GPF-C04 / synthetic_identity_bust_out")
  expect(appSource).not.toContain("Can it catch what it never saw?")
  expect(appSource).not.toContain("Score unseen evidence once.")
  expect(appSource).not.toContain("Residual generalization gap")
  expect(appSource).not.toContain("pristine holdout")
  expect(appSource).toContain("A+B benchmark · p50")
})

test("every non-hero section label is assigned to one shared heading", () => {
  expect(appSource).toContain('<h2 id={id} className="section-title">')
  const sectionLabels = [...appSource.matchAll(/<section id="(?!thesis)[^"]+"[^>]+aria-labelledby="([^"]+)"/g)].map((match) => match[1])
  const headingIds = [...appSource.matchAll(/<SectionHeader\s+(?:\n\s*)?id="([^"]+)"/g)].map((match) => match[1])
  expect(sectionLabels).toHaveLength(11)
  expect(new Set(headingIds).size).toBe(11)
  expect(headingIds.sort()).toEqual(sectionLabels.sort())
  for (const id of sectionLabels) {
    const sectionStart = appSource.indexOf(`aria-labelledby="${id}"`)
    const sectionEnd = appSource.indexOf("\nfunction ", sectionStart)
    const screen = appSource.slice(sectionStart, sectionEnd < 0 ? undefined : sectionEnd)
    const heading = screen.search(new RegExp(`<SectionHeader[\\s\\S]*?id="${id}"`))
    const resource = screen.indexOf("<ResourceView")
    expect(heading, `${id} heading must live outside async content`).toBeGreaterThan(-1)
    if (resource >= 0) expect(heading, `${id} heading must precede async content`).toBeLessThan(resource)
  }
})

test("either explicit miss signal marks a measured miss", () => {
  expect(appSource).toContain("(item.missCount !== undefined && item.missCount > 0) || (item.recall !== undefined && item.recall < 1)")
  expect(appSource).not.toContain("item.missCount !== undefined ?")
})

test("Live Wire starts paused for reduced motion and keeps row selection manual", () => {
  const liveWire = appSource.slice(appSource.indexOf("function LiveWireContent"), appSource.indexOf("function MandateScreen"))
  expect(liveWire).toContain("const reduce = useReducedMotion()")
  expect(liveWire).toContain("useState(() => !reduce)")
  expect(liveWire).toContain("if (reduce) setStreaming(false)")
  expect(liveWire).toContain("setSelectedId(event.eventId)")
  expect(liveWire).toContain("setStreaming(false)")
})
