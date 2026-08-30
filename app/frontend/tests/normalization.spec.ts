import { expect, test } from "@playwright/test"
import { composeFixture, fixtureResponses } from "../src/data/fixtures"
import { api, normalizers } from "../src/lib/api"
import { normalizeDecisionAction } from "../src/lib/utils"
import type { ComposeInput } from "../src/types"

test("decision actions normalize known values and reject unknown values", () => {
  expect(normalizeDecisionAction("step-up")).toBe("STEP_UP")
  expect(() => normalizeDecisionAction(undefined)).toThrow("invalid decision action")
  expect(() => normalizeDecisionAction("unknown")).toThrow("invalid decision action")
})

test("API normalizers reject missing evidence instead of inventing display values", () => {
  const composeInput: ComposeInput = {
    channel: "upi_collect",
    injection_style: "instruction_shift",
    amount_minor: 580_000,
    hops: 2,
    delegation_depth: 2,
    ttl_delay_seconds: 0,
    fanout: 1,
    evasion_aggressiveness: 0.5,
    lie_about_presence: true,
  }
  const valid = {
    atlas: fixtureResponses.atlas,
    stream: fixtureResponses.stream,
    metrics: fixtureResponses.metrics,
    generalization: fixtureResponses.generalization,
    arena: fixtureResponses.arena,
    phylogeny: fixtureResponses.phylogeny,
    case: fixtureResponses.case,
    precedent: fixtureResponses.precedent,
    vaccination: fixtureResponses.vaccination,
    compose: composeFixture(composeInput),
  }
  for (const name of Object.keys(valid) as Array<keyof typeof valid>) {
    expect(() => normalizers[name](valid[name] as never), name).not.toThrow()
  }

  const atlas = structuredClone(fixtureResponses.atlas) as unknown as { vectors: Array<Record<string, unknown>> }
  delete atlas.vectors[0]!.id
  expect(() => normalizers.atlas(atlas)).toThrow("missing atlas vector ID")

  const metrics = structuredClone(fixtureResponses.metrics) as unknown as { confusionMatrix: unknown[][] }
  metrics.confusionMatrix[0]![0] = "not-a-count"
  expect(() => normalizers.metrics(metrics)).toThrow("missing confusion matrix count")

  const compose = structuredClone(composeFixture(composeInput))
  compose.decision.reasonCodes = []
  expect(() => normalizers.compose(compose)).toThrow("missing decision reason codes")

  const partialStream = {
    count: 1,
    items: [{
      event_id: "evt_partial",
      ts: "2026-08-27T00:00:00Z",
      channel: "card_ecom",
      txn: { amount_minor: 10_000, currency: "INR", merchant_id: "mer_synthetic", beneficiary_id: "ben_synthetic" },
      scores: { fused: 0.2 },
      decision: { legacy_rules_action: "ALLOW", action: "ALLOW", reason_codes: ["Measured low risk"] },
      mandate_chain: null,
    }],
  }
  const normalizedPartial = normalizers.stream(partialStream).items[0]!
  expect(normalizedPartial.intent).toBeNull()
  expect(normalizedPartial.cart).toBeNull()
  expect(normalizedPartial.merchant).toBe("mer_synthetic")
  expect(normalizedPartial.payee).toBe("ben_synthetic")

  const malformedStream = structuredClone(partialStream) as unknown as { items: Array<Record<string, unknown>> }
  malformedStream.items[0]!.mandate_chain = "not-an-object"
  expect(() => normalizers.stream(malformedStream)).toThrow("API returned an invalid object")
})

test("production loader rejects fixture-tagged responses", async () => {
  const originalFetch = globalThis.fetch
  globalThis.fetch = async () => new Response(JSON.stringify(fixtureResponses.atlas), {
    status: 200,
    headers: { "content-type": "application/json", "x-ai-defense-fixture": "1" },
  })
  try {
    await expect(api.atlas()).rejects.toThrow("Fixture response rejected outside explicit development mode")
  } finally {
    globalThis.fetch = originalFetch
  }
})

test("metrics preserve censoring, observed successes, and explicit miss evidence", () => {
  const metrics = structuredClone(fixtureResponses.metrics) as unknown as {
    economics: {
      before: Record<string, unknown>
      after: Record<string, unknown>
    }
    knownUnknowns: Array<Record<string, unknown>>
  }
  Object.assign(metrics.economics.before, { observed_successes: 34, cost_per_success_censored: false })
  Object.assign(metrics.economics.after, { observed_successes: 0, cost_per_success_censored: true })
  Object.assign(metrics.knownUnknowns[0]!, { measured_recall: 1, miss_count: 0 })

  const normalized = normalizers.metrics(metrics)
  expect(normalized.economics.before.observedSuccesses).toBe(34)
  expect(normalized.economics.after.observedSuccesses).toBe(0)
  expect(normalized.economics.after.costPerSuccessCensored).toBe(true)
  expect(normalized.knownUnknowns[0]).toMatchObject({ recall: 1, missCount: 0 })
})

test("generalization preserves recall counts, intervals, and threshold evidence", () => {
  const generalization = structuredClone(fixtureResponses.generalization) as unknown as Record<string, unknown>
  generalization.recall_evidence = {
    known: { caught: 34, positive_examples: 34, recall: 1, ci95_low: 0.89, ci95_high: 1 },
    adaptive: { caught: 12, positive_examples: 12, recall: 1, ci95_low: 0.75, ci95_high: 1 },
    novel_composition: { caught: 39, positive_examples: 46, recall: 39 / 46, ci95_low: 0.71, ci95_high: 0.93 },
    heldout_family: { caught: 45, positive_examples: 45, recall: 1, ci95_low: 0.92, ci95_high: 1 },
  }
  const metrics = structuredClone(fixtureResponses.metrics) as unknown as Record<string, unknown>
  metrics.detection_threshold = 0.3
  const normalizedGeneralization = normalizers.generalization(generalization)
  const normalizedMetrics = normalizers.metrics(metrics)
  expect(normalizedGeneralization.recallEvidence?.novelComposition).toMatchObject({ caught: 39, positiveExamples: 46 })
  expect(normalizedMetrics.detectionThreshold).toBe(0.3)

  const invalid = structuredClone(generalization) as Record<string, Record<string, Record<string, number>>>
  invalid.recall_evidence.novel_composition.caught = 47
  expect(() => normalizers.generalization(invalid)).toThrow("inconsistent novel composition recall evidence")
})
