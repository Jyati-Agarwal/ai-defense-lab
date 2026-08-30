import type {
  ArenaData,
  AtlasData,
  CaseData,
  ComposeInput,
  ComposeResult,
  GeneralizationData,
  MetricsData,
  PhylogenyData,
  PrecedentData,
  StreamData,
  VaccinationData,
} from "../types"

const stages = [
  "Recon",
  "Context",
  "Intent",
  "Tool",
  "Merchant",
  "Payment",
  "Delay",
  "Cash-out",
]

const names = [
  "Synthetic identity drift",
  "Device velocity anomaly",
  "Beneficiary fan-out",
  "Card testing burst",
  "Presence mismatch",
  "Invisible context lure",
  "Instruction priority shift",
  "Zero-width context seam",
  "Retrieved-domain novelty",
  "Playback divergence",
  "Constraint reinterpretation",
  "Budget-edge inflation",
  "Category substitution",
  "Delegation depth jump",
  "Tool-topic pivot",
  "Unregistered tool hop",
  "Negotiation speed spike",
  "Merchant identity swap",
  "Payee allowlist miss",
  "Market-price divergence",
  "Cart signature break",
  "Cart hash decoupling",
  "Agent registry miss",
  "Amount split sequence",
  "UPI collect disguise",
  "Dormant mandate wake",
  "TTL boundary execution",
  "Circadian deviation",
  "New-beneficiary burst",
  "Shared-device cluster",
  "Component expansion",
  "Repudiation narrative",
  "Cross-rail pivot",
  "Currency mismatch",
  "MCC constraint escape",
  "Prompt playback loss",
  "Multi-merchant scatter",
  "Beneficiary in-degree jump",
  "Economic payoff shift",
  "Semantic precedent absence",
  "Synthetic media pretext",
  "Agent framework override",
  "Delayed cart mutation",
  "Registered-agent impersonation",
  "Intent specificity collapse",
  "Provenance artifact cluster",
  "Structural floor violation",
  "Novel composition path",
]

export const fixtureAtlas: AtlasData = {
  stages,
  vectors: names.map((name, index) => {
    const stage = stages[index % stages.length] ?? "Context"
    const id = `GPF-${String(index + 1).padStart(3, "0")}`
    const tier = index < 12 ? "classic" : "agentic"
    const severity = index % 7 === 0 ? "critical" : index % 3 === 0 ? "high" : "medium"
    const observableSignals = [
      index % 2 ? "semantic drift exceeds precedent support" : "structural mandate mismatch",
      index % 3 ? "payee or topic novelty" : "velocity outside entity baseline",
    ]
    return {
      id,
      name,
      stage,
      tier,
      severity,
      primitives: [stage.toLowerCase().replace("-", "_"), index % 2 ? "intent_manip" : "payment_manip"],
      observableSignals,
      defenseHooks: [index % 2 ? "semantic + precedent" : "deterministic structural floor"],
      yaml: [
        `id: ${id}`,
        `name: ${name}`,
        `kill_chain: ${stage.toLowerCase()}`,
        `tier: ${tier}`,
        `severity: ${severity}`,
        "observable_signals:",
        ...observableSignals.map((signal) => `  - ${signal}`),
        "safety: TRUNCATED · SYNTHETIC · NON-OPERATIONAL",
      ].join("\n"),
    }
  }),
}

export const fixtureStream: StreamData = {
  count: 6,
  items: [
    {
      eventId: "evt_flagship",
      time: "14:31:08",
      merchant: "mer_0001",
      channel: "AGENTIC CHECKOUT",
      amountMinor: 575_463,
      currency: "INR",
      score: 0.94,
      legacyAction: "ALLOW",
      action: "BLOCK",
      intent: "Find running shoes under INR 5755",
      cart: "Unrelated demo item",
      payee: "did:example:merchant-0001",
      reasonCodes: ["Intent and cart semantics diverge", "Retrieved page contains hidden-context artifacts"],
    },
    {
      eventId: "evt_borderline",
      time: "14:31:06",
      merchant: "Synthetic Trail Supply",
      channel: "AGENTIC CHECKOUT",
      amountMinor: 512_000,
      currency: "INR",
      score: 0.24,
      legacyAction: "ALLOW",
      action: "ALLOW",
      intent: "Trail-running essentials",
      cart: "Trail shoes + hydration vest",
      payee: "Known synthetic merchant",
      reasonCodes: ["Legitimate precedent support reduces semantic anomaly"],
    },
    {
      eventId: "evt_stepup",
      time: "14:31:02",
      merchant: "Synthetic Electronics",
      channel: "CARD ECOM",
      amountMinor: 289_900,
      currency: "INR",
      score: 0.46,
      legacyAction: "ALLOW",
      action: "STEP_UP",
      intent: "Replacement headphones",
      cart: "Wireless headphones",
      payee: "New synthetic merchant",
      reasonCodes: ["New payee lacks supporting purchase history"],
    },
    {
      eventId: "evt_hold",
      time: "14:30:58",
      merchant: "Synthetic Compute Store",
      channel: "UPI COLLECT",
      amountMinor: 30_000_000,
      currency: "INR",
      score: 0.71,
      legacyAction: "ALLOW",
      action: "HOLD",
      intent: "Workstation GPU",
      cart: "High-end GPU",
      payee: "New synthetic beneficiary",
      reasonCodes: ["High value at unusual hour requires review"],
    },
    {
      eventId: "evt_allow",
      time: "14:30:55",
      merchant: "Synthetic Grocer",
      channel: "UPI P2P",
      amountMinor: 84_900,
      currency: "INR",
      score: 0.08,
      legacyAction: "ALLOW",
      action: "ALLOW",
      intent: "Weekly groceries",
      cart: "Household groceries",
      payee: "Known synthetic merchant",
      reasonCodes: ["Purchase matches established intent and precedent"],
    },
    {
      eventId: "evt_block",
      time: "14:30:50",
      merchant: "Synthetic Marketplace",
      channel: "WIRE",
      amountMinor: 12_800_000,
      currency: "INR",
      score: 0.88,
      legacyAction: "HOLD",
      action: "BLOCK",
      intent: "Office subscription renewal",
      cart: "Unrelated transferable credit",
      payee: "Novel synthetic beneficiary",
      reasonCodes: ["Mandate constraint and beneficiary both diverge"],
    },
  ],
}

export const fixtureGeneralization: GeneralizationData = {
  knownRecall: 0.92,
  adaptiveRecall: 0.83,
  novelCompositionRecall: 0.69,
  heldoutFamilyRecall: 0.58,
  shuffledLabelPRAuc: 0.006,
  baseRate: 0.005,
  splitKind: "Temporal · frozen before C/D/E",
  frozenWeightsHash: "sha256:87a4…21cf",
  firewallStatus: "PASS · defend/ → heldout import denied",
}

export const fixtureArena: ArenaData = {
  label: "In-distribution replay security gate - not generalization",
  rounds: [
    { round: 0, asr: 0.78, mutation: "baseline composition", disposition: "caught after vaccine" },
    { round: 1, asr: 0.51, mutation: "topic padding", disposition: "semantic layer adapted" },
    { round: 2, asr: 0.34, mutation: "delayed execution", disposition: "temporal layer adapted" },
    { round: 3, asr: 0.22, mutation: "merchant fan-out", disposition: "economic layer adapted" },
    { round: 4, asr: 0.14, mutation: "presence mimic", disposition: "structural floor held" },
    { round: 5, asr: 0.09, mutation: "sparse provenance", disposition: "survivor quarantined" },
  ],
  survivors: ["Sparse provenance + long delay", "Low-value fan-out at mandate edge", "Context shift after trusted-domain hop"],
}

export const fixturePhylogeny: PhylogenyData = {
  nodes: [
    { id: "root", label: "baseline", status: "blocked" },
    { id: "pad", parent: "root", label: "topic padding", status: "blocked" },
    { id: "delay", parent: "root", label: "delay branch", status: "blocked" },
    { id: "fan", parent: "pad", label: "fan-out", status: "blocked" },
    { id: "mimic", parent: "delay", label: "presence mimic", status: "blocked" },
    { id: "emergent", parent: "fan", label: "sparse-delay hybrid", emergent: true, status: "quarantined" },
  ],
}

export const fixtureMetrics: MetricsData = {
  prAuc: 0.86,
  recallAtPointOneFpr: 0.61,
  recallAtOneFpr: 0.89,
  brier: 0.041,
  alertsPerThousand: 6.2,
  p50LatencyMs: 18.4,
  p99LatencyMs: 63.7,
  fraudValueCaught: 0.91,
  goodValueBlocked: 0.0018,
  prCurve: [
    { recall: 0, precision: 1 },
    { recall: 0.18, precision: 0.99 },
    { recall: 0.38, precision: 0.96 },
    { recall: 0.58, precision: 0.91 },
    { recall: 0.74, precision: 0.83 },
    { recall: 0.89, precision: 0.66 },
    { recall: 1, precision: 0.005 },
  ],
  confusionMatrix: [[198_812, 188], [112, 888]],
  ablations: [
    { layer: "Full defense", prAuc: 0.86, heldoutRecall: 0.58 },
    { layer: "Without semantic", prAuc: 0.72, heldoutRecall: 0.39 },
    { layer: "Without provenance", prAuc: 0.77, heldoutRecall: 0.44 },
  ],
  economics: {
    before: { costPerAttack: 190, successRate: 0.78, costPerSuccess: 244, roi: 4.8 },
    after: { costPerAttack: 670, successRate: 0.09, costPerSuccess: 7444, roi: -0.42 },
  },
  knownUnknowns: [
    { title: "Clean-context coercion", why: "No provenance artifact survives retrieval.", mitigation: "Bind intent to approved tool scopes." },
    { title: "Long-horizon low-value fan-out", why: "Each payment remains below entity thresholds.", mitigation: "Aggregate mandate-level value over longer windows." },
    { title: "New legitimate categories", why: "Precedent memory is sparse by construction.", mitigation: "Step up, collect consent, then add verified precedent." },
  ],
}

export const fixtureCase: CaseData = {
  eventId: "evt_flagship",
  promptPlayback: "Find running shoes under INR 5755",
  productPage: "https://merchant-mer_0001.example/product",
  productName: "Unrelated demo item",
  productPriceMinor: 575_463,
  merchantId: "mer_0001",
  hiddenTextSample: "[AI DEFENSE LAB · SAFE SYNTHETIC · TRUNCATED] Demonstration fragment: an embedded instruction attempted to change the merchant and payment topic. Actionable wording removed.",
  zeroWidthChars: 7,
  invisibleCss: true,
  userSignatureValid: true,
  merchantSignatureValid: true,
  paymentSignatureValid: true,
  cartHashMatches: true,
  agentRegistered: true,
  agentId: "agent:shopper-v3",
  intentCategory: "running_shoes",
  intentBudgetMinor: 575_463,
  intentMccs: ["5661"],
  intentPayeeAllowlist: ["did:example:merchant-0001"],
  cartItems: ["Unrelated demo item"],
  cartTotalMinor: 575_463,
  cartMcc: "6051",
  payee: "did:example:merchant-0001",
  paymentAmountMinor: 575_463,
  presenceFlag: "PRESENT",
}

export const fixturePrecedent: PrecedentData = {
  eventId: "evt_borderline",
  query: "Trail-running shoes + hydration vest",
  support: 0.91,
  baselineAction: "STEP_UP",
  contextualizedAction: "ALLOW",
  verdict: "suppressed",
  neighbors: [
    { title: "Trail kit · Pune", similarity: 0.94, outcome: "Verified legitimate", context: "Shoes + vest bought together" },
    { title: "Race prep · Bengaluru", similarity: 0.91, outcome: "Verified legitimate", context: "Hydration add-on after shoe search" },
    { title: "Monsoon run · Mumbai", similarity: 0.88, outcome: "Verified legitimate", context: "Trail shoes + safety gear" },
  ],
}

export const fixtureVaccination: VaccinationData = {
  candidate: "sparse-delay hybrid",
  stages: [
    { name: "Candidate isolated", status: "passed" },
    { name: "Old-attack regression", status: "passed" },
    { name: "Benign E stress test", status: "passed" },
    { name: "Calibration gate", status: "passed" },
    { name: "Human promotion", status: "promoted" },
  ],
  beforeScore: 0.42,
  afterScore: 0.87,
  benignRegressionDelta: -0.0002,
  modelVersion: "defense-v1.4-frozen",
  liveModelUnchanged: false,
}

export function composeFixture(input: ComposeInput): ComposeResult {
  const injection = input.injection_style === "none" ? 0 : input.injection_style === "instruction_shift" ? 0.68 : 0.86
  const semantic = Math.min(0.98, 0.18 + input.evasion_aggressiveness * 0.5 + input.fanout * 0.015)
  const provenance = Math.min(0.98, injection + (input.hops > 3 ? 0.04 : 0))
  const temporal = Math.min(0.95, 0.12 + input.ttl_delay_seconds / 604_800)
  const economic = Math.min(0.98, input.amount_minor / 100_000_000 + input.fanout * 0.035)
  const structural = Math.min(0.99, 0.12 + input.delegation_depth * 0.08 + (input.lie_about_presence ? 0.52 : 0))
  const fused = Math.min(0.99, semantic * 0.3 + provenance * 0.25 + temporal * 0.1 + economic * 0.12 + structural * 0.23)
  const action: ComposeResult["decision"]["action"] = fused >= 0.85 ? "BLOCK" : fused >= 0.6 ? "HOLD" : fused >= 0.3 ? "STEP_UP" : "ALLOW"
  return {
    eventId: "fixture_composed_event",
    scores: { semantic, provenance, temporal, economic, structural, fused },
    decision: {
      action,
      legacyAction: input.amount_minor > 10_000_000 ? "HOLD" : "ALLOW",
      reasonCodes: [
        input.injection_style === "none" ? "No page-level injection artifact detected" : "Retrieved page carries hidden-context artifacts",
        input.lie_about_presence ? "Claimed presence conflicts with agent trace" : "Delegation depth contributes structural risk",
        "Intent-to-payment consistency scored across five defense layers",
      ],
      latencyMs: 24.6,
    },
  }
}

export const fixtureResponses = {
  atlas: fixtureAtlas,
  stream: fixtureStream,
  metrics: fixtureMetrics,
  generalization: fixtureGeneralization,
  arena: fixtureArena,
  phylogeny: fixturePhylogeny,
  case: {
    event_id: fixtureCase.eventId,
    mandate_chain: {
      intent: {
        prompt_playback: fixtureCase.promptPlayback,
        signature_valid: fixtureCase.userSignatureValid,
        category: fixtureCase.intentCategory,
        budget_minor: fixtureCase.intentBudgetMinor,
        constraint_obj: { mcc: fixtureCase.intentMccs, payee_allowlist: fixtureCase.intentPayeeAllowlist },
      },
      cart: {
        items: [{ name: fixtureCase.productName, price_minor: fixtureCase.productPriceMinor }],
        total_minor: fixtureCase.cartTotalMinor,
        merchant_id: fixtureCase.merchantId,
        mcc: fixtureCase.cartMcc,
        payee_did: fixtureCase.payee,
        signature_valid: fixtureCase.merchantSignatureValid,
      },
      payment: {
        amount_minor: fixtureCase.paymentAmountMinor,
        signature_valid: fixtureCase.paymentSignatureValid,
        hash_matches_cart: fixtureCase.cartHashMatches,
        agent_registered: fixtureCase.agentRegistered,
        agent_id: fixtureCase.agentId,
        presence_flag: fixtureCase.presenceFlag,
      },
    },
    agent_trace: {
      pages_read: [
        { url: fixtureCase.productPage, hidden_text_present: false, invisible_css: false, zero_width_chars: 0 },
        {
          url: "https://safe-artifact.invalid/non-operational",
          hidden_text_present: true,
          invisible_css: fixtureCase.invisibleCss,
          zero_width_chars: fixtureCase.zeroWidthChars,
          hidden_text_sample: fixtureCase.hiddenTextSample,
        },
      ],
    },
  },
  precedent: fixturePrecedent,
  vaccination: fixtureVaccination,
}
