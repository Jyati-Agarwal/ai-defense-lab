import { useEffect, useState } from "react"
import { normalizeDecisionAction } from "./utils"
import type {
  ArenaData,
  AtlasData,
  AtlasVector,
  CaseData,
  ComposeInput,
  ComposeResult,
  DataSource,
  GeneralizationData,
  Loaded,
  LoadStatus,
  MetricsData,
  PhylogenyData,
  PrecedentData,
  RecallEvidence,
  StreamData,
  StreamEvent,
  VaccinationData,
} from "./types"

const ENV = import.meta.env ?? { DEV: false }
const API_BASE = ENV.VITE_API_BASE_URL ?? (ENV.DEV ? "http://127.0.0.1:8000" : "")

type RecordLike = Record<string, unknown>

function record(value: unknown): RecordLike {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("API returned an invalid object")
  return value as RecordLike
}

function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`API artifact is missing ${label}`)
  return value
}

function first(source: RecordLike, keys: string[]): unknown {
  for (const key of keys) {
    if (source[key] !== undefined) return source[key]
  }
}

function cleanString(value: unknown, label: string): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`API artifact is missing ${label}`)
  return value.replace(/[\u2013\u2014]/g, "-")
}

function optionalString(value: unknown, label: string): string | undefined {
  if (value === undefined || value === null) return undefined
  return cleanString(value, label)
}

function strings(value: unknown, label: string): string[] {
  const values = array(value, label).map((item) => cleanString(item, label))
  if (!values.length) throw new Error(`API artifact is missing ${label}`)
  return values
}

function number(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined
}

function finiteNumber(value: unknown, label: string): number {
  const parsed = number(value)
  if (parsed === undefined) throw new Error(`API artifact is missing ${label}`)
  return parsed
}

function firstNumber(source: RecordLike, keys: string[]): number | undefined {
  for (const key of keys) {
    const value = number(source[key])
    if (value !== undefined) return value
  }
}

function requiredNumber(source: RecordLike, keys: string[], label: string): number {
  const value = firstNumber(source, keys)
  if (value === undefined) throw new Error(`API artifact is missing ${label}`)
  return value
}

function requiredString(source: RecordLike, keys: string[], label: string): string {
  return cleanString(first(source, keys), label)
}

function requiredBoolean(source: RecordLike, keys: string[], label: string): boolean {
  for (const key of keys) {
    if (typeof source[key] === "boolean") return source[key]
  }
  throw new Error(`API artifact is missing ${label}`)
}

function optionalBoolean(value: unknown, label: string): boolean {
  if (value === undefined) return false
  if (typeof value !== "boolean") throw new Error(`API artifact has invalid ${label}`)
  return value
}

function optionalCount(value: unknown, label: string): number {
  if (value === undefined) return 0
  return integer(nonNegative(finiteNumber(value, label), label), label)
}

function optionalInteger(source: RecordLike, keys: string[], label: string): number | undefined {
  const value = first(source, keys)
  return value === undefined ? undefined : integer(nonNegative(finiteNumber(value, label), label), label)
}

function optionalFlag(source: RecordLike, keys: string[], label: string): boolean | undefined {
  const value = first(source, keys)
  if (value === undefined) return undefined
  if (typeof value !== "boolean") throw new Error(`API artifact has invalid ${label}`)
  return value
}

function optionalUnit(source: RecordLike, keys: string[], label: string): number | undefined {
  const value = first(source, keys)
  return value === undefined ? undefined : unit(finiteNumber(value, label), label)
}

function nested(source: RecordLike, key: string): RecordLike {
  return source[key] == null ? {} : record(source[key])
}

function requiredItems(source: RecordLike, keys: string[], label: string): unknown[] {
  const items = array(first(source, keys), label)
  if (!items.length) throw new Error(`API artifact is missing ${label}`)
  return items
}

function unit(value: number, label: string): number {
  if (value < 0 || value > 1) throw new Error(`API artifact has invalid ${label}: ${value}`)
  return value
}

function nonNegative(value: number, label: string): number {
  if (value < 0) throw new Error(`API artifact has invalid ${label}: ${value}`)
  return value
}

function integer(value: number, label: string): number {
  if (!Number.isInteger(value)) throw new Error(`API artifact has invalid ${label}: ${value}`)
  return value
}

function optionalRecallEvidence(value: unknown): GeneralizationData["recallEvidence"] {
  if (value === undefined) return undefined
  const source = record(value)
  const parse = (key: string, label: string): RecallEvidence => {
    const row = record(source[key])
    const caught = integer(nonNegative(finiteNumber(row.caught, `${label} caught`), `${label} caught`), `${label} caught`)
    const positiveExamples = integer(nonNegative(finiteNumber(row.positive_examples, `${label} positive examples`), `${label} positive examples`), `${label} positive examples`)
    const recall = unit(finiteNumber(row.recall, `${label} recall`), `${label} recall`)
    const ci95Low = unit(finiteNumber(row.ci95_low, `${label} lower confidence bound`), `${label} lower confidence bound`)
    const ci95High = unit(finiteNumber(row.ci95_high, `${label} upper confidence bound`), `${label} upper confidence bound`)
    if (caught > positiveExamples || ci95Low > ci95High || recall < ci95Low || recall > ci95High) {
      throw new Error(`API artifact has inconsistent ${label} recall evidence`)
    }
    return { caught, positiveExamples, recall, ci95Low, ci95High }
  }
  return {
    known: parse("known", "known"),
    adaptive: parse("adaptive", "adaptive"),
    novelComposition: parse("novel_composition", "novel composition"),
    heldoutFamily: parse("heldout_family", "held-out family"),
  }
}

function normalizeAtlas(value: unknown): AtlasData {
  const source = Array.isArray(value) ? { vectors: value } : record(value)
  const rawVectors = requiredItems(source, ["vectors", "items", "attacks", "matrix"], "atlas vectors").flat(2)
  const vectors = rawVectors.map((item): AtlasVector => {
    const row = record(item)
    const stage = requiredString(row, ["stage", "kill_chain_stage", "kill_chain", "killChain"], "atlas stage")
    const id = requiredString(row, ["id", "atlas_id"], "atlas vector ID")
    const name = requiredString(row, ["name", "title"], "atlas vector name")
    const rawTier = requiredString(row, ["tier", "type", "category"], "atlas tier").toLowerCase()
    if (rawTier !== "classic" && rawTier !== "agentic") throw new Error(`API artifact has invalid atlas tier: ${rawTier}`)
    const tier: AtlasVector["tier"] = rawTier
    const severityRaw = requiredString(row, ["severity"], "atlas severity").toLowerCase()
    if (severityRaw !== "low" && severityRaw !== "medium" && severityRaw !== "high" && severityRaw !== "critical") {
      throw new Error(`API artifact has invalid atlas severity: ${severityRaw}`)
    }
    const severity: AtlasVector["severity"] = severityRaw
    const observableSignals = strings(first(row, ["observableSignals", "observable_signals", "signals"]), "atlas observable signals")
    const defenseHooks = strings(first(row, ["defenseHooks", "defense_hooks", "defenses"]), "atlas defense hooks")
    return {
      id,
      name,
      stage,
      tier,
      severity,
      primitives: strings(first(row, ["primitives", "attack_primitives"]), "atlas primitives"),
      observableSignals,
      defenseHooks,
      yaml: optionalString(row.yaml, "atlas YAML") ?? JSON.stringify(row, null, 2),
    }
  })
  const rawStages = first(source, ["stages", "kill_chain_stages"])
  const stages = rawStages === undefined
    ? source.kill_chain === undefined
      ? [...new Set(vectors.map((item) => item.stage))]
      : requiredItems(source, ["kill_chain"], "atlas kill chain").map((item) => requiredString(record(item), ["stage"], "atlas stage"))
    : strings(rawStages, "atlas stages")
  return { stages, vectors }
}

function normalizeStream(value: unknown): StreamData {
  const source = Array.isArray(value) ? { items: value } : record(value)
  const rawItems = requiredItems(source, ["items", "events"], "stream events")
  const items = rawItems.map((item): StreamEvent => {
    const event = record(item)
    const decision = nested(event, "decision")
    const scores = nested(event, "scores")
    const txn = nested(event, "txn")
    const chain = nested(event, "mandate_chain")
    const intent = nested(chain, "intent")
    const cart = nested(chain, "cart")
    const payment = nested(chain, "payment")
    const cartEntries = cart.items === undefined ? undefined : array(cart.items, "stream cart items")
    const cartNames = cartEntries?.length
      ? cartEntries.map((entry) => requiredString(record(entry), ["name", "sku"], "stream cart item name")).join(" + ")
      : undefined
    const amountMinor = firstNumber(event, ["amountMinor", "amount_minor"])
      ?? firstNumber(txn, ["amount_minor"])
      ?? requiredNumber(cart, ["total_minor"], "stream event amount")
    const score = firstNumber(event, ["score"]) ?? requiredNumber(scores, ["fused"], "stream event score")
    return {
      eventId: requiredString(event, ["eventId", "event_id"], "stream event ID"),
      time: requiredString(event, ["time", "ts"], "stream event timestamp"),
      merchant: cleanString(event.merchant ?? event.merchant_name ?? cart.payee_did ?? payment.payee_did ?? txn.merchant_id, "stream merchant"),
      channel: requiredString(event, ["channel"], "stream channel").replaceAll("_", " ").toUpperCase(),
      amountMinor: integer(nonNegative(amountMinor, "stream event amount"), "stream event amount"),
      currency: cleanString(event.currency ?? txn.currency, "stream currency"),
      score: unit(score, "stream event score"),
      legacyAction: normalizeDecisionAction(event.legacyAction ?? decision.legacy_rules_action ?? decision.legacyAction),
      action: normalizeDecisionAction(event.action ?? decision.action),
      intent: optionalString(event.intent ?? intent.prompt_playback ?? intent.category, "stream intent") ?? null,
      cart: optionalString(event.cart ?? cartNames, "stream cart") ?? null,
      payee: cleanString(event.payee ?? cart.payee_did ?? payment.payee_did ?? txn.beneficiary_id, "stream payee"),
      reasonCodes: strings(event.reasonCodes ?? decision.reason_codes, "stream reason codes"),
    }
  })
  const count = integer(nonNegative(requiredNumber(source, ["count"], "stream count"), "stream count"), "stream count")
  if (count !== items.length) throw new Error(`API artifact has inconsistent stream count: ${count}`)
  return { items, count }
}

function normalizeGeneralization(value: unknown): GeneralizationData {
  const source = record(value)
  const controls = nested(source, "contamination_checks")
  return {
    knownRecall: unit(requiredNumber(source, ["knownRecall", "known_recall", "known_attack_recall"], "known recall"), "known recall"),
    adaptiveRecall: unit(requiredNumber(source, ["adaptiveRecall", "adaptive_recall"], "adaptive recall"), "adaptive recall"),
    novelCompositionRecall: unit(requiredNumber(source, ["novelCompositionRecall", "novel_composition_recall"], "novel composition recall"), "novel composition recall"),
    heldoutFamilyRecall: unit(requiredNumber(source, ["heldoutFamilyRecall", "heldout_family_recall"], "held-out family recall"), "held-out family recall"),
    shuffledLabelPRAuc: unit(firstNumber(source, ["shuffledLabelPRAuc", "shuffled_label_pr_auc"]) ?? requiredNumber(controls, ["shuffled_label_pr_auc"], "shuffled-label PR-AUC"), "shuffled-label PR-AUC"),
    baseRate: unit(firstNumber(source, ["baseRate", "base_rate"]) ?? requiredNumber(controls, ["base_rate"], "fraud base rate"), "fraud base rate"),
    splitKind: requiredString(source, ["splitKind", "split_kind", "temporal_split"], "evaluation split description"),
    frozenWeightsHash: requiredString(source, ["frozenWeightsHash", "frozen_weights_hash", "model_hash"], "frozen weights hash"),
    firewallStatus: cleanString(source.firewallStatus ?? source.firewall_status ?? controls.import_firewall, "import firewall status"),
    recallEvidence: optionalRecallEvidence(source.recall_evidence),
  }
}

function normalizeArena(value: unknown): ArenaData {
  const source = Array.isArray(value) ? { rounds: value } : record(value)
  const rounds = requiredItems(source, ["rounds", "log"], "arena rounds").map((item) => {
    const row = record(item)
    return {
      round: integer(nonNegative(requiredNumber(row, ["round"], "arena round"), "arena round"), "arena round"),
      asr: unit(requiredNumber(row, ["asr", "attack_success_rate"], "arena ASR"), "arena ASR"),
      mutation: requiredString(row, ["mutation", "best_mutation", "survivor"], "arena mutation"),
      disposition: requiredString(row, ["disposition", "status"], "arena disposition"),
    }
  })
  return {
    rounds,
    survivors: strings(first(source, ["survivors", "surviving_mutations"]), "arena survivors"),
    label: requiredString(source, ["label"], "arena scope label"),
  }
}

function normalizePhylogeny(value: unknown): PhylogenyData {
  const source = Array.isArray(value) ? { nodes: value } : record(value)
  const nodes = requiredItems(source, ["nodes"], "phylogeny nodes").map((item) => {
    const row = record(item)
    const emergentValue = first(row, ["emergent", "is_emergent"])
    if (emergentValue !== undefined && typeof emergentValue !== "boolean") throw new Error("API artifact has invalid phylogeny emergent flag")
    return {
      id: requiredString(row, ["id"], "phylogeny node ID"),
      label: requiredString(row, ["label", "name"], "phylogeny node label"),
      parent: optionalString(first(row, ["parent", "parent_id"]), "phylogeny parent ID"),
      emergent: emergentValue ?? false,
      status: optionalString(row.status, "phylogeny node status"),
    }
  })
  return { nodes }
}

function normalizeMetrics(value: unknown): MetricsData {
  const source = record(value)
  const latency = nested(source, "latency")
  const money = nested(source, "money_weighted")
  const economics = nested(source, "economics")
  const before = nested(economics, "before")
  const after = nested(economics, "after")
  const curve = requiredItems(source, ["prCurve", "pr_curve"], "PR curve").map((item) => {
    const point = record(item)
    return {
      recall: unit(requiredNumber(point, ["recall", "x"], "PR recall point"), "PR recall point"),
      precision: unit(requiredNumber(point, ["precision", "y"], "PR precision point"), "PR precision point"),
    }
  })
  const matrix = array(first(source, ["confusionMatrix", "confusion_matrix"]), "confusion matrix")
  if (matrix.length !== 2 || !Array.isArray(matrix[0]) || matrix[0].length !== 2 || !Array.isArray(matrix[1]) || matrix[1].length !== 2) {
    throw new Error("Metrics artifact contains no 2×2 confusion matrix")
  }
  const matrix0 = array(matrix[0], "confusion matrix row")
  const matrix1 = array(matrix[1], "confusion matrix row")
  const confusionMatrix: MetricsData["confusionMatrix"] = [
    [integer(nonNegative(finiteNumber(matrix0[0], "confusion matrix count"), "confusion matrix count"), "confusion matrix count"), integer(nonNegative(finiteNumber(matrix0[1], "confusion matrix count"), "confusion matrix count"), "confusion matrix count")],
    [integer(nonNegative(finiteNumber(matrix1[0], "confusion matrix count"), "confusion matrix count"), "confusion matrix count"), integer(nonNegative(finiteNumber(matrix1[1], "confusion matrix count"), "confusion matrix count"), "confusion matrix count")],
  ]
  const ablations = requiredItems(source, ["ablations", "ablation_table"], "ablation rows").map((item) => {
    const row = record(item)
    return {
      layer: requiredString(row, ["layer", "name", "configuration"], "ablation layer"),
      prAuc: unit(requiredNumber(row, ["prAuc", "pr_auc"], "ablation PR-AUC"), "ablation PR-AUC"),
      heldoutRecall: unit(requiredNumber(row, ["heldoutRecall", "heldout_recall", "heldout_family_recall"], "ablation held-out recall"), "ablation held-out recall"),
    }
  })
  const unknowns = requiredItems(source, ["knownUnknowns", "known_unknowns", "limitations"], "known limitations").map((item) => {
    const row = record(item)
    return {
      title: requiredString(row, ["title", "name"], "limitation title"),
      why: requiredString(row, ["why", "reason"], "limitation rationale"),
      mitigation: requiredString(row, ["mitigation", "next_step"], "limitation mitigation"),
      recall: optionalUnit(row, ["recall", "measured_recall"], "limitation recall"),
      missCount: optionalInteger(row, ["missCount", "miss_count", "false_negatives"], "limitation miss count"),
    }
  })
  const p50LatencyMs = nonNegative(firstNumber(source, ["p50LatencyMs", "p50_latency_ms"]) ?? requiredNumber(latency, ["p50_ms", "p50"], "p50 latency"), "p50 latency")
  const p99LatencyMs = nonNegative(firstNumber(source, ["p99LatencyMs", "p99_latency_ms"]) ?? requiredNumber(latency, ["p99_ms", "p99"], "p99 latency"), "p99 latency")
  if (p50LatencyMs > p99LatencyMs) throw new Error("API artifact has p50 latency above p99 latency")
  const rawDetectionThreshold = firstNumber(source, ["detectionThreshold", "detection_threshold"])
  return {
    prAuc: unit(requiredNumber(source, ["prAuc", "pr_auc"], "PR-AUC"), "PR-AUC"),
    recallAtPointOneFpr: unit(requiredNumber(source, ["recallAtPointOneFpr", "recall_at_0_1_fpr", "recall_at_0.1pct_fpr"], "recall at 0.1% FPR"), "recall at 0.1% FPR"),
    recallAtOneFpr: unit(requiredNumber(source, ["recallAtOneFpr", "recall_at_1_fpr", "recall_at_1pct_fpr"], "recall at 1% FPR"), "recall at 1% FPR"),
    brier: unit(requiredNumber(source, ["brier", "brier_score"], "Brier score"), "Brier score"),
    alertsPerThousand: nonNegative(requiredNumber(source, ["alertsPerThousand", "alerts_per_thousand", "alerts_per_1000"], "alerts per thousand"), "alerts per thousand"),
    p50LatencyMs,
    p99LatencyMs,
    detectionThreshold: rawDetectionThreshold === undefined ? undefined : unit(rawDetectionThreshold, "detection threshold"),
    fraudValueCaught: unit(firstNumber(source, ["fraudValueCaught", "fraud_value_caught"]) ?? requiredNumber(money, ["fraud_value_caught", "caught_pct"], "fraud value caught"), "fraud value caught"),
    goodValueBlocked: unit(firstNumber(source, ["goodValueBlocked", "good_value_blocked"]) ?? requiredNumber(money, ["good_value_blocked", "blocked_pct"], "good value blocked"), "good value blocked"),
    prCurve: curve,
    confusionMatrix,
    ablations,
    economics: {
      before: {
        costPerAttack: nonNegative(requiredNumber(before, ["costPerAttack", "cost_per_attack"], "pre-defense attack cost"), "pre-defense attack cost"),
        successRate: unit(requiredNumber(before, ["successRate", "success_rate"], "pre-defense success rate"), "pre-defense success rate"),
        costPerSuccess: nonNegative(requiredNumber(before, ["costPerSuccess", "cost_per_success"], "pre-defense cost per success"), "pre-defense cost per success"),
        roi: requiredNumber(before, ["roi"], "pre-defense ROI"),
        observedSuccesses: optionalInteger(before, ["observedSuccesses", "observed_successes"], "pre-defense observed successes"),
        costPerSuccessCensored: optionalFlag(before, ["costPerSuccessCensored", "cost_per_success_censored"], "pre-defense cost-per-success censoring flag"),
      },
      after: {
        costPerAttack: nonNegative(requiredNumber(after, ["costPerAttack", "cost_per_attack"], "post-defense attack cost"), "post-defense attack cost"),
        successRate: unit(requiredNumber(after, ["successRate", "success_rate"], "post-defense success rate"), "post-defense success rate"),
        costPerSuccess: nonNegative(requiredNumber(after, ["costPerSuccess", "cost_per_success"], "post-defense cost per success"), "post-defense cost per success"),
        roi: requiredNumber(after, ["roi"], "post-defense ROI"),
        observedSuccesses: optionalInteger(after, ["observedSuccesses", "observed_successes"], "post-defense observed successes"),
        costPerSuccessCensored: optionalFlag(after, ["costPerSuccessCensored", "cost_per_success_censored"], "post-defense cost-per-success censoring flag"),
      },
    },
    knownUnknowns: unknowns,
  }
}

function normalizeCase(value: unknown): CaseData {
  const source = record(value)
  const chain = nested(source, "mandate_chain")
  const intent = nested(chain, "intent")
  const constraints = nested(intent, "constraint_obj")
  const cart = nested(chain, "cart")
  const payment = nested(chain, "payment")
  const trace = nested(source, "agent_trace")
  const pages = requiredItems(trace, ["pages_read"], "retrieved pages").map(record)
  const productPage = pages[0]!
  const injectedPage = pages.find((page) => optionalBoolean(page.hidden_text_present, "hidden-text marker") || optionalBoolean(page.invisible_css, "invisible-CSS marker") || optionalCount(page.zero_width_chars, "zero-width character count") > 0) ?? productPage
  const items = requiredItems(cart, ["items"], "case cart items").map(record)
  const cartItems = items.map((item) => requiredString(item, ["name", "sku"], "cart item name"))
  const firstItem = items[0]!
  const productPriceMinor = firstNumber(firstItem, ["price_minor"]) ?? requiredNumber(cart, ["total_minor"], "cart total")
  return {
    eventId: requiredString(source, ["event_id", "eventId"], "case event ID"),
    promptPlayback: requiredString(intent, ["prompt_playback"], "prompt playback"),
    productPage: requiredString(productPage, ["url"], "retrieved product page URL"),
    productName: requiredString(firstItem, ["name", "sku"], "cart item name"),
    productPriceMinor: integer(nonNegative(productPriceMinor, "product price"), "product price"),
    merchantId: requiredString(cart, ["merchant_id"], "merchant ID"),
    hiddenTextSample: optionalString(injectedPage.hidden_text_sample, "hidden-text sample") ?? null,
    zeroWidthChars: optionalCount(injectedPage.zero_width_chars, "zero-width character count"),
    invisibleCss: optionalBoolean(injectedPage.invisible_css, "invisible-CSS marker"),
    userSignatureValid: requiredBoolean(intent, ["signature_valid"], "user signature result"),
    merchantSignatureValid: requiredBoolean(cart, ["signature_valid"], "merchant signature result"),
    paymentSignatureValid: requiredBoolean(payment, ["signature_valid"], "payment signature result"),
    cartHashMatches: requiredBoolean(payment, ["hash_matches_cart"], "cart hash result"),
    agentRegistered: requiredBoolean(payment, ["agent_registered"], "recorded agent-registration claim"),
    agentId: requiredString(payment, ["agent_id"], "agent ID"),
    intentCategory: requiredString(intent, ["category"], "intent category"),
    intentBudgetMinor: integer(nonNegative(requiredNumber(intent, ["budget_minor"], "intent budget"), "intent budget"), "intent budget"),
    intentMccs: strings(constraints.mcc, "intent MCC constraints"),
    intentPayeeAllowlist: strings(constraints.payee_allowlist, "intent payee allowlist"),
    cartItems,
    cartTotalMinor: integer(nonNegative(requiredNumber(cart, ["total_minor"], "cart total"), "cart total"), "cart total"),
    cartMcc: requiredString(cart, ["mcc"], "cart MCC"),
    payee: requiredString(cart, ["payee_did"], "cart payee"),
    paymentAmountMinor: integer(nonNegative(requiredNumber(payment, ["amount_minor"], "payment amount"), "payment amount"), "payment amount"),
    presenceFlag: requiredString(payment, ["presence_flag"], "presence flag"),
  }
}

function normalizePrecedent(value: unknown): PrecedentData {
  const source = record(value)
  const neighbors = requiredItems(source, ["neighbors", "precedents"], "precedent neighbors").map((item) => {
    const row = record(item)
    return {
      title: requiredString(row, ["title", "intent", "name"], "precedent title"),
      similarity: unit(requiredNumber(row, ["similarity", "score"], "precedent similarity"), "precedent similarity"),
      outcome: requiredString(row, ["outcome", "status"], "precedent outcome"),
      context: requiredString(row, ["context", "purchase", "cart"], "precedent context"),
    }
  })
  const rawVerdict = requiredString(source, ["verdict"], "precedent verdict").toLowerCase()
  if (rawVerdict !== "confirmed" && rawVerdict !== "suppressed") throw new Error(`API artifact has invalid precedent verdict: ${rawVerdict}`)
  const verdict: PrecedentData["verdict"] = rawVerdict
  return {
    eventId: requiredString(source, ["eventId", "event_id"], "precedent event ID"),
    query: requiredString(source, ["query", "intent"], "precedent query"),
    support: unit(requiredNumber(source, ["support", "precedent_support"], "precedent support"), "precedent support"),
    baselineAction: normalizeDecisionAction(source.baselineAction ?? source.baseline_action),
    contextualizedAction: normalizeDecisionAction(source.contextualizedAction ?? source.contextualized_action ?? source.action),
    verdict,
    neighbors,
  }
}

function normalizeVaccination(value: unknown): VaccinationData {
  const source = record(value)
  const liveModelUnchanged = requiredBoolean(source, ["liveModelUnchanged", "live_model_unchanged"], "live-model state")
  const stages = requiredItems(source, ["stages", "gates"], "vaccination gates").map((item) => {
    const row = record(item)
    const name = requiredString(row, ["name", "gate"], "vaccination gate name")
    const rawStatus = requiredString(row, ["status"], "vaccination gate status").toLowerCase()
    const pendingLivePromotion = liveModelUnchanged && rawStatus === "promoted" && /human promotion/i.test(name)
    let status: VaccinationData["stages"][number]["status"]
    if (pendingLivePromotion || rawStatus === "pending" || rawStatus === "pending_review") status = "pending"
    else if (rawStatus === "passed" || rawStatus === "rejected" || rawStatus === "promoted") status = rawStatus
    else throw new Error(`API artifact has invalid vaccination gate status: ${rawStatus}`)
    return { name, status }
  })
  const benignRegressionDelta = requiredNumber(source, ["benignRegressionDelta", "benign_regression_delta"], "benign regression delta")
  if (benignRegressionDelta < -1 || benignRegressionDelta > 1) throw new Error(`API artifact has invalid benign regression delta: ${benignRegressionDelta}`)
  return {
    candidate: requiredString(source, ["candidate", "attack"], "vaccination candidate"),
    stages,
    beforeScore: unit(requiredNumber(source, ["beforeScore", "before_score"], "pre-vaccine score"), "pre-vaccine score"),
    afterScore: unit(requiredNumber(source, ["afterScore", "after_score"], "post-vaccine score"), "post-vaccine score"),
    benignRegressionDelta,
    modelVersion: requiredString(source, ["modelVersion", "model_version"], "vaccination model version"),
    liveModelUnchanged,
  }
}

function normalizeCompose(value: unknown): ComposeResult {
  const source = record(value)
  const scores = nested(source, "scores")
  const decision = nested(source, "decision")
  return {
    eventId: requiredString(source, ["eventId", "event_id"], "composed event ID"),
    scores: {
      semantic: unit(requiredNumber(scores, ["semantic"], "semantic score"), "semantic score"),
      provenance: unit(requiredNumber(scores, ["provenance"], "provenance score"), "provenance score"),
      temporal: unit(requiredNumber(scores, ["temporal"], "temporal score"), "temporal score"),
      economic: unit(requiredNumber(scores, ["economic"], "economic score"), "economic score"),
      structural: unit(requiredNumber(scores, ["structural"], "structural score"), "structural score"),
      fused: unit(requiredNumber(scores, ["fused"], "fused score"), "fused score"),
    },
    decision: {
      action: normalizeDecisionAction(decision.action),
      legacyAction: normalizeDecisionAction(decision.legacyAction ?? decision.legacy_rules_action),
      reasonCodes: strings(decision.reasonCodes ?? decision.reason_codes, "decision reason codes"),
      latencyMs: nonNegative(requiredNumber(decision, ["latencyMs", "latency_ms"], "inference latency"), "inference latency"),
    },
  }
}

async function load<T>(path: string, normalize: (value: unknown) => T, init?: RequestInit): Promise<Loaded<T>> {
  const response = await fetch(`${API_BASE}${path}`, { ...init, signal: AbortSignal.timeout(6_000) })
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`)
  const acceptFixtureHeader = ENV.VITE_ACCEPT_FIXTURE_HEADER === "true"
  const fixtureResponse = response.headers.get("x-ai-defense-fixture") === "1"
  if (fixtureResponse && !acceptFixtureHeader) throw new Error("Fixture response rejected outside explicit development mode")
  const payload = await response.json()
  return {
    data: normalize(payload),
    source: fixtureResponse ? "fixture" : "api",
    warning: fixtureResponse ? "Development fixture server" : undefined,
  }
}

export interface ResourceState<T> {
  status: LoadStatus
  data?: T
  source?: DataSource
  error?: string
  warning?: string
}

export function useResource<T>(key: string, loader: () => Promise<Loaded<T>>): ResourceState<T> {
  const [state, setState] = useState<ResourceState<T>>({ status: "loading" })
  useEffect(() => {
    let active = true
    setState({ status: "loading" })
    loader()
      .then((result) => active && setState({ status: "ready", ...result }))
      .catch((error: unknown) => active && setState({ status: "error", error: error instanceof Error ? error.message : "Request failed" }))
    return () => {
      active = false
    }
    // loader is intentionally keyed by the stable endpoint string.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  return state
}

export const api = {
  atlas: () => load("/atlas", normalizeAtlas),
  stream: () => load("/stream?n=8", normalizeStream),
  metrics: () => load("/metrics", normalizeMetrics),
  generalization: () => load("/generalization", normalizeGeneralization),
  arena: () => load("/arena", normalizeArena),
  phylogeny: () => load("/phylogeny", normalizePhylogeny),
  case: (eventId: string) => load(`/case/${encodeURIComponent(eventId)}`, normalizeCase),
  precedent: (eventId: string) => load(`/precedent/${encodeURIComponent(eventId)}`, normalizePrecedent),
  vaccination: () => load("/vaccination", normalizeVaccination),
  compose: (input: ComposeInput) =>
    load("/compose", normalizeCompose, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(input),
  }),
}

export const normalizers = {
  atlas: normalizeAtlas,
  stream: normalizeStream,
  metrics: normalizeMetrics,
  generalization: normalizeGeneralization,
  arena: normalizeArena,
  phylogeny: normalizePhylogeny,
  case: normalizeCase,
  precedent: normalizePrecedent,
  vaccination: normalizeVaccination,
  compose: normalizeCompose,
}
