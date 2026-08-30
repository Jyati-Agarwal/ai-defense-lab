export type DataSource = "api" | "fixture"
export type LoadStatus = "loading" | "ready" | "error"
export type DecisionAction = "ALLOW" | "STEP_UP" | "HOLD" | "BLOCK"

export interface Loaded<T> {
  data: T
  source: DataSource
  warning?: string
}

export interface AtlasVector {
  id: string
  name: string
  stage: string
  tier: "classic" | "agentic"
  severity: "low" | "medium" | "high" | "critical"
  primitives: string[]
  observableSignals: string[]
  defenseHooks: string[]
  yaml: string
}

export interface AtlasData {
  stages: string[]
  vectors: AtlasVector[]
}

export interface StreamEvent {
  eventId: string
  time: string
  merchant: string
  channel: string
  amountMinor: number
  currency: string
  score: number
  legacyAction: DecisionAction
  action: DecisionAction
  intent: string | null
  cart: string | null
  payee: string
  reasonCodes: string[]
}

export interface StreamData {
  items: StreamEvent[]
  count: number
}

export interface RecallEvidence {
  caught: number
  positiveExamples: number
  recall: number
  ci95Low: number
  ci95High: number
}

export interface GeneralizationData {
  knownRecall: number
  adaptiveRecall: number
  novelCompositionRecall: number
  heldoutFamilyRecall: number
  shuffledLabelPRAuc: number
  baseRate: number
  splitKind: string
  frozenWeightsHash: string
  firewallStatus: string
  recallEvidence?: {
    known: RecallEvidence
    adaptive: RecallEvidence
    novelComposition: RecallEvidence
    heldoutFamily: RecallEvidence
  }
}

export interface ArenaRound {
  round: number
  asr: number
  mutation: string
  disposition: string
}

export interface ArenaData {
  rounds: ArenaRound[]
  survivors: string[]
  label: string
}

export interface PhylogenyNode {
  id: string
  label: string
  parent?: string
  emergent?: boolean
  status?: string
}

export interface PhylogenyData {
  nodes: PhylogenyNode[]
}

export interface PRPoint {
  recall: number
  precision: number
}

export interface AblationRow {
  layer: string
  prAuc: number
  heldoutRecall: number
}

export interface MetricsData {
  prAuc: number
  recallAtPointOneFpr: number
  recallAtOneFpr: number
  brier: number
  alertsPerThousand: number
  p50LatencyMs: number
  p99LatencyMs: number
  detectionThreshold?: number
  fraudValueCaught: number
  goodValueBlocked: number
  prCurve: PRPoint[]
  confusionMatrix: [[number, number], [number, number]]
  ablations: AblationRow[]
  economics: {
    before: { costPerAttack: number; successRate: number; costPerSuccess: number; roi: number }
    after: { costPerAttack: number; successRate: number; costPerSuccess: number; roi: number }
  }
  knownUnknowns: Array<{ title: string; why: string; mitigation: string }>
}

export interface CaseData {
  eventId: string
  promptPlayback: string
  productPage: string
  productName: string
  productPriceMinor: number
  merchantId: string
  hiddenTextSample: string | null
  zeroWidthChars: number
  invisibleCss: boolean
  userSignatureValid: boolean
  merchantSignatureValid: boolean
  paymentSignatureValid: boolean
  cartHashMatches: boolean
  agentRegistered: boolean
  agentId: string
  intentCategory: string
  intentBudgetMinor: number
  intentMccs: string[]
  intentPayeeAllowlist: string[]
  cartItems: string[]
  cartTotalMinor: number
  cartMcc: string
  payee: string
  paymentAmountMinor: number
  presenceFlag: string
}

export interface PrecedentItem {
  title: string
  similarity: number
  outcome: string
  context: string
}

export interface PrecedentData {
  eventId: string
  query: string
  support: number
  baselineAction: DecisionAction
  contextualizedAction: DecisionAction
  verdict: "suppressed" | "confirmed"
  neighbors: PrecedentItem[]
}

export interface VaccinationData {
  candidate: string
  stages: Array<{ name: string; status: "passed" | "rejected" | "promoted" | "pending" }>
  beforeScore: number
  afterScore: number
  benignRegressionDelta: number
  modelVersion: string
  liveModelUnchanged: boolean
}

export interface ComposeInput {
  channel: "agentic_checkout" | "card_ecom" | "upi_p2p" | "upi_collect" | "wire"
  injection_style: "hidden_text" | "zero_width" | "instruction_shift" | "none"
  amount_minor: number
  hops: number
  delegation_depth: number
  ttl_delay_seconds: number
  fanout: number
  evasion_aggressiveness: number
  lie_about_presence: boolean
}

export interface ComposeResult {
  eventId: string
  scores: {
    semantic: number
    provenance: number
    temporal: number
    economic: number
    structural: number
    fused: number
  }
  decision: {
    action: DecisionAction
    legacyAction: DecisionAction
    reasonCodes: string[]
    latencyMs: number
  }
}
