import type { MetricsData as BaseMetricsData } from "../types"

export type {
  ArenaData,
  AtlasData,
  AtlasVector,
  CaseData,
  ComposeInput,
  ComposeResult,
  DataSource,
  DecisionAction,
  GeneralizationData,
  Loaded,
  LoadStatus,
  PhylogenyData,
  PrecedentData,
  RecallEvidence,
  StreamData,
  StreamEvent,
  VaccinationData,
} from "../types"

type EconomicsPeriod = BaseMetricsData["economics"]["before"] & {
  observedSuccesses?: number
  costPerSuccessCensored?: boolean
}

export type MetricsData = Omit<BaseMetricsData, "economics" | "knownUnknowns"> & {
  economics: { before: EconomicsPeriod; after: EconomicsPeriod }
  knownUnknowns: Array<BaseMetricsData["knownUnknowns"][number] & {
    recall?: number
    missCount?: number
  }>
}
