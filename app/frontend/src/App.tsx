import {
  type CSSProperties,
  type FormEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react"
import { ReactLenis } from "lenis/react"
import { AnimatePresence, motion, useInView, useReducedMotion, useScroll, useSpring } from "motion/react"
import gsap from "gsap"
import { useGSAP } from "@gsap/react"
import { SplitText } from "gsap/SplitText"
import { DrawSVGPlugin } from "gsap/DrawSVGPlugin"
import {
  Background,
  ControlButton,
  Panel,
  Position,
  ReactFlow,
  useReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react"
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import {
  Activity,
  AlertTriangle,
  ArrowDown,
  ArrowRight,
  Bot,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleDollarSign,
  Clock3,
  Database,
  Eye,
  EyeOff,
  Fingerprint,
  FlaskConical,
  Gauge,
  GitBranch,
  KeyRound,
  LockKeyhole,
  Maximize2,
  Minus,
  Network,
  Pause,
  Play,
  Plus,
  Radar,
  RefreshCw,
  ScanLine,
  Shield,
  ShieldCheck,
  ShieldX,
  ShoppingBag,
  Swords,
  Terminal,
  Unlock,
  User,
  Vault,
  Waves,
  X,
  Zap,
} from "lucide-react"
import { api, type ResourceState, useResource } from "./lib/api"
import { cn, formatCompact, formatInrMinor, formatPercent } from "./lib/utils"
import {
  Badge,
  Button,
  ErrorBlock,
  LoadingBlock,
  SelectField,
  Slider,
  SourceBadge,
  Stat,
  Switch,
} from "./components/ui"
import type {
  ArenaData,
  AtlasData,
  CaseData,
  ComposeInput,
  ComposeResult,
  DataSource,
  GeneralizationData,
  Loaded,
  MetricsData,
  PhylogenyData,
  PrecedentData,
  StreamData,
  StreamEvent,
  VaccinationData,
} from "./lib/types"

gsap.registerPlugin(useGSAP, SplitText, DrawSVGPlugin)

const FIXTURE_MODE = import.meta.env.VITE_ACCEPT_FIXTURE_HEADER === "true"

const screens = [
  ["thesis", "Thesis"],
  ["atlas", "Atlas"],
  ["live-wire", "Live Wire"],
  ["mandate", "Mandate"],
  ["injection", "Injection"],
  ["precedent", "Memory"],
  ["arena", "Arena"],
  ["vault", "Vault"],
  ["metrics", "Metrics"],
  ["economics", "Economics"],
  ["composer", "Composer"],
  ["unknowns", "Unknowns"],
] as const

const panelEnter = {
  "data-panel-enter": true,
  initial: { opacity: 0, y: 22 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, amount: 0.16 },
}

function SmoothRoot({ children }: { children: ReactNode }) {
  const reduce = useReducedMotion()
  if (reduce) return <>{children}</>
  return (
    <ReactLenis root options={{ anchors: true, smoothWheel: true, duration: 1.05 }}>
      {children}
    </ReactLenis>
  )
}

function StoryNav() {
  const { scrollYProgress } = useScroll()
  const scaleX = useSpring(scrollYProgress, { stiffness: 120, damping: 28, mass: 0.25 })
  return (
    <header className="relative inset-x-0 top-0 z-50 border-b border-white/10 bg-[#08090d]/95 md:fixed">
      <motion.div className="absolute inset-x-0 bottom-0 h-px origin-left bg-cyan-300" style={{ scaleX }} aria-hidden="true" />
      <div className="mx-auto flex h-16 max-w-[1440px] items-center gap-5 px-4 sm:px-6">
        <a href="#thesis" aria-label="AI Defense Lab, return to thesis" className="flex min-h-11 min-w-11 shrink-0 items-center gap-2 rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300">
          <span className="grid size-8 place-items-center border border-cyan-300/40 bg-cyan-300/10 text-cyan-200">
            <ShieldCheck className="size-4" aria-hidden="true" />
          </span>
          <span className="text-[10px] font-semibold uppercase tracking-[0.13em] text-zinc-100 sm:text-xs sm:tracking-[0.15em]">AI Defense<span className="hidden sm:inline"> Lab</span></span>
        </a>
        <div className="story-nav-wrap min-w-0 flex-1">
          <nav className="story-nav-scroll min-w-0 overflow-x-auto" data-lenis-prevent aria-label="Story chapters; scroll horizontally for more">
            <ol className="flex min-w-max items-center gap-1">
              {screens.map(([id, label], index) => (
                <li key={id}>
                  <a
                    href={`#${id}`}
                    onFocus={(event) => event.currentTarget.scrollIntoView({ block: "nearest", inline: "nearest" })}
                    className="inline-flex min-h-11 items-center gap-1.5 rounded px-2.5 font-mono text-[10px] uppercase tracking-[0.12em] text-zinc-500 transition-colors hover:bg-white/[0.05] hover:text-zinc-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300"
                  >
                    <span className="text-zinc-700">{String(index + 1).padStart(2, "0")}</span> {label}
                  </a>
                </li>
              ))}
            </ol>
          </nav>
          <span className="story-nav-cue"><span className="sr-only">More sections:</span><span aria-hidden="true">MORE</span> <ArrowRight className="size-3" aria-hidden="true" /></span>
        </div>
        <a href="#composer" className="hidden min-h-11 shrink-0 items-center border border-[#ff3b5c]/50 px-3 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-[#ff8296] transition-colors hover:bg-[#ff3b5c]/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#ff3b5c] md:flex">
          Enter Red mode
        </a>
      </div>
    </header>
  )
}

function FixtureRail() {
  return (
    <div className="fixed inset-x-0 top-16 z-40 border-b border-amber-300/20 bg-[#17140b]/95 px-3 py-1.5 text-center font-mono text-[9px] font-semibold uppercase tracking-[0.14em] text-amber-200" role="status" data-testid="fixture-rail">
      Development fixture mode · values below are deterministic UI fixtures, not run evidence · live API artifacts replace them automatically
    </div>
  )
}

function SectionHeader({
  id,
  index,
  kicker,
  title,
  deck,
  happening,
  source,
}: {
  id: string
  index: number
  kicker: string
  title: string
  deck: string
  happening: string
  source?: DataSource
}) {
  return (
    <div className="mb-10 grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(280px,440px)] lg:items-end">
      <div>
        <p className="section-kicker"><span>{String(index).padStart(2, "0")}</span>{kicker}</p>
        <h2 id={id} className="section-title">{title}</h2>
        <p className="section-deck mt-6">{deck}</p>
      </div>
      <div className="space-y-4">
        {source ? <SourceBadge source={source} /> : null}
        <p className="happening"><strong>What’s happening</strong><span>{happening}</span></p>
      </div>
    </div>
  )
}

function ResourceView<T>({
  state,
  label,
  children,
}: {
  state: ResourceState<T>
  label: string
  children: (data: T, source: DataSource) => ReactNode
}) {
  if (state.status === "loading") return <LoadingBlock label={label} />
  if (state.status === "error" || !state.data || !state.source) return <ErrorBlock title={`${label} unavailable`} detail={state.error} />
  return children(state.data, state.source)
}

function DiagramControls({ label }: { label: string }) {
  const { fitView, zoomIn, zoomOut } = useReactFlow()
  return (
    <Panel position="bottom-left" className="react-flow__controls vertical" role="group" aria-label={label}>
      <ControlButton onClick={() => void zoomIn()} aria-label="Zoom in" title="Zoom in"><Plus aria-hidden="true" /></ControlButton>
      <ControlButton onClick={() => void zoomOut()} aria-label="Zoom out" title="Zoom out"><Minus aria-hidden="true" /></ControlButton>
      <ControlButton onClick={() => void fitView()} aria-label="Fit diagram" title="Fit diagram"><Maximize2 aria-hidden="true" /></ControlButton>
    </Panel>
  )
}

function Hero({ state }: { state: ResourceState<CaseData> }) {
  const root = useRef<HTMLElement>(null)
  const reduce = useReducedMotion()
  useGSAP(
    () => {
      if (reduce || state.status !== "ready") return
      const split = SplitText.create(".hero-title", { type: "words", aria: "auto" })
      const timeline = gsap.timeline({ defaults: { ease: "power3.out" } })
      timeline
        .from(split.words, { yPercent: 110, stagger: 0.055, duration: 0.72 })
        .from(".hero-deck", { y: 16, duration: 0.45 }, "-=0.32")
        .from(".hero-node", { y: 18, stagger: 0.12, duration: 0.4 }, "-=0.12")
        .from(".hero-draw", { drawSVG: 0, duration: 1.2, stagger: 0.12 }, "-=0.42")
        .from(".hero-verdict", { scale: 0.94, duration: 0.35 }, "-=0.28")
      return () => split.revert()
    },
    { scope: root, dependencies: [reduce, state.status], revertOnUpdate: true },
  )

  if (state.status === "loading") {
    return <section id="thesis" className="story-section min-h-[100svh] pt-32" aria-labelledby="hero-title"><div className="story-shell"><h1 id="hero-title" className="section-title">The signature is valid. The intent was stolen.</h1><LoadingBlock label="flagship case" /></div></section>
  }
  if (state.status === "error" || !state.data || !state.source) {
    return <section id="thesis" className="story-section min-h-[100svh] pt-32" aria-labelledby="hero-title"><div className="story-shell"><h1 id="hero-title" className="section-title">Intent integrity evidence unavailable.</h1><ErrorBlock title="Flagship case unavailable" detail={state.error} /></div></section>
  }

  const caseData = state.data
  const hasRetrievalArtifact = Boolean(caseData.zeroWidthChars > 0 || caseData.invisibleCss || caseData.hiddenTextSample)
  const steps = [
    { icon: User, label: "Recorded intent", detail: caseData.promptPlayback, tone: "blue" },
    { icon: Bot, label: "Recorded agent", detail: `${caseData.agentId} · registry claim: ${caseData.agentRegistered ? "registered" : "unregistered"}`, tone: "blue" },
    { icon: Terminal, label: "Retrieved page", detail: `${caseData.zeroWidthChars} zero-width chars · ${caseData.invisibleCss ? "invisible CSS" : "visible CSS only"}`, tone: hasRetrievalArtifact ? "red" : "blue" },
    { icon: CircleDollarSign, label: "Recorded payment", detail: `${formatInrMinor(caseData.paymentAmountMinor)} · ${caseData.payee}`, tone: "blue" },
  ] as const

  return (
    <section id="thesis" ref={root} className="story-section min-h-[100svh] overflow-hidden pt-32" aria-labelledby="hero-title">
      <div className="story-shell">
        <div className="grid gap-12 lg:grid-cols-[1.05fr_.95fr] lg:items-center">
          <div>
            <div className="flex flex-wrap gap-2">
              <Badge tone="blue"><Radar className="size-3" aria-hidden="true" /> Intent integrity · measured honestly</Badge>
              {state.source ? <SourceBadge source={state.source} /> : null}
            </div>
            <h1 id="hero-title" data-testid="hero-title" className="hero-title mt-7 overflow-hidden text-[clamp(3.1rem,13vw,4rem)] font-semibold leading-[0.88] tracking-[-0.065em] lg:text-[clamp(3.2rem,4vw,4rem)]">
              <span data-testid="hero-line" className="block lg:whitespace-nowrap">The signature is valid.</span>{" "}
              <span data-testid="hero-line" className="block text-[#ff3b5c] red-glow lg:whitespace-nowrap">The intent was stolen.</span>
            </h1>
            <p className="hero-deck mt-8 max-w-2xl text-base leading-7 text-zinc-400 sm:text-xl sm:leading-8">
              These are synthetic, signed records. The question is simple: did the recorded payment still match what the person meant?
            </p>
            <div className="hero-deck mt-8 flex flex-wrap items-center gap-3">
              <a href="#live-wire" data-testid="primary-cta" className="inline-flex min-h-12 items-center gap-2 rounded-md bg-cyan-300 px-5 text-sm font-semibold uppercase tracking-[0.09em] text-[#071015] transition-colors hover:bg-cyan-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white">
                Trace the intent mismatch <ArrowDown className="size-4" aria-hidden="true" />
              </a>
              <a href="#vault" className="inline-flex min-h-12 items-center gap-2 rounded-md border border-white/15 px-5 text-sm font-semibold uppercase tracking-[0.09em] text-zinc-200 hover:bg-white/[0.05] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300">
                Open evaluation scope
              </a>
            </div>
            <div className="mt-8 grid gap-px overflow-hidden border border-white/10 bg-white/10 sm:grid-cols-2" role="group" aria-label="Intent and payment comparison">
              <div className="bg-[#101219] p-4"><p className="mono-label text-cyan-200">User meant</p><p className="mt-2 text-sm font-medium leading-6 text-zinc-100">{caseData.intentCategory.replaceAll("_", " ")} under {formatInrMinor(caseData.intentBudgetMinor)}</p><p className="mt-1 break-all text-xs leading-5 text-zinc-500">Allowed payee: {caseData.intentPayeeAllowlist[0] ?? "not recorded"}</p></div>
              <div className="bg-[#190a10] p-4"><p className="mono-label text-[#ff9aac]">Recorded payment</p><p className="mt-2 text-sm font-medium leading-6 text-[#ffb3bf]">{caseData.productName} for {formatInrMinor(caseData.paymentAmountMinor)}</p><p className="mt-1 break-all text-xs leading-5 text-zinc-500">Charged payee: {caseData.payee}</p></div>
            </div>
          </div>

          <div className="relative mx-auto w-full max-w-xl" role="group" aria-label="Recorded case evidence from intent through payment">
            <div className="absolute -inset-16 -z-10 bg-[radial-gradient(circle,rgba(59,201,255,.08),transparent_62%)]" aria-hidden="true" />
            <svg viewBox="0 0 520 700" className="hero-evidence-path absolute inset-0 size-full" aria-hidden="true">
              <path className="hero-draw" d="M84 91 C275 91 240 222 430 222" fill="none" stroke="#3bc9ff" strokeWidth="2" />
              <path className="hero-draw" d="M430 222 C250 222 280 360 84 360" fill="none" stroke="#ff3b5c" strokeWidth="2" strokeDasharray="5 7" />
              <path className="hero-draw" d="M84 360 C260 360 270 505 430 505" fill="none" stroke="#ff3b5c" strokeWidth="3" />
            </svg>
            <div className="hero-evidence-grid relative grid min-h-[560px] grid-cols-2 grid-rows-4 items-center gap-x-4 sm:min-h-[700px] sm:gap-x-12">
              {steps.map((step, index) => {
                const Icon = step.icon
                const positions = ["col-start-1 row-start-1", "col-start-2 row-start-1", "col-start-1 row-start-2", "col-start-2 row-start-3"]
                return (
                  <div key={step.label} className={cn("hero-node story-panel relative min-w-0 p-3 sm:p-4", positions[index])}>
                    <div className={cn("mb-4 grid size-10 place-items-center rounded-full border", step.tone === "red" ? "border-[#ff3b5c]/50 bg-[#ff3b5c]/10 text-[#ff6d84]" : "border-cyan-300/50 bg-cyan-300/10 text-cyan-200")}>
                      <Icon className="size-5" aria-hidden="true" />
                    </div>
                    <p className="font-mono text-[10px] uppercase tracking-[0.13em] text-zinc-500">0{index + 1} · {step.label}</p>
                    <p className="mt-2 text-sm font-medium leading-5 text-zinc-100">{step.detail}</p>
                    {index === 3 ? <Badge tone={caseData.paymentSignatureValid ? "green" : "red"} className="mt-3">{caseData.paymentSignatureValid ? <Check className="size-3" /> : <X className="size-3" />} Payment signature {caseData.paymentSignatureValid ? "verified" : "failed"}</Badge> : null}
                  </div>
                )
              })}
              <div className="hero-verdict col-start-1 col-span-2 row-start-4 self-end justify-self-end border-l-2 border-[#ff3b5c] bg-[#ff3b5c]/10 px-4 py-3 font-mono text-xs text-[#ff9aac] sm:self-start">
                AUTHENTIC ≠ INTENDED
              </div>
            </div>
          </div>
        </div>
        <p className="happening mt-10"><strong>What’s happening</strong><span>The signature checks pass, but the allowed payee and recorded payment are separate pieces of evidence. Follow the payment to see where the intent mismatch enters.</span></p>
      </div>
    </section>
  )
}

function StageFlow({ stages, selectedStage = "all" }: { stages: string[]; selectedStage?: string }) {
  const nodes = useMemo<Node[]>(
    () => stages.slice(0, 8).map((stage, index) => ({
      id: stage,
      position: { x: index * 150, y: index % 2 ? 42 : 0 },
      data: { label: stage },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      style: { width: 116, borderColor: index < 2 ? "rgba(255,255,255,.18)" : "rgba(255,59,92,.45)" },
    })),
    [stages],
  )
  const edges = useMemo<Edge[]>(
    () => nodes.slice(1).map((node, index) => ({ id: `stage-${index}`, source: nodes[index]?.id ?? "", target: node.id })),
    [nodes],
  )
  return (
    <div className="border-y border-white/10">
      <ol className="stage-strip" aria-label="Attack kill-chain stages">
        {stages.slice(0, 8).map((stage, index) => <li key={stage} data-selected={selectedStage === stage} aria-current={selectedStage === stage ? "step" : undefined}><span>{String(index + 1).padStart(2, "0")}</span>{stage.replaceAll("_", " ")}</li>)}
      </ol>
      <div className="flow-canvas stage-flow-desktop h-44 w-full" data-lenis-prevent>
      <ReactFlow id="atlas-kill-chain" nodes={nodes} edges={edges} fitView nodesDraggable={false} nodesConnectable={false} edgesFocusable={false} elementsSelectable={false} panOnScroll zoomOnPinch minZoom={0.35} maxZoom={1.5} aria-label="Attack kill-chain flow">
        <Background gap={24} size={1} color="#3f414b" />
      </ReactFlow>
      </div>
    </div>
  )
}

function AtlasScreen({ state }: { state: ResourceState<AtlasData> }) {
  return (
    <section id="atlas" className="story-section" aria-labelledby="atlas-title">
      <div className="story-shell">
        <SectionHeader
          id="atlas-title"
          index={2}
          kicker="Attack atlas"
          title="Map the adversary before scoring it."
          deck={state.data ? `${state.data.vectors.length} vectors arranged across the payment kill chain. The agentic tier is the new terrain, not a recolor of card fraud.` : "Measured attack vectors arranged across the payment kill chain."}
          happening="Filter the atlas, then select a vector. You see only its safe metadata and observable evidence, not an operational recipe."
          source={state.source}
        />
        <ResourceView state={state} label="attack atlas">
          {(data) => <AtlasContent data={data} />}
        </ResourceView>
      </div>
    </section>
  )
}

function AtlasContent({ data }: { data: AtlasData }) {
  const [tier, setTier] = useState<"all" | "agentic" | "classic">("agentic")
  const firstAgentic = data.vectors.find((item) => item.tier === "agentic") ?? data.vectors[0]
  const [stage, setStage] = useState(() => {
    const narrow = typeof window !== "undefined" && window.matchMedia("(max-width: 900px)").matches
    return narrow ? (firstAgentic?.stage ?? "all") : "all"
  })
  const [selectedId, setSelectedId] = useState(firstAgentic?.id)
  const filtered = useMemo(() => data.vectors.filter((vector) => (tier === "all" || vector.tier === tier) && (stage === "all" || vector.stage === stage)), [data.vectors, tier, stage])
  const selected = filtered.find((vector) => vector.id === selectedId) ?? filtered[0]
  const displayStages = stage === "all" ? data.stages : [stage]
  return (
    <>
      <motion.div {...panelEnter} className="story-panel overflow-hidden">
        <div className="flex flex-col gap-4 border-b border-white/10 p-4 sm:flex-row sm:items-end sm:justify-between sm:p-6">
          <div className="flex flex-wrap gap-2" role="group" aria-label="Attack tier filter">
            {(["agentic", "classic", "all"] as const).map((item) => (
              <Button key={item} size="sm" variant={tier === item ? (item === "agentic" ? "attacker" : "defender") : "ghost"} aria-pressed={tier === item} onClick={() => setTier(item)}>
                {item === "agentic" ? <Zap className="size-3.5" /> : null}{item}
              </Button>
            ))}
          </div>
          <SelectField id="atlas-stage" label="Kill-chain stage" value={stage} onChange={setStage}>
            <option value="all">All stages</option>
            {data.stages.map((item) => <option key={item} value={item}>{item.replaceAll("_", " ")}</option>)}
          </SelectField>
        </div>
        <StageFlow stages={data.stages} selectedStage={stage} />
        <div className="grid lg:grid-cols-[minmax(0,1.6fr)_minmax(300px,.8fr)]">
          <div className="atlas-board-scroll min-w-0 overflow-x-auto border-b border-white/10 p-4 lg:border-b-0 lg:border-r lg:p-6" data-lenis-prevent tabIndex={0} role="region" aria-label="Attack vectors grouped by kill-chain stage">
            <p className="atlas-scroll-hint">Select a stage, then inspect each vector <ArrowRight className="size-3.5" aria-hidden="true" /></p>
            <div className="atlas-stage-grid grid gap-3" style={{ "--atlas-columns": displayStages.length } as CSSProperties}>
              {displayStages.map((stageName) => (
                <div key={stageName}>
                  <div className="mb-3 flex items-center justify-between border-b border-white/10 pb-2">
                    <span className="mono-label">{stageName.replaceAll("_", " ")}</span>
                    <span className="font-mono text-[10px] text-zinc-600">{filtered.filter((item) => item.stage === stageName).length}</span>
                  </div>
                  <div className="space-y-2">
                    {filtered.filter((item) => item.stage === stageName).map((vector) => (
                      <button
                        key={vector.id}
                        type="button"
                        onClick={() => setSelectedId(vector.id)}
                        aria-pressed={selected?.id === vector.id}
                        className={cn("min-h-16 w-full border-l-2 bg-white/[0.025] p-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300", vector.tier === "agentic" ? "border-[#ff3b5c]/70 hover:bg-[#ff3b5c]/10" : "border-zinc-600 hover:bg-white/[0.06]", selected?.id === vector.id && "bg-[#ff3b5c]/10")}
                      >
                        <span className="font-mono text-[9px] uppercase tracking-[0.13em] text-zinc-600">{vector.id}</span>
                        <span className="mt-1 block text-xs font-medium leading-4 text-zinc-200">{vector.name}</span>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="min-w-0 p-5 sm:p-6" aria-live="polite" data-testid="atlas-selected" data-tier={selected?.tier} data-stage={selected?.stage}>
            {selected ? (
              <>
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone={selected.tier === "agentic" ? "red" : "neutral"}>{selected.tier}</Badge>
                  <Badge tone={selected.severity === "critical" ? "red" : "amber"}>{selected.severity}</Badge>
                </div>
                <h3 className="mt-5 break-words text-2xl font-semibold tracking-tight">{selected.name}</h3>
                <p className="mt-2 font-mono text-xs text-zinc-500">{selected.id} · {selected.stage.replaceAll("_", " ")}</p>
                <div className="mt-6">
                  <p className="mono-label">Observable signals</p>
                  <ul className="mt-3 space-y-2 text-sm leading-6 text-zinc-300">
                    {selected.observableSignals.map((signal) => <li key={signal} className="flex gap-2"><ScanLine className="mt-1 size-4 shrink-0 text-cyan-300" aria-hidden="true" />{signal}</li>)}
                  </ul>
                </div>
                <a href="#live-wire" className="mt-6 inline-flex min-h-11 items-center gap-2 rounded-md border border-cyan-300/35 px-4 text-xs font-semibold uppercase tracking-[0.1em] text-cyan-200 hover:bg-cyan-300/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300">Continue to scored replay <ArrowDown className="size-4" aria-hidden="true" /></a>
                <pre tabIndex={0} role="region" className="mt-6 max-h-72 max-w-full overflow-auto border border-white/10 bg-black/30 p-4 font-mono text-[10px] leading-5 text-zinc-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300" data-lenis-prevent aria-label="Selected vector safe YAML metadata">{selected.yaml.slice(0, 1200)}{"\n"}# TRUNCATED · SYNTHETIC · NON-OPERATIONAL</pre>
              </>
            ) : <p className="text-sm text-zinc-500">No vector matches this filter.</p>}
          </div>
        </div>
      </motion.div>
    </>
  )
}

function actionTone(action: StreamEvent["action"]) {
  if (action === "BLOCK") return "text-[#ff6d84] border-[#ff3b5c]/40 bg-[#ff3b5c]/10"
  if (action === "HOLD") return "text-amber-200 border-amber-300/35 bg-amber-300/10"
  if (action === "STEP_UP") return "text-cyan-200 border-cyan-300/35 bg-cyan-300/10"
  return "text-emerald-300 border-emerald-400/35 bg-emerald-400/10"
}

function RiskCounter({ score, active }: { score: number; active: boolean }) {
  const ref = useRef<HTMLSpanElement>(null)
  const reduce = useReducedMotion()
  useGSAP(
    () => {
      if (!ref.current) return
      if (reduce) {
        ref.current.textContent = String(Math.round(score * 100))
        return
      }
      if (!active) {
        ref.current.textContent = String(Math.round(score * 100))
        return
      }
      const counter = { value: 0 }
      gsap.to(counter, {
        value: score * 100,
        duration: 0.82,
        ease: "power2.out",
        onUpdate: () => {
          if (ref.current) ref.current.textContent = String(Math.round(counter.value))
        },
      })
    },
    { scope: ref, dependencies: [score, reduce, active], revertOnUpdate: true },
  )
  return <span ref={ref} className="font-mono text-xl font-semibold tabular-nums" data-testid="risk-counter">{Math.round(score * 100)}</span>
}

function LiveWire({ state }: { state: ResourceState<StreamData> }) {
  return (
    <section id="live-wire" className="story-section" aria-labelledby="live-wire-title">
      <div className="story-shell">
        <SectionHeader id="live-wire-title" index={3} kicker="Live wire" title="One payment. Two very different answers." deck="The old rules see valid credentials and familiar limits. The intent-integrity stack sees where the delegated mission broke." happening="Select any cached transaction. Its fused risk is replayed, then legacy policy and our policy are compared on the same event." source={state.source} />
        <ResourceView state={state} label="scored event stream">
          {(data) => <LiveWireContent data={data} />}
        </ResourceView>
      </div>
    </section>
  )
}

function LiveWireContent({ data }: { data: StreamData }) {
  const reduce = useReducedMotion()
  const replayRef = useRef<HTMLDivElement>(null)
  const replayInView = useInView(replayRef, { amount: 0.15 })
  const flagship = data.items.find((item) => item.legacyAction === "ALLOW" && item.action === "BLOCK") ?? data.items[0]
  const [selectedId, setSelectedId] = useState(flagship?.eventId)
  const [streaming, setStreaming] = useState(() => !reduce)
  const selected = data.items.find((item) => item.eventId === selectedId) ?? flagship
  useEffect(() => {
    if (reduce) setStreaming(false)
  }, [reduce])
  useEffect(() => {
    if (!streaming || !replayInView || data.items.length < 2) return
    const timer = window.setInterval(() => {
      setSelectedId((current) => {
        const currentIndex = data.items.findIndex((item) => item.eventId === current)
        return data.items[(currentIndex + 1) % data.items.length]?.eventId
      })
    }, 1_800)
    return () => window.clearInterval(timer)
  }, [data.items, replayInView, streaming])
  return (
    <>
      <motion.div {...panelEnter} ref={replayRef} className="story-panel overflow-hidden">
        <div className="flex items-center justify-between border-b border-white/10 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.15em] text-zinc-500">
            <span className={cn("size-2 rounded-full", streaming ? "animate-pulse bg-emerald-400 motion-reduce:animate-none" : "bg-zinc-600")} /> cached event replay · {data.count} events
          </div>
          <Button size="icon" variant="ghost" onClick={() => setStreaming((value) => !value)} aria-label={streaming ? "Pause transaction replay" : "Resume transaction replay"}>
            {streaming ? <Pause className="size-4" /> : <Play className="size-4" />}
          </Button>
        </div>
        {selected ? (
          <div className="grid xl:grid-cols-[minmax(0,1.5fr)_minmax(360px,.8fr)]">
            <div className="min-w-0 border-b border-white/10 xl:border-b-0 xl:border-r">
              <div className="hidden grid-cols-[80px_1fr_120px_90px_100px] gap-3 border-b border-white/10 px-5 py-3 font-mono text-[9px] uppercase tracking-[0.12em] text-zinc-600 md:grid">
                <span>Time</span><span>Event</span><span>Amount</span><span>Risk</span><span>Decision</span>
              </div>
              <div className="divide-y divide-white/[0.07]">
                {data.items.map((event) => (
                  <button
                    key={event.eventId}
                    type="button"
                    onClick={() => {
                      setSelectedId(event.eventId)
                      setStreaming(false)
                    }}
                    className={cn("grid min-h-20 w-full gap-3 px-4 py-4 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-cyan-300 md:grid-cols-[80px_1fr_120px_90px_100px] md:items-center md:px-5", selected.eventId === event.eventId ? "bg-white/[0.055]" : "hover:bg-white/[0.03]", streaming && selected.eventId === event.eventId && "shadow-[inset_3px_0_0_#3bc9ff]")}
                    aria-label={`Inspect ${event.merchant} transaction ${event.eventId}`}
                    aria-pressed={selected.eventId === event.eventId}
                  >
                    <span className="font-mono text-[10px] text-zinc-600">{event.time.slice(-8)}</span>
                    <span className="min-w-0"><span className="block truncate text-sm font-medium text-zinc-200">{event.merchant}</span><span className="mt-1 block truncate font-mono text-[9px] uppercase tracking-[0.1em] text-zinc-600">{event.eventId} · {event.channel}</span><span className="mt-2 inline-flex items-center gap-1 font-mono text-[9px] uppercase tracking-[0.12em] text-cyan-200/75"><Eye className="size-3" aria-hidden="true" /> Inspect</span></span>
                    <span className="font-mono text-xs text-zinc-300">{formatInrMinor(event.amountMinor)}</span>
                    <span className={event.score >= 0.6 ? "text-[#ff6d84]" : "text-cyan-200"}><RiskCounter score={event.score} active={replayInView && streaming} /><span className="text-xs">/100</span></span>
                    <span className={cn("w-fit rounded-full border px-2 py-1 font-mono text-[9px] font-semibold tracking-[0.12em]", actionTone(event.action))}>{event.action}</span>
                  </button>
                ))}
              </div>
            </div>
            <aside className="p-5 sm:p-7" aria-live="polite">
              <div className="flex items-center justify-between gap-3"><Badge tone="red">{selected.eventId === flagship?.eventId ? "Flagship trace" : "Selected trace"}</Badge><span className="font-mono text-[10px] text-zinc-600" data-testid="selected-event-id">{selected.eventId}</span></div>
              <div className="mt-7 space-y-5">
                <div><p className="mono-label">Signed intent</p><p className="mt-2 text-lg font-medium text-cyan-100">{selected.intent ?? "No intent playback recorded in this event"}</p></div>
                <ArrowDown className="size-5 text-zinc-700" aria-hidden="true" />
                <div><p className="mono-label">Paid cart</p><p className="mt-2 text-lg font-medium text-[#ff8296]">{selected.cart ?? "No structured cart recorded in this event"}</p><p className="mt-1 text-xs text-zinc-500">{selected.payee} · {formatInrMinor(selected.amountMinor)}</p></div>
              </div>
              <div className="mt-8 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-white/10 bg-white/10">
                <div className="bg-[#101219] p-4"><p className="mono-label">Legacy</p><p className={cn("mt-3 text-xl font-semibold", selected.legacyAction === "ALLOW" ? "text-emerald-300" : "text-amber-200")}>{selected.legacyAction} <CheckCircle2 className="inline size-5" aria-hidden="true" /></p><p className="mt-2 text-xs leading-5 text-zinc-500">Credentials and amount checks pass.</p></div>
                <div className="bg-[#101219] p-4"><p className="mono-label">Intent integrity</p><p className={cn("mt-3 text-xl font-semibold", selected.action === "BLOCK" ? "text-[#ff6d84]" : "text-cyan-200")}>{selected.action} {selected.action === "BLOCK" ? <X className="inline size-5" aria-hidden="true" /> : <ShieldCheck className="inline size-5" aria-hidden="true" />}</p><p className="mt-2 text-xs leading-5 text-zinc-500">Five evidence layers vote together.</p></div>
              </div>
              <div className="policy-divergence mt-4">
                <p className="mono-label text-cyan-200">Same signed payment, different policy</p>
                <p><strong>Legacy {selected.legacyAction}.</strong> Intent Integrity {selected.action} after comparing the paid cart with the delegated mission.</p>
                <a href="#injection">Inspect the causal seam <ArrowDown className="size-4" aria-hidden="true" /></a>
              </div>
              <ul className="mt-6 space-y-2 text-xs leading-5 text-zinc-400">
                {selected.reasonCodes.map((reason) => <li key={reason} className="flex gap-2"><Fingerprint className="mt-0.5 size-3.5 shrink-0 text-cyan-300" aria-hidden="true" />{reason}</li>)}
              </ul>
            </aside>
          </div>
        ) : <ErrorBlock title="Stream is empty" />}
      </motion.div>
    </>
  )
}

function caseIntegrityChecks(data: CaseData) {
  return [
    ["User signature", data.userSignatureValid],
    ["Merchant signature", data.merchantSignatureValid],
    ["Payment signature", data.paymentSignatureValid],
    ["Cart hash", data.cartHashMatches],
  ] as const
}

function MandateScreen({ state }: { state: ResourceState<CaseData> }) {
  const allIntegrityChecksPass = state.data ? caseIntegrityChecks(state.data).every(([, valid]) => valid) : undefined
  return (
    <section id="mandate" className="story-section" aria-labelledby="mandate-title">
      <div className="story-shell">
        <SectionHeader id="mandate-title" index={4} kicker="Mandate chain" title={allIntegrityChecksPass === undefined ? "Synthetic identity and intent integrity are checked separately." : allIntegrityChecksPass ? "Every recorded signature and hash check holds. Meaning is checked separately." : "A recorded integrity check failed."} deck="The release-custodied scorer matches public-seed synthetic signer keys to its manifest registry. That fixture-only match does not prove real-world authorization or semantic alignment." happening="Follow the recorded intent, cart and payment values left to right, then inspect the signature and hash checks. The agent registry flag is a recorded claim, not proof." source={state.source} />
        <ResourceView state={state} label="flagship case">
          {(data) => <MandateContent data={data} />}
        </ResourceView>
      </div>
    </section>
  )
}

function MandateContent({ data }: { data: CaseData }) {
  const root = useRef<HTMLDivElement>(null)
  const reduce = useReducedMotion()
  const inView = useInView(root, { once: true, amount: 0.2 })
  useGSAP(() => {
    if (reduce || !inView) return
    gsap.from(".drift-line", { drawSVG: 0, duration: 1.25, ease: "power2.out" })
    gsap.from(".drift-wedge", { scaleY: 0, transformOrigin: "left center", duration: 0.9, ease: "power3.out" })
  }, { scope: root, dependencies: [reduce, inView], revertOnUpdate: true })
  const integrityChecks = caseIntegrityChecks(data)
  const intentCategory = data.intentCategory.replaceAll("_", " ")
  const nodes: Node[] = [
    { id: "intent", position: { x: 0, y: 55 }, data: { label: `01 · INTENT\n${intentCategory} · ${formatInrMinor(data.intentBudgetMinor)}` }, style: { width: 180, borderColor: "rgba(59,201,255,.55)" } },
    { id: "cart", position: { x: 250, y: 20 }, data: { label: `02 · CART\n${data.cartItems.join(" + ")} · ${formatInrMinor(data.cartTotalMinor)}` }, style: { width: 200, borderColor: "rgba(255,59,92,.55)" } },
    { id: "payment", position: { x: 530, y: 0 }, data: { label: `03 · PAYMENT\n${formatInrMinor(data.paymentAmountMinor)} · ${data.payee}` }, style: { width: 220, borderColor: "rgba(255,59,92,.75)" } },
  ]
  const edges: Edge[] = [
    { id: "i-c", source: "intent", target: "cart", style: { stroke: "#ff3b5c" } },
    { id: "c-p", source: "cart", target: "payment", style: { stroke: "#ff3b5c" } },
  ]
  const mandateSteps = [
    { label: "Authorized intent", value: data.promptPlayback, meta: `Budget ${formatInrMinor(data.intentBudgetMinor)}`, tone: "intent" },
    { label: "Recorded cart", value: data.cartItems.join(" + "), meta: `${formatInrMinor(data.cartTotalMinor)} · MCC ${data.cartMcc}`, tone: "drift" },
    { label: "Recorded payment", value: data.payee, meta: `${formatInrMinor(data.paymentAmountMinor)} · ${data.presenceFlag}`, tone: "drift" },
  ] as const
  return (
    <>
      <motion.div {...panelEnter} ref={root} className="story-panel overflow-hidden">
        <div className="grid lg:grid-cols-[1.35fr_.65fr]">
          <div className="min-w-0 border-b border-white/10 p-4 sm:p-6 lg:border-b-0 lg:border-r">
            <ol className="mandate-mobile-chain" aria-label="Intent to cart to payment sequence">
              {mandateSteps.map((step, index) => <li key={step.label} data-tone={step.tone}><span className="mandate-step-index">0{index + 1}</span><div><p>{step.label}</p><strong>{step.value}</strong><small>{step.meta}</small></div>{index < mandateSteps.length - 1 ? <ArrowDown aria-hidden="true" /> : null}</li>)}
            </ol>
            <div className="flow-canvas mandate-flow-desktop h-[280px] w-full" data-lenis-prevent>
              <ReactFlow id="mandate-chain" nodes={nodes} edges={edges} fitView nodesDraggable={false} nodesConnectable={false} edgesFocusable={false} elementsSelectable={false} panOnScroll zoomOnPinch minZoom={0.45} maxZoom={1.5} aria-label="Intent to cart to payment mandate chain">
                <Background gap={24} size={1} color="#333640" />
                <DiagramControls label="Mandate diagram controls" />
              </ReactFlow>
            </div>
            <div className="relative mt-5 overflow-hidden border border-white/10 bg-black/25 p-5">
              <svg viewBox="0 0 900 170" className="w-full" role="img" aria-label="Widening semantic drift from intent to payment">
                <defs><linearGradient id="wedge-red" x1="0" x2="1"><stop offset="0" stopColor="#ff3b5c" stopOpacity="0.05" /><stop offset="1" stopColor="#ff3b5c" stopOpacity="0.45" /></linearGradient></defs>
                <path className="drift-wedge" d="M40 86 L850 20 L850 152 Z" fill="url(#wedge-red)" />
                <path className="drift-line" d="M40 86 C270 86 560 70 850 20" fill="none" stroke="#3bc9ff" strokeWidth="3" />
                <path className="drift-line" d="M40 86 C300 90 560 120 850 152" fill="none" stroke="#ff3b5c" strokeWidth="3" />
                <text x="40" y="70" fill="#bcecff" fontSize="16">user intent</text>
                <text x="750" y="142" fill="#ff9aac" fontSize="16">paid outcome</text>
              </svg>
            </div>
          </div>
          <div className="p-5 sm:p-7">
            <p className="mono-label">Cryptographic checks</p>
            <div className="mt-5 space-y-3">
              {integrityChecks.map(([label, valid]) => (
                <div key={label} className="flex items-center justify-between gap-3 border-b border-white/[0.08] py-3">
                  <span className="text-sm text-zinc-300">{label}</span>
                  <Badge tone={valid ? "green" : "red"}>{valid ? <Check className="size-3" /> : <X className="size-3" />}{valid ? "verified" : "failed"}</Badge>
                </div>
              ))}
            </div>
            <a href="#semantic-mismatch" className="mt-7 inline-flex min-h-11 items-center gap-2 rounded-md border border-[#ff3b5c]/40 px-4 text-xs font-semibold uppercase tracking-[0.1em] text-[#ff9aac] hover:bg-[#ff3b5c]/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#ff3b5c]">Inspect semantic mismatch <ArrowDown className="size-4" aria-hidden="true" /></a>
            <div id="semantic-mismatch" className="mt-5 scroll-mt-24 border-l-2 border-[#ff3b5c] bg-[#ff3b5c]/[0.07] p-4">
              <p className="mono-label text-[#ff8296]">Semantic invariant</p>
              <p className="mt-3 text-sm leading-6 text-zinc-300">{data.promptPlayback}</p>
              <p className="mt-1 font-mono text-[10px] leading-5 text-zinc-500">Category {intentCategory} · budget {formatInrMinor(data.intentBudgetMinor)} · MCC {data.intentMccs.join(", ") || "none recorded"} · allowed payees {data.intentPayeeAllowlist.join(", ") || "none recorded"}</p>
              <ArrowDown className="my-3 size-4 text-[#ff3b5c]" aria-hidden="true" />
              <p className="text-sm leading-6 text-[#ff9aac]">{data.cartItems.join(" + ")} · {formatInrMinor(data.cartTotalMinor)} · MCC {data.cartMcc}</p>
              <p className="mt-1 font-mono text-[10px] leading-5 text-zinc-500">Payment {formatInrMinor(data.paymentAmountMinor)} → {data.payee} · {data.presenceFlag}</p>
            </div>
            <p className="mt-6 font-mono text-xs leading-5 text-zinc-500">ECDSA answers “was this object altered?” Intent integrity asks “was this still the right object to create?”</p>
          </div>
        </div>
      </motion.div>
    </>
  )
}

function InjectionScreen({ state }: { state: ResourceState<CaseData> }) {
  const [revealed, setRevealed] = useState(false)
  return (
    <section id="injection" className="story-section" aria-labelledby="injection-title">
      <div className="story-shell">
        <SectionHeader id="injection-title" index={5} kicker="Injection reveal" title="The page looked harmless. The agent read another layer." deck="Rendered pixels can say one thing while the retrieval context says another. Provenance evidence exposes that invisible instruction seam." happening="Toggle Agent vision. The safe, truncated red layer is what the agent retrieved but a human shopper would not have seen." source={state.source} />
        <ResourceView state={state} label="retrieved page evidence">
          {(data) => {
            const hasRetrievalArtifact = data.zeroWidthChars > 0 || data.invisibleCss || Boolean(data.hiddenTextSample)
            return (
              <>
              <motion.div {...panelEnter} className="story-panel overflow-hidden">
                <div className="grid lg:grid-cols-[.72fr_1.28fr]">
                  <div className="border-b border-white/10 p-5 sm:p-7 lg:border-b-0 lg:border-r">
                    <Badge tone={revealed ? "red" : "blue"}>{revealed ? <Eye className="size-3" /> : <EyeOff className="size-3" />}{revealed ? "Agent vision" : "Human vision"}</Badge>
                    <h3 className="mt-6 text-3xl font-semibold tracking-tight">Reveal the context seam.</h3>
                    <p className="mt-3 text-sm leading-6 text-zinc-400">The control is keyboard-operable. The exemplar stays truncated, watermarked and non-operational.</p>
                    <div className="mt-8 border-y border-white/10 py-4">
                      <Switch id="injection-reveal" checked={revealed} onCheckedChange={setRevealed} label="Reveal hidden injection" description="Show machine-visible retrieval artifacts" />
                    </div>
                    <dl className="mt-7 space-y-4 text-sm">
                      <div className="flex justify-between gap-4"><dt className="text-zinc-500">Zero-width chars</dt><dd className="font-mono text-[#ff8296]">{data.zeroWidthChars}</dd></div>
                      <div className="flex justify-between gap-4"><dt className="text-zinc-500">Invisible CSS</dt><dd className="font-mono text-[#ff8296]">{data.invisibleCss ? "DETECTED" : "NONE"}</dd></div>
                      <div className="flex justify-between gap-4"><dt className="text-zinc-500">Hidden artifact</dt><dd className="font-mono text-[#ff8296]">{hasRetrievalArtifact ? "RECORDED" : "NONE"}</dd></div>
                    </dl>
                    <div className="causal-proof mt-8" role="group" aria-label="Recorded injection causal proof">
                      <p className="mono-label text-cyan-200">Causal proof</p>
                      <div><span>Trusted intent</span><strong>{data.promptPlayback}</strong></div>
                      <div><span>Machine-only seam</span><strong>{data.zeroWidthChars} zero-width chars{data.invisibleCss ? " plus invisible CSS" : ""}</strong></div>
                      <div><span>Recorded outcome</span><strong>{data.cartItems.join(" + ")} to {data.payee}</strong></div>
                      <p>The signatures can remain valid while the object being authorized changes.</p>
                    </div>
                  </div>
                  <div className="relative min-h-[520px] bg-[#f5f3ec] p-5 text-[#191919] sm:p-10">
                    <div className="mx-auto max-w-xl">
                      <div className="flex items-center justify-between border-b border-black/10 pb-4"><span className="text-xs font-bold uppercase tracking-[0.14em]">{data.merchantId}</span><ShoppingBag className="size-5" /></div>
                      <div className="mt-8 grid gap-7 sm:grid-cols-[.8fr_1.2fr]">
                        <div className="aspect-square bg-[linear-gradient(135deg,#d7d4ca,#efede6)] p-5"><div className="grid size-full place-items-center border border-black/10"><Waves className="size-20 text-[#252525]" strokeWidth={1} aria-hidden="true" /></div></div>
                        <div><p className="text-xs font-bold uppercase tracking-[0.15em] text-black/65">Recorded product page</p><h3 className="mt-2 text-3xl font-semibold leading-tight">{data.productName}</h3><p className="mt-4 break-all font-mono text-xs leading-6 text-black/60">{data.productPage}</p><p className="mt-6 text-2xl font-semibold">{formatInrMinor(data.productPriceMinor)}</p><button type="button" disabled aria-describedby="recorded-case-note" className="mt-6 min-h-11 w-full cursor-not-allowed bg-black px-4 text-sm font-semibold uppercase tracking-[0.12em] text-white opacity-60">Add to mandate</button><p id="recorded-case-note" className="mt-2 text-xs leading-5 text-black/65">Read-only recorded evidence; this control cannot alter the mandate.</p></div>
                      </div>
                    </div>
                    <AnimatePresence>
                      {revealed ? (
                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="injection-layer absolute inset-0 grid place-items-center bg-[#16080d] p-6 text-white" data-testid="injection-layer">
                          <div className="w-full max-w-2xl border border-[#ff3b5c]/55 bg-black/45 p-5 shadow-[0_0_70px_rgba(255,59,92,.2)] sm:p-8">
                            <div className="flex items-center gap-2 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-[#ff8296]"><AlertTriangle className="size-4" /> machine-visible retrieval layer</div>
                            <p className="mt-6 font-mono text-sm leading-7 text-[#ffb3bf]">{data.hiddenTextSample || "No hidden text sample was recorded for this page."}</p>
                            <div className="mt-6 grid gap-px overflow-hidden border border-white/10 bg-white/10 sm:grid-cols-2">
                              <div className="bg-[#100c11] p-4"><p className="mono-label">Authorized</p><p className="mt-2 text-sm leading-6 text-zinc-300">{data.promptPlayback}</p></div>
                              <div className="bg-[#190a10] p-4"><p className="mono-label text-[#ff9aac]">Recorded outcome</p><p className="mt-2 text-sm leading-6 text-[#ffb3bf]">{data.cartItems.join(" + ")} · {data.payee}</p></div>
                            </div>
                            <p className="mt-4 border-l-2 border-cyan-300 pl-3 text-sm leading-6 text-cyan-100">Control point: verify semantic alignment before payment authorization.</p>
                            <div className="mt-7 flex flex-wrap gap-2">{data.zeroWidthChars > 0 ? <Badge tone="red">zero-width × {data.zeroWidthChars}</Badge> : null}{data.invisibleCss ? <Badge tone="red">invisible CSS</Badge> : null}{hasRetrievalArtifact ? <Badge tone="red">retrieval artifact recorded</Badge> : <Badge>no hidden artifact recorded</Badge>}</div>
                            <p className="mt-7 border-t border-white/10 pt-4 font-mono text-[10px] uppercase tracking-[0.13em] text-zinc-500">Watermarked safety preview · synthetic · payload truncated</p>
                          </div>
                        </motion.div>
                      ) : null}
                    </AnimatePresence>
                  </div>
                </div>
              </motion.div>
              </>
            )
          }}
        </ResourceView>
      </div>
    </section>
  )
}

function PrecedentScreen({ state }: { state: ResourceState<PrecedentData> }) {
  const suppressed = state.data?.verdict === "suppressed"
  return (
    <section id="precedent" className="story-section" aria-labelledby="precedent-title">
      <div className="story-shell">
        <SectionHeader id="precedent-title" index={6} kicker="Precedent memory" title="Anomaly without context becomes noise." deck="Nearest verified legitimate intent→purchase pairs add measured context to a borderline decision." happening={!state.data ? "Inspect the recorded precedent result and its effect on the defensive action." : suppressed ? "The recorded precedent result reduced the baseline action and marked the candidate false positive as suppressed." : "The recorded precedent result confirmed the anomaly; the contextualized action remains defensive."} source={state.source} />
        <ResourceView state={state} label="precedent memory">
          {(data) => {
            return (
              <>
              <motion.div {...panelEnter} className="story-panel overflow-hidden">
                <div className="grid lg:grid-cols-[.65fr_1.35fr]">
                  <div className="border-b border-white/10 p-5 sm:p-7 lg:border-b-0 lg:border-r">
                    <Badge tone="blue"><Database className="size-3" /> Legitimate-only index</Badge>
                    <p className="mono-label mt-8">Borderline query</p>
                    <h3 className="mt-3 text-2xl font-semibold leading-tight">{data.query}</h3>
                    <div className="mt-8"><Stat label="Precedent support" value={formatPercent(data.support)} note="k-nearest verified legitimate pairs" /></div>
                    <div className="precedent-decision-flow mt-8" role="group" aria-label={`Decision changes from ${data.baselineAction} to ${data.contextualizedAction}`}>
                      <div className="precedent-decision-card"><p className="mono-label">Raw drift</p><p className="mt-2 text-lg font-semibold text-amber-200">{data.baselineAction}</p></div>
                      <ArrowRight className="precedent-decision-arrow" aria-hidden="true" />
                      <div className="precedent-decision-card"><p className="mono-label">Contextualized</p><p className={cn("mt-2 text-lg font-semibold", suppressed ? "text-emerald-300" : "text-[#ff8296]")}>{data.contextualizedAction}</p></div>
                    </div>
                    <div className={cn("mt-5 flex items-center gap-2 text-sm font-medium", suppressed ? "text-emerald-300" : "text-[#ff8296]")}>{suppressed ? <ShieldCheck className="size-4" /> : <AlertTriangle className="size-4" />}{suppressed ? "False positive suppressed" : "Anomaly confirmed"}</div>
                  </div>
                  <div>
                    <div className="border-b border-white/10 px-5 py-4 font-mono text-[10px] uppercase tracking-[0.15em] text-zinc-500">Nearest legitimate precedents · similarity descending</div>
                    <ol className="divide-y divide-white/[0.08]">
                      {data.neighbors.map((neighbor, index) => (
                        <li key={neighbor.title} className="grid gap-4 p-5 sm:grid-cols-[60px_1fr_130px] sm:items-center sm:p-7">
                          <span className="font-mono text-3xl text-zinc-700">0{index + 1}</span>
                          <div><h4 className="font-medium text-zinc-100">{neighbor.title}</h4><p className="mt-1 text-sm leading-6 text-zinc-500">{neighbor.context}</p><Badge tone="green" className="mt-3">{neighbor.outcome}</Badge></div>
                          <div className="sm:text-right"><span className="font-mono text-2xl text-cyan-200">{formatPercent(neighbor.similarity)}</span><span className="block font-mono text-[9px] uppercase tracking-[0.12em] text-zinc-600">similarity</span></div>
                        </li>
                      ))}
                    </ol>
                  </div>
                </div>
              </motion.div>
              <div className="mt-5 flex flex-col justify-between gap-4 border-l-2 border-cyan-300/45 bg-cyan-300/[0.04] p-5 sm:flex-row sm:items-center sm:p-6">
                <p className="max-w-2xl text-sm leading-6 text-zinc-300"><strong className="text-cyan-100">Why the action changes:</strong> the legitimate-only neighbors supply context for this borderline query; they do not rewrite the recorded anomaly or its evidence.</p>
                <a href="#arena" className="inline-flex min-h-11 shrink-0 items-center justify-center gap-2 rounded-md border border-cyan-300/40 px-4 text-xs font-semibold uppercase tracking-[0.1em] text-cyan-200 hover:bg-cyan-300/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300">Continue to Arena <ArrowDown className="size-4" aria-hidden="true" /></a>
              </div>
              </>
            )
          }}
        </ResourceView>
      </div>
    </section>
  )
}

function ArenaScreen({ arenaState, phylogenyState, vaccinationState }: { arenaState: ResourceState<ArenaData>; phylogenyState: ResourceState<PhylogenyData>; vaccinationState: ResourceState<VaccinationData> }) {
  const allReady = [arenaState, phylogenyState, vaccinationState].every((state) => state.status === "ready")
  return (
    <section id="arena" className="story-section" aria-labelledby="arena-title">
      <div className="story-shell">
        <SectionHeader id="arena-title" index={7} kicker="Red vs Blue arena" title="Zero replay ASR is a floor, not proof of generalization." deck="Training-fold replay can reach zero while mutation candidates and fresh-row risk remain." happening="Each arena round mutates only against the training fold. The replay curve measures reduction on that fold; surviving mutations still need fresh-row testing. Partition D is post-remediation regression evidence with independent_holdout=false." source={allReady ? arenaState.source : undefined} />
        <ResourceView state={arenaState} label="arena log">
          {(arena) => (
            <ResourceView state={phylogenyState} label="fraud phylogeny">
              {(phylogeny) => (
                <ResourceView state={vaccinationState} label="guarded vaccination trace">
                  {(vaccination) => <ArenaContent arena={arena} phylogeny={phylogeny} vaccination={vaccination} />}
                </ResourceView>
              )}
            </ResourceView>
          )}
        </ResourceView>
      </div>
    </section>
  )
}

function ArenaContent({ arena, phylogeny, vaccination }: { arena: ArenaData; phylogeny: PhylogenyData; vaccination: VaccinationData }) {
  const root = useRef<HTMLDivElement>(null)
  const reduce = useReducedMotion()
  const inView = useInView(root, { once: true, amount: 0.12 })
  const path = useMemo(() => {
    const count = Math.max(1, arena.rounds.length - 1)
    return arena.rounds.map((row, index) => `${index ? "L" : "M"} ${35 + (index / count) * 530} ${25 + (1 - row.asr) * 170}`).join(" ")
  }, [arena.rounds])
  useGSAP(() => {
    if (reduce || !inView) return
    const timeline = gsap.timeline({ defaults: { ease: "power2.out" } })
    timeline.from(".arena-red", { x: -35, autoAlpha: 0, duration: 0.45 }).from(".arena-blue", { x: 35, autoAlpha: 0, duration: 0.45 }, "<").from(".asr-path", { drawSVG: 0, duration: 1.6 }, "-=0.12").from(".asr-dot", { scale: 0, stagger: 0.12, duration: 0.25 }, "-=0.65")
  }, { scope: root, dependencies: [reduce, inView, path], revertOnUpdate: true })

  const depthOf = (nodeId: string) => {
    let depth = 0
    let current = phylogeny.nodes.find((node) => node.id === nodeId)
    while (current?.parent && depth < 6) {
      depth += 1
      current = phylogeny.nodes.find((node) => node.id === current?.parent)
    }
    return depth
  }
  const flowNodes: Node[] = phylogeny.nodes.map((node, index) => {
    const depth = depthOf(node.id)
    const peers = phylogeny.nodes.filter((candidate) => depthOf(candidate.id) === depth)
    const peerIndex = peers.findIndex((candidate) => candidate.id === node.id)
    return {
      id: node.id,
      position: { x: depth * 190, y: peerIndex * 95 + (index % 2) * 8 },
      data: { label: node.emergent ? `${node.label}\nEMERGENT` : node.label },
      style: { width: 145, borderColor: node.emergent ? "#ff3b5c" : "rgba(59,201,255,.4)", color: node.emergent ? "#ff9aac" : "#d4f3ff" },
    }
  })
  const flowEdges: Edge[] = phylogeny.nodes.filter((node) => node.parent).map((node) => ({ id: `${node.parent}-${node.id}`, source: node.parent!, target: node.id, style: { stroke: node.emergent ? "#ff3b5c" : "#3bc9ff" } }))
  const survivorHighlights = arena.survivors.filter((item) => !item.includes("+")).slice(0, 6)
  survivorHighlights.push(...arena.survivors.filter((item) => item.includes("+")).slice(0, 6 - survivorHighlights.length))
  const additionalSurvivors = arena.survivors.filter((item) => !survivorHighlights.includes(item))
  const emergentCount = phylogeny.nodes.filter((node) => node.emergent).length
  const finalAttackSuccess = arena.rounds.at(-1)?.asr ?? 0
  const mobilePhylogeny = phylogeny.nodes.filter((node) => node.emergent).slice(0, 4)

  return (
    <>
      <motion.div {...panelEnter} ref={root} className="space-y-5">
        <div className="story-panel overflow-hidden">
          <div className="grid lg:grid-cols-[.55fr_1.45fr_.55fr] lg:items-stretch">
            <div className="arena-red flex min-h-48 flex-col items-center justify-center border-b border-white/10 bg-[#ff3b5c]/[0.045] p-6 text-center lg:border-b-0 lg:border-r">
              <div className="grid size-20 place-items-center rounded-full border border-[#ff3b5c]/50 bg-[#ff3b5c]/10 text-[#ff6d84] shadow-[0_0_45px_rgba(255,59,92,.18)]"><Swords className="size-9" /></div>
              <p className="mt-4 font-mono text-xs font-semibold uppercase tracking-[0.16em] text-[#ff8296]">Red mutator</p>
              <p className="mt-2 text-xs leading-5 text-zinc-500">payoff × evasion fitness</p>
            </div>
            <div className="min-w-0 p-4 sm:p-7">
              <div className="flex items-center justify-between gap-3"><span className="mono-label">Attack success rate</span><Badge tone="amber">training-fold only</Badge></div>
              <svg viewBox="0 0 600 230" className="mt-4 w-full" role="img" aria-label={`Attack success rate falls from ${formatPercent(arena.rounds[0]?.asr ?? 0)} to ${formatPercent(arena.rounds.at(-1)?.asr ?? 0)}`}>
                {[0, 0.25, 0.5, 0.75, 1].map((tick) => <g key={tick}><line x1="35" x2="565" y1={25 + (1 - tick) * 170} y2={25 + (1 - tick) * 170} stroke="rgba(255,255,255,.08)" /><text x="4" y={29 + (1 - tick) * 170} fill="#62636d" fontSize="10">{Math.round(tick * 100)}%</text></g>)}
                <path className="asr-path" data-testid="asr-path" d={path} fill="none" stroke="#ff3b5c" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
                {arena.rounds.map((row, index) => {
                  const x = 35 + (index / Math.max(1, arena.rounds.length - 1)) * 530
                  const y = 25 + (1 - row.asr) * 170
                  return <g key={row.round} className="asr-dot"><circle cx={x} cy={y} r="6" fill="#08090d" stroke="#ff3b5c" strokeWidth="3" /><text x={x} y="220" textAnchor="middle" fill="#777984" fontSize="10">R{row.round}</text></g>
                })}
              </svg>
              <div className="flex justify-between font-mono text-xs"><span className="text-[#ff8296]">{formatPercent(arena.rounds[0]?.asr ?? 0)} initial</span><span className="text-cyan-200">{formatPercent(arena.rounds.at(-1)?.asr ?? 0)} final replay</span></div>
                <div className="arena-verdict mt-5">
                  <div><span>Training replay verdict</span><strong>{formatPercent(finalAttackSuccess)} replay ASR does not clear {arena.survivors.length} surviving mutation candidates or fresh-row risk.</strong><p className="mt-2 text-xs leading-5 text-zinc-400">ASR is attack success on the training fold. Surviving mutations stay unresolved until the fresh-row regression is inspected; promotion remains gated.</p></div>
                  <a href="#vault">View fresh-row regression evidence <ArrowDown className="size-4" aria-hidden="true" /></a>
                </div>
            </div>
            <div className="arena-blue flex min-h-48 flex-col items-center justify-center border-t border-white/10 bg-cyan-300/[0.035] p-6 text-center lg:border-l lg:border-t-0">
              <div className="grid size-20 place-items-center rounded-full border border-cyan-300/50 bg-cyan-300/10 text-cyan-200 shadow-[0_0_45px_rgba(59,201,255,.16)]"><Shield className="size-9" /></div>
              <p className="mt-4 font-mono text-xs font-semibold uppercase tracking-[0.16em] text-cyan-200">Blue defense</p>
              <p className="mt-2 text-xs leading-5 text-zinc-500">guarded promotion only</p>
            </div>
          </div>
          <div className="grid border-t border-white/10 lg:grid-cols-[1.4fr_.6fr]">
            <div className="min-w-0 border-b border-white/10 p-4 sm:p-6 lg:border-b-0 lg:border-r">
              <div className="mb-4 flex items-center justify-between"><span className="mono-label">Fraud phylogeny</span><Badge tone="red"><GitBranch className="size-3" /> {emergentCount} emergent</Badge></div>
              <ol className="phylogeny-mobile-summary" aria-label="Emergent fraud mutation summary">
                {mobilePhylogeny.map((node, index) => <li key={node.id}><span>0{index + 1}</span><strong>{node.label}</strong></li>)}
                {emergentCount > mobilePhylogeny.length ? <li className="phylogeny-more">+{emergentCount - mobilePhylogeny.length} additional emergent mutations in the full map</li> : null}
              </ol>
              <div className="flow-canvas phylogeny-flow-desktop h-80" data-lenis-prevent>
                <ReactFlow id="arena-phylogeny" nodes={flowNodes} edges={flowEdges} fitView nodesDraggable={false} nodesConnectable={false} edgesFocusable={false} panOnScroll zoomOnPinch minZoom={0.4} maxZoom={1.5} aria-label="Fraud mutation phylogeny">
                  <Background gap={22} size={1} color="#343640" />
                  <DiagramControls label="Fraud phylogeny controls" />
                </ReactFlow>
              </div>
            </div>
            <div className="p-5 sm:p-6">
              <p className="mono-label">Surviving mutations</p>
              <p className="mt-2 text-xs leading-5 text-zinc-500">Candidates that survive mutation search remain unresolved until tested outside the replay fold.</p>
              <ol className="mt-4 space-y-3" data-testid="survivor-highlights">{survivorHighlights.map((item, index) => <li key={item} className="flex gap-3 border-b border-white/[0.08] pb-3 text-sm leading-5 text-zinc-300"><span className="font-mono text-[#ff6d84]">{String(index + 1).padStart(2, "0")}</span>{item}</li>)}</ol>
              {additionalSurvivors.length ? (
                <details className="mt-3 border-t border-white/10" data-testid="additional-survivors">
                  <summary className="min-h-11 cursor-pointer py-3 font-mono text-[10px] font-semibold uppercase tracking-[0.12em] text-cyan-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300">
                    Show {additionalSurvivors.length} additional survivors
                  </summary>
                  <ol className="space-y-3 pb-2">{additionalSurvivors.map((item, index) => <li key={item} className="flex gap-3 border-b border-white/[0.08] pb-3 text-sm leading-5 text-zinc-400"><span className="font-mono text-[#ff6d84]">{String(index + survivorHighlights.length + 1).padStart(2, "0")}</span>{item}</li>)}</ol>
                </details>
              ) : null}
            </div>
          </div>
        </div>
        <div className="story-panel overflow-hidden p-5 sm:p-7">
          <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center"><div><p className="mono-label">Guarded vaccination · {vaccination.candidate}</p><p className="mt-2 text-sm text-zinc-400">Attack traffic never updates the live model by itself.</p></div><Badge tone={vaccination.liveModelUnchanged ? "amber" : "green"}>{vaccination.modelVersion} · {vaccination.liveModelUnchanged ? "live unchanged" : "promoted"}</Badge></div>
          <div className="mt-6 grid gap-3 md:grid-cols-5">
            {vaccination.stages.map((stage, index) => <div key={stage.name} className="relative border border-white/10 bg-white/[0.025] p-4"><span className="font-mono text-[9px] text-zinc-600">0{index + 1}</span><p className="mt-2 text-xs font-medium leading-5 text-zinc-200">{stage.name}</p><Badge tone={stage.status === "promoted" ? "blue" : stage.status === "pending" ? "amber" : stage.status === "rejected" ? "red" : "green"} className="mt-3">{stage.status === "pending" ? "pending review" : stage.status}</Badge>{index < vaccination.stages.length - 1 ? <ArrowRight className="absolute -right-5 top-1/2 z-10 hidden size-4 text-zinc-700 md:block" /> : null}</div>)}
          </div>
        </div>
      </motion.div>
    </>
  )
}

function VaultScreen({ state }: { state: ResourceState<GeneralizationData> }) {
  const [open, setOpen] = useState(false)
  const reduce = useReducedMotion()
  return (
    <section id="vault" className="story-section overflow-hidden" aria-labelledby="vault-title">
      <div className="story-shell">
        <SectionHeader id="vault-title" index={8} kicker="Evaluation scope" title="Fresh rows after remediation - not an independent holdout." deck="This is a fresh-row post-remediation regression. independent_holdout=false. Prior families were disclosed before remediation." happening="Use the scope control to compare fresh-row composition and fresh-row D recall. The gap below compares known recall with D recall only." source={state.source} />
        <ResourceView state={state} label="fresh-row regression evidence">
          {(data) => {
            const evidence = data.recallEvidence
            const recallRows = evidence ? [
              ["Known", evidence.known],
              ["Adaptive", evidence.adaptive],
              ["Fresh-row composition", evidence.novelComposition],
              ["Fresh-row D", evidence.heldoutFamily],
            ] as const : []
            const lowest = recallRows.reduce<(typeof recallRows)[number] | undefined>((current, row) => !current || row[1].recall < current[1].recall ? row : current, undefined)
            return (
              <>
              <motion.div {...panelEnter} className="story-panel overflow-hidden">
                <div className="grid lg:grid-cols-[.9fr_1.1fr]">
                  <div className="vault-door relative grid min-h-[300px] place-items-center overflow-hidden border-b border-white/10 p-6 sm:min-h-[420px] lg:border-b-0 lg:border-r">
                    <motion.div data-testid="vault-door-panel" animate={{ rotateY: open ? -72 : 0, x: open ? -30 : 0, opacity: open ? 0.45 : 1 }} transition={reduce ? { duration: 0 } : { duration: 0.7, ease: [0.22, 1, 0.36, 1] }} className="relative grid aspect-square w-full max-w-sm place-items-center rounded-full border-2 border-white/15 bg-[#0b0d12] shadow-[inset_0_0_80px_rgba(0,0,0,.65),0_28px_80px_rgba(0,0,0,.5)]">
                      <div className="absolute inset-[12%] rounded-full border border-cyan-300/20" />
                      <div className="absolute inset-[24%] rounded-full border border-dashed border-white/20" />
                      <div className="grid size-24 place-items-center rounded-full border border-[#ff3b5c]/45 bg-[#ff3b5c]/10 text-[#ff8296]"><LockKeyhole className="size-10" /></div>
                      <span className="absolute bottom-[18%] font-mono text-[10px] uppercase tracking-[0.16em] text-zinc-500">D · FRESH-ROW REGRESSION</span>
                    </motion.div>
                    <Button variant={open ? "outline" : "attacker"} size="lg" onClick={() => setOpen((value) => !value)} aria-expanded={open} aria-controls="vault-evidence" className="absolute bottom-6 w-[calc(100%-2rem)] whitespace-normal text-center text-xs leading-4 sm:bottom-8 sm:w-auto" data-testid="vault-toggle">
                      {open ? <><LockKeyhole className="size-4" /> Hide regression results</> : <><Unlock className="size-4" /> Show regression results</>}
                    </Button>
                  </div>
                  <div id="vault-evidence" className="relative min-h-0 p-5 sm:min-h-[520px] sm:p-10" aria-live="polite">
                    <AnimatePresence mode="wait">
                      {!open ? (
                        <motion.div key="closed" initial={reduce ? false : { opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={reduce ? { duration: 0 } : undefined} className="flex h-full min-h-[280px] flex-col items-center justify-center text-center sm:min-h-[440px]">
                          <Vault className="size-12 text-zinc-700" aria-hidden="true" />
                          <p className="mt-5 max-w-sm text-xl font-medium text-zinc-300">Frozen weights. Fresh rows. Previously disclosed families.</p>
                          <p className="mt-3 font-mono text-xs leading-6 text-zinc-600">One consumed attempt · no threshold retuning · independent_holdout=false</p>
                        </motion.div>
                      ) : (
                        <motion.div key="open" initial={reduce ? false : { opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={reduce ? { duration: 0 } : undefined} data-testid="vault-result">
                          <Badge tone="green"><ShieldCheck className="size-3" /> fresh-row regression · frozen weights</Badge>
                          <div className="mt-5 border-l-2 border-amber-300/60 bg-amber-300/[0.06] p-4" data-testid="evaluation-scope-disclosure"><p className="break-words font-mono text-xs text-amber-200">Fresh-row post-remediation regression · independent_holdout=false · prior families disclosed</p><p className="mt-2 break-words text-xs leading-5 text-zinc-500">GPF-A20 / ttl_dormancy · GPF-C04 / synthetic_identity_bust_out</p></div>
                          <div className={cn("vault-recall-grid mt-8 grid gap-px overflow-hidden border border-white/10 bg-white/10", evidence ? "sm:grid-cols-3" : "sm:grid-cols-2")}>
                            {lowest ? <div className="bg-[#160f12] p-5 sm:p-7"><p className="mono-label text-amber-200">Lowest measured recall</p><p className="mt-4 font-mono text-[clamp(2.8rem,7vw,5.5rem)] font-semibold leading-none text-amber-200">{formatPercent(lowest[1].recall)}</p><p className="mt-3 text-xs leading-5 text-zinc-500">{lowest[1].caught}/{lowest[1].positiveExamples} positives · 95% CI {formatPercent(lowest[1].ci95Low, 1)}-{formatPercent(lowest[1].ci95High, 1)}</p></div> : null}
                            <div className="bg-[#101219] p-5 sm:p-7"><p className="mono-label">Fresh-row D recall</p><p className="mt-4 font-mono text-[clamp(2.8rem,7vw,5.5rem)] font-semibold leading-none text-cyan-200 blue-glow" data-testid="heldout-recall">{formatPercent(data.heldoutFamilyRecall)}</p><p className="mt-3 text-xs leading-5 text-zinc-500">Post-remediation measurement</p></div>
                            <div className="bg-[#101219] p-5 sm:p-7"><p className="mono-label">Known recall</p><p className="mt-4 font-mono text-[clamp(2.8rem,7vw,5.5rem)] font-semibold leading-none text-zinc-200">{formatPercent(data.knownRecall)}</p><p className="mt-3 text-xs leading-5 text-zinc-500">Training-scope comparison</p></div>
                          </div>
                          <div className="mt-5 border-l-2 border-amber-300/60 bg-amber-300/[0.06] p-4"><p className="font-mono text-xs text-amber-200">Fresh-row D gap (known vs D): {formatPercent(data.knownRecall - data.heldoutFamilyRecall)}</p><p className="mt-2 text-xs leading-5 text-zinc-500">The composition and D rows are separate fresh-row checks; this number compares known recall with D recall only.</p></div>
                          <div className="mt-5 grid grid-cols-2 gap-px border border-white/10 bg-white/10 sm:grid-cols-4" role="group" aria-label="Fresh-row regression recall sequence">
                            {(recallRows.length ? recallRows.map(([label, row]) => [label, row.recall] as const) : [["Known", data.knownRecall], ["Adaptive", data.adaptiveRecall], ["Fresh-row composition", data.novelCompositionRecall], ["Fresh-row D", data.heldoutFamilyRecall]] as const).map(([label, value]) => <div key={String(label)} className="bg-[#101219] p-3"><p className="font-mono text-[8px] uppercase tracking-[0.11em] text-zinc-600">{label}</p><p className="mt-1 font-mono text-sm text-zinc-200">{formatPercent(Number(value))}</p></div>)}
                          </div>
                          {evidence ? <div className="mt-5 border border-white/10" role="region" aria-label="Recall evidence counts and confidence intervals"><table className="hidden w-full text-left text-xs sm:table"><thead className="border-b border-white/10 font-mono text-[9px] uppercase tracking-[0.12em] text-zinc-600"><tr><th className="p-3">Scope</th><th className="p-3">Caught / positives</th><th className="p-3">Recall</th><th className="p-3">95% CI</th></tr></thead><tbody>{recallRows.map(([label, row]) => <tr key={label} className="border-b border-white/[0.07] last:border-0"><th className="p-3 font-medium text-zinc-300">{label}</th><td className="p-3 font-mono text-zinc-400">{row.caught} / {row.positiveExamples}</td><td className="p-3 font-mono text-cyan-200">{formatPercent(row.recall)}</td><td className="p-3 font-mono text-zinc-400">{formatPercent(row.ci95Low, 1)}-{formatPercent(row.ci95High, 1)}</td></tr>)}</tbody></table><div className="space-y-2 p-3 sm:hidden">{recallRows.map(([label, row]) => <p key={label} className="break-words text-xs leading-5 text-zinc-300"><strong>{label}:</strong> <span className="font-mono text-zinc-400">{row.caught}/{row.positiveExamples}</span> · {formatPercent(row.recall)} · CI {formatPercent(row.ci95Low, 1)}-{formatPercent(row.ci95High, 1)}</p>)}</div></div> : null}
                          <div className="mt-7 space-y-3 font-mono text-[10px] leading-5 text-zinc-500">
                            <p className="flex items-start gap-2"><CheckCircle2 className="mt-0.5 size-3.5 shrink-0 text-emerald-300" />{data.firewallStatus}</p>
                            <p className="flex min-w-0 items-start gap-2"><Fingerprint className="mt-0.5 size-3.5 shrink-0 text-cyan-300" /><span className="min-w-0 break-all">Frozen weights: {data.frozenWeightsHash}</span></p>
                            <p className="flex items-start gap-2"><Clock3 className="mt-0.5 size-3.5 shrink-0 text-cyan-300" />{data.splitKind}</p>
                            <p className="flex items-start gap-2"><FlaskConical className="mt-0.5 size-3.5 shrink-0 text-cyan-300" />Label shuffle PR-AUC {formatPercent(data.shuffledLabelPRAuc, 2)} ≈ base rate {formatPercent(data.baseRate, 2)}</p>
                          </div>
                          <div className="mt-6 border border-amber-300/25 bg-amber-300/[0.05] p-4"><p className="mono-label text-amber-200">Deployment decision</p><p className="mt-2 text-sm leading-6 text-zinc-300">This regression is evidence of stability, not production readiness. An independent post-remediation holdout is still required.</p><a href="#unknowns" className="mt-3 inline-flex min-h-11 items-center gap-2 font-mono text-[10px] uppercase tracking-[0.12em] text-cyan-200 hover:text-cyan-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300">Review independent-holdout requirement <ArrowDown className="size-3.5" aria-hidden="true" /></a></div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </div>
              </motion.div>
              </>
            )
          }}
        </ResourceView>
      </div>
    </section>
  )
}

const tooltipStyle = { background: "#0d0f15", border: "1px solid rgba(255,255,255,.15)", borderRadius: "8px", fontFamily: "var(--font-mono)", fontSize: "11px" }
const tooltipTextStyle = { color: "#f4f4f5" }

function MetricsScreen({ state }: { state: ResourceState<MetricsData> }) {
  return (
    <section id="metrics" className="story-section" aria-labelledby="metrics-title">
      <div className="story-shell">
        <SectionHeader id="metrics-title" index={9} kicker="Evaluation ledger" title="Accuracy is the wrong question." deck="PR-AUC, recall at constrained false-positive rates, calibration, money and latency tell the real operating story." happening="Read each metric at its operating constraint. The ablation rows report measured PR-AUC and fresh-row D recall for each configuration." source={state.source} />
        <ResourceView state={state} label="evaluation metrics">
          {(data) => <MetricsContent data={data} />}
        </ResourceView>
      </div>
    </section>
  )
}

function MetricsContent({ data }: { data: MetricsData }) {
  const reduce = useReducedMotion()
  const recallBars = [
    { label: "@ 0.1% FPR", value: data.recallAtPointOneFpr },
    { label: "@ 1% FPR", value: data.recallAtOneFpr },
  ]
  const maxMatrix = Math.max(...data.confusionMatrix.flat())
  const baselineRecall = data.ablations[0]?.heldoutRecall
  const hasRecallDrop = baselineRecall !== undefined && data.ablations.slice(1).some((row) => baselineRecall - row.heldoutRecall > 0.000_001)
  const weakestAblation = data.ablations.reduce<(typeof data.ablations)[number] | undefined>(
    (lowest, row) => !lowest || row.prAuc < lowest.prAuc ? row : lowest,
    undefined,
  )
  const ablationGap = weakestAblation ? Math.max(0, data.prAuc - weakestAblation.prAuc) : 0
  const prCurve = [...data.prCurve].sort((left, right) => left.recall - right.recall)
  return (
    <>
      <motion.div {...panelEnter} className="space-y-5">
        <div className="story-panel grid gap-px overflow-hidden bg-white/10 sm:grid-cols-2 lg:grid-cols-4">
          <div className="bg-[#101219] p-5"><Stat label="PR-AUC" value={data.prAuc.toFixed(3)} note="precision-recall area" /></div>
          <div className="bg-[#101219] p-5"><Stat label="Brier" value={data.brier.toFixed(3)} note="lower calibration error" tone="neutral" /></div>
          <div className="bg-[#101219] p-5"><Stat label="Alerts / 1,000" value={data.alertsPerThousand.toFixed(1)} note="operational load" tone="neutral" /></div>
          <div className="bg-[#101219] p-5"><Stat label="p99 latency" value={`${data.p99LatencyMs.toFixed(1)} ms`} note={`A+B benchmark · p50 ${data.p50LatencyMs.toFixed(1)} ms · 128 warmed feature-ready scores · HTTP measured separately · ${data.p99LatencyMs < 100 ? "under 100 ms target" : "above target"}`} tone={data.p99LatencyMs < 100 ? "blue" : "red"} /></div>
        </div>
        <div className="grid gap-5 lg:grid-cols-[1.25fr_.75fr]">
          <div className="story-panel p-4 sm:p-6">
            <div className="flex items-center justify-between"><p className="mono-label">Precision-recall curve</p><Badge tone="blue">PR, not accuracy</Badge></div>
            <p className="mt-2 text-xs leading-5 text-zinc-400">As recall rises, precision is the tradeoff to watch; the constrained operating points are listed beside the curve.</p>
            <div className="mt-4 h-80 w-full" role="img" aria-label={`Precision recall curve with PR-AUC ${data.prAuc.toFixed(3)}`}>
              <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                <AreaChart data={prCurve} accessibilityLayer margin={{ top: 10, right: 12, bottom: 8, left: -8 }}>
                  <defs><linearGradient id="pr-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#3bc9ff" stopOpacity={0.35} /><stop offset="100%" stopColor="#3bc9ff" stopOpacity={0.01} /></linearGradient></defs>
                  <CartesianGrid stroke="rgba(255,255,255,.08)" vertical={false} />
                  <XAxis dataKey="recall" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} stroke="#62636d" tick={{ fontSize: 10 }} />
                  <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} stroke="#62636d" tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={tooltipStyle} itemStyle={tooltipTextStyle} labelStyle={tooltipTextStyle} formatter={(value) => formatPercent(Number(value), 1)} labelFormatter={(value) => `Recall ${formatPercent(Number(value), 1)}`} />
                  <Area type="monotone" dataKey="precision" stroke="#3bc9ff" strokeWidth={3} fill="url(#pr-fill)" isAnimationActive={!reduce} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-axis-key" aria-hidden="true"><span>Y · precision</span><span>X · recall</span></div>
          </div>
          <div className="story-panel p-4 sm:p-6">
            <p className="mono-label">Recall under false-positive caps</p>
            <p className="mt-2 text-xs leading-5 text-zinc-400">These are the recall levels the operation gets at fixed false-positive budgets, not an unconstrained accuracy score.</p>
            <div className="mt-4 h-80 w-full" role="img" aria-label="Recall comparison at false-positive-rate thresholds">
              <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                <BarChart data={recallBars} layout="vertical" accessibilityLayer margin={{ top: 18, right: 18, bottom: 18, left: 10 }}>
                  <CartesianGrid stroke="rgba(255,255,255,.08)" horizontal={false} />
                  <XAxis type="number" domain={[0, 1]} hide />
                  <YAxis type="category" dataKey="label" width={82} tick={{ fill: "#a1a1aa", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={tooltipStyle} itemStyle={tooltipTextStyle} labelStyle={tooltipTextStyle} formatter={(value) => formatPercent(Number(value))} />
                  <Bar dataKey="value" radius={[0, 5, 5, 0]} isAnimationActive={!reduce}>{recallBars.map((row, index) => <Cell key={row.label} fill={index ? "#3bc9ff" : "#ff3b5c"} />)}</Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-3 grid grid-cols-2 gap-2">{recallBars.map((row) => <div key={row.label} className="border-l border-cyan-300/35 pl-3"><span className="block font-mono text-[9px] uppercase tracking-[0.1em] text-zinc-500">{row.label}</span><strong className="mt-1 block font-mono text-lg text-zinc-100">{formatPercent(row.value)}</strong></div>)}</div>
          </div>
        </div>
        <div className="grid gap-5 lg:grid-cols-[.75fr_1.25fr]">
          <div className="story-panel p-5 sm:p-6">
            <p className="mono-label">Confusion matrix · counts{data.detectionThreshold === undefined ? "" : ` at threshold ${data.detectionThreshold.toFixed(2)}`}</p>
            <table className="mt-5 w-full table-fixed border-separate border-spacing-2 text-center" aria-label="Confusion matrix heatmap">
              <thead><tr><th className="p-2" /><th className="p-2 font-mono text-[9px] uppercase text-zinc-600">Pred legit</th><th className="p-2 font-mono text-[9px] uppercase text-zinc-600">Pred fraud</th></tr></thead>
              <tbody>{data.confusionMatrix.map((row, rowIndex) => <tr key={rowIndex}><th className="p-2 text-left font-mono text-[9px] uppercase text-zinc-600">{rowIndex ? "Fraud" : "Legit"}</th>{row.map((value, colIndex) => <td key={colIndex} className="h-24 border border-white/10 font-mono text-lg font-semibold" style={{ backgroundColor: rowIndex === colIndex ? `rgba(59,201,255,${0.08 + value / maxMatrix * 0.28})` : `rgba(255,59,92,${0.08 + value / maxMatrix * 0.28})` }}>{formatCompact(value)}</td>)}</tr>)}</tbody>
            </table>
          </div>
          <div id="metrics-ablation" className="story-panel scroll-mt-24 overflow-hidden">
            <div className="border-b border-white/10 p-5 sm:p-6"><p className="mono-label">{hasRecallDrop ? "Ablation · measured fresh-row D recall drop" : "Ablation · fresh-row D recall comparison"}</p></div>
            {weakestAblation ? <div className="border-b border-white/10 bg-cyan-300/[0.035] p-5 text-sm leading-6 text-zinc-300 sm:px-6"><strong className="text-cyan-100">Measured takeaway:</strong> {weakestAblation.layer} records PR-AUC {weakestAblation.prAuc.toFixed(3)}, {ablationGap.toFixed(3)} below the full model.</div> : null}
            <dl id="ablation-definitions" className="grid gap-4 border-b border-white/10 bg-white/[0.02] p-5 text-xs leading-5 text-zinc-400 sm:grid-cols-2 sm:px-6" aria-label="Ablation metric definitions">
              <div><dt className="font-semibold text-zinc-200">Fresh-row D</dt><dd className="mt-1">Recall on partition D rows generated after remediation. It is regression evidence with independent_holdout=false, not an independent evaluation.</dd></div>
              <div><dt className="font-semibold text-zinc-200">D recall delta</dt><dd className="mt-1">This row's Fresh-row D recall minus the full model recall. Negative means recall fell after removing that layer; no change means no measured D-recall difference.</dd></div>
            </dl>
            <div className="ablation-scroll overflow-x-auto focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300" data-lenis-prevent tabIndex={0} role="region" aria-label="Ablation results table"><table className="w-full text-left" aria-describedby="ablation-definitions"><thead><tr className="border-b border-white/10 font-mono text-[9px] uppercase tracking-[0.12em] text-zinc-600"><th className="p-4 sm:px-6">Configuration</th><th className="p-4">PR-AUC</th><th className="p-4">Fresh-row D recall</th><th className="p-4 sm:pr-6">D recall delta</th></tr></thead><tbody>{data.ablations.map((row, index) => {
              const delta = baselineRecall === undefined ? 0 : row.heldoutRecall - baselineRecall
              const deltaLabel = index === 0 ? "baseline" : Math.abs(delta) <= 0.000_001 ? "no change" : `${delta > 0 ? "+" : "-"}${formatPercent(Math.abs(delta), 1)}`
              return <tr key={row.layer} className="border-b border-white/[0.07] last:border-0"><td className="p-4 text-sm font-medium sm:px-6" data-label="Configuration">{row.layer}</td><td className="p-4 font-mono text-cyan-200" data-label="PR-AUC">{row.prAuc.toFixed(3)}</td><td className="p-4 font-mono text-zinc-200" data-label="Fresh-row D recall">{formatPercent(row.heldoutRecall)}</td><td className="p-4 font-mono text-[#ff8296] sm:pr-6" data-label="Delta" data-testid="ablation-delta">{deltaLabel}</td></tr>
            })}</tbody></table></div>
            <div className="grid grid-cols-2 gap-px border-t border-white/10 bg-white/10"><div className="bg-[#101219] p-5"><Stat label="Fraud value caught" value={formatPercent(data.fraudValueCaught, 1)} /></div><div className="bg-[#101219] p-5"><Stat label="Good value blocked" value={formatPercent(data.goodValueBlocked, 2)} tone="neutral" /></div></div>
          </div>
        </div>
        <a href="#metrics-ablation" className="inline-flex min-h-11 items-center gap-2 rounded-md border border-cyan-300/35 px-4 text-xs font-semibold uppercase tracking-[0.1em] text-cyan-200 hover:bg-cyan-300/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300">Compare configurations <ArrowDown className="size-4" aria-hidden="true" /></a>
      </motion.div>
    </>
  )
}

function EconomicsScreen({ state }: { state: ResourceState<MetricsData> }) {
  const measuredRoi = state.data?.economics.after.roi
  return (
    <section id="economics" className="story-section" aria-labelledby="economics-title">
      <div className="story-shell">
        <SectionHeader id="economics-title" index={10} kicker="Attacker economics" title="Defense compresses profitability. Deterrence is not yet proven." deck={measuredRoi === undefined ? "The ledger measures economic pressure without claiming campaign victory." : measuredRoi < 0 ? "Measured attacker ROI is below break-even in this simulation; production deterrence is not established." : `Measured attacker ROI remains ${measuredRoi.toFixed(2)}×, above break-even. The defense applies pressure but has not removed the incentive.`} happening="The ledger translates measured simulation outcomes into attacker unit economics. It does not report a live Mastercard or payment-network result." source={state.source} />
        <ResourceView state={state} label="attacker economics">
          {(data) => {
            const { before, after } = data.economics
            const belowBreakEven = after.roi < 0
            const beforeCostPerSuccess = before.costPerSuccessCensored || before.observedSuccesses === 0 ? "Censored estimate omitted" : `₹${before.costPerSuccess.toLocaleString("en-IN")}`
            const afterCostPerSuccess = after.costPerSuccessCensored || after.observedSuccesses === 0 ? "No observed successes; censored estimate omitted" : `₹${after.costPerSuccess.toLocaleString("en-IN")}`
            const rows = [
              ["Cost per attempt", `₹${before.costPerAttack.toLocaleString("en-IN")}`, `₹${after.costPerAttack.toLocaleString("en-IN")}`],
              ["Success rate", formatPercent(before.successRate), formatPercent(after.successRate)],
              ["Observed successes", before.observedSuccesses?.toLocaleString("en-IN") ?? "Not reported", after.observedSuccesses?.toLocaleString("en-IN") ?? "Not reported"],
              ["Cost per successful fraud", beforeCostPerSuccess, afterCostPerSuccess],
              ["Attacker ROI", `${before.roi.toFixed(2)}×`, `${after.roi.toFixed(2)}×`],
            ]
            return (
              <>
                <motion.div {...panelEnter} className="story-panel overflow-hidden">
                  <div className="grid lg:grid-cols-[1.35fr_.65fr]">
                    <div className="border-b border-white/10 lg:border-b-0 lg:border-r">
                      <div className="grid grid-cols-[1fr_100px_100px] border-b border-white/10 px-4 py-4 font-mono text-[9px] uppercase tracking-[0.13em] text-zinc-600 sm:grid-cols-[1fr_180px_180px] sm:px-7"><span>Attacker ledger</span><span className="text-right text-[#ff8296]">Before</span><span className="text-right text-cyan-200">After Blue</span></div>
                      {rows.map(([label, beforeValue, afterValue]) => <div key={label} className="grid min-h-24 grid-cols-[1fr_100px_100px] items-center border-b border-white/[0.08] px-4 sm:grid-cols-[1fr_180px_180px] sm:px-7"><span className="text-sm text-zinc-300">{label}</span><span className="text-right font-mono text-sm text-[#ff8296] sm:text-xl">{beforeValue}</span><span className={cn("text-right font-mono text-sm sm:text-xl", label === "Cost per successful fraud" ? "text-amber-200" : "text-cyan-200")}>{afterValue}</span></div>)}
                    </div>
                    <div className="flex flex-col justify-center p-7 sm:p-10">
                      <Badge tone={belowBreakEven ? "blue" : "red"}>{belowBreakEven ? "Below break-even" : "Above break-even"}</Badge>
                      <div className="mx-auto grid size-44 place-items-center rounded-full border border-cyan-300/30 bg-cyan-300/[0.05] text-center shadow-[0_0_60px_rgba(59,201,255,.1)]"><div><Gauge className="mx-auto size-8 text-cyan-200" /><p className="mt-3 font-mono text-4xl font-semibold text-cyan-200">{after.roi.toFixed(2)}×</p><p className="mt-1 font-mono text-[9px] uppercase tracking-[0.14em] text-zinc-500">attacker ROI</p></div></div>
                      <p className="mt-8 text-center text-2xl font-semibold leading-tight">{belowBreakEven ? <>This simulation records attacker ROI <span className="text-cyan-200">below break-even.</span></> : <>Defense compresses modeled profitability. Measured post-defense ROI: <span className="text-cyan-200">{after.roi.toFixed(2)}×.</span> It remains above break-even.</>}</p>
                      <p className="mt-4 text-center font-mono text-[10px] leading-5 text-zinc-500">ROI = successful fraud value / total attack cost - 1. Break-even is 0×.</p>
                      <a href="#network-control" className="mt-6 inline-flex min-h-11 items-center justify-center gap-2 rounded-md border border-cyan-300/35 px-4 text-center text-xs font-semibold uppercase tracking-[0.1em] text-cyan-200 hover:bg-cyan-300/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300">Next control: authorization intervention <ArrowDown className="size-4" aria-hidden="true" /></a>
                    </div>
                  </div>
                </motion.div>
                <aside id="network-control" className="story-panel scroll-mt-24 p-5 sm:p-7" aria-labelledby="network-control-title">
                  <div className="grid gap-5 lg:grid-cols-[.7fr_1.3fr] lg:items-start">
                    <div>
                      <Badge tone="amber"><Network className="size-3" aria-hidden="true" /> Proposed, not deployed</Badge>
                      <h3 id="network-control-title" className="mt-4 text-xl font-semibold tracking-tight text-zinc-100">Mastercard brief mapping: authorization-time intent signal</h3>
                      <p className="mt-3 text-sm leading-6 text-zinc-400">No live Mastercard or payment-network integration or outcome is claimed.</p>
                    </div>
                    <dl className="grid gap-4 sm:grid-cols-3">
                      <div className="border-l border-cyan-300/30 pl-4"><dt className="mono-label text-cyan-200">Intervention</dt><dd className="mt-2 text-sm leading-6 text-zinc-300">Pass signed-mandate match, calibrated risk, and reason codes into issuer authorization decisioning before approval.</dd></div>
                      <div className="border-l border-cyan-300/30 pl-4"><dt className="mono-label text-cyan-200">Stakeholders</dt><dd className="mt-2 text-sm leading-6 text-zinc-300">Cardholder, issuer, merchant or acquirer, and payment-network risk teams.</dd></div>
                      <div className="border-l border-cyan-300/30 pl-4"><dt className="mono-label text-cyan-200">Proposed outcome</dt><dd className="mt-2 text-sm leading-6 text-zinc-300">Inform approve, step-up, review, or decline decisions alongside existing controls. No production outcome is measured.</dd></div>
                    </dl>
                  </div>
                </aside>
                <details className="story-panel mt-5 overflow-hidden">
                  <summary className="cursor-pointer list-none p-5 font-mono text-[10px] font-semibold uppercase tracking-[0.13em] text-cyan-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 sm:p-7">Measured ledger inputs · metrics artifact</summary>
                  <div className="grid gap-4 border-t border-white/10 p-5 text-sm leading-6 text-zinc-300 sm:grid-cols-2 sm:p-7">
                    <div><p className="mono-label">Before defense</p><p className="mt-2">{before.observedSuccesses?.toLocaleString("en-IN") ?? "Not reported"} observed successes · {formatPercent(before.successRate)} success rate · ₹{before.costPerAttack.toLocaleString("en-IN")} per attempt</p></div>
                    <div><p className="mono-label">After Blue</p><p className="mt-2">{after.observedSuccesses?.toLocaleString("en-IN") ?? "Not reported"} observed successes · {formatPercent(after.successRate)} success rate · ₹{after.costPerAttack.toLocaleString("en-IN")} per attempt</p></div>
                    <p className="font-mono text-[10px] leading-5 text-zinc-500 sm:col-span-2">ROI = successful fraud value / total attack cost - 1. These are measured simulation inputs from the metrics artifact; they do not establish a production Mastercard outcome or causal deployment effect.</p>
                  </div>
                </details>
              </>
            )
          }}
        </ResourceView>
      </div>
    </section>
  )
}

const initialCompose: ComposeInput = {
  channel: "agentic_checkout",
  injection_style: "hidden_text",
  amount_minor: 575_463,
  hops: 2,
  delegation_depth: 2,
  ttl_delay_seconds: 0,
  fanout: 1,
  evasion_aggressiveness: 0.5,
  lie_about_presence: false,
}

function ComposerScreen() {
  const [input, setInput] = useState<ComposeInput>(initialCompose)
  const [result, setResult] = useState<Loaded<ComposeResult> | null>(null)
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle")
  const [error, setError] = useState("")
  const update = <K extends keyof ComposeInput>(key: K, value: ComposeInput[K]) => setInput((current) => ({ ...current, [key]: value }))
  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setStatus("loading")
    setError("")
    try {
      const response = await api.compose(input)
      setResult(response)
      setStatus("idle")
    } catch (caught) {
      setResult(null)
      setError(caught instanceof Error ? caught.message : "Composer request failed")
      setStatus("error")
    }
  }
  return (
    <section id="composer" className="story-section" aria-labelledby="composer-title">
      <div className="story-shell">
        <SectionHeader id="composer-title" index={11} kicker="Attack composer" title="Your turn, Red." deck="Change one synthetic event, then compare Legacy Rules with Intent Integrity on that same event." happening="Every control maps directly to POST /compose. LAUNCH sends one bounded synthetic request; the backend performs the only live inference in the demo." />
        <motion.div {...panelEnter} className="story-panel overflow-hidden">
          <div className="grid xl:grid-cols-[.95fr_1.05fr]">
            <form onSubmit={submit} className="border-b border-white/10 p-5 sm:p-7 xl:border-b-0 xl:border-r" data-testid="composer-form">
              <fieldset disabled={status === "loading"}>
                <legend className="flex w-full items-center justify-between border-b border-white/10 pb-4"><span className="font-mono text-xs font-semibold uppercase tracking-[0.15em] text-[#ff8296]">Synthetic attack controls</span><Badge tone="red"><Swords className="size-3" /> Red mode</Badge></legend>
                <div className="mt-6 grid gap-6 sm:grid-cols-2">
                  <SelectField id="compose-channel" label="Channel" value={input.channel} onChange={(value) => update("channel", value as ComposeInput["channel"])}>
                    <option value="agentic_checkout">Agentic checkout</option><option value="card_ecom">Card e-commerce</option><option value="upi_p2p">UPI P2P</option><option value="upi_collect">UPI collect</option><option value="wire">Wire</option>
                  </SelectField>
                  <SelectField id="compose-injection" label="Injection style" value={input.injection_style} onChange={(value) => update("injection_style", value as ComposeInput["injection_style"])}>
                    <option value="hidden_text">Hidden text</option><option value="zero_width">Zero-width seam</option><option value="instruction_shift">Instruction shift</option><option value="none">No injection</option>
                  </SelectField>
                </div>
                <div className="mt-8 space-y-7">
                  <Slider id="compose-amount" label="Amount" value={input.amount_minor} onValueChange={(value) => update("amount_minor", value)} min={10_000} max={100_000_000} step={10_000} valueLabel={formatInrMinor(input.amount_minor)} />
                  <div className="grid gap-7 sm:grid-cols-2">
                    <Slider id="compose-hops" label="Tool hops" value={input.hops} onValueChange={(value) => update("hops", value)} min={0} max={8} step={1} valueLabel={`${input.hops}`} />
                    <Slider id="compose-depth" label="Delegation depth" value={input.delegation_depth} onValueChange={(value) => update("delegation_depth", value)} min={0} max={8} step={1} valueLabel={`${input.delegation_depth}`} />
                    <Slider id="compose-ttl" label="TTL delay" value={input.ttl_delay_seconds} onValueChange={(value) => update("ttl_delay_seconds", value)} min={0} max={604_800} step={86_400} valueLabel={`${Math.round(input.ttl_delay_seconds / 86_400)} d`} />
                    <Slider id="compose-fanout" label="Fan-out" value={input.fanout} onValueChange={(value) => update("fanout", value)} min={1} max={20} step={1} valueLabel={`${input.fanout} payee${input.fanout === 1 ? "" : "s"}`} />
                  </div>
                  <Slider id="compose-evasion" label="Evasion aggressiveness" value={input.evasion_aggressiveness} onValueChange={(value) => update("evasion_aggressiveness", value)} min={0} max={1} step={0.05} valueLabel={formatPercent(input.evasion_aggressiveness)} />
                  <div className="border-y border-white/10 py-3"><Switch id="compose-presence" checked={input.lie_about_presence} onCheckedChange={(value) => update("lie_about_presence", value)} label="Lie about human presence" description="Claim PRESENT while the trace remains autonomous" /></div>
                </div>
                <Button type="submit" variant="attacker" size="lg" className="mt-8 w-full" data-testid="launch-composer">
                  {status === "loading" ? <><RefreshCw className="size-4 animate-spin motion-reduce:animate-none" /> Scoring frozen model…</> : <><Zap className="size-4" /> Launch event</>}
                </Button>
                <p className="mt-3 text-center font-mono text-[9px] uppercase tracking-[0.12em] text-zinc-600">Synthetic only · bounded inputs · no operational payload generated</p>
              </fieldset>
            </form>
            <div className="relative min-h-[360px] bg-black/15 p-5 sm:p-7 xl:min-h-[680px]" aria-live="polite">
              <AnimatePresence mode="wait">
                {status === "error" ? <motion.div key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }}><ErrorBlock title="Inference unavailable" detail={error} /></motion.div> : result ? (
                  <motion.div key={result.data.eventId + result.data.decision.latencyMs} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} data-testid="composer-result">
                    <div className="flex flex-wrap items-center justify-between gap-3"><SourceBadge source={result.source} /><Badge tone="blue"><Clock3 className="size-3" /> {result.data.decision.latencyMs.toFixed(1)} ms</Badge></div>
                    <p className="mt-6 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-zinc-400">Same event, two policy decisions</p>
                    <div className="mt-3 grid grid-cols-2 gap-px overflow-hidden border border-white/10 bg-white/10">
                      <div className="bg-[#101219] p-5"><p className="mono-label">Legacy rules</p><p className={cn("mt-3 text-3xl font-semibold", result.data.decision.legacyAction === "ALLOW" ? "text-emerald-300" : "text-amber-200")}>{result.data.decision.legacyAction}</p></div>
                      <div className="bg-[#101219] p-5"><p className="mono-label">Intent integrity</p><p className={cn("mt-3 text-3xl font-semibold", result.data.decision.action === "BLOCK" ? "text-[#ff6d84]" : result.data.decision.action === "HOLD" ? "text-amber-200" : "text-cyan-200")}>{result.data.decision.action}</p></div>
                    </div>
                    <div className="mt-7 flex items-end justify-between border-b border-white/10 pb-4"><div><p className="mono-label">Fused risk</p><p className="mt-2 font-mono text-5xl font-semibold text-[#ff6d84]">{Math.round(result.data.scores.fused * 100)}<span className="text-lg text-zinc-600">/100</span></p></div><ShieldX className="size-12 text-[#ff3b5c]/60" /></div>
                    <div className="mt-6 space-y-4">
                      {(Object.entries(result.data.scores) as Array<[keyof ComposeResult["scores"], number]>).filter(([key]) => key !== "fused").map(([key, value]) => (
                        <div key={key}><div className="mb-2 flex justify-between font-mono text-[10px] uppercase tracking-[0.12em]"><span className="text-zinc-500">{key}</span><span className="text-zinc-300">{formatPercent(value)}</span></div><div className="h-1.5 overflow-hidden rounded-full bg-white/10"><motion.div initial={{ width: 0 }} animate={{ width: `${value * 100}%` }} className={cn("h-full", key === "provenance" || key === "structural" ? "bg-[#ff3b5c]" : "bg-cyan-300")} /></div></div>
                      ))}
                    </div>
                    <div className="mt-8"><p className="mono-label">Why Blue decided</p><ul className="mt-4 space-y-3">{result.data.decision.reasonCodes.map((reason) => <li key={reason} className="flex gap-3 text-sm leading-6 text-zinc-300"><Fingerprint className="mt-1 size-4 shrink-0 text-cyan-300" />{reason}</li>)}</ul></div>
                    <p className="mt-7 border-l-2 border-cyan-300 bg-cyan-300/[0.05] p-4 text-sm leading-6 text-zinc-300"><strong className="text-cyan-100">Control outcome:</strong> Intent Integrity returned {result.data.decision.action} after fusing five recorded evidence layers for this event.</p>
                  </motion.div>
                ) : (
                  <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex min-h-[300px] flex-col items-center justify-center text-center xl:min-h-[620px]">
                    <div className="grid size-28 place-items-center rounded-full border border-[#ff3b5c]/30 bg-[#ff3b5c]/[0.05]"><Radar className="size-12 text-[#ff3b5c]/70" /></div>
                    <h3 className="mt-7 text-2xl font-semibold">Awaiting synthetic event.</h3>
                    <p className="mt-3 max-w-sm text-sm leading-6 text-zinc-500">Tune Red’s path, then launch. One inference returns five layer scores, policy actions and reason codes.</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

function isMeasuredMiss(item: MetricsData["knownUnknowns"][number]) {
  return (item.missCount !== undefined && item.missCount > 0) || (item.recall !== undefined && item.recall < 1)
}

function limitationTitle(title: string) {
  return title === "Residual evidence gap: GPF-A21" ? "Single-intent multi-merchant fanout (GPF-A21)" : title
}

function UnknownsScreen({ state }: { state: ResourceState<MetricsData> }) {
  const hasMeasuredMisses = state.data?.knownUnknowns.some(isMeasuredMiss)
  const limitationCount = state.data?.knownUnknowns.length
  const title = hasMeasuredMisses === undefined
    ? "Known unknowns and residual evidence gaps."
    : hasMeasuredMisses
      ? limitationCount === 1 ? "One observed miss." : "Observed misses and residual evidence gaps."
      : limitationCount === 1 ? "One residual evidence gap." : "Residual evidence gaps."
  return (
    <section id="unknowns" className="story-section" aria-labelledby="unknowns-title">
      <div className="story-shell">
        <SectionHeader id="unknowns-title" index={12} kicker="Known unknowns" title={title} deck="A serious defense separates measured misses from evidence-quality caveats, then attaches each limitation to a concrete next control." happening="A row is marked as an observed miss only when the artifact reports a miss count or recall below 100%." source={state.source} />
        <ResourceView state={state} label="known unknowns">
          {(data) => {
            const observedMiss = data.knownUnknowns.find(isMeasuredMiss)
            return <>
              <div className="mb-5 grid gap-px overflow-hidden border border-white/10 bg-white/10 sm:grid-cols-3" role="group" aria-label="Residual evidence map">
                <div className="bg-[#160f12] p-5"><p className="mono-label text-[#ff9aac]">Observed signal</p><p className="mt-2 text-sm font-medium text-zinc-100">{observedMiss ? limitationTitle(observedMiss.title) : "Evidence gap recorded"}</p><p className="mt-2 text-xs leading-5 text-zinc-500">{observedMiss?.recall === undefined ? "The artifact records this limitation but exposes no structured numerator or denominator." : `Measured recall ${formatPercent(observedMiss.recall)}.`}</p></div>
                <div className="border-y border-dashed border-amber-300/35 bg-[#131116] p-5 sm:border-x sm:border-y-0"><p className="mono-label text-amber-200">Evidence seam</p><p className="mt-2 text-sm font-medium text-zinc-100">Sparse signal at authorization time</p><p className="mt-2 text-xs leading-5 text-zinc-500">A payment can remain below single-transaction thresholds while the mandate-level pattern is still uncertain.</p></div>
                <div className="bg-[#101219] p-5"><p className="mono-label text-cyan-200">Next control</p><p className="mt-2 text-sm font-medium text-zinc-100">Verify intent at authorization time</p><p className="mt-2 text-xs leading-5 text-zinc-500">Route the uncertain case to step-up, consent, and verified post-review evidence before approval.</p></div>
              </div>
              <motion.div {...panelEnter} className="story-panel overflow-hidden">
                <div className="divide-y divide-white/[0.08]">
                  {data.knownUnknowns.map((item, index) => {
                    const measuredMiss = isMeasuredMiss(item)
                    return <article key={item.title} className="grid gap-5 p-5 sm:p-7 lg:grid-cols-[90px_1fr_1fr] lg:items-start">
                      <div className="flex items-center gap-3 lg:block"><span className="font-mono text-4xl text-[#ff3b5c]/70">0{index + 1}</span><Badge tone={measuredMiss ? "red" : "amber"} className="lg:mt-3">{measuredMiss ? "observed miss" : "evidence gap"}</Badge></div>
                      <div><h3 className="text-xl font-semibold tracking-tight">{limitationTitle(item.title)}</h3><p className="mono-label mt-4 text-amber-200">Evidence boundary</p><p className="mt-2 text-sm leading-6 text-zinc-400">{item.why}</p><p className="mt-3 font-mono text-[10px] leading-5 text-zinc-500">Source: evaluation metrics artifact. {item.recall === undefined && item.missCount === undefined ? "No structured miss numerator or denominator is exposed for this row." : "Structured miss evidence is exposed for this row."}</p></div>
                      <div className="border-l border-cyan-300/30 pl-5"><p className="mono-label text-cyan-200">Next control</p><p className="mt-3 text-sm leading-6 text-zinc-300">{item.mitigation}</p></div>
                    </article>
                  })}
                </div>
                <div className="flex flex-col justify-between gap-5 border-t border-white/10 bg-cyan-300/[0.035] p-6 sm:flex-row sm:items-center sm:p-8"><div><p className="font-mono text-xs font-semibold uppercase tracking-[0.14em] text-cyan-200">Measured residual risk</p><p className="mt-2 max-w-2xl text-sm leading-6 text-zinc-400">The vault number is a floor we can defend, not a ceiling we pretend to have reached. Use the composer to probe one uncertain case at a time.</p></div><a href="#composer" className="inline-flex min-h-11 shrink-0 items-center justify-center gap-2 rounded-md border border-cyan-300/40 px-4 text-xs font-semibold uppercase tracking-[0.1em] text-cyan-200 hover:bg-cyan-300/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300">Test an uncertain case <ArrowDown className="size-4 rotate-180" aria-hidden="true" /></a></div>
              </motion.div>
              <footer className="mt-10 flex flex-col justify-between gap-4 border-t border-white/10 pt-6 font-mono text-[10px] uppercase tracking-[0.12em] text-zinc-600 sm:flex-row"><span>AI Defense Lab · Intent integrity</span><span>Synthetic PII only · attack examples truncated, watermarked, non-operational</span></footer>
            </>
          }}
        </ResourceView>
      </div>
    </section>
  )
}

function App() {
  const atlas = useResource("/atlas", api.atlas)
  const stream = useResource("/stream?n=8", api.stream)
  const metrics = useResource("/metrics", api.metrics)
  const generalization = useResource("/generalization", api.generalization)
  const arena = useResource("/arena", api.arena)
  const phylogeny = useResource("/phylogeny", api.phylogeny)
  const caseData = useResource("/case/evt_flagship", () => api.case("evt_flagship"))
  const precedent = useResource("/precedent/evt_borderline", () => api.precedent("evt_borderline"))
  const vaccination = useResource("/vaccination", api.vaccination)
  const resourceStates = [atlas, stream, metrics, generalization, arena, phylogeny, caseData, precedent, vaccination]
  const hasFixture = FIXTURE_MODE || resourceStates.some((state) => state.source === "fixture")

  return (
    <SmoothRoot>
      <StoryNav />
      {hasFixture ? <FixtureRail /> : null}
      <main id="main-content" tabIndex={-1} className={hasFixture ? "pt-6" : undefined}>
        <Hero state={caseData} />
        {caseData.status !== "loading" ? <>
          <AtlasScreen state={atlas} />
          <LiveWire state={stream} />
          <MandateScreen state={caseData} />
          <InjectionScreen state={caseData} />
          <PrecedentScreen state={precedent} />
          <ArenaScreen arenaState={arena} phylogenyState={phylogeny} vaccinationState={vaccination} />
          <VaultScreen state={generalization} />
          <MetricsScreen state={metrics} />
          <EconomicsScreen state={metrics} />
          <ComposerScreen />
          <UnknownsScreen state={metrics} />
        </> : null}
      </main>
    </SmoothRoot>
  )
}

export default App
