import * as React from "react"
import * as SliderPrimitive from "@radix-ui/react-slider"
import * as SwitchPrimitive from "@radix-ui/react-switch"
import { cva, type VariantProps } from "class-variance-authority"
import { AlertTriangle, Check, FlaskConical, LoaderCircle } from "lucide-react"
import type { DataSource } from "../types"
import { cn } from "../lib/utils"

const buttonVariants = cva(
  "inline-flex min-h-11 items-center justify-center gap-2 rounded-md border px-4 py-2 text-sm font-semibold tracking-[0.08em] uppercase transition-[background-color,border-color,color,transform,box-shadow] duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 focus-visible:ring-offset-2 focus-visible:ring-offset-[#08090d] disabled:pointer-events-none disabled:opacity-50 active:translate-y-px",
  {
    variants: {
      variant: {
        defender: "border-cyan-300/60 bg-cyan-300 text-[#071015] shadow-[0_0_28px_rgba(59,201,255,.18)] hover:bg-cyan-100",
        attacker: "border-[#ff3b5c] bg-[#ff3b5c] text-[#180409] shadow-[0_0_28px_rgba(255,59,92,.22)] hover:bg-[#ff637c]",
        outline: "border-white/18 bg-white/[0.035] text-white hover:border-cyan-300/50 hover:bg-cyan-300/[0.08]",
        ghost: "border-transparent bg-transparent text-zinc-300 hover:bg-white/[0.06] hover:text-white",
      },
      size: {
        default: "h-11",
        sm: "h-11 min-h-11 px-3 text-xs",
        lg: "h-14 px-7 text-base",
        icon: "size-11 px-0",
      },
    },
    defaultVariants: { variant: "outline", size: "default" },
  },
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, type = "button", ...props }, ref) => (
    <button ref={ref} type={type} className={cn(buttonVariants({ variant, size }), className)} {...props} />
  ),
)
Button.displayName = "Button"

export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: React.ReactNode
  tone?: "neutral" | "red" | "blue" | "green" | "amber"
  className?: string
}) {
  const tones = {
    neutral: "border-white/15 bg-white/[0.045] text-zinc-300",
    red: "border-[#ff3b5c]/40 bg-[#ff3b5c]/10 text-[#ff8397]",
    blue: "border-cyan-300/40 bg-cyan-300/10 text-cyan-200",
    green: "border-emerald-400/35 bg-emerald-400/10 text-emerald-300",
    amber: "border-amber-300/35 bg-amber-300/10 text-amber-200",
  }
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[10px] font-semibold uppercase tracking-[0.14em]", tones[tone], className)}>
      {children}
    </span>
  )
}

export function SourceBadge({ source }: { source?: DataSource }) {
  return source === "fixture" ? (
    <Badge tone="amber" className="shrink-0" data-testid="fixture-badge">
      <FlaskConical className="size-3" aria-hidden="true" /> Development fixture · not run evidence
    </Badge>
  ) : (
    <Badge tone="green" className="shrink-0" data-testid="api-badge">
      <Check className="size-3" aria-hidden="true" /> Results artifact
    </Badge>
  )
}

export function Switch({
  checked,
  onCheckedChange,
  id,
  label,
  description,
}: {
  checked: boolean
  onCheckedChange: (checked: boolean) => void
  id: string
  label: string
  description?: string
}) {
  return (
    <div className="flex min-h-12 items-center justify-between gap-4">
      <div>
        <label htmlFor={id} className="block cursor-pointer text-sm font-medium text-zinc-100">
          {label}
        </label>
        {description ? <p className="mt-0.5 text-xs leading-5 text-zinc-500">{description}</p> : null}
      </div>
      <SwitchPrimitive.Root
        id={id}
        checked={checked}
        onCheckedChange={onCheckedChange}
        className="relative h-11 w-14 shrink-0 rounded-full border border-white/15 bg-zinc-800 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 data-[state=checked]:border-[#ff3b5c]/60 data-[state=checked]:bg-[#ff3b5c]/35"
      >
        <SwitchPrimitive.Thumb className="block size-6 translate-x-1.5 rounded-full bg-zinc-300 shadow transition-transform data-[state=checked]:translate-x-[26px] data-[state=checked]:bg-white" />
      </SwitchPrimitive.Root>
    </div>
  )
}

export function Slider({
  value,
  onValueChange,
  min,
  max,
  step,
  id,
  label,
  valueLabel,
}: {
  value: number
  onValueChange: (value: number) => void
  min: number
  max: number
  step: number
  id: string
  label: string
  valueLabel: string
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between gap-3">
        <label id={`${id}-label`} className="text-sm font-medium text-zinc-200">
          {label}
        </label>
        <output className="font-mono text-xs text-cyan-200" htmlFor={id}>
          {valueLabel}
        </output>
      </div>
      <SliderPrimitive.Root
        id={id}
        value={[value]}
        onValueChange={([next]) => next !== undefined && onValueChange(next)}
        min={min}
        max={max}
        step={step}
        className="relative flex h-11 w-full touch-none select-none items-center"
      >
        <SliderPrimitive.Track className="relative h-1.5 grow overflow-hidden rounded-full bg-white/10">
          <SliderPrimitive.Range className="absolute h-full bg-gradient-to-r from-cyan-400 to-[#ff3b5c]" />
        </SliderPrimitive.Track>
        <SliderPrimitive.Thumb aria-labelledby={`${id}-label`} aria-valuetext={valueLabel} className="relative block size-11 rounded-full bg-transparent transition-transform before:absolute before:inset-2.5 before:rounded-full before:border-2 before:border-cyan-200 before:bg-[#08090d] before:shadow-[0_0_16px_rgba(59,201,255,.45)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 focus-visible:ring-offset-2 focus-visible:ring-offset-[#08090d]" />
      </SliderPrimitive.Root>
    </div>
  )
}

export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden="true" className={cn("animate-pulse rounded-md bg-white/[0.07]", className)} />
}

export function LoadingBlock({ label }: { label: string }) {
  return (
    <div className="flex min-h-56 items-center justify-center gap-3 rounded-xl border border-white/10 bg-white/[0.025] text-sm text-zinc-400" role="status">
      <LoaderCircle className="size-4 animate-spin text-cyan-300 motion-reduce:animate-none" aria-hidden="true" />
      Loading {label}…
    </div>
  )
}

export function ErrorBlock({ title, detail }: { title: string; detail?: string }) {
  return (
    <div className="rounded-xl border border-[#ff3b5c]/35 bg-[#ff3b5c]/[0.07] p-5" role="alert">
      <div className="flex items-center gap-2 text-sm font-semibold text-[#ff9aac]">
        <AlertTriangle className="size-4" aria-hidden="true" /> {title}
      </div>
      {detail ? <p className="mt-2 font-mono text-xs leading-5 text-zinc-400">{detail}</p> : null}
    </div>
  )
}

export function SelectField({
  id,
  label,
  value,
  onChange,
  children,
}: {
  id: string
  label: string
  value: string
  onChange: (value: string) => void
  children: React.ReactNode
}) {
  return (
    <div className="space-y-2">
      <label htmlFor={id} className="block text-sm font-medium text-zinc-200">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="min-h-11 w-full rounded-md border border-white/15 bg-[#0c0f14] px-3 text-sm text-white outline-none transition-colors focus:border-cyan-300 focus:ring-2 focus:ring-cyan-300/25"
      >
        {children}
      </select>
    </div>
  )
}

export function Stat({ label, value, note, tone = "blue" }: { label: string; value: string; note?: string; tone?: "blue" | "red" | "neutral" }) {
  const color = tone === "red" ? "text-[#ff6d84]" : tone === "blue" ? "text-cyan-200" : "text-white"
  return (
    <div className="border-l border-white/15 pl-4">
      <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-zinc-500">{label}</p>
      <p className={cn("mt-2 font-mono text-2xl font-semibold tracking-tight sm:text-3xl", color)}>{value}</p>
      {note ? <p className="mt-1 text-xs leading-5 text-zinc-500">{note}</p> : null}
    </div>
  )
}
