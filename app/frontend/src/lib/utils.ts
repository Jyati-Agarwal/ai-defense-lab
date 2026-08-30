import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import type { DecisionAction } from "../types"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function normalizeDecisionAction(value: unknown): DecisionAction {
  const action = typeof value === "string" ? value.toUpperCase().replace("-", "_") : ""
  if (action === "ALLOW" || action === "STEP_UP" || action === "HOLD" || action === "BLOCK") return action
  throw new Error("API artifact contains an invalid decision action")
}

export function formatPercent(value: number, digits = 0) {
  return new Intl.NumberFormat("en-IN", {
    style: "percent",
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value)
}

export function formatInrMinor(value: number) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value / 100)
}

export function formatCompact(value: number) {
  return new Intl.NumberFormat("en-IN", { notation: "compact", maximumFractionDigits: 1 }).format(value)
}
