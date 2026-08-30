# AI Defense Lab design notes

The twelve-screen story uses one visual grammar: defense cyan, attacker red, evidence amber, near-black surfaces, oversized editorial claims, and compact monospace telemetry. Each chapter has one claim, one evidence surface, and one short “What’s happening” explanation so the demo can be followed without presenter narration.

All displayed run values come from API payloads normalized at the browser trust boundary. Missing required evidence, malformed values, out-of-range metrics, or fixture-tagged production responses render an error instead of a substitute value; genuinely optional stream context is labeled as not recorded. Deterministic fixtures exist only in the isolated Playwright suite and are visibly watermarked.

Desktop uses a fixed one-line chapter rail and wide evidence layouts. Tablet and mobile collapse the same content into readable single-column flows, preserve horizontal exploration only inside bounded canvases, and keep interactive targets at least 44px. Keyboard focus, a skip link, semantic controls, WCAG AA contrast, overflow checks, and reduced-motion behavior are release assertions.

Motion is explanatory: the hero traces intent-to-payment drift, stream counters replay scored events, and the scope panel opens before revealing the fresh-row regression evidence. The panel states `independent_holdout=false`, identifies the previously disclosed families, and avoids novelty claims. The real-browser suite records start/mid/end states and verifies that reduced motion preserves the complete story without animation.

The build-time design jury uses seven separate structured OpenAI vision calls per screen—one independent critic for aesthetics, motion quality, clarity, relevance, memorability, performance, and accessibility. Every call is bound to real-server desktop, tablet, and mobile screenshot hashes plus measured Playwright evidence; cache-only builds fail on any missing or stale binding. No model call occurs in the browser.

Release evidence remains fail-closed until the V4 PASS terminal, exact historical lineage, active runtime bundle, and signed scope all verify. The frontend tests do not bypass that gate, and design-jury evidence must be regenerated whenever the scope copy changes.

## Local design-jury exception register — 2026-08-29

The current local jury artifact records 84 independent critics, a minimum score of 6, and `gate_passed=false`. This is intentionally visible rather than padded or hidden. The functional real-browser run is green at desktop, tablet, and mobile; the remaining below-8 findings are presentation and evidence-quality work: axe contrast nodes marked incomplete require manual composited-background verification, motion evidence covers the measured entrance/interaction paths but not every possible transition, and the bundle has not been profiled under throttled mobile conditions. Relevance scores also cannot honestly claim an independent holdout or live Mastercard outcome: the signed V4 evidence explicitly says `independent_holdout=false` and is a post-remediation regression.

This exception applies only to local review. No AWS deployment or production-readiness claim is made until the jury findings are either resolved and re-run or explicitly accepted by the reviewer with the underlying evidence retained.
